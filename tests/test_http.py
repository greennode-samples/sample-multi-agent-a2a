"""HTTP layer: authentication, validation, error handling, streaming, A2A server, readiness."""
import asyncio
import json
import threading
import time
from types import SimpleNamespace

import httpx
import pytest
import uvicorn
from starlette.routing import Route
from starlette.testclient import TestClient

import a2a_client
import agent
import main
import mcp_client

USER = "X-GreenNode-AgentBase-User-Id"
SESSION = "X-GreenNode-AgentBase-Session-Id"
IDENTITY = {USER: "alice", SESSION: "s-1"}
KEY = "k-" + "x" * 30
DELEGATIONS = [{"agent": "travel-buddy", "ok": True, "elapsed_ms": 12}]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Anything a test forgets to fake fails fast instead of calling GreenNode or an agent."""

    def offline(*args, **kwargs):
        raise RuntimeError("network disabled in tests")

    monkeypatch.setattr(mcp_client, "list_tools", offline)
    monkeypatch.setattr(agent, "DIRECTORY", a2a_client.AgentDirectory([]))  # no agent to call
    monkeypatch.setattr(agent, "_tools", [])


@pytest.fixture()
def client():
    return TestClient(main.app, raise_server_exceptions=False)


@pytest.fixture()
def turns(monkeypatch):
    """Replace the agent; `turns.calls` records (text, user, session, request id)."""
    state = SimpleNamespace(
        calls=[], reply="a reply", delegations=DELEGATIONS, fail=None, tokens=["a ", "reply"],
        events=[{"type": "delegate", "agent": "travel-buddy", "request": "plan"},
                {"type": "delegate_done", "agent": "travel-buddy", "ok": True, "elapsed_ms": 12}],
    )

    async def run_turn(text, user_id, session_id, *, request_id, callbacks=None):
        state.calls.append((text, user_id, session_id, request_id))
        if state.fail:
            raise state.fail
        return agent.TurnResult(state.reply, state.delegations)

    async def stream_turn(text, user_id, session_id, *, request_id, callbacks=None):
        state.calls.append((text, user_id, session_id, request_id))
        if state.fail:
            raise state.fail
        for event in state.events:
            yield event
        for token in state.tokens:
            yield token
        yield agent.TurnResult(state.reply, state.delegations)

    monkeypatch.setattr(agent, "run_turn", run_turn)
    monkeypatch.setattr(agent, "stream_turn", stream_turn)
    return state


def sse_events(response) -> list[dict]:
    return [json.loads(chunk[5:]) for chunk in response.text.split("\n\n") if chunk.startswith("data:")]


# --- authentication ----------------------------------------------------------------------------

PROTECTED = [("POST", "/invocations"), ("POST", "/a2a"), ("GET", "/ready"), ("POST", "/api/chat/stream")]
PUBLIC = ["/", "/health", "/api/info", "/.well-known/agent-card.json"]


@pytest.fixture()
def keyed(monkeypatch):
    monkeypatch.setattr(main, "AGENT_API_KEY", KEY)


@pytest.mark.parametrize(("method", "path"), PROTECTED)
def test_protected_paths_need_the_api_key(client, keyed, method, path):
    assert client.request(method, path, json={}).status_code == 401
    assert client.request(method, path, json={}, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.request(method, path, json={}, headers={"X-API-Key": KEY}).status_code != 401


@pytest.mark.parametrize("path", PUBLIC)
def test_ui_health_info_and_agent_card_stay_public(client, keyed, path):
    assert client.get(path).status_code == 200


def test_every_registered_route_is_either_protected_or_deliberately_public():
    paths = {r.path for r in main.app.routes if isinstance(r, Route)}
    assert {"/invocations", "/a2a", "/ready", "/api/chat/stream"} <= paths
    for path in paths:
        assert main._requires_key(path) or path in PUBLIC, f"{path} is neither protected nor listed as public"


def test_a_trailing_slash_is_not_a_way_around_the_key(client, keyed):
    assert main._requires_key("/a2a/") and main._requires_key("/api/chat/stream/") and not main._requires_key("/api/info/")
    assert client.post("/a2a/", json={}, follow_redirects=False).status_code in (307, 401)
    assert client.post("/a2a/", json={}).status_code == 401


def test_without_a_key_everything_is_open(client, monkeypatch):
    monkeypatch.setattr(main, "AGENT_API_KEY", "")
    assert client.get("/ready").status_code != 401


def test_the_key_is_compared_in_constant_time_and_non_ascii_does_not_crash(client, keyed, monkeypatch):
    seen = []
    real = main.hmac.compare_digest
    monkeypatch.setattr(main.hmac, "compare_digest", lambda a, b: seen.append((a, b)) or real(a, b))
    client.get("/ready", headers={"X-API-Key": KEY})
    assert seen == [(KEY.encode(), KEY.encode())]
    response = client.get("/ready", headers={"X-API-Key": "café".encode("latin-1")})
    assert response.status_code == 401


def test_api_info_hides_resource_ids_and_agents_from_unauthorized_callers(client, keyed, monkeypatch):
    monkeypatch.setattr(agent, "DIRECTORY", SimpleNamespace(status=lambda: [("TRAVEL", "travel-buddy"), ("RESTAURANT", None)]))
    public = client.get("/api/info").json()
    assert public == {"agent": "trip-concierge", "auth_required": True}
    full = client.get("/api/info", headers={"X-API-Key": KEY}).json()
    assert full["memory_id"] == "memory-test" and full["gateway"] == "https://gw.example"
    assert full["llm_model"] == "test-model"
    assert full["agents"] == [{"label": "TRAVEL", "name": "travel-buddy"}, {"label": "RESTAURANT", "name": None}]


def test_api_info_without_a_key_shows_everything(client, monkeypatch):
    monkeypatch.setattr(main, "AGENT_API_KEY", "")
    monkeypatch.setattr(agent, "DIRECTORY", SimpleNamespace(status=lambda: []))
    info = client.get("/api/info").json()
    assert info["auth_required"] is False and info["memory_id"] == "memory-test" and info["agents"] == []


# --- /invocations ------------------------------------------------------------------------------

def test_invocation_success(client, turns):
    r = client.post("/invocations", json={"message": "  hi  "}, headers=IDENTITY)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success" and body["agent"] == "trip-concierge" and body["response"] == "a reply"
    assert body["delegations"] == DELEGATIONS and body["timestamp"]
    assert turns.calls == [("hi", "alice", "s-1", body["request_id"])] and len(body["request_id"]) == 12


def test_every_turn_gets_its_own_request_id(client, turns):
    ids = {client.post("/invocations", json={"message": "hi"}, headers=IDENTITY).json()["request_id"] for _ in range(3)}
    assert len(ids) == 3


def test_invocation_accepts_input_as_an_alias(client, turns):
    assert client.post("/invocations", json={"input": "hello"}, headers=IDENTITY).status_code == 200


@pytest.mark.parametrize("headers", [{}, {USER: "alice"}, {SESSION: "s-1"}], ids=["none", "only-user", "only-session"])
def test_missing_identity_headers_are_a_400_and_never_reach_the_agent(client, turns, headers):
    for path in ("/invocations", "/api/chat/stream"):
        r = client.post(path, json={"message": "hi"}, headers=headers)
        assert r.status_code == 400 and USER in r.text
    assert turns.calls == []


@pytest.mark.parametrize("bad", ["a/b", "x y", "../etc", "a" * 129, "café"])
def test_malformed_user_or_session_ids_are_rejected(client, turns, bad):
    bad = bad.encode("latin-1")  # HTTP header values are bytes
    for headers in ({USER: bad, SESSION: "s-1"}, {USER: "alice", SESSION: bad}):
        assert client.post("/invocations", json={"message": "hi"}, headers=headers).status_code == 400
    assert turns.calls == []


@pytest.mark.parametrize("body", [{}, {"message": ""}, {"message": "   \n"}, {"message": 5}, {"message": None}, [1, 2], "text"])
def test_empty_or_malformed_messages_are_a_400(client, turns, body):
    assert client.post("/invocations", json=body, headers=IDENTITY).status_code == 400
    assert client.post("/api/chat/stream", json=body, headers=IDENTITY).status_code == 400
    assert turns.calls == []


def test_oversized_messages_are_a_400(client, turns):
    too_long = {"message": "x" * (main.MAX_MESSAGE_CHARS + 1)}
    assert client.post("/invocations", json=too_long, headers=IDENTITY).status_code == 400
    assert client.post("/api/chat/stream", json=too_long, headers=IDENTITY).status_code == 400
    assert client.post("/invocations", json={"message": "x" * main.MAX_MESSAGE_CHARS}, headers=IDENTITY).status_code == 200


def test_invalid_json_is_a_400(client):
    r = client.post("/api/chat/stream", content=b"{not json", headers={**IDENTITY, "Content-Type": "application/json"})
    assert r.status_code == 400


def test_agent_errors_return_a_generic_message_and_the_request_id_of_the_turn(client, turns, caplog):
    turns.fail = RuntimeError("secret-db-password leaked in a trace")
    r = client.post("/invocations", json={"message": "hi"}, headers=IDENTITY)
    assert r.status_code == 500
    assert "secret-db-password" not in r.text and "leaked in a trace" not in r.text
    request_id = r.json()["details"]["request_id"]
    assert r.json()["error"] == main.GENERIC_ERROR
    assert request_id == turns.calls[0][3]  # the id the delegation calls and logs carry
    assert request_id in caplog.text and "secret-db-password" in caplog.text  # the detail is logged, not returned


def test_whoami_is_disabled_by_default(client, monkeypatch):
    monkeypatch.setattr(main, "DEBUG_OPS", False)
    assert client.post("/invocations", json={"op": "whoami"}).status_code == 403


def test_whoami_when_enabled(client, monkeypatch):
    monkeypatch.setattr(main, "DEBUG_OPS", True)
    monkeypatch.setattr(agent, "whoami", lambda: {"token_sub": "svc"})
    r = client.post("/invocations", json={"op": "whoami"})
    assert r.status_code == 200 and r.json()["token_sub"] == "svc"


def test_whoami_failures_are_generic(client, monkeypatch):
    def boom():
        raise RuntimeError("secret")

    monkeypatch.setattr(main, "DEBUG_OPS", True)
    monkeypatch.setattr(agent, "whoami", boom)
    r = client.post("/invocations", json={"op": "whoami"})
    assert r.status_code == 500 and "secret" not in r.text and r.json()["details"]["request_id"]


# --- /api/chat/stream ----------------------------------------------------------------------------

def test_stream_sends_progress_tokens_then_done(client, turns):
    r = client.post("/api/chat/stream", json={"message": "hi"}, headers=IDENTITY)
    assert r.headers["content-type"].startswith("text/event-stream")
    request_id = turns.calls[0][3]
    assert sse_events(r) == [
        {"type": "delegate", "agent": "travel-buddy", "request": "plan"},
        {"type": "delegate_done", "agent": "travel-buddy", "ok": True, "elapsed_ms": 12},
        {"type": "token", "text": "a "}, {"type": "token", "text": "reply"},
        {"type": "done", "response": "a reply", "delegations": DELEGATIONS, "request_id": request_id},
    ]


def test_stream_setup_errors_end_the_stream_with_an_error_event(client, turns, caplog):
    turns.fail = RuntimeError("could not build the agent: secret")
    r = client.post("/api/chat/stream", json={"message": "hi"}, headers=IDENTITY)
    assert r.status_code == 200
    (event,) = sse_events(r)
    assert event["type"] == "error" and event["error"] == main.GENERIC_ERROR
    assert "secret" not in r.text and event["request_id"] == turns.calls[0][3] and event["request_id"] in caplog.text


# --- A2A server ----------------------------------------------------------------------------------

def rpc(text="hello", **message):
    return {"jsonrpc": "2.0", "id": 7, "method": "message/send",
            "params": {"message": {"parts": [{"kind": "text", "text": text}], **message}}}


def test_a2a_card_shape():
    card = main._a2a_card()
    assert card["name"] == "trip-concierge" and card["protocolVersion"] == "0.3.0"
    assert card["preferredTransport"] == "JSONRPC" and card["url"] == "/a2a"
    assert card["capabilities"]["streaming"] is True
    assert [s["id"] for s in card["skills"]] == ["trip-concierge"]
    for skill in card["skills"]:
        assert skill["name"] and skill["description"] and skill["tags"] and skill["examples"]
    assert "securitySchemes" not in card


def test_a2a_card_uses_the_public_url_and_declares_the_key_when_one_is_set(monkeypatch):
    monkeypatch.setattr(main, "A2A_PUBLIC_URL", "https://agent.example")
    monkeypatch.setattr(main, "AGENT_API_KEY", KEY)
    card = main._a2a_card()
    assert card["url"] == "https://agent.example/a2a"
    assert card["securitySchemes"] == {"apiKey": {"type": "apiKey", "in": "header", "name": "X-API-Key"}}
    assert card["security"] == [{"apiKey": []}] and KEY not in json.dumps(card)


def test_a2a_send(client, turns):
    r = client.post("/a2a", json=rpc(contextId="c-1"), headers={USER: "alice"})
    assert r.status_code == 200
    result = r.json()["result"]
    assert result["kind"] == "message" and result["role"] == "agent" and result["contextId"] == "c-1"
    assert result["parts"] == [{"kind": "text", "text": "a reply"}] and r.json()["id"] == 7
    (text, user, session, request_id) = turns.calls[0]
    assert (text, user, session) == ("hello", "alice", "c-1") and request_id


def test_a2a_needs_the_user_header_and_has_no_shared_user(client, turns):
    r = client.post("/a2a", json=rpc())
    assert r.status_code == 400 and r.json()["error"]["code"] == -32602
    assert turns.calls == []


@pytest.mark.parametrize(
    "body",
    [
        [1, 2],
        {"jsonrpc": "2.0", "id": 1, "method": "message/send", "params": []},
        {"jsonrpc": "2.0", "id": 1, "method": "message/send", "params": {"message": {"parts": ["text"]}}},
        {"jsonrpc": "2.0", "id": 1, "method": "message/send", "params": {"message": {"parts": "text"}}},
        {"jsonrpc": "2.0", "id": 1, "method": "message/send", "params": {"message": "text"}},
        {"jsonrpc": "2.0", "id": 1, "method": "message/send", "params": {"message": {"parts": []}}},
        {"jsonrpc": "2.0", "id": 1, "method": "message/send", "params": {"message": {"contextId": 5, "parts": [{"text": "x"}]}}},
        {"jsonrpc": "2.0", "id": 1, "method": "message/send", "params": {"message": {"contextId": "a/b", "parts": [{"text": "x"}]}}},
        {"jsonrpc": "2.0", "id": 1, "method": "message/send"},
        {"jsonrpc": "2.0", "id": 1, "method": "tasks/get", "params": {}},
    ],
    ids=["body-list", "params-list", "part-not-object", "parts-not-list", "message-not-object",
         "no-text", "context-id-int", "context-id-path", "no-params", "unknown-method"],
)
def test_malformed_a2a_requests_are_a_400_not_a_500(client, turns, body):
    r = client.post("/a2a", json=body, headers={USER: "alice"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] in (-32600, -32601, -32602)
    assert turns.calls == []


def test_a2a_unparseable_body_is_a_parse_error(client):
    r = client.post("/a2a", content=b"nope", headers={USER: "alice", "Content-Type": "application/json"})
    assert r.status_code == 400 and r.json()["error"]["code"] == -32700


def test_a2a_oversized_text_is_rejected(client, turns):
    r = client.post("/a2a", json=rpc("x" * (main.MAX_MESSAGE_CHARS + 1)), headers={USER: "alice"})
    assert r.status_code == 400 and turns.calls == []


def test_a2a_context_falls_back_to_the_session_header_then_a_fresh_id(client, turns):
    client.post("/a2a", json=rpc(), headers={USER: "alice", SESSION: "from-header"})
    client.post("/a2a", json=rpc(), headers={USER: "alice"})
    assert turns.calls[0][2] == "from-header" and turns.calls[1][2].startswith("a2a-")


def test_a2a_send_errors_are_generic_and_carry_the_request_id(client, turns):
    turns.fail = RuntimeError("secret")
    r = client.post("/a2a", json=rpc(), headers={USER: "alice"})
    assert r.status_code == 500 and r.json()["error"]["code"] == -32603 and "secret" not in r.text
    assert r.json()["error"]["data"]["request_id"] == turns.calls[0][3]


def test_a2a_stream_events_follow_the_spec_shape_and_show_the_delegation_progress(client, turns):
    r = client.post("/a2a", json={**rpc(contextId="c-2"), "method": "message/stream"}, headers={USER: "alice"})
    events = [e["result"] for e in sse_events(r)]
    assert [(e["kind"], e.get("status", {}).get("state")) for e in events] == [
        ("status-update", "working"),
        ("status-update", "working"), ("status-update", "working"),  # delegate, delegate_done
        ("artifact-update", None), ("artifact-update", None),
        ("artifact-update", None), ("status-update", "completed"),
    ]
    for event in events:
        assert event["contextId"] == "c-2" and event["taskId"].startswith("task-") and "task" not in event
    assert events[0]["final"] is False and events[-1]["final"] is True and "message" not in events[0]["status"]
    progress = [e["status"]["message"] for e in events[1:3]]
    assert [m["parts"][0]["text"] for m in progress] == ["Asking travel-buddy", "travel-buddy answered"]
    assert all(m["kind"] == "message" and m["role"] == "agent" for m in progress)
    tokens = events[3:5]
    assert [(e["append"], e["lastChunk"]) for e in tokens] == [(False, False), (True, False)]
    last = events[5]
    assert last["lastChunk"] is True and last["append"] is False
    assert last["artifact"] == {"artifactId": "reply", "parts": [{"kind": "text", "text": "a reply"}]}


def test_a2a_stream_does_not_show_the_partial_text_of_another_agent(client, turns):
    turns.events = [{"type": "delegate_text", "agent": "travel-buddy", "text": "private partial", "append": False}]
    r = client.post("/a2a", json={**rpc(), "method": "message/stream"}, headers={USER: "alice"})
    assert "private partial" not in r.text


def test_a2a_stream_failure_is_a_failed_task_with_a_message_object(client, turns):
    turns.fail = RuntimeError("secret")
    r = client.post("/a2a", json={**rpc(), "method": "message/stream"}, headers={USER: "alice"})
    final = [e["result"] for e in sse_events(r)][-1]
    assert final["final"] is True and final["status"]["state"] == "failed"
    message = final["status"]["message"]
    assert message["kind"] == "message" and message["role"] == "agent" and "secret" not in r.text
    assert f"(request {turns.calls[0][3]})" in message["parts"][0]["text"]


@pytest.mark.parametrize(
    ("event", "text"),
    [
        ({"type": "delegate", "agent": "a"}, "Asking a"),
        ({"type": "delegate_done", "agent": "a", "ok": True}, "a answered"),
        ({"type": "delegate_done", "agent": "a", "ok": False}, "a failed"),
        ({"type": "tool", "name": "top_gainers"}, "Calling top_gainers"),
        ({"type": "delegate_text", "agent": "a", "text": "x"}, None),
        ({"type": "unknown"}, None),
    ],
)
def test_progress_text(event, text):
    assert main._progress_text(event) == text


def test_a2a_msg_result_envelope():
    out = main._a2a_msg_result("req-1", "ctx-9", "Da Lat is lovely")
    assert out["jsonrpc"] == "2.0" and out["id"] == "req-1"
    assert out["result"]["parts"] == [{"kind": "text", "text": "Da Lat is lovely"}] and out["result"]["messageId"]


def test_a2a_text_helper():
    parts = {"message": {"parts": [{"kind": "text", "text": "xin "}, {"text": "chao"}, {"kind": "file", "file": {}}]}}
    assert main._a2a_text(parts) == "xin chao"
    assert main._a2a_text(None) == main._a2a_text({}) == main._a2a_text({"message": {}}) == ""


def test_a2a_ctx_helper():
    assert main._a2a_ctx({"params": {"message": {"contextId": "ctx-x"}}}, "hdr") == "ctx-x"
    assert main._a2a_ctx({}, "hdr") == "hdr"
    assert main._a2a_ctx({}, "").startswith("a2a-")


class ServedApp:
    """The real orchestrator app on 127.0.0.1, so its A2A server can be called by the A2A client."""

    def __enter__(self):
        config = uvicorn.Config(main.app, host="127.0.0.1", port=0, log_level="warning", lifespan="off")
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        while not self.server.started:
            time.sleep(0.01)
        self.url = f"http://127.0.0.1:{self.server.servers[0].sockets[0].getsockname()[1]}"
        return self

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(10)


def test_another_orchestrator_can_compose_this_one_over_a2a(turns, monkeypatch):
    """The card and the shapes this agent serves are what the A2A client of the orchestrator reads."""
    monkeypatch.setattr(main, "AGENT_API_KEY", KEY)
    turns.reply = "Itinerary, booking and stocks: all done."
    with ServedApp() as served:
        remote = a2a_client.fetch_agent(a2a_client.AgentConfig("CONCIERGE", served.url, KEY))
        assert remote.name == "trip-concierge" and remote.streaming and remote.rpc_url == f"{served.url}/a2a"

        def delegate(on_text=None):
            return asyncio.run(a2a_client.delegate(
                remote, "Plan, book and report", user_id="alice", context_id="s1:trip-concierge",
                request_id="r1", timeout=10, on_text=on_text,
            ))

        assert delegate() == "Itinerary, booking and stocks: all done."
        chunks = []
        assert delegate(lambda text, append: chunks.append((text, append))) == "Itinerary, booking and stocks: all done."
        assert chunks[-1] == ("Itinerary, booking and stocks: all done.", False) and chunks[0][1] is False
        assert [c[1] for c in turns.calls] == ["alice", "alice"] and {c[2] for c in turns.calls} == {"s1:trip-concierge"}
        wrong = a2a_client.RemoteAgent(a2a_client.AgentConfig("X", served.url, "wrong"), "x", "", remote.rpc_url, False)
        with pytest.raises(a2a_client.A2AError, match="HTTP 401"):
            asyncio.run(a2a_client.delegate(wrong, "hi", user_id="a", context_id="c", request_id="r", timeout=5))


# --- /ready --------------------------------------------------------------------------------------

@pytest.fixture()
def ready_deps(monkeypatch):
    state = SimpleNamespace(memory_error=None, mcp_tools=[object()], remote=[SimpleNamespace(name="travel-buddy")],
                            status=[("TRAVEL", "travel-buddy")])

    async def ping():
        if state.memory_error:
            raise state.memory_error

    monkeypatch.setattr(agent, "ping_memory", ping)
    monkeypatch.setattr(agent, "get_mcp_tools", lambda: state.mcp_tools)
    monkeypatch.setattr(agent, "get_remote_tools", lambda: state.remote)
    monkeypatch.setattr(agent, "DIRECTORY", SimpleNamespace(status=lambda: state.status))
    return state


def test_ready_when_everything_is_up(client, ready_deps):
    r = client.get("/ready")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert r.json()["checks"]["agents"] == {"ok": True, "available": ["travel-buddy"], "unavailable": []}
    assert r.json()["checks"]["mcp"] == {"ok": True, "tools": 1}


def test_ready_reflects_the_recovery_of_the_mcp_tools_and_of_an_agent(client, ready_deps):
    ready_deps.mcp_tools = []
    ready_deps.remote, ready_deps.status = [], [("TRAVEL", None)]
    degraded = client.get("/ready")
    assert degraded.status_code == 503 and degraded.json()["checks"]["mcp"] == {"ok": False, "tools": 0}
    assert degraded.json()["checks"]["agents"] == {"ok": False, "available": [], "unavailable": ["TRAVEL"]}
    ready_deps.mcp_tools = [object()]
    ready_deps.remote, ready_deps.status = [SimpleNamespace(name="travel-buddy")], [("TRAVEL", "travel-buddy")]
    assert client.get("/ready").status_code == 200


def test_ready_reports_a_memory_outage_without_details(client, ready_deps):
    ready_deps.memory_error = RuntimeError("secret")
    r = client.get("/ready")
    assert r.status_code == 503 and r.json()["checks"]["memory"] == {"ok": False, "error": "RuntimeError"}
    assert "secret" not in r.text


# --- no route blocks the event loop ----------------------------------------------------------------

@pytest.mark.parametrize("slow", ["memory", "mcp", "agents"])
def test_slow_dependencies_do_not_stall_other_requests(monkeypatch, slow):
    async def slow_ping():
        await asyncio.sleep(0.4)

    monkeypatch.setattr(agent, "ping_memory", slow_ping if slow == "memory" else (lambda: asyncio.sleep(0)))
    monkeypatch.setattr(agent, "get_mcp_tools", lambda: (time.sleep(0.4 if slow == "mcp" else 0), [object()])[1])  # blocking calls
    monkeypatch.setattr(agent, "get_remote_tools", lambda: (time.sleep(0.4 if slow == "agents" else 0), [])[1])
    monkeypatch.setattr(agent, "DIRECTORY", SimpleNamespace(status=lambda: []))

    async def probe():
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            started = time.monotonic()

            async def info():
                await asyncio.sleep(0.05)
                await http.get("/api/info")
                return time.monotonic() - started

            _, info_latency = await asyncio.gather(http.get("/ready"), info())
            return info_latency

    assert asyncio.run(probe()) < 0.3


# --- Langfuse helpers: tracing off by default -------------------------------------------------------

def test_tracing_is_off_without_langfuse_env():
    assert main._lf_tracing() is False and main._lf_callback() is None
    with main._traced("t", "u", "s", ["x"]) as callbacks:
        assert callbacks == []
