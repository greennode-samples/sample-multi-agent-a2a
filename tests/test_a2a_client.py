"""a2a_client: agent cards, discovery, message/send and message/stream, errors, retries, credentials.

Stub agents on 127.0.0.1 have the shapes of the sample agents; httpx.MockTransport covers the
failures a real server cannot produce on demand.
"""
import asyncio
import json
import time
from types import SimpleNamespace

import httpx
import pytest
from starlette.responses import JSONResponse, Response, StreamingResponse

import a2a_client
import mcp_client
from a2a_client import A2AError, AgentConfig, RemoteAgent
from fakes import rpc_message

BASE = "http://agent.test"


def card(**overrides) -> dict:
    return {
        "name": "travel-buddy", "description": "Plans trips.", "url": "/a2a", "protocolVersion": "0.3.0",
        "capabilities": {"streaming": True}, "preferredTransport": "JSONRPC",
        "skills": [{"id": "s", "name": "Planning", "description": "Itineraries.", "tags": [], "examples": ["Plan Hue"]}],
        **overrides,
    }


def remote(streaming: bool = False, **config) -> RemoteAgent:
    cfg = AgentConfig("TRAVEL", BASE, **config)
    return RemoteAgent(cfg, "travel-buddy", "Plans trips.", f"{BASE}/a2a", streaming)


def call(agent: RemoteAgent, text: str = "hi", *, timeout: float = 5, on_text=None, user: str = "alice",
         context_id: str = "s1:travel-buddy", request_id: str = "req-1") -> str:
    return asyncio.run(a2a_client.delegate(
        agent, text, user_id=user, context_id=context_id, request_id=request_id, timeout=timeout, on_text=on_text,
    ))


@pytest.fixture()
def http(monkeypatch):
    """Route a2a_client's httpx clients through MockTransport. `http.respond(fn)` sets
    fn(request, attempt) -> Response (it may raise); `http.requests` records the requests."""
    state = {"handler": None}
    requests: list[httpx.Request] = []
    real_async, real_sync = httpx.AsyncClient, httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return state["handler"](request, len(requests))

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_async(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(httpx, "Client", lambda **kw: real_sync(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(a2a_client, "_RETRY_BASE_DELAY", 0)
    monkeypatch.setattr(a2a_client, "_MAX_RETRY_AFTER_SECONDS", 0)
    api = SimpleNamespace(requests=requests, respond=lambda fn: state.update(handler=fn))
    api.body = lambda i=-1: json.loads(requests[i].content)
    return api


def reply(text: str = "ok", rid="1") -> httpx.Response:
    return httpx.Response(200, json=rpc_message(rid, "ctx", text))


# --- agent cards ---------------------------------------------------------------------------------

def test_a_card_becomes_a_remote_agent():
    agent = a2a_client.agent_from_card(AgentConfig("TRAVEL", BASE), card())
    assert agent.name == "travel-buddy" and agent.rpc_url == f"{BASE}/a2a" and agent.streaming is True
    assert agent.description.splitlines() == ['Plans trips.', '- Planning: Itineraries. Examples: "Plan Hue"']


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("travel-buddy", "travel-buddy"), ("Travel Buddy!", "Travel_Buddy"), ("zalo.restaurant/bot", "zalo_restaurant_bot"),
     ("Quán Ngon", "Qu_n_Ngon"), ("x" * 100, "x" * 64), ("--a--", "a")],
)
def test_the_card_name_becomes_a_valid_tool_name(raw, expected):
    assert a2a_client.agent_from_card(AgentConfig("T", BASE), card(name=raw)).name == expected


@pytest.mark.parametrize("bad", [None, "", "  ", "___", 5, ["x"]])
def test_a_card_without_a_usable_name_is_rejected(bad):
    with pytest.raises(A2AError, match="usable name"):
        a2a_client.agent_from_card(AgentConfig("T", BASE), card(name=bad))


@pytest.mark.parametrize("bad", [None, [], "text", 5])
def test_a_card_that_is_not_an_object_is_rejected(bad):
    with pytest.raises(A2AError, match="not a JSON object"):
        a2a_client.agent_from_card(AgentConfig("T", BASE), bad)


def test_a_card_for_another_transport_is_rejected():
    with pytest.raises(A2AError, match="GRPC"):
        a2a_client.agent_from_card(AgentConfig("T", BASE), card(preferredTransport="GRPC"))


def test_a_card_without_a_transport_defaults_to_jsonrpc():
    card_without = card()
    del card_without["preferredTransport"]
    assert a2a_client.agent_from_card(AgentConfig("T", BASE), card_without).rpc_url == f"{BASE}/a2a"


@pytest.mark.parametrize("url", [None, "", "  ", 5])
def test_a_card_without_a_url_is_rejected(url):
    with pytest.raises(A2AError, match="no url"):
        a2a_client.agent_from_card(AgentConfig("T", BASE), card(url=url))


@pytest.mark.parametrize(
    ("base", "card_url", "rpc_url"),
    [
        ("http://h:8080", "/a2a", "http://h:8080/a2a"),
        ("https://h", "https://h/a2a", "https://h/a2a"),
        ("https://h", "https://H:443/a2a", "https://H:443/a2a"),  # same origin: host case and default port
        ("https://h/prefix", "https://h/prefix/a2a", "https://h/prefix/a2a"),
        ("https://h/prefix", "a2a", "https://h/prefix/a2a"),
    ],
)
def test_the_rpc_url_is_resolved_against_the_configured_url(base, card_url, rpc_url):
    assert a2a_client.agent_from_card(AgentConfig("T", base), card(url=card_url)).rpc_url == rpc_url


@pytest.mark.parametrize("url", ["https://evil.example/a2a", "http://agent.test:9999/a2a", "https://agent.test/a2a", "//evil.example/a2a"])
def test_the_rpc_url_must_stay_on_the_configured_origin(url):
    """The API key and the IAM token go to the RPC url: a card must not be able to redirect them."""
    with pytest.raises(A2AError, match="not on the configured origin"):
        a2a_client.agent_from_card(AgentConfig("T", BASE), card(url=url))


def test_card_text_is_bounded_and_flattened():
    noisy = card(
        description="A" * 5000,
        skills=[{"name": f"skill {i}\nwith newline", "description": "d" * 1000, "examples": ["e" * 500] * 5} for i in range(20)]
        + ["not a skill", {"name": "", "description": ""}],
    )
    description = a2a_client.agent_from_card(AgentConfig("T", BASE), noisy).description
    assert len(description) <= a2a_client._MAX_DESCRIPTION_CHARS
    assert "\n- skill 0 with newline" in description and "skill 6" not in description  # at most 6 skills
    assert description.count("Examples:") <= 6


def test_a_card_with_only_a_name_still_works():
    agent = a2a_client.agent_from_card(AgentConfig("T", BASE), {"name": "bare", "url": "/a2a"})
    assert agent.description == "" and agent.streaming is False


@pytest.mark.parametrize("capabilities", [None, [], {}, {"streaming": "yes"}, {"streaming": False}])
def test_streaming_needs_an_explicit_true(capabilities):
    assert a2a_client.agent_from_card(AgentConfig("T", BASE), card(capabilities=capabilities)).streaming is False


def test_the_api_key_is_not_in_the_repr_of_the_config():
    assert "secret-key" not in repr(AgentConfig("T", BASE, "secret-key")) + repr(remote(api_key="secret-key"))


# --- fetching cards ------------------------------------------------------------------------------

def test_fetch_agent_reads_the_card_of_a_real_agent(stub_agents):
    stub = stub_agents("travel-buddy")
    agent = a2a_client.fetch_agent(AgentConfig("TRAVEL", stub.url))
    assert agent.name == "travel-buddy" and agent.rpc_url == f"{stub.url}/a2a" and agent.streaming


def test_fetch_agent_sends_the_credentials(http, monkeypatch):
    monkeypatch.setattr(mcp_client, "get_token", lambda force=False: "iam-token")
    http.respond(lambda request, n: httpx.Response(200, json=card()))
    a2a_client.fetch_agent(AgentConfig("T", BASE, "key-1", iam_auth=True))
    request = http.requests[0]
    assert str(request.url) == f"{BASE}/.well-known/agent-card.json"
    assert request.headers["x-api-key"] == "key-1" and request.headers["authorization"] == "Bearer iam-token"


def test_fetch_agent_sends_no_credentials_when_there_are_none(http):
    http.respond(lambda request, n: httpx.Response(200, json=card()))
    a2a_client.fetch_agent(AgentConfig("T", BASE))
    assert "x-api-key" not in http.requests[0].headers and "authorization" not in http.requests[0].headers


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(404), "HTTP 404"),
        (httpx.Response(503), "HTTP 503"),
        (httpx.Response(302, headers={"Location": "https://evil.example/"}), "HTTP 302"),  # redirects are not followed
        (httpx.Response(200, text="<html>"), "not valid JSON"),
        (httpx.Response(200, json=["x"]), "not a JSON object"),
    ],
)
def test_fetch_agent_failures_are_a2a_errors(http, response, message):
    http.respond(lambda request, n: response)
    with pytest.raises(A2AError, match=message):
        a2a_client.fetch_agent(AgentConfig("T", BASE))
    assert len(http.requests) == 1


def test_fetch_agent_when_the_agent_is_down():
    with pytest.raises(A2AError, match="could not be fetched"):
        a2a_client.fetch_agent(AgentConfig("T", "http://127.0.0.1:9"))


def test_fetch_agent_without_an_iam_token(monkeypatch):
    def no_credentials(force=False):
        raise RuntimeError("GREENNODE_CLIENT_ID / GREENNODE_CLIENT_SECRET are not set")

    monkeypatch.setattr(mcp_client, "get_token", no_credentials)
    with pytest.raises(A2AError, match="could not get an IAM token .*GREENNODE_CLIENT_ID"):
        a2a_client.fetch_agent(AgentConfig("T", BASE, iam_auth=True))


def test_an_iam_outage_is_an_a2a_error_that_does_not_leak_the_response(monkeypatch):
    def iam_down(force=False):
        raise httpx.HTTPStatusError("401 secret-iam-detail", request=None, response=None)

    monkeypatch.setattr(mcp_client, "get_token", iam_down)
    with pytest.raises(A2AError) as info:
        a2a_client.fetch_agent(AgentConfig("T", BASE, iam_auth=True))
    assert "HTTPStatusError" in str(info.value) and "secret-iam-detail" not in str(info.value)


# --- the agent directory -------------------------------------------------------------------------

class Clock:
    def __init__(self):
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now


@pytest.fixture()
def directory(monkeypatch):
    """A directory over two agents whose card fetch is scripted: outcomes[label] is a list of
    results, a RemoteAgent or an exception, consumed one per fetch."""
    clock = Clock()
    monkeypatch.setattr(a2a_client, "time", clock)
    outcomes: dict[str, list] = {"A": [], "B": []}
    fetches: list[str] = []

    def fetch(config):
        fetches.append(config.label)
        outcome = outcomes[config.label].pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(a2a_client, "fetch_agent", fetch)
    configs = [AgentConfig("A", "http://a"), AgentConfig("B", "http://b")]
    api = SimpleNamespace(
        clock=clock, outcomes=outcomes, fetches=fetches, configs=configs,
        directory=a2a_client.AgentDirectory(configs, ttl=600, retry=30),
        agent=lambda label, name=None: RemoteAgent(next(c for c in configs if c.label == label), name or label.lower(), "d", "http://x/a2a", False),
    )
    return api


def test_all_agents_load_and_are_cached_until_the_ttl(directory):
    directory.outcomes["A"] += [directory.agent("A"), directory.agent("A")]
    directory.outcomes["B"] += [directory.agent("B"), directory.agent("B")]
    assert [a.name for a in directory.directory.get()] == ["a", "b"]
    directory.clock.now += 599
    directory.directory.get()
    assert directory.fetches == ["A", "B"]  # still cached
    directory.clock.now += 2
    directory.directory.get()
    assert directory.fetches == ["A", "B", "A", "B"]


def test_a_failing_card_is_skipped_and_retried_after_the_retry_delay_never_cached(directory, caplog):
    directory.outcomes["A"] += [directory.agent("A")]
    directory.outcomes["B"] += [A2AError("the agent card request returned HTTP 503"), A2AError("still down"), directory.agent("B")]
    with caplog.at_level("WARNING", logger="a2a-client"):
        assert [a.name for a in directory.directory.get()] == ["a"]
    assert "agent B: the agent card request returned HTTP 503" in caplog.text and "retrying in 30 s" in caplog.text
    assert directory.directory.status() == [("A", "a"), ("B", None)]
    directory.clock.now += 29  # not yet: a dead agent must not slow every turn down
    directory.directory.get()
    assert directory.fetches == ["A", "B"]
    directory.clock.now += 2  # retried, fails again
    assert [a.name for a in directory.directory.get()] == ["a"]
    assert directory.fetches == ["A", "B", "B"]
    directory.clock.now += 31  # recovered
    assert [a.name for a in directory.directory.get()] == ["a", "b"]
    assert directory.directory.status() == [("A", "a"), ("B", "b")]


def test_a_failed_refresh_keeps_the_previous_card(directory, caplog):
    directory.outcomes["A"] += [directory.agent("A", "old"), A2AError("down"), directory.agent("A", "new")]
    directory.outcomes["B"] += [directory.agent("B")]
    directory.directory.get()
    directory.clock.now += 601
    with caplog.at_level("WARNING", logger="a2a-client"):
        assert [a.name for a in directory.directory.get()] == ["old", "b"]
    assert "keeping the previous card" in caplog.text
    directory.clock.now += 31
    assert [a.name for a in directory.directory.get()] == ["new", "b"]


def test_an_unexpected_error_in_one_card_does_not_break_the_others(directory, caplog):
    directory.outcomes["A"] += [KeyError("bug")]
    directory.outcomes["B"] += [directory.agent("B")]
    with caplog.at_level("WARNING", logger="a2a-client"):
        assert [a.name for a in directory.directory.get()] == ["b"]
    assert "unexpected error while loading the card" in caplog.text and "KeyError" in caplog.text


def test_the_cards_load_in_parallel(stub_agents):
    stubs = [stub_agents(f"agent-{i}") for i in range(3)]
    for stub in stubs:
        original = stub.card
        stub.card = lambda original=original: (time.sleep(0.4), original())[1]
    directory = a2a_client.AgentDirectory([AgentConfig(f"A{i}", s.url) for i, s in enumerate(stubs)])
    started = time.monotonic()
    assert len(directory.get()) == 3
    assert time.monotonic() - started < 1.0  # three cards of 0.4 s one after the other would take 1.2 s


def test_status_needs_no_network(directory):
    assert directory.directory.status() == [("A", None), ("B", None)]
    assert directory.fetches == []


# --- message/send ----------------------------------------------------------------------------------

def test_send_returns_the_text_of_a_message_reply(stub_agents):
    stub = stub_agents("travel-buddy")
    agent = a2a_client.fetch_agent(AgentConfig("TRAVEL", stub.url))
    assert call(agent, "Plan 2 days in Da Nang") == "travel-buddy answers: Plan 2 days in Da Nang"
    (recorded,) = stub.calls
    message = recorded.body["params"]["message"]
    assert recorded.body["jsonrpc"] == "2.0" and recorded.body["method"] == "message/send"
    assert message["kind"] == "message" and message["role"] == "user" and message["messageId"]
    assert message["contextId"] == "s1:travel-buddy"
    assert message["parts"] == [{"kind": "text", "text": "Plan 2 days in Da Nang"}]
    assert recorded.body["params"]["configuration"] == {"blocking": True, "acceptedOutputModes": ["text/plain"]}


def test_the_user_id_the_request_id_and_the_key_are_sent_as_headers(stub_agents):
    stub = stub_agents("zalo-restaurant-bot", api_key="k" * 20)
    agent = a2a_client.fetch_agent(AgentConfig("ZALO", stub.url, "k" * 20))
    call(agent, "book", user="guest-42", request_id="abc123", context_id="s5:zalo")
    headers = stub.calls[0].headers
    assert headers["x-greennode-agentbase-user-id"] == "guest-42"
    assert headers["x-greennode-agentbase-session-id"] == "s5:zalo"  # the contextId, like the platform's session header
    assert headers["x-greennode-agentbase-request-id"] == "abc123"
    assert headers["x-api-key"] == "k" * 20 and "authorization" not in headers


def test_the_same_context_id_is_sent_every_time_and_each_message_has_its_own_id(stub_agents):
    stub = stub_agents()
    agent = a2a_client.fetch_agent(AgentConfig("TRAVEL", stub.url))
    call(agent, "one", context_id="s1:travel-buddy")
    call(agent, "two", context_id="s1:travel-buddy")
    ids = [c.body["params"]["message"] for c in stub.calls]
    assert {m["contextId"] for m in ids} == {"s1:travel-buddy"} and len({m["messageId"] for m in ids}) == 2


def test_a_wrong_key_is_reported_with_a_hint(stub_agents):
    stub = stub_agents(api_key="right-key-right-key")
    agent = a2a_client.fetch_agent(AgentConfig("TRAVEL", stub.url, "wrong"))
    with pytest.raises(A2AError, match=r"HTTP 401: the agent refused the credentials"):
        call(agent)
    assert len(stub.calls) == 1


def task(state="completed", artifacts=None, message=None, **extra) -> dict:
    result = {"kind": "task", "id": "t1", "contextId": "c", "status": {"state": state}, **extra}
    if artifacts is not None:
        result["artifacts"] = artifacts
    if message is not None:
        result["status"]["message"] = {"kind": "message", "role": "agent", "messageId": "m", "parts": [{"kind": "text", "text": message}]}
    return result


def art(*texts, id_="a") -> dict:
    return {"artifactId": id_, "parts": [{"kind": "text", "text": t} for t in texts]}


def send(http, result) -> str:
    http.respond(lambda request, n: httpx.Response(200, json={"jsonrpc": "2.0", "id": "1", "result": result}))
    return call(remote())


def test_a_task_result_is_read_from_its_artifacts(http):
    assert send(http, task(artifacts=[art("Day 1: beach.", " Day 2: hills."), art("Budget: 5M", id_="b")])) == (
        "Day 1: beach. Day 2: hills.\n\nBudget: 5M"
    )


def test_a_task_without_artifacts_answers_with_its_status_message(http):
    assert send(http, task(message="Done: booked.")) == "Done: booked."


def test_artifacts_win_over_the_status_message(http):
    assert send(http, task(artifacts=[art("the artifact")], message="the status")) == "the artifact"


def test_data_parts_are_returned_as_json_and_file_parts_are_ignored(http):
    parts = [
        {"kind": "data", "data": {"price": 12.5, "name": "Quán"}},
        {"kind": "file", "file": {"uri": "https://x/y.png"}},
        {"kind": "text", "text": " tail"},
    ]
    assert send(http, task(artifacts=[{"artifactId": "a", "parts": parts}])) == '{"price": 12.5, "name": "Quán"} tail'


def test_input_required_is_passed_on_as_a_question(http):
    assert send(http, task("input-required", message="Which date?")) == "INPUT_REQUIRED: Which date?"


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (task("failed", message="The agent could not complete the request."), "the task failed: The agent could not complete"),
        (task("rejected"), "the task rejected"),
        (task("canceled"), "the task canceled"),
        (task("auth-required"), "needs additional authentication"),
        (task("working"), "did not finish .state: working"),
        (task("submitted"), "did not finish .state: submitted"),
        (task("unknown"), "did not finish .state: unknown"),
        ({"kind": "task", "id": "t", "contextId": "c", "status": {}}, "state: missing"),
        (task("completed"), "returned no text"),
        (task("completed", artifacts=[art("  ")]), "returned no text"),
        ({"kind": "message", "parts": []}, "returned no text"),
        ({"kind": "message", "parts": [{"kind": "file", "file": {}}]}, "returned no text"),
        ({"kind": "something-else"}, "unexpected result"),
        ("just text", "unexpected result"),
        (None, "unexpected result"),
    ],
)
def test_results_that_are_not_an_answer_are_errors(http, result, message):
    with pytest.raises(A2AError, match=message):
        send(http, result)


def test_the_reply_is_capped(http):
    out = send(http, rpc_message("1", "c", "X" * 20_000)["result"])
    assert out.startswith("X" * mcp_client.MAX_TOOL_OUTPUT_CHARS) == (len(out) > mcp_client.MAX_TOOL_OUTPUT_CHARS)
    assert out.endswith(f"...[truncated {20_000 - mcp_client.MAX_TOOL_OUTPUT_CHARS} chars]")
    assert len(out) < mcp_client.MAX_TOOL_OUTPUT_CHARS + 60


def test_a_reply_at_the_limit_is_not_truncated(http):
    text = "Y" * mcp_client.MAX_TOOL_OUTPUT_CHARS
    assert send(http, rpc_message("1", "c", text)["result"]) == text


# --- errors ----------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("response", "message"),
    [
        # what the sample agents answer: HTTP 400 / 500 with a JSON-RPC error object
        (httpx.Response(400, json={"jsonrpc": "2.0", "id": 1, "error": {"code": -32602, "message": "The message is empty."}}),
         "HTTP 400: The message is empty."),
        (httpx.Response(500, json={"jsonrpc": "2.0", "id": 1, "error": {"code": -32603, "message": "The agent could not complete the request. Please try again.", "data": {"request_id": "ab12cd"}}}),
         r"HTTP 500: The agent could not complete the request. Please try again. \(request ab12cd\)"),
        (httpx.Response(401, json={"status": "error", "error": "Unauthorized: missing or invalid X-API-Key header."}),
         "HTTP 401: the agent refused the credentials"),
        (httpx.Response(403, text="forbidden"), "HTTP 403: the agent refused the credentials"),
        (httpx.Response(404, text="nope"), "HTTP 404: no A2A endpoint"),
        (httpx.Response(502, text="<html>Bad gateway secret-proxy-detail</html>"), "^HTTP 502$"),
        (httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "error": {"code": -32000, "message": "Task not found"}}),
         "returned an error: Task not found"),
        (httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "error": "plain"}), "returned an error: no message"),
        (httpx.Response(200, text="not json"), "not JSON"),
        (httpx.Response(200, json=[1, 2]), "not a JSON-RPC response"),
        (httpx.Response(200, json={"jsonrpc": "2.0", "id": 1}), "unexpected result"),
    ],
)
def test_error_responses_become_short_a2a_errors(http, response, message):
    http.respond(lambda request, n: response)
    with pytest.raises(A2AError, match=message) as info:
        call(remote())
    assert "secret-proxy-detail" not in str(info.value)


def test_the_error_message_of_a_remote_agent_is_bounded_and_flattened(http):
    http.respond(lambda request, n: httpx.Response(500, json={"error": {"message": "line one\nline two " + "z" * 1000}}))
    with pytest.raises(A2AError) as info:
        call(remote())
    assert "\n" not in str(info.value) and len(str(info.value)) < 260


def test_a_slow_agent_times_out_and_is_not_retried(stub_agents):
    stub = stub_agents()
    stub.delay = 2
    agent = a2a_client.fetch_agent(AgentConfig("TRAVEL", stub.url))
    with pytest.raises(A2AError, match=r"did not answer within 0\.3 s"):
        call(agent, timeout=0.3)
    assert len(stub.calls) == 1  # message/send is not idempotent


def test_a_read_timeout_is_not_retried(http):
    def hang(request, n):
        raise httpx.ReadTimeout("slow", request=request)

    http.respond(hang)
    with pytest.raises(A2AError, match="did not answer"):
        call(remote())
    assert len(http.requests) == 1


def test_server_errors_are_not_retried(http):
    """A 5xx may come after the agent did the work (a booking): repeating it could do it twice."""
    http.respond(lambda request, n: httpx.Response(503, text="down"))
    with pytest.raises(A2AError, match="HTTP 503"):
        call(remote())
    assert len(http.requests) == 1


def test_a_connection_that_cannot_be_opened_is_retried(http):
    def flaky(request, n):
        if n < 3:
            raise httpx.ConnectError("refused", request=request)
        return reply("finally")

    http.respond(flaky)
    assert call(remote()) == "finally"
    assert len(http.requests) == 3


def test_connection_failures_stop_after_three_attempts(http):
    def refused(request, n):
        raise httpx.ConnectError("refused", request=request)

    http.respond(refused)
    with pytest.raises(A2AError, match="could not be reached"):
        call(remote())
    assert len(http.requests) == 3


def test_a_connect_timeout_is_retried_too(http):
    def flaky(request, n):
        if n == 1:
            raise httpx.ConnectTimeout("slow", request=request)
        return reply()

    http.respond(flaky)
    assert call(remote()) == "ok" and len(http.requests) == 2


def test_rate_limiting_is_retried_because_the_agent_did_not_process_the_request(http):
    http.respond(lambda request, n: httpx.Response(429, headers={"Retry-After": "2"}) if n < 3 else reply("after the wait"))
    assert call(remote()) == "after the wait"
    assert len(http.requests) == 3


def test_endless_rate_limiting_ends_in_an_error(http):
    http.respond(lambda request, n: httpx.Response(429))
    with pytest.raises(A2AError, match="HTTP 429: the agent is rate limiting"):
        call(remote())
    assert len(http.requests) == 3


def test_a_connection_that_breaks_after_sending_is_not_retried(http):
    def broken(request, n):
        raise httpx.RemoteProtocolError("peer closed", request=request)

    http.respond(broken)
    with pytest.raises(A2AError, match="connection to the agent failed .RemoteProtocolError"):
        call(remote())
    assert len(http.requests) == 1


def test_an_agent_that_is_down_is_reported_not_raised():
    with pytest.raises(A2AError, match="could not be reached"):
        asyncio.run(a2a_client.delegate(
            RemoteAgent(AgentConfig("T", "http://127.0.0.1:9"), "x", "", "http://127.0.0.1:9/a2a", False),
            "hi", user_id="u", context_id="c", request_id="r", timeout=5,
        ))


# --- IAM bearer token ------------------------------------------------------------------------------

def test_the_iam_token_is_sent_as_a_bearer(http, monkeypatch):
    monkeypatch.setattr(mcp_client, "get_token", lambda force=False: "iam-token")
    http.respond(lambda request, n: reply())
    call(remote(api_key="key-1", iam_auth=True))
    assert http.requests[0].headers["authorization"] == "Bearer iam-token"
    assert http.requests[0].headers["x-api-key"] == "key-1"


def test_an_expired_iam_token_is_refreshed_once(http, monkeypatch):
    forced = []
    monkeypatch.setattr(mcp_client, "get_token", lambda force=False: forced.append(force) or f"token-{len(forced)}")
    http.respond(lambda request, n: httpx.Response(401) if n == 1 else reply("ok"))
    assert call(remote(iam_auth=True)) == "ok"
    assert forced == [False, True]
    assert [r.headers["authorization"] for r in http.requests] == ["Bearer token-1", "Bearer token-2"]


def test_a_second_401_is_an_error_and_the_token_is_not_refreshed_again(http, monkeypatch):
    forced = []
    monkeypatch.setattr(mcp_client, "get_token", lambda force=False: forced.append(force) or "t")
    http.respond(lambda request, n: httpx.Response(401))
    with pytest.raises(A2AError, match="HTTP 401"):
        call(remote(iam_auth=True))
    assert forced == [False, True] and len(http.requests) == 2


def test_a_401_without_iam_auth_is_not_retried(http):
    http.respond(lambda request, n: httpx.Response(401))
    with pytest.raises(A2AError, match="HTTP 401"):
        call(remote())
    assert len(http.requests) == 1


def test_a_missing_iam_token_is_an_a2a_error(http, monkeypatch):
    def no_token(force=False):
        raise RuntimeError("GREENNODE_CLIENT_ID / GREENNODE_CLIENT_SECRET are not set")

    monkeypatch.setattr(mcp_client, "get_token", no_token)
    with pytest.raises(A2AError, match="could not get an IAM token"):
        call(remote(iam_auth=True))
    assert http.requests == []


# --- message/stream --------------------------------------------------------------------------------

def test_streaming_reports_the_partial_text_and_returns_the_complete_reply(stub_agents):
    stub = stub_agents()
    agent = a2a_client.fetch_agent(AgentConfig("TRAVEL", stub.url))
    seen: list[tuple[str, bool]] = []
    out = call(agent, "plan Hue now", on_text=lambda text, append: seen.append((text, append)))
    reply_text = "travel-buddy answers: plan Hue now"
    assert out == reply_text
    assert stub.calls[0].body["method"] == "message/stream" and "configuration" not in stub.calls[0].body["params"]
    # the chunks as the sample agents send them: first chunk append=false, the others append=true,
    # then the whole reply replaces them (append=false)
    assert [a for _, a in seen] == [False, True, True, True, True, False]
    assert "".join(t for t, a in seen[:-1]) == reply_text + " " and seen[-1] == (reply_text, False)


def test_streaming_is_used_only_with_a_callback_and_an_agent_that_streams(stub_agents):
    stub = stub_agents(streaming=False)
    agent = a2a_client.fetch_agent(AgentConfig("TRAVEL", stub.url))
    seen = []
    call(agent, on_text=lambda *a: seen.append(a))
    call(a2a_client.fetch_agent(AgentConfig("TRAVEL", stub_agents().url)))
    methods = [c.body["method"] for c in stub.calls]
    assert methods == ["message/send"] and seen == []


def stream(http, *events: str, status=200):
    http.respond(lambda request, n: httpx.Response(status, text="".join(events), headers={"Content-Type": "text/event-stream"}))
    seen = []
    return call(remote(streaming=True), on_text=lambda text, append: seen.append((text, append))), seen


def ev(result: dict) -> str:
    return f"data: {json.dumps({'jsonrpc': '2.0', 'id': '1', 'result': result})}\n\n"


def artifact_ev(text, append=False, id_="a") -> str:
    return ev({"kind": "artifact-update", "taskId": "t", "contextId": "c", "append": append, "artifact": art(text, id_=id_)})


def status_ev(state, final=True, message=None) -> str:
    status = {"state": state}
    if message:
        status["message"] = {"kind": "message", "role": "agent", "messageId": "m", "parts": [{"kind": "text", "text": message}]}
    return ev({"kind": "status-update", "taskId": "t", "contextId": "c", "final": final, "status": status})


def test_appended_chunks_make_the_reply_and_a_replacing_update_wins(http):
    out, seen = stream(http, status_ev("working", False), artifact_ev("Hel"), artifact_ev("lo", append=True), status_ev("completed"))
    assert out == "Hello" and seen == [("Hel", False), ("lo", True)]
    out, _ = stream(http, artifact_ev("Hello "), artifact_ev("big ", append=True), artifact_ev("world", append=True), status_ev("completed"))
    assert out == "Hello big world"  # chunks keep their spaces; only the whole reply is trimmed
    out, _ = stream(http, artifact_ev("par"), artifact_ev("tial", append=True), artifact_ev("The full reply"), status_ev("completed"))
    assert out == "The full reply"


def test_several_artifacts_are_joined_in_order(http):
    out, _ = stream(http, artifact_ev("one", id_="a"), artifact_ev("two", id_="b"), artifact_ev("1+", append=True, id_="a"), status_ev("completed"))
    assert out == "one1+\n\ntwo"


def test_a_stream_may_end_with_a_message(http):
    out, seen = stream(http, status_ev("working", False), ev(rpc_message("1", "c", "final message")["result"]))
    assert out == "final message" and seen == []


def test_a_stream_may_end_with_a_finished_task(http):
    out, _ = stream(http, ev(task("submitted")), ev(task("completed", artifacts=[art("from the task")])))
    assert out == "from the task"


def test_a_stream_that_asks_for_input(http):
    out, _ = stream(http, status_ev("working", False), status_ev("input-required", message="Which date?"))
    assert out == "INPUT_REQUIRED: Which date?"


@pytest.mark.parametrize(
    ("events", "message"),
    [
        ((status_ev("working", False), status_ev("failed", message="The agent could not complete the request. (request ab)")),
         "the task failed: The agent could not complete the request. .request ab."),
        ((artifact_ev("some text"), status_ev("failed")), "the task failed"),
        ((artifact_ev("partial text"), ), "stream ended before the agent finished"),
        ((), "stream ended before the agent finished"),
        ((status_ev("working", False),), "stream ended before the agent finished"),
        ((status_ev("completed"),), "returned no text"),
        ((status_ev("rejected", final=False),), "the task rejected"),
        ((status_ev("auth-required"),), "additional authentication"),
        (("data: {not json}\n\n",), "not JSON"),
        (("data: [1]\n\n",), "unexpected stream event"),
        ((f"data: {json.dumps({'jsonrpc': '2.0', 'id': '1', 'error': {'code': -32603, 'message': 'boom'}})}\n\n",), "returned an error: boom"),
    ],
)
def test_streams_that_do_not_complete_are_errors(http, events, message):
    with pytest.raises(A2AError, match=message):
        stream(http, *events)


def test_partial_text_of_a_stream_that_never_completes_is_not_returned_as_an_answer(http):
    """Never present a half reply as the answer: the model would build on text the agent did not finish."""
    with pytest.raises(A2AError):
        stream(http, artifact_ev("The booking is conf"))


def test_sse_framing_comments_ids_and_events_without_a_trailing_blank_line(http):
    raw = (
        ": keep-alive\n\n"
        "event: message\nid: 1\ndata: " + json.dumps({"jsonrpc": "2.0", "id": "1", "result": {"kind": "artifact-update", "taskId": "t", "contextId": "c", "artifact": art("ok")}}) + "\n\n"
        'data: {"jsonrpc": "2.0", "id": "1",\ndata: "result": {"kind": "status-update", "taskId": "t", "contextId": "c", "final": true,\ndata: "status": {"state": "completed"}}}\n'
    )
    out, _ = stream(http, raw)
    assert out == "ok"


def test_an_http_error_before_the_stream_starts_is_an_error(http):
    with pytest.raises(A2AError, match="HTTP 500: Internal"):
        http.respond(lambda request, n: httpx.Response(500, json={"error": {"message": "Internal"}}))
        call(remote(streaming=True), on_text=lambda *a: None)


def test_a_stream_that_is_cut_off_is_an_error(stub_agents):
    stub = stub_agents()

    async def cut(stub_, body, request):
        async def events():
            yield "data: " + json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": {"kind": "artifact-update", "taskId": "t", "contextId": "c", "artifact": art("part")}}) + "\n\n"
            raise ConnectionResetError

        return StreamingResponse(events(), media_type="text/event-stream")

    stub.respond = cut
    agent = a2a_client.fetch_agent(AgentConfig("TRAVEL", stub.url))
    with pytest.raises(A2AError):
        call(agent, on_text=lambda *a: None)


def test_a_stream_is_bounded_by_the_overall_timeout(stub_agents):
    stub = stub_agents()

    async def endless(stub_, body, request):
        async def events():
            while True:
                yield ": keep-alive\n\n"
                await asyncio.sleep(0.05)

        return StreamingResponse(events(), media_type="text/event-stream")

    stub.respond = endless
    agent = a2a_client.fetch_agent(AgentConfig("TRAVEL", stub.url))
    with pytest.raises(A2AError, match="did not answer within 0.5 s"):
        call(agent, timeout=0.5, on_text=lambda *a: None)


def test_streaming_calls_are_not_retried_after_the_connection_was_opened(stub_agents):
    stub = stub_agents()

    async def fail(stub_, body, request):
        return JSONResponse({"jsonrpc": "2.0", "id": body["id"], "error": {"code": -32603, "message": "boom"}}, status_code=500)

    stub.respond = fail
    agent = a2a_client.fetch_agent(AgentConfig("TRAVEL", stub.url))
    with pytest.raises(A2AError, match="HTTP 500: boom"):
        call(agent, on_text=lambda *a: None)
    assert len(stub.calls) == 1


def test_a_stub_hook_can_return_a_plain_response(stub_agents):
    stub = stub_agents()

    async def teapot(stub_, body, request):
        return Response(status_code=418)

    stub.respond = teapot
    with pytest.raises(A2AError, match="HTTP 418"):
        call(a2a_client.fetch_agent(AgentConfig("TRAVEL", stub.url)))
