"""The orchestrator against a scripted fake LLM, a fake Memory API, stub A2A agents and fake MCP tools."""
import asyncio
import base64
import json
import re
import time
from datetime import datetime
from types import SimpleNamespace

import httpx
import openai
import pytest
from greennode_agent_bridge import AgentBaseMemoryEvents
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from starlette.responses import JSONResponse

import a2a_client
import agent
import agent_loop
import mcp_client
from a2a_client import AgentConfig
from fakes import FakeCheckpointApi, ScriptedChatModel, tool_call

STOCK_TOOL = {
    "name": "top_gainers",
    "description": "Top gainers of the day",
    "inputSchema": {"type": "object", "properties": {"limit": {"type": "integer"}}},
}
TRAVEL, ZALO = "travel-buddy", "zalo-restaurant-bot"


@pytest.fixture()
def rig(monkeypatch, stub_agents):
    model = ScriptedChatModel()
    api = FakeCheckpointApi()
    rig = SimpleNamespace(model=model, api=api, llm_kwargs={}, tool_defs=[STOCK_TOOL], mcp_calls=[])
    rig.travel = stub_agents(TRAVEL)
    rig.zalo = stub_agents(ZALO)

    def use_agents(*stubs, ttl=600, retry=0):
        """Point the orchestrator at these stubs (labels T0, T1, ...)."""
        configs = [AgentConfig(f"T{i}", s.url, s.api_key) for i, s in enumerate(stubs)]
        monkeypatch.setattr(agent, "DIRECTORY", a2a_client.AgentDirectory(configs, ttl=ttl, retry=retry))

    rig.use_agents = use_agents
    use_agents(rig.travel, rig.zalo)

    monkeypatch.setattr(agent, "ChatOpenAI", lambda **kwargs: rig.llm_kwargs.update(kwargs) or model)
    monkeypatch.setattr(agent, "_checkpointer", AgentBaseMemoryEvents(memory_id="memory-test", memory_client=api))
    for name, value in (
        ("_agent", None), ("_agent_tools", None), ("_tools", []), ("_tool_defs", []), ("_tools_loaded_at", 0.0),
        ("_remote_agents", ()), ("_remote_tools", []),
    ):
        monkeypatch.setattr(agent, name, value)
    monkeypatch.setattr(mcp_client, "list_tools", lambda url: rig.tool_defs)

    def call_tool(url, name, arguments):
        rig.mcp_calls.append((name, arguments))
        return f"result for {arguments}"

    monkeypatch.setattr(mcp_client, "call_tool", call_tool)

    def turn(text, user="alice", session="s1", request_id="req-1"):
        return asyncio.run(agent_loop.arun_coro(agent.run_turn(text, user, session, request_id=request_id)))

    rig.turn = turn
    return rig


def tool_messages(messages) -> list[ToolMessage]:
    return [m for m in messages if isinstance(m, ToolMessage)]


def valid_for_openai(messages):
    """Every ToolMessage answers an earlier AI tool call; the first message is not mid-pair."""
    first = next(m for m in messages if not isinstance(m, SystemMessage))
    assert not isinstance(first, ToolMessage)
    assert not (isinstance(first, AIMessage) and first.tool_calls)
    announced: set[str] = set()
    for m in messages:
        if isinstance(m, AIMessage):
            announced.update(c["id"] for c in m.tool_calls)
        elif isinstance(m, ToolMessage):
            assert m.tool_call_id in announced, "orphan ToolMessage"


def call_both(**requests):
    return AIMessage("", tool_calls=[tool_call(f"c-{name}", name, request=text) for name, text in requests.items()])


# --- tools from the agent cards ----------------------------------------------------------------

def test_every_agent_card_becomes_a_tool_next_to_the_mcp_tools(rig):
    rig.turn("hi")
    tools = {t.name: t for t in rig.model.bound}
    assert list(tools) == [TRAVEL, ZALO, "top_gainers"]
    assert "Main skill: Does the main thing." in tools[TRAVEL].description
    assert set(tools[TRAVEL].args_schema.model_fields) == {"request"}  # one argument: the user, session and ids are not the model's
    assert tools[TRAVEL].metadata == {"a2a_agent": TRAVEL, "a2a_url": f"{rig.travel.url}/a2a"}


def test_a_card_that_fails_to_load_skips_only_that_tool_and_the_prompt_says_so(rig, caplog):
    rig.zalo.card_status = 503
    with caplog.at_level("WARNING", logger="a2a-client"):
        rig.turn("hi")
    assert [t.name for t in rig.model.bound] == [TRAVEL, "top_gainers"]
    prompt = rig.model.seen[-1][0].content
    assert "# Unreachable agents" in prompt and "T1" in prompt and "T0" not in prompt
    assert "agent T1: the agent card request returned HTTP 503" in caplog.text


def test_a_failed_card_is_retried_and_the_agent_is_rebuilt_with_the_recovered_tool(rig):
    rig.zalo.card_status = 503
    rig.turn("hi")
    first = agent.get_agent()
    rig.zalo.card_status = 200
    rig.turn("hi again")
    assert [t.name for t in rig.model.bound] == [TRAVEL, ZALO, "top_gainers"]
    assert agent.get_agent() is not first
    assert "Unreachable agents" not in rig.model.seen[-1][0].content


def test_unchanged_cards_keep_the_same_agent_and_a_changed_card_rebuilds_it(rig):
    rig.use_agents(rig.travel, rig.zalo, ttl=0)  # refetch every card on every call
    first = agent.get_agent()
    assert agent.get_agent() is first
    rig.travel.card = lambda: {**type(rig.travel).card(rig.travel), "description": "Now with a new description."}
    second = agent.get_agent()
    assert second is not first
    assert "new description" in next(t for t in agent.get_remote_tools() if t.name == TRAVEL).description


def test_two_agents_with_the_same_card_name_get_one_tool(rig, stub_agents, caplog):
    twin = stub_agents(TRAVEL)
    rig.use_agents(rig.travel, twin)
    with caplog.at_level("WARNING", logger="trip-concierge"):
        assert [t.name for t in agent.get_remote_tools()] == [TRAVEL]
    assert "agent T1: skipped, another agent already uses the name travel-buddy" in caplog.text


def test_with_no_agent_reachable_the_orchestrator_still_answers(rig):
    rig.travel.card_status = rig.zalo.card_status = 503
    rig.model.script = [AIMessage("The travel and restaurant agents are unavailable.")]
    assert rig.turn("plan a trip").reply == "The travel and restaurant agents are unavailable."
    assert [t.name for t in rig.model.bound] == ["top_gainers"]
    assert "T0, T1" in rig.model.seen[-1][0].content


# --- delegation ----------------------------------------------------------------------------------

def test_a_delegation_reaches_the_agent_with_user_context_and_request_id(rig):
    rig.model.script = [call_both(**{TRAVEL: "Plan 2 days in Da Nang for 2 people"}), AIMessage("Here is your plan.")]
    result = rig.turn("Plan 2 days in Da Nang", user="alice", session="s-7", request_id="req-abc")
    assert result.reply == "Here is your plan."
    (recorded,) = rig.travel.calls
    message = recorded.body["params"]["message"]
    assert message["parts"] == [{"kind": "text", "text": "Plan 2 days in Da Nang for 2 people"}]
    assert message["contextId"] == f"s-7:{TRAVEL}"
    assert recorded.headers["x-greennode-agentbase-user-id"] == "alice"
    assert recorded.headers["x-greennode-agentbase-request-id"] == "req-abc"
    assert recorded.headers["x-greennode-agentbase-session-id"] == f"s-7:{TRAVEL}"
    sent_back = tool_messages(rig.model.seen[-1])[0]
    assert sent_back.content == "travel-buddy answers: Plan 2 days in Da Nang for 2 people" and sent_back.status == "success"
    assert result.delegations == [{"agent": TRAVEL, "ok": True, "elapsed_ms": result.delegations[0]["elapsed_ms"]}]
    assert isinstance(result.delegations[0]["elapsed_ms"], int)


def test_the_user_id_of_the_turn_goes_to_every_agent(rig):
    rig.model.script = [call_both(**{TRAVEL: "a", ZALO: "b"}), AIMessage("ok")]
    rig.turn("both", user="guest-42")
    assert rig.travel.calls[0].headers["x-greennode-agentbase-user-id"] == "guest-42"
    assert rig.zalo.calls[0].headers["x-greennode-agentbase-user-id"] == "guest-42"
    rig.model.script = [call_both(**{TRAVEL: "a"}), AIMessage("ok")]
    rig.turn("again", user="bob")
    assert rig.travel.calls[-1].headers["x-greennode-agentbase-user-id"] == "bob"


def test_the_model_cannot_choose_the_user_or_the_conversation(rig):
    """Extra arguments the model invents are dropped by the schema, the ids come from the run config."""
    bad = AIMessage("", tool_calls=[{
        "name": TRAVEL, "id": "c1", "type": "tool_call",
        "args": {"request": "hi", "user_id": "mallory", "context_id": "other-session"},
    }])
    rig.model.script = [bad, AIMessage("ok")]
    rig.turn("hi", user="alice", session="s1")
    recorded = rig.travel.calls[0]
    assert recorded.headers["x-greennode-agentbase-user-id"] == "alice"
    assert recorded.body["params"]["message"]["contextId"] == f"s1:{TRAVEL}"


def test_the_context_id_is_stable_per_session_and_agent(rig):
    for session in ("s1", "s1", "s2"):
        rig.model.script = [call_both(**{TRAVEL: "q", ZALO: "q"}), AIMessage("ok")]
        rig.turn("go", session=session)
    travel = [c.body["params"]["message"]["contextId"] for c in rig.travel.calls]
    zalo = [c.body["params"]["message"]["contextId"] for c in rig.zalo.calls]
    assert travel == [f"s1:{TRAVEL}", f"s1:{TRAVEL}", f"s2:{TRAVEL}"]
    assert zalo == [f"s1:{ZALO}", f"s1:{ZALO}", f"s2:{ZALO}"]


def test_the_context_id_survives_a_restart_because_it_is_derived_not_stored(rig, monkeypatch):
    rig.model.script = [call_both(**{TRAVEL: "q"}), AIMessage("ok")]
    rig.turn("one", session="s9")
    for name, value in (("_agent", None), ("_agent_tools", None), ("_remote_agents", ()), ("_remote_tools", [])):
        monkeypatch.setattr(agent, name, value)  # a fresh process
    rig.model.script = [call_both(**{TRAVEL: "q"}), AIMessage("ok")]
    rig.turn("two", session="s9")
    contexts = {c.body["params"]["message"]["contextId"] for c in rig.travel.calls}
    assert contexts == {f"s9:{TRAVEL}"}


def test_conversation_id_is_valid_for_the_sample_agents_and_bounded():
    assert agent.conversation_id("s1", "travel-buddy") == "s1:travel-buddy"
    long_session = "s" * 128
    first = agent.conversation_id(long_session, "a" * 64)
    assert first == agent.conversation_id(long_session, "a" * 64)
    assert len(first) <= agent.MAX_CONTEXT_ID_CHARS and first != agent.conversation_id("t" * 128, "a" * 64)
    assert re.fullmatch(r"[A-Za-z0-9._:@+=~-]{1,128}", first)


def test_a_failing_agent_becomes_tool_error_and_the_turn_continues(rig):
    rig.zalo.api_key = "the-real-key-of-zalo"  # the orchestrator has no key for it: HTTP 401
    rig.model.script = [call_both(**{TRAVEL: "plan", ZALO: "book"}), AIMessage("Plan ready; the booking failed.")]
    result = rig.turn("plan and book")
    assert result.reply == "Plan ready; the booking failed."
    by_name = {m.name: m for m in tool_messages(rig.model.seen[-1])}
    assert by_name[TRAVEL].content == "travel-buddy answers: plan"
    assert by_name[ZALO].content.startswith(f"TOOL_ERROR: agent '{ZALO}' failed: HTTP 401")
    assert "the-real-key-of-zalo" not in by_name[ZALO].content
    assert [(d["agent"], d["ok"]) for d in result.delegations] == [(TRAVEL, True), (ZALO, False)]


def test_a_slow_agent_times_out_and_the_others_still_answer(rig, monkeypatch):
    monkeypatch.setattr(agent, "A2A_TIMEOUT_SECONDS", 0.4)
    rig.zalo.delay = 3
    rig.model.script = [call_both(**{TRAVEL: "plan", ZALO: "book"}), AIMessage("partial answer")]
    started = time.monotonic()
    result = rig.turn("both")
    assert time.monotonic() - started < 2.5  # not 3 s: the call was cut at the timeout
    by_name = {m.name: m for m in tool_messages(rig.model.seen[-1])}
    assert by_name[TRAVEL].content == "travel-buddy answers: plan"
    assert by_name[ZALO].content == f"TOOL_ERROR: agent '{ZALO}' failed: the agent did not answer within 0.4 s"
    assert result.reply == "partial answer" and len(rig.zalo.calls) == 1  # never retried


def test_an_unexpected_error_in_a_delegation_does_not_kill_the_turn(rig, monkeypatch):
    async def broken(*args, **kwargs):
        raise ValueError("secret-internal-detail")

    monkeypatch.setattr(a2a_client, "delegate", broken)
    rig.model.script = [call_both(**{TRAVEL: "plan"}), AIMessage("the agent is down")]
    assert rig.turn("plan").reply == "the agent is down"
    content = tool_messages(rig.model.seen[-1])[0].content
    assert "ValueError" in content and "secret-internal-detail" not in content


def test_the_request_is_validated_before_anything_is_sent(rig):
    rig.model.script = [
        AIMessage("", tool_calls=[tool_call("c1", TRAVEL, request="   "), tool_call("c2", TRAVEL, request="x" * 4001)]),
        AIMessage("ok"),
    ]
    rig.turn("hi")
    messages = [m.content for m in tool_messages(rig.model.seen[-1])]
    assert "request must not be empty" in messages[0] and "request is too long (max 4000 characters)" in messages[1]
    assert rig.travel.calls == []


def test_a_request_of_exactly_the_limit_is_sent(rig):
    rig.model.script = [call_both(**{TRAVEL: "x" * 4000}), AIMessage("ok")]
    rig.turn("hi")
    assert len(rig.travel.calls) == 1


def test_a_tool_cannot_run_without_the_turn_ids(rig):
    tool = agent.get_remote_tools()[0]
    with pytest.raises(RuntimeError, match="actor_id, thread_id and request_id"):
        asyncio.run(agent_loop.arun_coro(tool.ainvoke({"request": "hi"}, config={"configurable": {"actor_id": "alice"}})))
    assert rig.travel.calls == []


def test_a_long_reply_is_capped(rig):
    rig.travel.answer = lambda text: "X" * 30_000
    rig.model.script = [call_both(**{TRAVEL: "plan"}), AIMessage("ok")]
    rig.turn("plan")
    content = tool_messages(rig.model.seen[-1])[0].content
    assert len(content) < mcp_client.MAX_TOOL_OUTPUT_CHARS + 60 and "...[truncated 22000 chars]" in content


def test_a_task_reply_with_artifacts_is_understood(rig):
    async def task_reply(stub, body, request):
        return JSONResponse({"jsonrpc": "2.0", "id": body["id"], "result": {
            "kind": "task", "id": "t1", "contextId": "c", "status": {"state": "completed"},
            "artifacts": [{"artifactId": "a", "parts": [{"kind": "text", "text": "Booked: table 7 at 19:00."}]}],
        }})

    rig.zalo.respond = task_reply
    rig.model.script = [call_both(**{ZALO: "book"}), AIMessage("ok")]
    rig.turn("book")
    assert tool_messages(rig.model.seen[-1])[0].content == "Booked: table 7 at 19:00."


def test_an_agent_that_asks_a_question_is_relayed_not_failed(rig):
    rig.zalo.answer = lambda text: "unused"

    async def ask(stub, body, request):
        return JSONResponse({"jsonrpc": "2.0", "id": body["id"], "result": {
            "kind": "task", "id": "t1", "contextId": "c",
            "status": {"state": "input-required", "message": {"kind": "message", "role": "agent", "messageId": "m", "parts": [{"kind": "text", "text": "How many guests?"}]}},
        }})

    rig.zalo.respond = ask
    rig.model.script = [call_both(**{ZALO: "book"}), AIMessage("How many guests?")]
    result = rig.turn("book")
    assert tool_messages(rig.model.seen[-1])[0].content == "INPUT_REQUIRED: How many guests?"
    assert result.delegations[0]["ok"] is True


# --- guardrails ----------------------------------------------------------------------------------------

def test_calls_to_the_same_agent_are_capped_per_turn_and_counted_per_agent(rig):
    calls = [tool_call(f"c{i}", TRAVEL, request=f"plan {i}") for i in range(agent.AGENT_CALL_LIMIT + 2)]
    calls += [tool_call(f"z{i}", ZALO, request=f"book {i}") for i in range(2)]
    rig.model.script = [AIMessage("", tool_calls=calls), AIMessage("done anyway")]
    result = rig.turn("many requests")
    assert result.reply == "done anyway"
    assert len(rig.travel.calls) == agent.AGENT_CALL_LIMIT and len(rig.zalo.calls) == 2
    refused = [m for m in tool_messages(rig.model.seen[-1]) if m.status == "error"]
    assert [m.name for m in refused] == [TRAVEL, TRAVEL]
    assert "Do not call 'travel-buddy' again" in refused[0].content
    assert all(d["agent"] != TRAVEL or d["ok"] for d in result.delegations)  # the refused calls never ran


def test_the_per_agent_cap_starts_again_every_turn(rig):
    for _ in range(2):
        rig.model.script = [
            AIMessage("", tool_calls=[tool_call(f"c{i}", TRAVEL, request=f"r{i}") for i in range(agent.AGENT_CALL_LIMIT)]),
            AIMessage("ok"),
        ]
        rig.turn("go")
    assert len(rig.travel.calls) == 2 * agent.AGENT_CALL_LIMIT


def test_agents_called_in_one_step_run_in_parallel(rig):
    rig.travel.delay = rig.zalo.delay = 0.6
    rig.model.script = [call_both(**{TRAVEL: "plan", ZALO: "book"}), AIMessage("ok")]
    started = time.monotonic()
    rig.turn("both")
    elapsed = time.monotonic() - started
    t, z = rig.travel.calls[0], rig.zalo.calls[0]
    assert elapsed < 1.1  # one after the other would be 1.2 s plus the model
    assert t.started < z.finished and z.started < t.finished  # their calls overlapped


def test_calls_to_the_same_agent_in_one_step_run_one_after_the_other(rig):
    """Two concurrent turns on one remote thread would fork its conversation history."""
    rig.travel.delay = 0.3
    rig.model.script = [
        AIMessage("", tool_calls=[tool_call("a", TRAVEL, request="plan Hue"), tool_call("b", TRAVEL, request="plan Hoi An")]),
        AIMessage("ok"),
    ]
    rig.turn("two plans")
    assert rig.travel.max_active == 1 and len(rig.travel.calls) == 2
    first, second = rig.travel.calls
    assert first.finished <= second.started


def test_the_same_agent_for_different_sessions_is_not_serialized(rig):
    rig.travel.delay = 0.5
    # both turns ask the model first, then both get their answer
    rig.model.script = [call_both(**{TRAVEL: "a"}), call_both(**{TRAVEL: "b"}), AIMessage("ok"), AIMessage("ok")]

    async def both():
        await asyncio.gather(
            agent_loop.arun_coro(agent.run_turn("x", "alice", "s1", request_id="r1")),
            agent_loop.arun_coro(agent.run_turn("y", "bob", "s2", request_id="r2")),
        )

    asyncio.run(both())
    assert rig.travel.max_active == 2


# --- loop caps and tools (the ported core) -----------------------------------------------------------------

def test_a_model_that_always_calls_tools_stops_at_the_model_call_limit(rig):
    rig.model.script = [AIMessage("", tool_calls=[tool_call(f"c{i}", "top_gainers", limit=i)]) for i in range(40)]
    result = rig.turn("loop forever")
    assert len(rig.model.seen) == agent.MODEL_CALL_LIMIT
    assert result.reply == agent.LIMIT_REPLY and "limit" not in result.reply.lower()


def test_tool_calls_over_the_global_limit_are_refused_and_the_model_still_answers(rig):
    calls = [tool_call(f"c{i}", "top_gainers", limit=i + 1) for i in range(agent.TOOL_CALL_LIMIT + 3)]
    rig.model.script = [AIMessage("", tool_calls=calls), AIMessage("done anyway")]
    result = rig.turn("many tools at once")
    assert result.reply == "done anyway" and len(rig.mcp_calls) == agent.TOOL_CALL_LIMIT
    refused = [m for m in tool_messages(rig.model.seen[-1]) if m.status == "error"]
    assert len(refused) == 3


def test_a_failing_mcp_tool_does_not_kill_the_turn(rig, monkeypatch):
    def broken(url, name, arguments):
        raise httpx.ConnectError("gateway unreachable")

    monkeypatch.setattr(mcp_client, "call_tool", broken)
    rig.model.script = [AIMessage("", tool_calls=[tool_call("c1", "top_gainers", limit=3)]), AIMessage("stock data is down")]
    assert rig.turn("top gainers").reply == "stock data is down"


def test_a_capped_turn_does_not_poison_the_next_one(rig):
    rig.model.script = [AIMessage("", tool_calls=[tool_call(f"c{i}", "top_gainers", limit=i)]) for i in range(40)]
    assert rig.turn("loop").reply == agent.LIMIT_REPLY
    rig.model.script = [AIMessage("a normal answer")]
    assert rig.turn("something simple").reply == "a normal answer"


def test_model_retry_is_the_only_retry_layer(rig):
    """ChatOpenAI retrying on top of ModelRetryMiddleware would multiply the attempts."""
    rig.turn("hi")
    assert rig.llm_kwargs["max_retries"] == 0
    assert rig.llm_kwargs["timeout"] == agent.LLM_TIMEOUT_SECONDS == 60


def test_transient_llm_errors_are_retried(rig):
    request = httpx.Request("POST", "https://llm.example/v1/chat/completions")
    rig.model.fail_with = [openai.APIConnectionError(request=request)]
    rig.model.script = [AIMessage("recovered")]
    assert rig.turn("hello").reply == "recovered"


def test_retries_stop_after_max_retries_and_the_error_propagates(rig):
    request = httpx.Request("POST", "https://llm.example/v1/chat/completions")
    rig.model.fail_with = [openai.APIConnectionError(request=request)] * 5
    with pytest.raises(openai.APIConnectionError):
        rig.turn("hello")
    assert len(rig.model.seen) == 3  # 1 call + max_retries=2


def test_permanent_llm_errors_propagate_without_retry(rig):
    rig.model.fail_with = [ValueError("bad request")]
    with pytest.raises(ValueError, match="bad request"):
        rig.turn("hello")
    assert len(rig.model.seen) == 1


def test_summarization_bounds_the_history_and_never_leaves_an_orphan_tool_message(rig, monkeypatch):
    monkeypatch.setattr(agent, "SUMMARIZE_AT_TOKENS", 400)
    monkeypatch.setattr(agent, "KEEP_MESSAGES", 3)  # small, so the cut lands between AI and tool messages
    for i in range(8):
        parallel = [tool_call(f"t{i}-{j}", "top_gainers", limit=j + 1) for j in range(1 + i % 3)]
        parallel.append(tool_call(f"a{i}", TRAVEL, request=f"question {i}"))
        rig.model.script = [AIMessage("", tool_calls=parallel), AIMessage("answer " + "word " * 60)]
        rig.turn(f"question {i} " + "blah " * 60)
    assert rig.model.summaries, "summarization never triggered"
    for messages in rig.model.seen:
        valid_for_openai(messages)
        assert len(messages) <= 14  # bounded: summary + kept messages + the current turn
    assert any("Here is a summary of the conversation" in str(m.content) for m in rig.model.seen[-1])


# --- MCP tool cache ----------------------------------------------------------------------------------------

def test_a_failed_tools_list_is_retried_on_the_next_turn_and_the_agent_rebuilt(rig, monkeypatch):
    attempts = []

    def flaky_list(url):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("gateway 503 at boot")
        return [STOCK_TOOL]

    monkeypatch.setattr(mcp_client, "list_tools", flaky_list)
    first = asyncio.run(agent_loop.arun_coro(agent.aget_agent()))
    assert len(attempts) == 1 and agent._tools == []
    rig.model.script = [AIMessage("no stock tools yet")]
    assert rig.turn("hi").reply == "no stock tools yet"
    second = asyncio.run(agent_loop.arun_coro(agent.aget_agent()))
    assert second is not first  # rebuilt with the recovered tool
    assert [t.name for t in agent.get_mcp_tools()] == ["top_gainers"]
    assert len(attempts) == 2  # retried while failing, cached once it worked
    agent.get_mcp_tools()
    assert len(attempts) == 2


def test_an_empty_tools_list_is_not_cached(rig):
    rig.tool_defs = []
    assert agent.get_mcp_tools() == []
    rig.tool_defs = [STOCK_TOOL]
    assert [t.name for t in agent.get_mcp_tools()] == ["top_gainers"]


def test_tools_are_refreshed_after_the_ttl_and_a_failed_refresh_keeps_the_old_list(rig, monkeypatch):
    assert len(agent.get_mcp_tools()) == 1
    monkeypatch.setattr(agent, "TOOLS_TTL_SECONDS", 0)

    def broken(url):
        raise RuntimeError("gateway down")

    monkeypatch.setattr(mcp_client, "list_tools", broken)
    assert [t.name for t in agent.get_mcp_tools()] == ["top_gainers"]
    rig.tool_defs = [STOCK_TOOL, {**STOCK_TOOL, "name": "top_losers"}]
    monkeypatch.setattr(mcp_client, "list_tools", lambda url: rig.tool_defs)
    assert [t.name for t in agent.get_mcp_tools()] == ["top_gainers", "top_losers"]


def test_unchanged_tools_and_agents_keep_the_same_agent(rig):
    first = agent.get_agent()
    assert agent.get_agent() is first


def test_one_bad_tool_schema_skips_only_that_tool(rig, caplog):
    rig.tool_defs = [
        {"name": "broken", "inputSchema": {"properties": {"_private": {"type": "string"}}}},
        {"inputSchema": {}},  # no name at all
        STOCK_TOOL,
    ]
    with caplog.at_level("WARNING", logger="trip-concierge"):
        tools = agent.get_mcp_tools()
    assert [t.name for t in tools] == ["top_gainers"]
    assert "skipping MCP tool" in caplog.text


def test_a_stock_tool_drops_unset_optional_arguments(rig):
    tool = agent._build_tools([STOCK_TOOL])[0]
    rig.model.script = [AIMessage("", tool_calls=[tool_call("c1", "top_gainers")]), AIMessage("ok")]
    rig.turn("top gainers")
    assert rig.mcp_calls == [("top_gainers", {})]
    assert tool.name == "top_gainers"


# --- JSON schema -> pydantic -------------------------------------------------------------------------------

def model_for(properties, required=()):
    return agent._schema_to_model(
        {"name": "t", "inputSchema": {"type": "object", "required": list(required), "properties": properties}}
    )


def test_schema_required_and_optional_fields():
    model = model_for({"query": {"type": "string", "description": "q"}, "n": {"type": "integer"}}, ["query"])
    assert model(query="x").n is None
    with pytest.raises(ValueError):
        model()


def test_schema_type_given_as_a_list():
    model = model_for({"days": {"type": ["integer", "null"]}}, ["days"])
    assert model(days=3).days == 3
    assert model(days=None).days is None
    with pytest.raises(ValueError):
        model(days="three")


def test_schema_enum():
    model = model_for({"sort": {"type": "string", "enum": ["default", "volume"]}})
    assert model(sort="volume").sort == "volume"
    with pytest.raises(ValueError):
        model(sort="deep")


def test_schema_arrays_with_items():
    model = model_for({"symbols": {"type": "array", "items": {"type": "string"}}, "any": {"type": "array"}})
    assert model(symbols=["VNM"], any=[1, "x"]).symbols == ["VNM"]
    with pytest.raises(ValueError):
        model(symbols=[1])


def test_schema_nested_objects_fall_back_to_dict():
    model = model_for({"filters": {"type": "object", "properties": {"a": {"type": "string"}}}})
    assert model(filters={"a": "x", "b": 1}).filters == {"a": "x", "b": 1}


def test_schema_anyof_and_missing_type():
    model = model_for({"x": {"anyOf": [{"type": "integer"}, {"type": "string"}]}, "y": {"description": "free"}})
    assert model(x="s", y=object()).x == "s"


def test_schema_without_properties_has_no_fake_argument():
    model = agent._schema_to_model({"name": "no-args", "inputSchema": {}})
    assert model.model_fields == {}


# --- prompt ------------------------------------------------------------------------------------------------

def test_the_system_prompt_has_the_current_vietnam_date_on_every_call(rig, monkeypatch):
    clock = {"now": datetime(2030, 3, 4, 8, 15, tzinfo=agent.TZ_VN)}
    monkeypatch.setattr(agent, "now_vn", lambda: clock["now"])
    rig.turn("hello")
    assert "Monday, 2030-03-04 08:15" in rig.model.seen[-1][0].content
    clock["now"] = datetime(2030, 3, 5, 9, 0, tzinfo=agent.TZ_VN)  # the next day, same cached agent
    rig.turn("hello again")
    assert "Tuesday, 2030-03-05 09:00" in rig.model.seen[-1][0].content
    assert "Asia/Ho_Chi_Minh" in rig.model.seen[-1][0].content


def test_the_system_prompt_covers_the_required_behaviours():
    prompt = agent.build_system_prompt().lower()
    for needle in (
        "vietnamese", "independent parts", "same step", "self-contained", "never invent a result", "tool_error",
        "input_required", "denied_by_policy", "untrusted data", "explicitly confirmed", "do not repeat it",
        "which agent or tool each part came from",
    ):
        assert needle in prompt, needle
    assert "unreachable agents" not in prompt


def test_the_prompt_names_the_unreachable_agents():
    prompt = agent.build_system_prompt(["TRAVEL_BUDDY", "STOCKS"])
    assert "# Unreachable agents" in prompt and "TRAVEL_BUDDY, STOCKS" in prompt


# --- delegations of the turn ---------------------------------------------------------------------------------

def test_delegations_cover_the_current_turn_only(rig):
    rig.model.script = [call_both(**{TRAVEL: "plan"}), AIMessage("planned")]
    assert [d["agent"] for d in rig.turn("plan").delegations] == [TRAVEL]
    rig.model.script = [AIMessage("plain answer")]
    assert rig.turn("thanks").delegations == []  # same thread: the first turn's tool message is in the history
    rig.model.script = [call_both(**{ZALO: "book"}), AIMessage("booked")]
    assert [d["agent"] for d in rig.turn("book").delegations] == [ZALO]


def test_delegations_ignore_the_summary_message_and_failed_calls():
    summary = HumanMessage("Here is a summary", additional_kwargs={"lc_source": "summarization"})
    artifact = {"agent": "a", "ok": True, "elapsed_ms": 5}
    old = ToolMessage("x", tool_call_id="1", name="a", artifact={"agent": "old", "ok": True, "elapsed_ms": 1})
    refused = ToolMessage("limit", tool_call_id="2", name="a", status="error", artifact=artifact)
    fresh = ToolMessage("y", tool_call_id="3", name="a", artifact=artifact)
    not_a_delegation = ToolMessage("z", tool_call_id="4", name="top_gainers")
    messages = [HumanMessage("turn 1"), old, HumanMessage("turn 2"), refused, fresh, not_a_delegation]
    assert agent.delegations_used(messages) == [artifact]
    # the summarization message does not start a new turn
    assert agent.delegations_used([HumanMessage("turn"), fresh, summary, fresh]) == [artifact, artifact]


# --- checkpoint traffic ----------------------------------------------------------------------------------------

def test_one_checkpoint_is_written_per_turn_and_state_survives_between_turns(rig):
    rig.model.script = [call_both(**{TRAVEL: "plan"}), AIMessage("first answer")]
    rig.turn("first question")
    assert rig.api.creates > 0
    rig.model.script = [AIMessage("second answer")]
    before = rig.api.creates
    rig.turn("second question")
    assert rig.api.creates - before <= 12  # one put (channel blobs + 1 checkpoint), not one per graph step
    sent = rig.model.seen[-1]
    assert [m.content for m in sent if isinstance(m, HumanMessage)] == ["first question", "second question"]


def test_async_durability_would_write_far_more(rig, monkeypatch):
    rig.model.script = [call_both(**{TRAVEL: "plan"}), AIMessage("answer")]
    rig.turn("question", session="exit")
    exit_writes = rig.api.creates
    monkeypatch.setattr(agent, "_DURABILITY", "async")
    rig.api.creates = 0
    rig.model.script = [call_both(**{TRAVEL: "plan"}), AIMessage("answer")]
    rig.turn("question", session="async")
    assert rig.api.creates > exit_writes * 2


# --- streaming ----------------------------------------------------------------------------------------------------

def collect_stream(text, user="alice", session="s1", request_id="req-1"):
    async def main():
        return [item async for item in agent.stream_turn(text, user, session, request_id=request_id)]

    return asyncio.run(agent_loop.arun_coro(main()))


def test_stream_yields_tokens_progress_events_and_the_result(rig):
    rig.model.script = [
        AIMessage("", tool_calls=[tool_call("c1", TRAVEL, request="plan"), tool_call("c2", "top_gainers", limit=3)]),
        AIMessage("the final answer"),
    ]
    items = collect_stream("go")
    tokens = [i for i in items if isinstance(i, str)]
    events = [i for i in items if isinstance(i, dict)]
    assert tokens == ["the ", "final ", "answer "]
    assert isinstance(items[-1], agent.TurnResult) and items[-1].reply == "the final answer"
    assert [e["type"] for e in events if e.get("agent") == TRAVEL] == ["delegate", "delegate_done"]
    assert {"type": "tool", "name": "top_gainers"} in events
    delegate = next(e for e in events if e["type"] == "delegate")
    assert delegate == {"type": "delegate", "agent": TRAVEL, "request": "plan"}
    done = next(e for e in events if e["type"] == "delegate_done")
    assert done["ok"] is True and isinstance(done["elapsed_ms"], int)


def test_stream_events_of_a_failed_delegation_and_of_refused_calls(rig):
    rig.zalo.api_key = "secret-secret-secret"
    calls = [tool_call(f"c{i}", TRAVEL, request="r") for i in range(agent.AGENT_CALL_LIMIT + 1)] + [tool_call("z", ZALO, request="b")]
    rig.model.script = [AIMessage("", tool_calls=calls), AIMessage("ok")]
    events = [i for i in collect_stream("go") if isinstance(i, dict)]
    assert [e["type"] for e in events if e["agent"] == TRAVEL].count("delegate") == agent.AGENT_CALL_LIMIT  # the refused call never started
    assert next(e for e in events if e["type"] == "delegate_done" and e["agent"] == ZALO)["ok"] is False


def test_a_long_request_is_shortened_in_the_event(rig):
    rig.model.script = [call_both(**{TRAVEL: "x" * 1000}), AIMessage("ok")]
    delegate = next(i for i in collect_stream("go") if isinstance(i, dict) and i["type"] == "delegate")
    assert len(delegate["request"]) == 300 and delegate["request"].endswith("…")


def test_partial_text_of_the_agent_is_streamed_only_with_the_flag(rig, monkeypatch):
    rig.model.script = [call_both(**{TRAVEL: "plan Hue"}), AIMessage("ok")]
    assert not [i for i in collect_stream("go") if isinstance(i, dict) and i["type"] == "delegate_text"]
    assert rig.travel.calls[0].body["method"] == "message/send"

    monkeypatch.setattr(agent, "A2A_STREAMING", True)
    rig.model.script = [call_both(**{TRAVEL: "plan Hue"}), AIMessage("ok")]
    items = collect_stream("go again", session="s2")
    texts = [i for i in items if isinstance(i, dict) and i["type"] == "delegate_text"]
    assert rig.travel.calls[1].body["method"] == "message/stream"
    assert all(e["agent"] == TRAVEL for e in texts)
    assert "".join(e["text"] for e in texts if e["append"] or e is texts[0]).startswith("travel-buddy answers: plan Hue")
    assert texts[-1] == {"type": "delegate_text", "agent": TRAVEL, "text": "travel-buddy answers: plan Hue", "append": False}
    order = [i["type"] for i in items if isinstance(i, dict) and i.get("agent") == TRAVEL]
    assert order[0] == "delegate" and order[-1] == "delegate_done" and set(order[1:-1]) == {"delegate_text"}
    assert tool_messages(rig.model.seen[-1])[-1].content == "travel-buddy answers: plan Hue"  # the complete reply, not a chunk


def test_streaming_delegation_falls_back_to_send_for_an_agent_without_streaming(rig, monkeypatch, stub_agents):
    quiet = stub_agents("quiet-agent", streaming=False)
    rig.use_agents(quiet)
    monkeypatch.setattr(agent, "A2A_STREAMING", True)
    rig.model.script = [call_both(**{"quiet-agent": "hi"}), AIMessage("ok")]
    items = collect_stream("go")
    assert quiet.calls[0].body["method"] == "message/send"
    assert not [i for i in items if isinstance(i, dict) and i["type"] == "delegate_text"]


def test_stream_does_not_leak_summarization_tokens(rig, monkeypatch):
    monkeypatch.setattr(agent, "SUMMARIZE_AT_TOKENS", 100)
    monkeypatch.setattr(agent, "KEEP_MESSAGES", 2)
    for i in range(3):
        rig.model.script = [AIMessage("answer " * 40)]
        rig.turn(f"question {i} " + "blah " * 40)
    rig.model.script = [AIMessage("fresh reply")]
    items = collect_stream("one more " + "blah " * 40)
    assert rig.model.summaries
    assert "".join(i for i in items if isinstance(i, str)) == "fresh reply "


def test_stream_reply_comes_from_the_final_message_when_the_cap_ends_the_turn(rig):
    rig.model.script = [AIMessage("", tool_calls=[tool_call(f"c{i}", "top_gainers", limit=i)]) for i in range(40)]
    result = collect_stream("loop")[-1]
    assert result.reply == agent.LIMIT_REPLY and len(rig.model.seen) == agent.MODEL_CALL_LIMIT


def test_an_empty_model_answer_falls_back_to_a_message(rig):
    rig.model.script = [AIMessage("")]
    assert rig.turn("hi").reply == agent.FALLBACK_REPLY


def test_abandoning_the_stream_cancels_the_call_to_the_agent(rig):
    rig.travel.delay = 5
    rig.model.script = [call_both(**{TRAVEL: "plan"}), AIMessage("ok")]
    cancelled = []

    async def job(emit):
        try:
            async for item in agent.stream_turn("go", "alice", "s1", request_id="r"):
                emit(item)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    async def consume():
        async for item in agent_loop.stream_on_loop(job):
            if isinstance(item, dict) and item["type"] == "delegate":
                break  # the client went away while the agent was working

    started = time.monotonic()
    asyncio.run(consume())
    deadline = time.monotonic() + 3
    while not cancelled and time.monotonic() < deadline:
        time.sleep(0.02)
    assert cancelled and time.monotonic() - started < 4


# --- misc ------------------------------------------------------------------------------------------------------------

def test_warm_up_loads_the_agents_and_the_tools(rig, caplog):
    with caplog.at_level("INFO", logger="trip-concierge"):
        agent.warm_up()
    assert "startup: 2 of 2 remote agent(s) and 1 MCP tool(s) loaded" in caplog.text


def test_run_config_sets_identity_and_a_recursion_limit_above_langgraphs_default():
    config = agent._run_config("alice", "s1", "req-9", None)
    assert config["configurable"] == {"thread_id": "s1", "actor_id": "alice", "request_id": "req-9"}
    assert config["metadata"] == {"request_id": "req-9"}
    assert config["recursion_limit"] == agent.RECURSION_LIMIT > 25
    assert config["callbacks"] == []


def test_whoami_decodes_the_runtime_token(monkeypatch):
    payload = base64.urlsafe_b64encode(json.dumps({"sub": "svc-1", "azp": "client"}).encode()).decode().rstrip("=")
    monkeypatch.setattr(mcp_client, "get_token", lambda force=False: f"h.{payload}.s")
    monkeypatch.setenv("GREENNODE_CLIENT_ID", "cid")
    assert agent.whoami() == {"client_id": "cid", "token_sub": "svc-1", "azp": "client", "authAccountId": ""}


def test_ping_memory_asks_the_memory_service_for_one_actor(monkeypatch):
    calls = []

    class FakeClient:
        async def list_actors_async(self, id, page, size):  # noqa: A002 - SDK keyword names
            calls.append((id, page, size))

    monkeypatch.setattr(agent, "_memory_client", FakeClient())
    asyncio.run(agent_loop.arun_coro(agent.ping_memory()))
    assert calls == [("memory-test", 1, 1)]
