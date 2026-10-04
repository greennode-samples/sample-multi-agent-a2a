"""mcp_client: retries, SSE parsing, policy denial, output truncation (httpx.MockTransport)."""
import base64
import json

import httpx
import pytest

import mcp_client


@pytest.fixture()
def gateway(monkeypatch):
    """Route mcp_client's httpx.Client through a MockTransport; `calls` records the JSON-RPC bodies."""
    calls: list[dict] = []
    state = {"handler": None}
    real_client = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return state["handler"](request, len(calls))

    monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(mcp_client, "get_token", lambda force=False: "token")
    monkeypatch.setattr(mcp_client.time, "sleep", lambda s: None)
    gw = type("Gateway", (), {})()
    gw.calls = calls
    gw.respond = lambda fn: state.update(handler=fn)
    return gw


def rpc(result=None, error=None) -> httpx.Response:
    body = {"jsonrpc": "2.0", "id": 1}
    body.update({"error": error} if error else {"result": result})
    return httpx.Response(200, json=body)


def test_jwt_helpers():
    payload = base64.urlsafe_b64encode(json.dumps({"sub": "u1", "exp": 1790000000}).encode()).decode().rstrip("=")
    assert mcp_client.jwt_claims(f"h.{payload}.s")["sub"] == "u1"
    assert mcp_client.jwt_claims("not-a-jwt") == {}
    assert mcp_client.jwt_claims("a.!!!.c") == {}


def test_tool_output_is_truncated_with_marker(gateway):
    gateway.respond(lambda req, n: rpc({"content": [{"type": "text", "text": "X" * 50_000}]}))
    out = mcp_client.call_tool("https://gw/mcp", "tavily_search", {"query": "q"})
    assert out.startswith("X" * mcp_client.MAX_TOOL_OUTPUT_CHARS)
    assert out.endswith(f"...[truncated {50_000 - mcp_client.MAX_TOOL_OUTPUT_CHARS} chars]")
    assert len(out) < mcp_client.MAX_TOOL_OUTPUT_CHARS + 60


def test_short_output_is_untouched(gateway):
    gateway.respond(lambda req, n: rpc({"content": [{"type": "text", "text": "hello"}]}))
    assert mcp_client.call_tool("https://gw/mcp", "t", {}) == "hello"


def test_denied_by_policy_http_403(gateway):
    gateway.respond(lambda req, n: httpx.Response(403, text="Request denied by policy."))
    out = mcp_client.call_tool("https://gw/mcp", "tavily_search", {})
    assert out.startswith("DENIED_BY_POLICY")


def test_denied_by_policy_in_error_result(gateway):
    gateway.respond(lambda req, n: rpc({"isError": True, "content": [{"type": "text", "text": "Denied by policy"}]}))
    assert mcp_client.call_tool("https://gw/mcp", "t", {}).startswith("DENIED_BY_POLICY")


def test_other_error_results_are_flagged_so_the_model_can_tell_they_failed(gateway):
    gateway.respond(lambda req, n: rpc({"isError": True, "content": [{"type": "text", "text": "Invalid API key"}]}))
    assert mcp_client.call_tool("https://gw/mcp", "t", {}) == "TOOL_ERROR: Invalid API key"
    gateway.respond(lambda req, n: rpc({"isError": True, "content": []}))
    assert mcp_client.call_tool("https://gw/mcp", "t", {}).startswith("TOOL_ERROR: {")
    gateway.respond(lambda req, n: rpc({"isError": False, "content": [{"type": "text", "text": "fine"}]}))
    assert mcp_client.call_tool("https://gw/mcp", "t", {}) == "fine"


def test_sse_response_skips_notifications(gateway):
    sse = (
        'data: {"jsonrpc":"2.0","method":"notifications/progress","params":{}}\n\n'
        'data: {"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"from sse"}]}}\n\n'
    )
    gateway.respond(lambda req, n: httpx.Response(200, text=sse))
    assert mcp_client.call_tool("https://gw/mcp", "t", {}) == "from sse"


def test_tools_list_retries_5xx_then_succeeds(gateway):
    gateway.respond(lambda req, n: httpx.Response(503) if n < 3 else rpc({"tools": [{"name": "a"}]}))
    assert mcp_client.list_tools("https://gw/mcp") == [{"name": "a"}]
    assert len(gateway.calls) == 3


def test_tools_list_gives_up_after_three_attempts(gateway):
    gateway.respond(lambda req, n: httpx.Response(503, text="down"))
    with pytest.raises(RuntimeError, match="503"):
        mcp_client.list_tools("https://gw/mcp")
    assert len(gateway.calls) == 3


def test_tools_call_is_not_retried_on_5xx(gateway):
    """Repeating a tools/call could run the tool twice."""
    gateway.respond(lambda req, n: httpx.Response(502, text="bad gateway"))
    out = mcp_client.call_tool("https://gw/mcp", "t", {})
    assert out.startswith("MCP_ERROR (HTTP 502)")
    assert len(gateway.calls) == 1


def test_tools_call_is_not_retried_on_read_timeout(gateway):
    def boom(req, n):
        raise httpx.ReadTimeout("slow", request=req)

    gateway.respond(boom)
    with pytest.raises(httpx.ReadTimeout):
        mcp_client.call_tool("https://gw/mcp", "t", {})
    assert len(gateway.calls) == 1


def test_connect_errors_are_retried_for_every_method(gateway):
    def flaky(req, n):
        if n < 3:
            raise httpx.ConnectError("refused", request=req)
        return rpc({"content": [{"type": "text", "text": "ok"}]})

    gateway.respond(flaky)
    assert mcp_client.call_tool("https://gw/mcp", "t", {}) == "ok"
    assert len(gateway.calls) == 3


def test_expired_token_is_refreshed_once(gateway, monkeypatch):
    forced = []
    monkeypatch.setattr(mcp_client, "get_token", lambda force=False: forced.append(force) or "tok")
    gateway.respond(lambda req, n: httpx.Response(401) if n == 1 else rpc({"content": []}))
    mcp_client.call_tool("https://gw/mcp", "t", {})
    assert forced == [False, True]


def test_list_tools_raises_on_jsonrpc_error(gateway):
    gateway.respond(lambda req, n: rpc(error={"code": -32000, "message": "nope"}))
    with pytest.raises(RuntimeError, match="nope"):
        mcp_client.list_tools("https://gw/mcp")
