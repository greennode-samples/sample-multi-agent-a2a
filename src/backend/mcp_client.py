"""Thin stateless MCP client for an MCP Gateway / Connector endpoint.

Sends JSON-RPC requests straight to the MCP URL (no initialize handshake, no keep-alive).
The IAM Bearer token comes from GREENNODE_CLIENT_ID / GREENNODE_CLIENT_SECRET, which
AgentBase Runtime injects automatically.
"""

from __future__ import annotations

import base64
import itertools
import json
import logging
import os
import threading
import time

import httpx

logger = logging.getLogger("mcp-client")

IAM_TOKEN_URL = "https://iam.api.vngcloud.vn/accounts-api/v2/auth/token"

# Upper bound for a single tool result handed back to the model (MCP tools and remote agents).
# Market-data tools can return tens of thousands of characters, which would flood the context window.
MAX_TOOL_OUTPUT_CHARS = 8000

_HTTP_TIMEOUT = httpx.Timeout(120.0, connect=10.0)
_RETRY_ATTEMPTS = 3
_RETRY_BASE_DELAY = 0.5

_token_lock = threading.Lock()
_token_cache: dict = {"token": None, "exp": 0.0}
_request_ids = itertools.count(1)


def jwt_claims(token: str) -> dict:
    """Decode the payload of a JWT without verifying it. Returns {} if it is malformed."""
    try:
        part = token.split(".")[1]
        part += "=" * (-len(part) % 4)
        claims = json.loads(base64.urlsafe_b64decode(part))
    except (IndexError, ValueError):
        return {}
    return claims if isinstance(claims, dict) else {}


def get_token(force: bool = False) -> str:
    """Return an IAM access token (client credentials), cached until 60 s before expiry."""
    with _token_lock:
        now = time.time()
        if not force and _token_cache["token"] and now < _token_cache["exp"] - 60:
            return _token_cache["token"]

        client_id = os.environ.get("GREENNODE_CLIENT_ID")
        client_secret = os.environ.get("GREENNODE_CLIENT_SECRET")
        if not client_id or not client_secret:
            raise RuntimeError(
                "GREENNODE_CLIENT_ID / GREENNODE_CLIENT_SECRET are not set "
                "(AgentBase Runtime injects them automatically)."
            )
        r = httpx.post(
            IAM_TOKEN_URL,
            auth=(client_id, client_secret),
            data={"grant_type": "client_credentials"},
            timeout=30,
        )
        r.raise_for_status()
        token = r.json()["access_token"]
        _token_cache["token"] = token
        _token_cache["exp"] = float(jwt_claims(token).get("exp") or now + 1500)
        return token


def _post_with_retry(
    client: httpx.Client, mcp_url: str, headers: dict, body: dict, *, idempotent: bool
) -> httpx.Response:
    """POST with exponential backoff on transient failures.

    - Connection errors (the request never reached the server) are retried for every method.
    - Read timeouts, HTTP 429 and 5xx are retried only for idempotent methods (tools/list):
      repeating a tools/call could run the tool twice.
    """
    delay = _RETRY_BASE_DELAY
    for attempt in range(1, _RETRY_ATTEMPTS):
        try:
            response = client.post(mcp_url, headers=headers, json=body)
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            reason = repr(e)
        except httpx.ReadTimeout as e:
            if not idempotent:
                raise
            reason = repr(e)
        else:
            transient = response.status_code == 429 or response.status_code >= 500
            if not (idempotent and transient):
                return response
            reason = f"HTTP {response.status_code}"
        logger.warning(
            "%s transient failure (%s) - retry %d/%d in %.1fs",
            body["method"], reason, attempt, _RETRY_ATTEMPTS - 1, delay,
        )
        time.sleep(delay)
        delay *= 3
    # Final attempt: whatever it returns or raises goes back to the caller.
    return client.post(mcp_url, headers=headers, json=body)


def _parse_body(raw: str) -> dict | str:
    """Parse a JSON-RPC response that is either plain JSON or an SSE stream (`data:` lines)."""
    try:
        if raw.lstrip().startswith("{"):
            return json.loads(raw)
        for line in raw.splitlines():
            if not line.startswith("data:"):
                continue
            message = json.loads(line[5:].strip())
            # Skip server notifications that may precede the actual response.
            if isinstance(message, dict) and ("result" in message or "error" in message):
                return message
    except ValueError:
        logger.warning("MCP response is not valid JSON/SSE (%d chars)", len(raw))
    return raw


def mcp_request(
    mcp_url: str, method: str, params: dict | None = None
) -> tuple[int, dict | str]:
    """POST one JSON-RPC request to the MCP URL. Returns (http_status, parsed_body)."""
    body: dict = {"jsonrpc": "2.0", "id": next(_request_ids), "method": method}
    if params is not None:
        body["params"] = params

    headers = {
        "Authorization": f"Bearer {get_token()}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    idempotent = method == "tools/list"
    with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
        r = _post_with_retry(client, mcp_url, headers, body, idempotent=idempotent)
        if r.status_code == 401:  # token expired or revoked: refresh once and retry
            headers["Authorization"] = f"Bearer {get_token(force=True)}"
            r = _post_with_retry(client, mcp_url, headers, body, idempotent=idempotent)

    if r.status_code != 200:
        return r.status_code, r.text[:2000]
    return 200, _parse_body(r.text)


def list_tools(mcp_url: str) -> list[dict]:
    """tools/list - return the tool definitions."""
    status, body = mcp_request(mcp_url, "tools/list")
    if status != 200 or not isinstance(body, dict):
        raise RuntimeError(f"tools/list failed ({status}): {str(body)[:300]}")
    if "error" in body:
        raise RuntimeError(f"tools/list returned an error: {json.dumps(body['error'])[:300]}")
    return body.get("result", {}).get("tools", [])


def truncate(text: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    """Cap a tool result (an MCP tool or a remote agent's reply) at `limit` characters."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n...[truncated {len(text) - limit} chars]"


def call_tool(mcp_url: str, tool: str, arguments: dict) -> str:
    """tools/call - return the tool's text content, capped at MAX_TOOL_OUTPUT_CHARS.

    The MCP Gateway reports a policy denial in one of two ways (depending on its version):
      - HTTP 403
      - HTTP 200 + result.isError=true + text containing "denied by policy"
    Both are returned as a string starting with DENIED_BY_POLICY, which the system prompt
    tells the model to treat as "this tool is not allowed". Any other `isError` result is
    returned with a TOOL_ERROR prefix so the model can tell the tool failed.
    """
    status, body = mcp_request(mcp_url, "tools/call", {"name": tool, "arguments": arguments})
    if status == 403:
        logger.info("tools/call %s denied by policy (HTTP 403)", tool)
        return (
            f"DENIED_BY_POLICY (HTTP 403): tool '{tool}' is not allowed for this agent "
            "by the MCP Gateway Policy Group."
        )
    if status != 200:
        return truncate(f"MCP_ERROR (HTTP {status}) calling tool '{tool}': {body}")
    if not isinstance(body, dict):
        return truncate(str(body))
    if "error" in body:
        return truncate(f"MCP_RPC_ERROR: {json.dumps(body['error'])}")

    result = body.get("result", {})
    texts = [
        item.get("text", "")
        for item in result.get("content", [])
        if isinstance(item, dict) and item.get("type") == "text"
    ]
    joined = "\n".join(texts)
    if result.get("isError") and "denied by policy" in joined.lower():
        logger.info("tools/call %s denied by policy (isError result)", tool)
        return (
            f"DENIED_BY_POLICY: tool '{tool}' is not allowed for this agent "
            "by the MCP Gateway Policy Group."
        )
    if result.get("isError"):
        return truncate(f"TOOL_ERROR: {joined or json.dumps(result)}")
    return truncate(joined or json.dumps(result))
