"""Thin A2A (Agent2Agent protocol v0.3, JSON-RPC transport) client: agent cards and message calls.

Like mcp_client it is stateless: every call opens its own HTTP client. It knows nothing about
LangChain; agent.py turns the discovered agents into tools.

  - `fetch_agent`   GET <base-url>/.well-known/agent-card.json  ->  RemoteAgent
  - `AgentDirectory`  the discovered agents, refreshed on a TTL; a card that fails to load is
                    skipped and retried later (a failure is never cached as success)
  - `delegate`      one `message/send` (or `message/stream`) call -> the reply text

Credentials go only to the origin configured for the agent: the RPC URL in the card must be on
that origin, otherwise a malicious or misconfigured card could redirect the API key and the IAM
token somewhere else.

Retries: `message/send` and `message/stream` are NOT idempotent (a booking request could run
twice), so a call is repeated only when it provably did not reach the agent: a connection that
could not be opened, or HTTP 429. A timeout or a 5xx is never retried.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import aclosing
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlsplit

import httpx

import mcp_client

logger = logging.getLogger("a2a-client")

CARD_PATH = "/.well-known/agent-card.json"
API_KEY_HEADER = "X-API-Key"
USER_HEADER = "X-GreenNode-AgentBase-User-Id"
SESSION_HEADER = "X-GreenNode-AgentBase-Session-Id"
REQUEST_ID_HEADER = "X-GreenNode-AgentBase-Request-Id"

# Agent cards: a small GET, so a short timeout. A card that loads is trusted for CARD_TTL_SECONDS;
# one that fails is retried after CARD_RETRY_SECONDS (not on every turn: a dead agent must not
# slow down every request with its connect timeout).
CARD_TIMEOUT_SECONDS = 10
CARD_TTL_SECONDS = 600
CARD_RETRY_SECONDS = 30
CONNECT_TIMEOUT_SECONDS = 10

_SEND_ATTEMPTS = 3
_RETRY_BASE_DELAY = 0.5
_MAX_RETRY_AFTER_SECONDS = 5

# Cards are data from another system: cap what reaches the model through the tool description.
_MAX_NAME_CHARS = 64  # the limit of a function name in OpenAI-compatible APIs
_MAX_DESCRIPTION_CHARS = 1000
_MAX_SKILLS = 6
_MAX_EXAMPLES = 2

_FAILED_STATES = {"failed", "rejected", "canceled"}
_STOP_STATES = _FAILED_STATES | {"completed", "input-required", "auth-required"}


class A2AError(Exception):
    """A card or a call failed. The message is short and safe to show to the model (no trace)."""


@dataclass(frozen=True)
class AgentConfig:
    """One remote agent as configured by the operator."""

    label: str  # the LABEL in A2A_AGENTS, used in logs until the card gives the agent its name
    base_url: str  # no trailing slash
    api_key: str = field(default="", repr=False)  # sent as X-API-Key; empty = the agent has no key
    iam_auth: bool = False  # also send the runtime's IAM token as a bearer (Inbound Identity: IAM)


@dataclass(frozen=True)
class RemoteAgent:
    """A remote agent as described by its card. Equal cards compare equal, so a refresh that
    changes nothing does not rebuild the orchestrator."""

    config: AgentConfig
    name: str  # also the name of the tool
    description: str  # the description of the tool
    rpc_url: str
    streaming: bool  # the card advertises message/stream


# --- agent cards -----------------------------------------------------------------------------

def _clean(value: object, limit: int) -> str:
    """One line of printable text, cut to `limit` characters."""
    return " ".join(str(value or "").split())[:limit]


def _origin(url: str) -> tuple[str, str, int | None]:
    parts = urlsplit(url)
    default_port = {"http": 80, "https": 443}.get(parts.scheme)
    return parts.scheme, (parts.hostname or "").lower(), parts.port or default_port


def _tool_name(card_name: object) -> str:
    """The card name as a valid function name: letters, digits, `_` and `-` only ("" if unusable)."""
    if not isinstance(card_name, str):
        return ""
    safe = "".join(c if c.isascii() and (c.isalnum() or c in "_-") else "_" for c in card_name)
    return safe.strip("_-")[:_MAX_NAME_CHARS]


def _describe(card: dict) -> str:
    """The tool description: the card's description plus one line per skill with its examples."""
    description = _clean(card.get("description"), 400)
    lines = [description] if description else []
    skills = card.get("skills")
    for skill in (skills if isinstance(skills, list) else [])[:_MAX_SKILLS]:
        if not isinstance(skill, dict):
            continue
        name, summary = _clean(skill.get("name"), 80), _clean(skill.get("description"), 200)
        if not (name or summary):
            continue
        line = f"- {name}: {summary}"
        examples = skill.get("examples")
        shown = [_clean(e, 120) for e in (examples if isinstance(examples, list) else [])[:_MAX_EXAMPLES]]
        if shown:
            line += " Examples: " + "; ".join(f'"{e}"' for e in shown)
        lines.append(line)
    return "\n".join(lines)[:_MAX_DESCRIPTION_CHARS]


def agent_from_card(config: AgentConfig, card: object) -> RemoteAgent:
    """Validate an agent card and describe it as a RemoteAgent. Raises A2AError."""
    if not isinstance(card, dict):
        raise A2AError("the agent card is not a JSON object")
    name = _tool_name(card.get("name"))
    if not name:
        raise A2AError("the agent card has no usable name")
    transport = card.get("preferredTransport") or "JSONRPC"
    if transport != "JSONRPC":
        raise A2AError(f"the agent's preferred transport is {transport}; only JSONRPC is supported")
    url = card.get("url")
    if not isinstance(url, str) or not url.strip():
        raise A2AError("the agent card has no url")
    rpc_url = urljoin(config.base_url + "/", url.strip())
    if _origin(rpc_url) != _origin(config.base_url):
        raise A2AError(
            f"the card url {rpc_url} is not on the configured origin {config.base_url} "
            "(credentials are never sent elsewhere): set A2A_PUBLIC_URL of that agent to its real address"
        )
    capabilities = card.get("capabilities")
    streaming = isinstance(capabilities, dict) and capabilities.get("streaming") is True
    return RemoteAgent(config, name, _describe(card), rpc_url, streaming)


def _iam_token(force: bool = False) -> str:
    """The runtime's IAM token (the same one the MCP client uses). Raises A2AError."""
    try:
        return mcp_client.get_token(force)
    except (RuntimeError, httpx.HTTPError) as e:
        reason = str(e) if isinstance(e, RuntimeError) else type(e).__name__
        raise A2AError(f"could not get an IAM token ({reason})") from None


def _credentials(config: AgentConfig, *, force_token: bool = False) -> dict[str, str]:
    """The headers that authenticate this caller to the agent."""
    headers = {}
    if config.api_key:
        headers[API_KEY_HEADER] = config.api_key
    if config.iam_auth:
        headers["Authorization"] = f"Bearer {_iam_token(force_token)}"
    return headers


def fetch_agent(config: AgentConfig) -> RemoteAgent:
    """Load the agent card of `config`. Raises A2AError."""
    headers = _credentials(config)
    timeout = httpx.Timeout(CARD_TIMEOUT_SECONDS, connect=CONNECT_TIMEOUT_SECONDS)
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(config.base_url + CARD_PATH, headers=headers)
    except httpx.HTTPError as e:
        raise A2AError(f"the agent card could not be fetched ({type(e).__name__})") from None
    if response.status_code != 200:
        raise A2AError(f"the agent card request returned HTTP {response.status_code}")
    try:
        card = response.json()
    except ValueError:
        raise A2AError("the agent card is not valid JSON") from None
    return agent_from_card(config, card)


class AgentDirectory:
    """The remote agents discovered from their cards.

    `get()` returns the agents that are loaded now. Cards are fetched in parallel when they are
    due: never loaded, past the TTL, or past the retry delay after a failure. A failed refresh
    keeps the previous card.
    """

    def __init__(
        self, configs: list[AgentConfig], *, ttl: float = CARD_TTL_SECONDS, retry: float = CARD_RETRY_SECONDS
    ):
        self.configs = configs
        self._ttl, self._retry = ttl, retry
        self._lock = threading.Lock()
        self._agents: dict[str, RemoteAgent] = {}
        self._next_try: dict[str, float] = {}

    def _load(self, config: AgentConfig) -> RemoteAgent | None:
        try:
            return fetch_agent(config)
        except A2AError as e:
            reason = str(e)
        except Exception as e:  # a bug must not take the turn down
            logger.exception("agent %s: unexpected error while loading the card", config.label)
            reason = type(e).__name__
        with self._lock:
            kept = config.label in self._agents
        logger.warning(
            "agent %s: %s; %s", config.label, reason,
            "keeping the previous card" if kept else f"its tool is skipped, retrying in {self._retry:g} s",
        )
        return None

    def get(self) -> list[RemoteAgent]:
        now = time.monotonic()
        with self._lock:
            due = [c for c in self.configs if now >= self._next_try.get(c.label, 0.0)]
        if due:
            with ThreadPoolExecutor(max_workers=len(due)) as pool:  # network I/O: outside the lock
                loaded = list(pool.map(self._load, due))
            with self._lock:
                for config, agent in zip(due, loaded, strict=True):
                    if agent is None:
                        self._next_try[config.label] = now + self._retry
                    else:
                        self._agents[config.label] = agent
                        self._next_try[config.label] = now + self._ttl
        with self._lock:
            return [self._agents[c.label] for c in self.configs if c.label in self._agents]

    def status(self) -> list[tuple[str, str | None]]:
        """(label, agent name or None while its card is not loaded), without any network I/O."""
        with self._lock:
            return [
                (c.label, self._agents[c.label].name if c.label in self._agents else None)
                for c in self.configs
            ]


# --- calls -----------------------------------------------------------------------------------

async def _headers(
    config: AgentConfig, user_id: str, context_id: str, request_id: str, *, force_token: bool
) -> dict[str, str]:
    # get_token does blocking HTTP and takes a lock: keep it off the event loop
    credentials = await asyncio.to_thread(_credentials, config, force_token=force_token)
    return {USER_HEADER: user_id, SESSION_HEADER: context_id, REQUEST_ID_HEADER: request_id, **credentials}


def _retry_after(response: httpx.Response) -> float | None:
    try:
        return float(response.headers["Retry-After"])
    except (KeyError, ValueError):
        return None


async def _send_with_retry(client: httpx.AsyncClient, request: httpx.Request, *, stream: bool) -> httpx.Response:
    """Send `request`, repeating it only when the agent cannot have processed it: the connection
    could not be opened, or the answer is HTTP 429 (see the module docstring)."""
    delay = _RETRY_BASE_DELAY
    for attempt in range(1, _SEND_ATTEMPTS):
        try:
            response = await client.send(request, stream=stream)
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            reason, wait = type(e).__name__, delay
        else:
            if response.status_code != 429:
                return response
            await response.aclose()
            reason, wait = "HTTP 429", min(_retry_after(response) or delay, _MAX_RETRY_AFTER_SECONDS)
        logger.warning("%s: %s - retry %d/%d in %.1fs", request.url, reason, attempt, _SEND_ATTEMPTS - 1, wait)
        await asyncio.sleep(wait)
        delay *= 3
    return await client.send(request, stream=stream)  # final attempt: whatever happens goes to the caller


def _rpc_error_text(error: object) -> str:
    """The message of a JSON-RPC error object, with the request id the agent quotes, if any."""
    if not isinstance(error, dict):
        return ""
    text = _clean(error.get("message"), 200)
    data = error.get("data")
    if isinstance(data, dict) and data.get("request_id"):
        text += f" (request {_clean(data['request_id'], 40)})"
    return text


_HTTP_HINTS = {
    401: "the agent refused the credentials: check its API key and Inbound Identity settings",
    403: "the agent refused the credentials: check its API key and Inbound Identity settings",
    404: "no A2A endpoint at the agent's url",
    429: "the agent is rate limiting this caller",
}


def _http_error(response: httpx.Response) -> A2AError:
    detail = ""
    try:
        payload = response.json()
        detail = _rpc_error_text(payload.get("error")) if isinstance(payload, dict) else ""
    except ValueError:
        pass
    detail = detail or _HTTP_HINTS.get(response.status_code, "")
    return A2AError(f"HTTP {response.status_code}" + (f": {detail}" if detail else ""))


async def _post(
    client: httpx.AsyncClient, agent: RemoteAgent, body: dict, *,
    user_id: str, context_id: str, request_id: str, stream: bool,
) -> httpx.Response:
    """POST the JSON-RPC body to the agent. Returns an HTTP 200 response (still open when
    `stream`), raises A2AError for any other status."""
    force_token = False
    while True:
        headers = await _headers(agent.config, user_id, context_id, request_id, force_token=force_token)
        request = client.build_request("POST", agent.rpc_url, headers=headers, json=body)
        response = await _send_with_retry(client, request, stream=stream)
        if response.status_code == 200:
            return response
        try:
            await response.aread()
        finally:
            await response.aclose()
        if response.status_code == 401 and agent.config.iam_auth and not force_token:
            force_token = True  # the token may have expired or been revoked: fetch a new one, once
            continue
        raise _http_error(response)


def _parts_text(parts: object) -> str:
    """Text of A2A parts: text parts as they are (whitespace included: streamed chunks end in
    spaces), data parts as JSON, files ignored."""
    texts = []
    for part in parts if isinstance(parts, list) else []:
        if not isinstance(part, dict):
            continue
        if part.get("kind") == "data":
            texts.append(json.dumps(part.get("data"), ensure_ascii=False))
        elif part.get("kind") == "text" and isinstance(part.get("text"), str):
            texts.append(part["text"])
    return "".join(texts)


def _message_text(message: object) -> str:
    return _parts_text(message.get("parts")).strip() if isinstance(message, dict) else ""


def _finish(state: object, status_text: str, artifact_text: str) -> str:
    """The reply of a task that stopped in `state`; raises A2AError unless it completed (or asks
    the user something)."""
    if state in _FAILED_STATES:
        raise A2AError(f"the task {state}" + (f": {_clean(status_text, 300)}" if status_text else ""))
    if state == "auth-required":
        raise A2AError("the agent needs additional authentication (task state auth-required)")
    if state == "input-required":
        return f"INPUT_REQUIRED: {status_text or artifact_text or '(the agent gave no question)'}"
    if state != "completed":
        raise A2AError(
            f"the task did not finish (state: {state or 'missing'}); "
            "only agents that answer within the call are supported"
        )
    text = artifact_text or status_text
    if not text:
        raise A2AError("the agent returned no text")
    return text


def _result_text(result: object) -> str:
    """Reply text of a `message/send` result: a Message, or a Task with artifacts."""
    if not isinstance(result, dict):
        raise A2AError("the agent returned an unexpected result")
    if result.get("kind") == "message":
        text = _parts_text(result.get("parts")).strip()
        if not text:
            raise A2AError("the agent returned no text")
        return text
    if result.get("kind") == "task":
        status = result.get("status") if isinstance(result.get("status"), dict) else {}
        artifacts = result.get("artifacts")
        artifact_text = "\n\n".join(
            t for a in (artifacts if isinstance(artifacts, list) else [])
            if isinstance(a, dict) and (t := _parts_text(a.get("parts")).strip())
        )
        return _finish(status.get("state"), _message_text(status.get("message")), artifact_text)
    raise A2AError("the agent returned an unexpected result")


def _rpc_payload(response: httpx.Response) -> dict:
    try:
        payload = response.json()
    except ValueError:
        raise A2AError("the agent answered with something that is not JSON") from None
    if not isinstance(payload, dict):
        raise A2AError("the agent answered with something that is not a JSON-RPC response")
    if "error" in payload:
        raise A2AError(f"the agent returned an error: {_rpc_error_text(payload['error']) or 'no message'}")
    return payload


async def _sse_data(response: httpx.Response) -> AsyncIterator[str]:
    """The `data` of every server-sent event of a streamed response (other fields and comments are ignored)."""
    data: list[str] = []
    async for line in response.aiter_lines():
        if line == "":
            if data:
                yield "\n".join(data)
                data = []
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
    if data:
        yield "\n".join(data)


async def _stream_text(response: httpx.Response, on_text: Callable[[str, bool], None]) -> str:
    """Read a `message/stream` response and return the reply text.

    Artifact chunks are forwarded to `on_text(text, append)` as they arrive; the reply is the
    artifact content after the last update (a chunk with append=false replaces the artifact).
    """
    artifacts: dict[str, str] = {}
    outcome: tuple[object, str] | None = None  # (state, status text) of the update that ended the task
    async with aclosing(_sse_data(response)) as events:
        async for data in events:
            try:
                event = json.loads(data)
            except ValueError:
                raise A2AError("the agent sent a stream event that is not JSON") from None
            if not isinstance(event, dict):
                raise A2AError("the agent sent an unexpected stream event")
            if "error" in event:
                raise A2AError(f"the agent returned an error: {_rpc_error_text(event['error']) or 'no message'}")
            result = event.get("result")
            if not isinstance(result, dict):
                continue
            kind = result.get("kind")
            if kind == "artifact-update" and isinstance(result.get("artifact"), dict):
                artifact = result["artifact"]
                text = _parts_text(artifact.get("parts"))
                key, append = str(artifact.get("artifactId", "")), result.get("append") is True
                artifacts[key] = artifacts.get(key, "") + text if append else text
                if text:
                    on_text(text, append)
            elif kind == "status-update" and isinstance(result.get("status"), dict):
                state = result["status"].get("state")
                if result.get("final") is True or state in _STOP_STATES:
                    outcome = (state, _message_text(result["status"].get("message")))
                    break
            elif kind == "message" or (
                kind == "task" and isinstance(result.get("status"), dict) and result["status"].get("state") in _STOP_STATES
            ):  # a server may end a stream with a Message or a finished Task instead of status updates
                return _result_text(result)
    if outcome is None:
        raise A2AError("the stream ended before the agent finished")
    artifact_text = "\n\n".join(t for t in (text.strip() for text in artifacts.values()) if t)
    return _finish(outcome[0], outcome[1], artifact_text)


async def delegate(
    agent: RemoteAgent,
    request: str,
    *,
    user_id: str,
    context_id: str,
    request_id: str,
    timeout: float,
    on_text: Callable[[str, bool], None] | None = None,
) -> str:
    """Send `request` to `agent` and return its reply text, capped at mcp_client.MAX_TOOL_OUTPUT_CHARS.

    `user_id` goes to the agent as X-GreenNode-AgentBase-User-Id (it partitions the agent's
    memory); `context_id` names the conversation, so repeating it continues the same
    conversation (it is the message's contextId and also the X-GreenNode-AgentBase-Session-Id
    header: the sample agents treat the two as the same thing). With `on_text` and an agent whose card advertises streaming the call uses
    `message/stream` and reports the partial text as it arrives; otherwise `message/send`.
    `timeout` bounds the whole call. Raises A2AError.
    """
    stream = on_text is not None and agent.streaming
    message = {
        "kind": "message",
        "messageId": uuid.uuid4().hex,
        "role": "user",
        "contextId": context_id,
        "parts": [{"kind": "text", "text": request}],
    }
    params: dict = {"message": message}
    if not stream:
        params["configuration"] = {"blocking": True, "acceptedOutputModes": ["text/plain"]}
    body = {
        "jsonrpc": "2.0",
        "id": uuid.uuid4().hex,
        "method": "message/stream" if stream else "message/send",
        "params": params,
    }
    try:
        async with (
            asyncio.timeout(timeout),
            httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=CONNECT_TIMEOUT_SECONDS)) as client,
        ):
            response = await _post(
                client, agent, body, user_id=user_id, context_id=context_id, request_id=request_id, stream=stream
            )
            try:
                if stream:
                    text = await _stream_text(response, on_text)
                else:
                    text = _result_text(_rpc_payload(response).get("result"))
            finally:
                await response.aclose()
    except (httpx.ConnectError, httpx.ConnectTimeout):
        raise A2AError("the agent could not be reached (connection failed)") from None
    except (TimeoutError, httpx.TimeoutException):
        raise A2AError(f"the agent did not answer within {timeout:g} s") from None
    except httpx.TransportError as e:
        raise A2AError(f"the connection to the agent failed ({type(e).__name__})") from None
    return mcp_client.truncate(text)
