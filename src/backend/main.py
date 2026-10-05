"""Trip Concierge - GreenNode AgentBase sample: an orchestrator that delegates over A2A and MCP (backend).

Endpoints:
  POST /invocations         SDK entrypoint: one chat turn (needs the user and session headers)
  GET  /health              SDK liveness probe
  GET  /                    bundled chat UI (src/frontend)
  GET  /api/info            UI configuration (details only for authorized callers)
  POST /api/chat/stream     the same chat turn as an SSE stream: tokens, delegation progress, the answer
  GET  /ready               deep readiness check (memory, MCP tools, remote agents, LLM config)
  GET  /.well-known/agent-card.json, POST /a2a    A2A protocol (JSON-RPC): this agent can be composed further

Local run: `python main.py` (the SDK serves on port 8080; a .env file is loaded).
When AGENT_API_KEY is set, everything except the UI, /health, /api/info (summary only) and
the agent card requires the X-API-Key header.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import re
import threading
import uuid
from contextlib import contextmanager, nullcontext
from functools import partial
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv

# Only the repo's own .env (<repo>/.env). load_dotenv() without a path would walk UP the directory
# tree and could pick up an unrelated .env from a parent folder.
# Loaded before the imports below: agent.py reads its configuration from the environment.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from greennode_agentbase import (  # noqa: E402
    GreenNodeAgentBaseApp,
    GreenNodeRequestError,
    GreenNodeRuntimeError,
    PingStatus,
    RequestContext,
)
from starlette.datastructures import Headers  # noqa: E402
from starlette.requests import Request  # noqa: E402
from starlette.responses import JSONResponse, StreamingResponse  # noqa: E402
from starlette.staticfiles import StaticFiles  # noqa: E402

import agent as agent_mod  # noqa: E402
import agent_loop  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("trip-concierge")

app = GreenNodeAgentBaseApp()

AGENT_NAME = "trip-concierge"

# AGENT_API_KEY (optional locally, REQUIRED for any public deployment): callers must send it in
# the X-API-Key header. Every turn spends LLM credits and calls the remote agents on behalf of a
# user id the CLIENT chooses, so without a key anyone who can reach the endpoint can do both.
AGENT_API_KEY = os.environ.get("AGENT_API_KEY", "").strip()
if AGENT_API_KEY.startswith(agent_mod.PLACEHOLDER):
    raise ValueError("AGENT_API_KEY still has the .env.example placeholder: set a random secret or remove it.")
# DEBUG_OPS=1 enables the `whoami` op (exposes the runtime identity; only for policy setup).
DEBUG_OPS = os.environ.get("DEBUG_OPS", "0").strip().lower() in ("1", "true", "yes")
# Public URL of this runtime, written into the A2A agent card.
A2A_PUBLIC_URL = os.environ.get("A2A_PUBLIC_URL", "").rstrip("/")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

MAX_MESSAGE_CHARS = 4000
GENERIC_ERROR = "The agent could not complete the request. Please try again."

# --- Langfuse tracing (optional) -----------------------------------------------------------
# Set LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and LANGFUSE_HOST to trace every turn: one trace per
# turn, with the delegation calls as tool spans (agent name and URL in their metadata).


def _lf_tracing() -> bool:
    """Langfuse v4 tracing is on when all three env vars are set (the SDK reads them itself)."""
    return bool(
        os.environ.get("LANGFUSE_PUBLIC_KEY")
        and os.environ.get("LANGFUSE_SECRET_KEY")
        and os.environ.get("LANGFUSE_HOST")
    )


def _lf_scope(trace_name: str, user_id: str = "", session_id: str = "", tags: list | None = None):
    """Langfuse v4 `propagate_attributes` scope: trace name, user, session and tags apply to the
    root observation and every child (including cost-bearing generations).

    Enter it BEFORE creating the CallbackHandler and running the agent (same task/context).
    With tracing off it is a no-op context manager."""
    if not _lf_tracing():
        return nullcontext()
    try:
        from langfuse import propagate_attributes

        kwargs: dict = {"trace_name": trace_name, "tags": tags or []}
        if user_id:
            kwargs["user_id"] = user_id
        if session_id:
            kwargs["session_id"] = session_id
        return propagate_attributes(**kwargs)
    except Exception as e:
        logger.warning("Langfuse scope disabled: %s", e)
        return nullcontext()


def _lf_callback():
    """Langfuse v4 CallbackHandler (OTel, auth from env). Create it INSIDE the scope so it
    inherits the trace context. None means tracing is off."""
    if not _lf_tracing():
        return None
    try:
        from langfuse.langchain import CallbackHandler

        return CallbackHandler()
    except Exception as e:
        logger.warning("Langfuse callback disabled: %s", e)
        return None


@contextmanager
def _traced(trace_name: str, user_id: str, session_id: str, tags: list[str]):
    """Trace one turn; yields the callbacks list for the agent run ([] when tracing is off)."""
    with _lf_scope(trace_name, user_id, session_id, tags):
        callback = _lf_callback()
        yield [callback] if callback else []


# --- authentication ------------------------------------------------------------------------

_PROTECTED_PATHS = {"/invocations", "/a2a", "/ready"}


def _requires_key(path: str) -> bool:
    """Everything that runs the LLM or calls the remote agents. Public: /, static files, /health,
    /api/info (summary only) and the A2A agent card."""
    path = path.rstrip("/")  # "/a2a/" must not be a way around the key
    return path in _PROTECTED_PATHS or (path.startswith("/api/") and path != "/api/info")


def _key_ok(headers: Headers) -> bool:
    """True when no key is configured or the request carries the right X-API-Key."""
    if not AGENT_API_KEY:
        return True
    provided = headers.get("x-api-key", "")
    return hmac.compare_digest(provided.encode(), AGENT_API_KEY.encode())


class ApiKeyMiddleware:
    """Pure-ASGI middleware enforcing X-API-Key on the protected paths (no-op without a key)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] == "http"
            and _requires_key(scope.get("path", ""))
            and not _key_ok(Headers(scope=scope))
        ):
            response = JSONResponse(
                {"status": "error", "error": "Unauthorized: missing or invalid X-API-Key header."},
                status_code=401,
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


# --- request validation --------------------------------------------------------------------

USER_HEADER = "X-GreenNode-AgentBase-User-Id"
SESSION_HEADER = "X-GreenNode-AgentBase-Session-Id"
MISSING_HEADERS_MSG = (
    f"Missing required headers: {USER_HEADER} and {SESSION_HEADER} (the user id is passed on to the "
    "remote agents, which partition their memory with it; the session id names the conversation). "
    "There are no defaults, to avoid mixing data between users."
)
MISSING_A2A_USER_MSG = (
    f"Missing header {USER_HEADER} (the user id passed on to the remote agents): an A2A caller must "
    "go through AgentBase Runtime, which attaches it, or send it itself."
)
# User and session ids end up in the memory of this agent and of the remote agents, in their
# namespaces and URL paths: keep them boring.
_ID_RE = re.compile(r"[A-Za-z0-9._:@+=~-]{1,128}")


class BadRequest(ValueError):
    """The client sent something we refuse to process (maps to HTTP 400)."""


def _valid_id(value: str) -> bool:
    return bool(_ID_RE.fullmatch(value or ""))


def _identity_error(user_id: str, session_id: str) -> str | None:
    """Error message when the user/session ids are missing or malformed, else None."""
    if not (user_id or "").strip() or not (session_id or "").strip():
        return MISSING_HEADERS_MSG
    if not (_valid_id(user_id) and _valid_id(session_id)):
        return "Invalid user or session id: use 1-128 of letters, digits and . _ : @ + = ~ -"
    return None


def _check_text(text: str) -> str:
    text = text.strip()
    if not text:
        raise BadRequest("The message is empty.")
    if len(text) > MAX_MESSAGE_CHARS:
        raise BadRequest(f"The message is too long (max {MAX_MESSAGE_CHARS} characters).")
    return text


def _message_text(body) -> str:
    """The user message of a chat request body (`message`, or `input` as an alias)."""
    if not isinstance(body, dict):
        raise BadRequest("The request body must be a JSON object.")
    text = body.get("message", body.get("input"))
    if not isinstance(text, str):
        raise BadRequest('"message" must be a string.')
    return _check_text(text)


def _new_request_id() -> str:
    """One id per turn: it is in the response, in every log line of the turn, and travels to the
    remote agents in the X-GreenNode-AgentBase-Request-Id header."""
    return uuid.uuid4().hex[:12]


def _log_failure(what: str, request_id: str = "") -> str:
    """Log the exception being handled and return the request id for the client to quote."""
    request_id = request_id or _new_request_id()
    logger.exception("%s failed (request_id=%s)", what, request_id)
    return request_id


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


_SSE_HEADERS = {"Cache-Control": "no-store", "X-Accel-Buffering": "no"}


# --- one chat turn (shared by /invocations, /api/chat/stream and A2A) ----------------------
# Both jobs run on the persistent agent loop (see agent_loop.arun_coro / stream_on_loop).

async def _turn_job(
    trace_name: str, tags: list[str], text: str, user_id: str, session_id: str, request_id: str
) -> agent_mod.TurnResult:
    with _traced(trace_name, user_id, session_id, tags) as callbacks:
        return await agent_mod.run_turn(text, user_id, session_id, request_id=request_id, callbacks=callbacks)


async def _stream_job(
    trace_name: str, tags: list[str], text: str, user_id: str, session_id: str, request_id: str, emit
) -> None:
    """Emits text tokens (str), progress events (dict), then the final TurnResult."""
    with _traced(trace_name, user_id, session_id, tags) as callbacks:
        async for item in agent_mod.stream_turn(
            text, user_id, session_id, request_id=request_id, callbacks=callbacks
        ):
            emit(item)


# --- POST /invocations ---------------------------------------------------------------------

@app.entrypoint
async def handler(payload: dict, context: RequestContext) -> dict:
    """Chat entrypoint. The user and session headers are mandatory (they name the remote
    agents' memory partition and conversation)."""
    if not isinstance(payload, dict):
        raise GreenNodeRequestError("The request body must be a JSON object.", status_code=400)

    if payload.get("op") == "whoami":
        if not DEBUG_OPS:
            raise GreenNodeRequestError(
                "whoami is disabled. Set DEBUG_OPS=1 (only while setting up policies) and restart.",
                status_code=403,
            )
        try:
            return {"status": "success", "agent": AGENT_NAME, **await asyncio.to_thread(agent_mod.whoami)}
        except Exception:
            raise GreenNodeRuntimeError(
                GENERIC_ERROR, details={"request_id": _log_failure("whoami")}
            ) from None

    user_id, session_id = context.user_id or "", context.session_id or ""
    error = _identity_error(user_id, session_id)
    if error:
        raise GreenNodeRequestError(error, status_code=400)
    try:
        text = _message_text(payload)
    except BadRequest as e:
        raise GreenNodeRequestError(str(e), status_code=400) from None

    request_id = _new_request_id()
    try:
        result = await agent_loop.arun_coro(
            _turn_job("trip-concierge-chat", ["chat"], text, user_id, session_id, request_id)
        )
    except Exception:
        raise GreenNodeRuntimeError(
            GENERIC_ERROR, details={"request_id": _log_failure("chat turn", request_id)}
        ) from None
    return {
        "status": "success",
        "agent": AGENT_NAME,
        "response": result.reply,
        "delegations": result.delegations,
        "request_id": request_id,
        "timestamp": agent_mod.now_vn().isoformat(),
    }


@app.ping
def health_check() -> PingStatus:
    return PingStatus.HEALTHY


# --- REST helpers for the bundled UI (same origin, no CORS) --------------------------------

def _json_error(message: str, status_code: int, **extra) -> JSONResponse:
    return JSONResponse({"status": "error", "error": message, **extra}, status_code=status_code)


async def _api_info(request: Request) -> JSONResponse:
    info = {"agent": AGENT_NAME, "auth_required": bool(AGENT_API_KEY)}
    if _key_ok(request.headers):  # resource identifiers only for authorized callers
        gateway = urlsplit(agent_mod.MCP_STOCK_URL)
        info.update(
            memory_id=agent_mod.MEMORY_ID,
            llm_model=agent_mod.LLM_MODEL,
            gateway=f"{gateway.scheme}://{gateway.netloc}",
            agents=[{"label": label, "name": name} for label, name in agent_mod.DIRECTORY.status()],
        )
    return JSONResponse(info)


# --- POST /api/chat/stream -----------------------------------------------------------------

async def _chat_stream(request: Request):
    """POST /api/chat/stream - body {"message": ...} plus the user and session headers.

    SSE events, in the order they happen:
      {"type":"token","text":...}                      the model writes
      {"type":"delegate","agent":...,"request":...}    a remote agent is being asked
      {"type":"delegate_text","agent":...,"text":...,"append":bool}
                                                       its partial answer (only with A2A_STREAMING=1)
      {"type":"delegate_done","agent":...,"ok":bool,"elapsed_ms":int}
      {"type":"tool","name":...}                       an MCP tool is being called
    and at the end {"type":"done","response":...,"delegations":[...],"request_id":...} or
    {"type":"error","error":...,"request_id":...}. If the client disconnects, the turn is
    cancelled (and with it the calls to the remote agents).
    """
    user_id = request.headers.get(USER_HEADER, "")
    session_id = request.headers.get(SESSION_HEADER, "")
    error = _identity_error(user_id, session_id)
    if error:
        return _json_error(error, 400)
    try:
        text = _message_text(await request.json())
    except ValueError as e:  # BadRequest, or a body that is not valid JSON
        return _json_error(str(e) if isinstance(e, BadRequest) else "Invalid JSON body.", 400)

    request_id = _new_request_id()

    async def events():
        job = partial(
            _stream_job, "trip-concierge-stream", ["chat", "stream"], text, user_id, session_id, request_id
        )
        try:
            async for item in agent_loop.stream_on_loop(job):
                if isinstance(item, agent_mod.TurnResult):
                    yield _sse({
                        "type": "done", "response": item.reply, "delegations": item.delegations,
                        "request_id": request_id,
                    })
                elif isinstance(item, dict):
                    yield _sse(item)
                else:
                    yield _sse({"type": "token", "text": item})
        except Exception:
            yield _sse({"type": "error", "error": GENERIC_ERROR, "request_id": _log_failure("chat stream", request_id)})

    return StreamingResponse(events(), media_type="text/event-stream", headers=_SSE_HEADERS)


# --- A2A (Agent-to-Agent protocol): agent card + JSON-RPC /a2a -------------------------------
# The card is served at /.well-known/agent-card.json; POST /a2a accepts JSON-RPC 2.0
# message/send (returns a Message) and message/stream (SSE of status-update / artifact-update
# events). The A2A contextId is the conversation thread (session) id. The shapes are the ones of
# the sample agents this orchestrator calls, so orchestrators can be stacked.

def _a2a_card() -> dict:
    card = {
        "name": AGENT_NAME,
        "description": (
            "Trip concierge: answers composite travel requests (itinerary, restaurant booking, "
            "market data) by delegating each part to a specialist agent over A2A or to a tool over MCP."
        ),
        "url": f"{A2A_PUBLIC_URL}/a2a" if A2A_PUBLIC_URL else "/a2a",
        "version": "1.0.0",
        "protocolVersion": "0.3.0",
        "capabilities": {"streaming": True, "pushNotifications": False, "stateTransitionHistory": False},
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": [
            {
                "id": "trip-concierge",
                "name": "Composite trip requests",
                "description": (
                    "Splits a request into an itinerary, a restaurant booking and market-data parts, "
                    "asks the right agent or tool for each and answers once."
                ),
                "tags": ["travel", "booking", "stocks", "orchestration"],
                "examples": [
                    "Plan 2 days in Da Nang next weekend for 2 people, book a table at Quán Ngon 123 "
                    "on Saturday at 19:00, and tell me how tourism stocks did this week."
                ],
            },
        ],
        "preferredTransport": "JSONRPC",
    }
    if AGENT_API_KEY:
        card["securitySchemes"] = {"apiKey": {"type": "apiKey", "in": "header", "name": "X-API-Key"}}
        card["security"] = [{"apiKey": []}]
    return card


async def _agent_card_route(request: Request) -> JSONResponse:
    return JSONResponse(_a2a_card())


def _a2a_text(params) -> str:
    """Concatenated text parts of params.message; "" when there are none."""
    if params is None:
        return ""
    if not isinstance(params, dict):
        raise BadRequest("params must be an object.")
    message = params.get("message") or {}
    parts = message.get("parts", []) if isinstance(message, dict) else None
    if not isinstance(parts, list) or not all(isinstance(p, dict) for p in parts):
        raise BadRequest("params.message.parts must be a list of objects.")
    return "".join(
        str(p.get("text", "")) for p in parts if p.get("kind") == "text" or "text" in p
    ).strip()


def _a2a_ctx(body: dict, header_session: str = "") -> str:
    """contextId = thread id: the message's contextId, else the runtime's Session-Id header,
    else a fresh id (a new, unshared conversation)."""
    message = (body.get("params") or {}).get("message") or {}
    context_id = message.get("contextId") or header_session or f"a2a-{uuid.uuid4().hex[:12]}"
    if not isinstance(context_id, str) or not _valid_id(context_id):
        raise BadRequest("Invalid contextId.")
    return context_id


def _rpc_error(rid, code: int, message: str, status_code: int = 400, **data) -> JSONResponse:
    error = {"code": code, "message": message, **({"data": data} if data else {})}
    return JSONResponse({"jsonrpc": "2.0", "id": rid, "error": error}, status_code=status_code)


def _a2a_msg_result(rid, ctx: str, text: str) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": rid,
        "result": {
            "kind": "message",
            "messageId": f"msg-{uuid.uuid4()}",
            "contextId": ctx,
            "role": "agent",
            "parts": [{"kind": "text", "text": text}],
        },
    }


def _progress_text(event: dict) -> str | None:
    """What a caller of this agent reads while it works; None for events that are not shown over A2A
    (the partial text of another agent is not an artifact of this one)."""
    kind = event.get("type")
    if kind == "delegate":
        return f"Asking {event.get('agent')}"
    if kind == "delegate_done":
        return f"{event.get('agent')} {'answered' if event.get('ok') else 'failed'}"
    if kind == "tool":
        return f"Calling {event.get('name')}"
    return None


async def _a2a_route(request: Request):
    try:
        body = await request.json()
    except ValueError:
        return _rpc_error(None, -32700, "Parse error")
    if not isinstance(body, dict):
        return _rpc_error(None, -32600, "Invalid request: the body must be a JSON object")
    rid = body.get("id")
    method = body.get("method", "")
    if method not in ("message/send", "message/stream"):
        return _rpc_error(rid, -32601, f"Method not found: {method}")
    try:
        text = _check_text(_a2a_text(body.get("params")))
    except BadRequest as e:
        return _rpc_error(rid, -32602, str(e))
    # The user id is the real user from the runtime's header: a shared default would mix the
    # memories the remote agents keep per user.
    user_id = request.headers.get(USER_HEADER, "").strip()
    if not user_id:
        return _rpc_error(rid, -32602, MISSING_A2A_USER_MSG)
    try:
        ctx = _a2a_ctx(body, request.headers.get(SESSION_HEADER, "").strip())
    except BadRequest as e:
        return _rpc_error(rid, -32602, str(e))
    error = _identity_error(user_id, ctx)
    if error:
        return _rpc_error(rid, -32602, error)

    request_id = _new_request_id()
    if method == "message/send":
        try:
            result = await agent_loop.arun_coro(
                _turn_job("trip-concierge-a2a", ["a2a"], text, user_id, ctx, request_id)
            )
        except Exception:
            return _rpc_error(
                rid, -32603, GENERIC_ERROR, status_code=500,
                request_id=_log_failure("a2a message/send", request_id),
            )
        return JSONResponse(_a2a_msg_result(rid, ctx, result.reply))

    task_id = f"task-{uuid.uuid4()}"

    def event(kind: str, **fields) -> str:
        result = {"kind": kind, "taskId": task_id, "contextId": ctx, **fields}
        return _sse({"jsonrpc": "2.0", "id": rid, "result": result})

    def status(state: str, final: bool = False, message: str | None = None) -> str:
        task_status: dict = {"state": state, "timestamp": agent_mod.now_vn().isoformat()}
        if message:
            task_status["message"] = _a2a_msg_result(rid, ctx, message)["result"]
        return event("status-update", status=task_status, final=final)

    def artifact(text: str, append: bool, last_chunk: bool) -> str:
        return event(
            "artifact-update",
            artifact={"artifactId": "reply", "parts": [{"kind": "text", "text": text}]},
            append=append, lastChunk=last_chunk,
        )

    async def events():
        yield status("working")
        job = partial(
            _stream_job, "trip-concierge-a2a-stream", ["a2a", "stream"], text, user_id, ctx, request_id
        )
        appended = False
        try:
            async for item in agent_loop.stream_on_loop(job):
                if isinstance(item, agent_mod.TurnResult):
                    # The authoritative reply replaces the streamed chunks.
                    yield artifact(item.reply, append=False, last_chunk=True)
                    yield status("completed", final=True)
                elif isinstance(item, dict):
                    progress = _progress_text(item)
                    if progress:
                        yield status("working", message=progress)
                else:
                    yield artifact(item, append=appended, last_chunk=False)
                    appended = True
        except Exception:
            _log_failure("a2a message/stream", request_id)
            yield status("failed", final=True, message=f"{GENERIC_ERROR} (request {request_id})")

    return StreamingResponse(events(), media_type="text/event-stream", headers=_SSE_HEADERS)


# --- GET /ready: deep health check for operators -------------------------------------------

async def _ready(request: Request) -> JSONResponse:
    async def check_memory() -> dict:
        try:
            await agent_loop.arun_coro(agent_mod.ping_memory(), timeout=15)
            return {"ok": True}
        except Exception as e:
            logger.warning("readiness: memory check failed: %s", e)
            return {"ok": False, "error": type(e).__name__}

    async def check_mcp() -> dict:
        tools = await asyncio.to_thread(agent_mod.get_mcp_tools)  # retried until the gateway answers
        return {"ok": bool(tools), "tools": len(tools)}

    async def check_agents() -> dict:
        tools = await asyncio.to_thread(agent_mod.get_remote_tools)  # cards of failed agents are retried
        unavailable = [label for label, name in agent_mod.DIRECTORY.status() if name is None]
        return {"ok": not unavailable, "available": [t.name for t in tools], "unavailable": unavailable}

    memory, mcp, agents = await asyncio.gather(check_memory(), check_mcp(), check_agents())
    # Configuration only: checking the LLM for real would spend tokens on every probe.
    checks = {"memory": memory, "mcp": mcp, "agents": agents, "llm": {"ok": True, "model": agent_mod.LLM_MODEL}}
    ok = all(c["ok"] for c in checks.values())
    return JSONResponse({"status": "ok" if ok else "degraded", "checks": checks}, status_code=200 if ok else 503)


app.add_route("/ready", _ready, methods=["GET"])
app.add_route("/.well-known/agent-card.json", _agent_card_route, methods=["GET"])
app.add_route("/a2a", _a2a_route, methods=["POST"])
app.add_route("/api/chat/stream", _chat_stream, methods=["POST"])
app.add_route("/api/info", _api_info, methods=["GET"])
app.add_middleware(ApiKeyMiddleware)
# Static frontend: mounted last so it does not shadow the routes above.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="ui")


if __name__ == "__main__":
    # Discover the remote agents and the MCP tools now, in the background: the first turn finds
    # them loaded and a slow agent does not delay /health.
    threading.Thread(target=agent_mod.warm_up, daemon=True, name="warm-up").start()
    app.run(port=8080, host="0.0.0.0")
