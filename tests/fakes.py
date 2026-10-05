"""Test doubles: scripted chat model, in-memory AgentBase Memory API, stub A2A agents on 127.0.0.1."""
from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Awaitable, Callable
from types import SimpleNamespace

import uvicorn
from greennode_agentbase.memory.models import EventEntity, ListResponseEventEntity
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route


def tool_call(call_id: str, name: str, **args) -> dict:
    return {"name": name, "args": args or {"request": call_id}, "id": call_id, "type": "tool_call"}


class ScriptedChatModel(BaseChatModel):
    """Plays back `script` (a list of AIMessage). The summarization middleware's call is
    answered separately and recorded in `summaries`. `seen` has the messages of every agent call,
    `bound` the tools of the last call."""

    script: list = []
    seen: list = []
    summaries: list = []
    bound: list = []
    fail_with: list = []  # exceptions raised by the next calls, in order

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        self.bound[:] = list(tools)
        return self

    def _next(self, messages) -> AIMessage:
        if len(messages) == 1 and "Context Extraction Assistant" in str(messages[0].content):
            self.summaries.append(messages[0])
            return AIMessage("SUMMARY")
        self.seen.append(list(messages))
        if self.fail_with:
            raise self.fail_with.pop(0)
        return self.script.pop(0) if self.script else AIMessage("default answer")

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        message = self._next(messages)
        if message.tool_calls:
            chunks = [
                {"name": c["name"], "args": json.dumps(c["args"]), "id": c["id"], "index": i}
                for i, c in enumerate(message.tool_calls)
            ]
            yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=chunks))
            return
        for word in str(message.content).split(" "):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=word + " "))
            if run_manager:
                run_manager.on_llm_new_token(word + " ", chunk=chunk)
            yield chunk


class FakeCheckpointApi:
    """Sync part of the Memory API used by the AgentBaseMemoryEvents checkpointer."""

    def __init__(self):
        self.events: list = []
        self.creates = 0

    def create_event(self, id, actorId, sessionId, request):  # noqa: A002 - SDK keyword names
        self.creates += 1
        self.events.append(EventEntity(id=str(len(self.events)), payload=request.payload.model_dump()))

    def list_events(self, id, actorId, sessionId, page, size):  # noqa: A002
        data = self.events[(page - 1) * size : page * size]
        return ListResponseEventEntity(listData=data, totalPage=max(1, -(-len(self.events) // size)))


# --- stub A2A agents -------------------------------------------------------------------------

def rpc_message(rid, context_id: str, text: str) -> dict:
    """The Message result of the sample agents (their `_a2a_msg_result`)."""
    return {
        "jsonrpc": "2.0",
        "id": rid,
        "result": {
            "kind": "message",
            "messageId": "msg-stub",
            "contextId": context_id,
            "role": "agent",
            "parts": [{"kind": "text", "text": text}],
        },
    }


def sse(rid, result: dict) -> str:
    return f"data: {json.dumps({'jsonrpc': '2.0', 'id': rid, 'result': result})}\n\n"


def stream_events(rid, context_id: str, text: str) -> list[str]:
    """The events of a sample agent's message/stream: working, one artifact chunk per word,
    the complete reply (append=false, lastChunk=true), completed."""
    base = {"taskId": "task-stub", "contextId": context_id}

    def artifact(chunk: str, append: bool, last: bool) -> str:
        return sse(rid, {
            "kind": "artifact-update", **base, "append": append, "lastChunk": last,
            "artifact": {"artifactId": "reply", "parts": [{"kind": "text", "text": chunk}]},
        })

    def status(state: str, final: bool) -> str:
        return sse(rid, {"kind": "status-update", **base, "final": final, "status": {"state": state}})

    words = text.split(" ")
    return [
        status("working", False),
        *(artifact(w + " ", i > 0, False) for i, w in enumerate(words)),
        artifact(text, False, True),
        status("completed", True),
    ]


class StubAgent:
    """An A2A server on 127.0.0.1 with the shapes of the sample agents (travel-buddy, zalo-restaurant).

    `calls` records every JSON-RPC request (body, lower-cased headers, start/end time); `max_active`
    is the highest number of requests that were in progress at once. The behaviour is changed
    through the attributes: `answer` (reply text), `delay` (seconds before answering),
    `card_status`, and `respond`, an async hook (stub, body, request) -> Response | None that may
    replace the answer of one request.
    """

    def __init__(self, name: str = "travel-buddy", *, api_key: str = "", streaming: bool = True, card_url: str = ""):
        self.name, self.api_key, self.streaming, self.card_url = name, api_key, streaming, card_url
        self.card_status = 200
        self.delay = 0.0
        self.answer: Callable[[str], str] = lambda text: f"{name} answers: {text}"
        self.respond: Callable[..., Awaitable[Response | None]] | None = None
        self.calls: list[SimpleNamespace] = []
        self.active = self.max_active = 0
        self.app = Starlette(routes=[
            Route("/.well-known/agent-card.json", self._card),
            Route("/a2a", self._rpc, methods=["POST"]),
        ])
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self.url = ""

    # the card of the sample agents (`_a2a_card`), with their security scheme when a key is set
    def card(self) -> dict:
        card = {
            "name": self.name,
            "description": f"{self.name} stub agent.",
            "url": self.card_url or "/a2a",
            "version": "1.0.0",
            "protocolVersion": "0.3.0",
            "capabilities": {"streaming": self.streaming, "pushNotifications": False, "stateTransitionHistory": False},
            "defaultInputModes": ["text/plain"],
            "defaultOutputModes": ["text/plain"],
            "skills": [{
                "id": "skill-1", "name": "Main skill", "description": "Does the main thing.",
                "tags": ["stub"], "examples": ["Do the main thing"],
            }],
            "preferredTransport": "JSONRPC",
        }
        if self.api_key:
            card["securitySchemes"] = {"apiKey": {"type": "apiKey", "in": "header", "name": "X-API-Key"}}
            card["security"] = [{"apiKey": []}]
        return card

    async def _card(self, request: Request) -> Response:
        if self.card_status != 200:
            return Response(status_code=self.card_status)
        return JSONResponse(self.card())

    async def _rpc(self, request: Request) -> Response:
        body = await request.json()
        call = SimpleNamespace(body=body, headers=dict(request.headers), started=time.monotonic(), finished=0.0)
        self.calls.append(call)
        if self.api_key and request.headers.get("x-api-key") != self.api_key:
            return JSONResponse({"status": "error", "error": "Unauthorized: missing or invalid X-API-Key header."}, 401)
        message = body["params"]["message"]
        text = "".join(p["text"] for p in message["parts"])
        context_id = message.get("contextId", "ctx")
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        streaming = False  # True once the stream owns the "active" count
        try:
            await asyncio.sleep(self.delay)
            if self.respond and (response := await self.respond(self, body, request)) is not None:
                return response
            reply = self.answer(text)
            if body["method"] == "message/stream":
                streaming = True
                return StreamingResponse(
                    self._stream(call, body["id"], context_id, reply), media_type="text/event-stream"
                )
            return JSONResponse(rpc_message(body["id"], context_id, reply))
        finally:
            if not streaming:
                self.active -= 1
                call.finished = time.monotonic()

    async def _stream(self, call: SimpleNamespace, rid, context_id: str, reply: str):
        try:
            for event in stream_events(rid, context_id, reply):
                yield event
                await asyncio.sleep(0)
        finally:
            self.active -= 1
            call.finished = time.monotonic()

    def start(self) -> StubAgent:
        config = uvicorn.Config(self.app, host="127.0.0.1", port=0, log_level="warning", lifespan="off")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True, name=f"stub-{self.name}")
        self._thread.start()
        deadline = time.monotonic() + 10
        while not self._server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("the stub agent did not start")
            time.sleep(0.01)
        port = self._server.servers[0].sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
            self._thread.join(10)
            self._server = None
