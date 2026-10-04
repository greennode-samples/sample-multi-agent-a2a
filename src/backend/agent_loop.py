"""The persistent event loop that runs every agent turn.

The AgentBase SDK caches its async HTTP client per event loop, so every SDK / agent call runs on
one persistent background loop (see `arun_coro` and `stream_on_loop`); using several loops
fails with "Event loop is closed". The delegation tools also run here, which is why their
per-conversation locks (agent.py) can safely be `asyncio.Lock`s.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
from typing import Any

_loop: asyncio.AbstractEventLoop | None = None
_loop_lock = threading.Lock()


def agent_loop() -> asyncio.AbstractEventLoop:
    """The persistent background loop that runs every agent and memory SDK call."""
    global _loop
    with _loop_lock:
        if _loop is None or _loop.is_closed():
            _loop = asyncio.new_event_loop()
            threading.Thread(target=_loop.run_forever, daemon=True, name="agent-loop").start()
        return _loop


async def arun_coro(coro: Coroutine, timeout: float = 600):
    """Run `coro` on the agent loop and await its result without blocking the caller's loop.

    Cancelling the caller (or hitting `timeout`) cancels the coroutine on the agent loop.
    """
    future = asyncio.run_coroutine_threadsafe(coro, agent_loop())
    return await asyncio.wait_for(asyncio.wrap_future(future), timeout)


_ITEM, _ERROR, _END = "item", "error", "end"


async def stream_on_loop(produce: Callable[[Callable[[Any], None]], Awaitable[None]]) -> AsyncIterator:
    """Run `produce(emit)` on the agent loop and yield every emitted item on the caller's loop.

    `emit` is thread-safe. An exception raised by `produce` is re-raised here. When the
    consumer stops early (e.g. the HTTP client disconnected) the producer is cancelled, so an
    abandoned request stops spending LLM tokens and stops waiting for remote agents.
    """
    here = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def emit(item: Any) -> None:
        here.call_soon_threadsafe(queue.put_nowait, (_ITEM, item))

    async def run() -> None:
        try:
            await produce(emit)
        except Exception as e:
            here.call_soon_threadsafe(queue.put_nowait, (_ERROR, e))
        else:
            here.call_soon_threadsafe(queue.put_nowait, (_END, None))

    future = asyncio.run_coroutine_threadsafe(run(), agent_loop())
    try:
        while True:
            kind, value = await queue.get()
            if kind == _END:
                return
            if kind == _ERROR:
                raise value
            yield value
    finally:
        future.cancel()
