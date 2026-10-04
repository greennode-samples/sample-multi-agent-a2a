"""agent_loop: the persistent agent loop and the stream bridge to the caller's loop."""
import asyncio
import threading

import pytest

import agent_loop


def test_arun_coro_runs_on_the_agent_loop_without_blocking_the_caller():
    async def where():
        await asyncio.sleep(0)
        return asyncio.get_running_loop()

    async def main():
        ticks = []

        async def ticker():
            for _ in range(3):
                await asyncio.sleep(0.01)
                ticks.append(1)

        async def slow():
            await asyncio.sleep(0.1)
            return await where()

        loop, _ = await asyncio.gather(agent_loop.arun_coro(slow()), ticker())
        return loop, ticks

    loop, ticks = asyncio.run(main())
    assert loop is agent_loop.agent_loop()
    assert len(ticks) == 3  # the caller's loop stayed responsive


def test_arun_coro_times_out_and_cancels_the_coroutine():
    cancelled = threading.Event()

    async def hang():
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    with pytest.raises(TimeoutError):
        asyncio.run(agent_loop.arun_coro(hang(), timeout=0.2))
    assert cancelled.wait(2)


def test_stream_on_loop_relays_items_and_errors():
    async def produce(emit):
        emit("a")
        await asyncio.sleep(0)
        emit("b")

    async def fail(emit):
        emit("x")
        raise ValueError("boom")

    async def collect(job):
        out = []
        async for item in agent_loop.stream_on_loop(job):
            out.append(item)
        return out

    assert asyncio.run(collect(produce)) == ["a", "b"]

    async def failing():
        out = []
        with pytest.raises(ValueError, match="boom"):
            async for item in agent_loop.stream_on_loop(fail):
                out.append(item)
        return out

    assert asyncio.run(failing()) == ["x"]


def test_stream_on_loop_cancels_the_producer_when_the_consumer_stops():
    started = threading.Event()
    cancelled = threading.Event()

    async def endless(emit):
        started.set()
        try:
            while True:
                emit("tick")
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    async def consume_one():
        async for _ in agent_loop.stream_on_loop(endless):
            break

    asyncio.run(consume_one())
    assert started.wait(2) and cancelled.wait(2)
