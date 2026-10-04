"""Test setup: dummy environment BEFORE the backend modules are imported (agent.py validates it)."""
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("LLM_MODEL", "test-model")
os.environ.setdefault("AGENTBASE_MEMORY_ID", "memory-test")
os.environ.setdefault("MCP_STOCK_URL", "https://gw.example/stock")
# The agents the orchestrator imports with. Tests that call agents replace agent.DIRECTORY with
# one that points at stub servers on 127.0.0.1 (see fakes.StubAgent).
os.environ.setdefault("A2A_AGENTS", "TRAVEL,RESTAURANT")
os.environ.setdefault("A2A_AGENT_TRAVEL_URL", "http://127.0.0.1:9")
os.environ.setdefault("A2A_AGENT_RESTAURANT_URL", "http://127.0.0.1:9")
for name in (
    "AGENT_API_KEY", "A2A_IAM_AUTH", "A2A_STREAMING", "A2A_TIMEOUT_SECONDS", "A2A_PUBLIC_URL",
    "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST",
):
    os.environ.pop(name, None)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture()
def stub_agents():
    """Factory for stub A2A agents on 127.0.0.1; they are stopped when the test ends."""
    from fakes import StubAgent

    started: list = []

    def make(name: str = "travel-buddy", **kwargs):
        stub = StubAgent(name, **kwargs).start()
        started.append(stub)
        return stub

    yield make
    for stub in started:
        stub.stop()
