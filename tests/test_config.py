"""Startup validation: missing values and the .env.example placeholders are rejected, .env handling,
the A2A agent configuration, and .env.example in step with the code."""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from dotenv import dotenv_values

import agent
from a2a_client import AgentConfig

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "src" / "backend"
GOOD_ENV = {
    "LLM_API_KEY": "lap-real", "AGENTBASE_MEMORY_ID": "memory-1", "MCP_STOCK_URL": "https://gw.example/stock",
    "A2A_AGENTS": "TRAVEL", "A2A_AGENT_TRAVEL_URL": "https://travel.example",
}
CLEARED = (*GOOD_ENV, "AGENT_API_KEY", "A2A_IAM_AUTH", "A2A_TIMEOUT_SECONDS", "A2A_STREAMING", "A2A_AGENT_TRAVEL_API_KEY")


def start(module: str, **overrides) -> subprocess.CompletedProcess:
    """Import a backend module in a clean interpreter with the given environment."""
    env = {k: v for k, v in os.environ.items() if k not in CLEARED and not k.startswith("A2A_AGENT_")}
    env.update({**GOOD_ENV, **overrides, "PYTHONPATH": str(BACKEND)})
    env = {k: v for k, v in env.items() if v is not None}
    return subprocess.run(
        [sys.executable, "-c", f"import {module}"], env=env, cwd=ROOT / "tests",
        capture_output=True, text=True, timeout=60,
    )


def test_a_complete_configuration_starts():
    assert start("agent").returncode == 0
    assert start("main").returncode == 0


@pytest.mark.parametrize(
    ("name", "value"),
    [(name, "change-me") for name in ("LLM_API_KEY", "AGENTBASE_MEMORY_ID", "MCP_STOCK_URL", "A2A_AGENT_TRAVEL_URL")]
    + [("LLM_API_KEY", ""), ("MCP_STOCK_URL", "change-me-url")],
)
def test_each_required_value_is_checked(name, value):
    result = start("agent", **{name: value})
    assert result.returncode != 0 and f"{name} is not configured" in result.stderr


def test_at_least_one_agent_is_required():
    result = start("agent", A2A_AGENTS="")
    assert result.returncode != 0 and "A2A_AGENTS is not configured" in result.stderr


def test_the_example_api_key_placeholders_are_rejected():
    result = start("main", AGENT_API_KEY="change-me")
    assert result.returncode != 0 and "AGENT_API_KEY still has" in result.stderr
    assert start("main", AGENT_API_KEY="a-real-secret").returncode == 0
    result = start("agent", A2A_AGENT_TRAVEL_API_KEY="change-me")
    assert result.returncode != 0 and "A2A_AGENT_TRAVEL_API_KEY still has the .env.example placeholder" in result.stderr
    assert start("agent", A2A_AGENT_TRAVEL_API_KEY="the-key-of-travel").returncode == 0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"A2A_TIMEOUT_SECONDS": "abc"}, "A2A_TIMEOUT_SECONDS must be a number between 1 and 600"),
        ({"A2A_AGENTS": "TRAVEL,travel"}, "invalid label 'travel'"),
        ({"A2A_IAM_AUTH": "1", "A2A_AGENT_TRAVEL_URL": "http://travel.example"}, "must be an https URL"),
    ],
)
def test_a_wrong_agent_setting_fails_at_startup(overrides, message):
    result = start("agent", **overrides)
    assert result.returncode != 0 and message in result.stderr


def test_the_optional_settings_are_accepted():
    assert start("agent", A2A_TIMEOUT_SECONDS="12.5", A2A_STREAMING="1", A2A_IAM_AUTH="1").returncode == 0


def import_main_in_a_copy_of_the_repo(tmp_path, repo_env: str | None) -> str:
    """Import main from a copy at <tmp>/parent/repo with a .env in <tmp>/parent; return what it exported."""
    repo = tmp_path / "parent" / "repo"
    shutil.copytree(BACKEND, repo / "src" / "backend", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(BACKEND.parent / "frontend", repo / "src" / "frontend")
    (tmp_path / "parent" / ".env").write_text("TC_PROBE_PARENT=leaked\n")
    if repo_env is not None:
        (repo / ".env").write_text(repo_env)
    env = {k: v for k, v in os.environ.items() if not k.startswith("TC_PROBE")}
    env.update({**GOOD_ENV, "PYTHONPATH": str(repo / "src" / "backend")})
    result = subprocess.run(
        [sys.executable, "-c", "import main, os; print(os.environ.get('TC_PROBE_PARENT'), os.environ.get('TC_PROBE_REPO'))"],
        env=env, cwd=repo / "src" / "backend", capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip().splitlines()[-1]


def test_a_env_file_in_a_parent_directory_is_never_read(tmp_path):
    """python-dotenv's default search walks up the tree and would load unrelated secrets."""
    assert import_main_in_a_copy_of_the_repo(tmp_path, repo_env=None) == "None None"


def test_the_repo_env_file_is_read(tmp_path):
    assert import_main_in_a_copy_of_the_repo(tmp_path, repo_env="TC_PROBE_REPO=loaded\n") == "None loaded"


# --- parse_agents ----------------------------------------------------------------------------

def parse(**env):
    return agent.parse_agents({"A2A_AGENTS": "TRAVEL,RESTAURANT", "A2A_AGENT_TRAVEL_URL": "https://t.example",
                               "A2A_AGENT_RESTAURANT_URL": "https://r.example", **env})


def test_parse_agents_reads_the_urls_and_the_optional_keys():
    configs = parse(A2A_AGENT_RESTAURANT_API_KEY="  key-r  ")
    assert configs == [AgentConfig("TRAVEL", "https://t.example"), AgentConfig("RESTAURANT", "https://r.example", "key-r")]
    assert not any(c.iam_auth for c in configs)


def test_parse_agents_keeps_the_order_strips_blanks_and_trailing_slashes():
    configs = parse(A2A_AGENTS=" RESTAURANT , TRAVEL ,", A2A_AGENT_TRAVEL_URL="https://t.example/prefix/")
    assert [c.label for c in configs] == ["RESTAURANT", "TRAVEL"] and configs[1].base_url == "https://t.example/prefix"


@pytest.mark.parametrize(("value", "expected"), [("1", True), ("true", True), ("YES", True), ("0", False), ("", False), ("no", False)])
def test_parse_agents_iam_flag(value, expected):
    assert all(c.iam_auth is expected for c in parse(A2A_IAM_AUTH=value))


def test_parse_agents_allows_http_urls_without_iam_and_rejects_them_with_iam():
    assert parse(A2A_AGENT_TRAVEL_URL="http://travel:8080")[0].base_url == "http://travel:8080"
    with pytest.raises(ValueError, match="A2A_IAM_AUTH .* A2A_AGENT_TRAVEL_URL must be an https URL"):
        parse(A2A_AGENT_TRAVEL_URL="http://travel:8080", A2A_IAM_AUTH="1")


@pytest.mark.parametrize(
    "url",
    ["travel.example", "https://", "https://t.example?x=1", "https://t.example#frag", "file:///etc/passwd",
     "ftp://t.example", "https://user:pw@t.example"],
)
def test_parse_agents_rejects_malformed_urls(url):
    with pytest.raises(ValueError, match="A2A_AGENT_TRAVEL_URL must be an http.s. URL"):
        parse(A2A_AGENT_TRAVEL_URL=url)


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"A2A_AGENTS": ""}, "A2A_AGENTS is not configured"),
        ({"A2A_AGENTS": " , "}, "A2A_AGENTS is not configured"),
        ({"A2A_AGENTS": "TRAVEL,travel"}, "invalid label 'travel'"),
        ({"A2A_AGENTS": "TRAVEL,1ST"}, "invalid label '1ST'"),
        ({"A2A_AGENTS": "TRAVEL,A-B"}, "invalid label 'A-B'"),
        ({"A2A_AGENTS": "TRAVEL,TRAVEL"}, "listed twice"),
        ({"A2A_AGENTS": "TRAVEL,OTHER"}, "A2A_AGENT_OTHER_URL is not configured"),
        ({"A2A_AGENT_TRAVEL_URL": "change-me"}, "A2A_AGENT_TRAVEL_URL is not configured"),
        ({"A2A_AGENT_RESTAURANT_API_KEY": "change-me-too"}, "A2A_AGENT_RESTAURANT_API_KEY still has the .env.example placeholder"),
    ],
)
def test_parse_agents_rejects_a_wrong_configuration(env, message):
    with pytest.raises(ValueError, match=message):
        parse(**env)


def test_the_timeout_parser():
    assert agent._timeout_seconds("90") == 90 and agent._timeout_seconds("0.5e1") == 5
    for bad in ("", "x", "0.9", "601", "nan"):
        with pytest.raises(ValueError, match="A2A_TIMEOUT_SECONDS"):
            agent._timeout_seconds(bad)


# --- .env.example in step with the code ------------------------------------------------------------

EXAMPLE = (ROOT / ".env.example").read_text()
# Every variable assignment in the example, active or commented out.
DOCUMENTED = set(re.findall(r"^(?:# ?)?([A-Z][A-Z0-9_]*)=", EXAMPLE, re.M))
SOURCE = "\n".join(p.read_text() for p in BACKEND.glob("*.py"))
READ_BY_CODE = set(re.findall(r"""(?:environ|env)\.get\(\s*["']([A-Z][A-Z0-9_]*)["']""", SOURCE))
# The names of the per-agent variables are built from the labels in A2A_AGENTS.
PER_AGENT = re.compile(r"A2A_AGENT_[A-Z0-9_]+_(?:URL|API_KEY)")


def test_env_example_lists_every_variable_the_code_reads():
    assert READ_BY_CODE, "the pattern found no variable: update it"
    assert READ_BY_CODE <= DOCUMENTED, sorted(READ_BY_CODE - DOCUMENTED)
    assert 'f"A2A_AGENT_{label}_URL"' in SOURCE or "A2A_AGENT_{label}_URL" in SOURCE
    assert "A2A_AGENT_{label}_API_KEY" in SOURCE


def test_env_example_lists_nothing_the_code_does_not_read():
    unread = {n for n in DOCUMENTED if n not in READ_BY_CODE and not PER_AGENT.fullmatch(n)}
    assert not unread, sorted(unread)
    labels = dotenv_values(ROOT / ".env.example")["A2A_AGENTS"].split(",")
    for name in filter(PER_AGENT.fullmatch, DOCUMENTED):
        assert any(name in (f"A2A_AGENT_{label}_URL", f"A2A_AGENT_{label}_API_KEY") for label in labels), name


def test_every_placeholder_in_env_example_is_rejected_by_the_code():
    """Active and commented-out `change-me` values: a customer who forgets one fails at startup."""
    placeholders = dict(re.findall(r"^(?:# ?)?([A-Z][A-Z0-9_]*)=(change-me\S*)$", EXAMPLE, re.M))
    assert {"LLM_API_KEY", "AGENTBASE_MEMORY_ID", "AGENT_API_KEY"} <= set(placeholders)
    for name, value in placeholders.items():
        agents = {"A2A_AGENTS": "TRAVEL_BUDDY,ZALO_RESTAURANT", "A2A_AGENT_TRAVEL_BUDDY_URL": "https://t.example",
                  "A2A_AGENT_ZALO_RESTAURANT_URL": "https://z.example"}
        result = start("main", **{**agents, name: value})
        assert result.returncode != 0 and name in result.stderr, name


def test_the_active_values_of_env_example_other_than_placeholders_are_valid():
    """What a customer gets after filling in only the placeholders must start."""
    values = {k: v for k, v in dotenv_values(ROOT / ".env.example").items() if v}
    values = {k: ("lap-real" if v.startswith("change-me") else v) for k, v in values.items()}
    values["AGENTBASE_MEMORY_ID"] = "memory-1"
    assert start("main", **values).returncode == 0
