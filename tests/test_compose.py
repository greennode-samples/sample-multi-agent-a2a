"""docker-compose.yml: the services, the `all` profile, the env files and the addresses in .env.example."""
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

import pytest
import yaml
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
COMPOSE = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
SERVICES = COMPOSE["services"]
SIBLINGS = {
    "travel-buddy": "https://github.com/greennode-samples/sample-travel-buddy.git#main",
    "zalo-restaurant": "https://github.com/greennode-samples/sample-zalo-restaurant.git#main",
    "restaurant-mcp-server": "https://github.com/greennode-samples/sample-zalo-restaurant.git#main:src/mcp-server",
    "stock-mcp-server": "https://github.com/greennode-samples/sample-mcp-stock-server.git#main",
}


def test_the_orchestrator_always_runs_and_the_siblings_only_with_the_all_profile():
    assert "profiles" not in SERVICES["orchestrator"]
    assert set(SERVICES) - {"orchestrator"} == set(SIBLINGS)
    assert all(SERVICES[name]["profiles"] == ["all"] for name in SIBLINGS)


def test_the_siblings_are_built_from_github_and_the_orchestrator_from_this_repo():
    assert SERVICES["orchestrator"]["build"] == {"context": "."}
    assert {name: SERVICES[name]["build"]["context"] for name in SIBLINGS} == SIBLINGS


def test_every_published_port_is_loopback_only():
    for name, service in SERVICES.items():
        for port in service.get("ports", []):
            assert port.startswith("127.0.0.1:"), f"{name} publishes {port} on every interface"


def test_every_env_file_has_a_committed_example():
    for service in SERVICES.values():
        env_file = service.get("env_file")
        if env_file and env_file != ".env":
            assert (ROOT / f"{env_file}.example").exists(), env_file
    assert (ROOT / ".env.example").exists()


def test_the_local_mcp_servers_are_anonymous_only_for_local_use():
    for name in ("restaurant-mcp-server", "stock-mcp-server"):
        assert SERVICES[name]["environment"]["ALLOW_ANONYMOUS"] == "true"


def test_the_urls_in_env_example_point_at_compose_services_on_the_container_port():
    values = dotenv_values(ROOT / ".env.example")
    urls = [values["MCP_STOCK_URL"], values["A2A_AGENT_TRAVEL_BUDDY_URL"], values["A2A_AGENT_ZALO_RESTAURANT_URL"]]
    for url in urls:
        parts = urlsplit(url)
        assert parts.hostname in SERVICES and parts.port == 8080, url
    assert urlsplit(urls[0]).hostname == "stock-mcp-server" and urlsplit(urls[0]).path == "/mcp"
    assert SERVICES["zalo-restaurant"]["environment"]["MCP_RESTAURANT_URL"] == "http://restaurant-mcp-server:8080/mcp"


def test_the_per_service_env_examples_have_no_inline_comments():
    """`docker run --env-file` keeps a comment after a value as part of the value."""
    for path in [ROOT / ".env.example", *sorted((ROOT / "compose").glob("*.env.example"))]:
        for line in path.read_text().splitlines():
            if re.match(r"^[A-Z][A-Z0-9_]*=", line):
                assert " #" not in line, f"{path.name}: {line}"


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker is not installed")
def test_docker_compose_accepts_the_file(tmp_path):
    shutil.copy(ROOT / "docker-compose.yml", tmp_path)
    shutil.copy(ROOT / ".env.example", tmp_path / ".env")

    def config(*args):
        return subprocess.run(["docker", "compose", *args, "config", "-q"], cwd=tmp_path, capture_output=True, text=True)

    assert config().returncode == 0  # the orchestrator alone needs only .env
    missing = config("--profile", "all")
    assert missing.returncode != 0 and "compose/travel-buddy.env" in missing.stderr  # the siblings need their env files
    (tmp_path / "compose").mkdir()
    for example in (ROOT / "compose").glob("*.env.example"):
        shutil.copy(example, tmp_path / "compose" / example.name.removesuffix(".example"))
    assert config("--profile", "all").returncode == 0
