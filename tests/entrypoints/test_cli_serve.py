"""`social-tracker serve`: loopback by default, anything else only on request (G-5)."""

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from typer.main import get_command

from social_tracker.entrypoints.cli import app
from social_tracker.entrypoints.web.security import check_bind_host, is_loopback
from social_tracker.shared.errors import ValidationError
from tests.entrypoints.conftest import Cli


@pytest.fixture
def launched(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replaces the server with a recorder: nothing listens during the tests."""
    calls: list[dict[str, Any]] = []

    def fake_run(application: Any, **kwargs: Any) -> None:
        calls.append({"app": application, **kwargs})

    monkeypatch.setattr("uvicorn.run", fake_run)
    return calls


def test_the_defaults_listen_on_loopback_port_8765(cli: Cli, launched: list[dict[str, Any]]):
    result = cli("serve")
    assert result.exit_code == 0, result.stderr
    (call,) = launched
    assert (call["host"], call["port"]) == ("127.0.0.1", 8765)
    assert isinstance(call["app"], FastAPI)
    assert call["server_header"] is False


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "127.0.0.2", "LocalHost"])
def test_every_loopback_host_is_accepted_without_a_flag(
    cli: Cli, launched: list[dict[str, Any]], host: str
):
    assert cli("serve", "--host", host, "--port", "9000").exit_code == 0
    assert (launched[0]["host"], launched[0]["port"]) == (host, 9000)


@pytest.mark.parametrize(
    "host", ["0.0.0.0", "::", "192.168.1.20", "example.org", "127.0.0.1.evil.example", "", " "]
)
def test_a_non_loopback_host_is_refused_without_the_flag_and_nothing_starts(
    cli: Cli, launched: list[dict[str, Any]], host: str
):
    result = cli("serve", "--host", host)
    assert result.exit_code == 1
    assert result.stderr.startswith("error: refusing to listen on a non-loopback address")
    assert "--allow-non-loopback" in result.stderr
    assert host.strip() == "" or host not in result.stderr  # the typed value is not echoed
    assert result.stdout == ""
    assert launched == []


def test_the_flag_allows_a_non_loopback_host_and_warns(cli: Cli, launched: list[dict[str, Any]]):
    result = cli("serve", "--host", "192.168.1.20", "--allow-non-loopback")
    assert result.exit_code == 0, result.stderr
    assert launched[0]["host"] == "192.168.1.20"
    assert "warning: the UI has no login" in result.stderr


def test_the_flag_alone_does_not_warn_on_loopback(cli: Cli, launched: list[dict[str, Any]]):
    result = cli("serve", "--allow-non-loopback")
    assert result.exit_code == 0
    assert "warning" not in result.stderr


@pytest.mark.parametrize("port", ["0", "65536", "-1", "http", ""])
def test_a_port_outside_1_to_65535_is_a_usage_error(
    cli: Cli, launched: list[dict[str, Any]], port: str
):
    assert cli("serve", "--port", port).exit_code == 2
    assert launched == []


@pytest.mark.parametrize("port", ["1", "65535"])
def test_the_port_bounds_are_accepted(cli: Cli, launched: list[dict[str, Any]], port: str):
    assert cli("serve", "--port", port).exit_code == 0


def test_the_help_warns_about_the_flag():
    command = get_command(app).commands["serve"]  # type: ignore[attr-defined]
    flag = next(p for p in command.params if "--allow-non-loopback" in p.opts)
    assert "WARNING" in flag.help
    assert "no login" in flag.help


def test_the_launched_app_serves_the_database_of_the_environment(
    cli: Cli, monkeypatch: pytest.MonkeyPatch
):
    cli.add("city_news", "City News")
    pages: list[str] = []

    def fake_run(application: Any, **_: Any) -> None:
        # inside the command, so the environment of the invocation is still in force
        client = TestClient(application, base_url="http://127.0.0.1:8765")
        pages.append(client.get("/").text)

    monkeypatch.setattr("uvicorn.run", fake_run)
    assert cli("serve").exit_code == 0
    assert "City News" in pages[0]


def test_an_unusable_database_fails_at_startup_with_a_refusal_not_at_the_first_request(
    cli: Cli, launched: list[dict[str, Any]], tmp_path: Any
):
    broken = tmp_path / "not-a-db.db"
    broken.write_text("this is not sqlite" * 100, encoding="utf-8")
    result = cli("serve", env={"SOCIAL_TRACKER_DB_PATH": str(broken)})
    assert result.exit_code == 1
    assert result.stderr.startswith("error:")
    assert launched == []


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("127.0.0.1", True),
        ("127.255.255.254", True),
        ("::1", True),
        ("[::1]", True),
        ("localhost", True),
        ("LOCALHOST", True),
        ("0.0.0.0", False),
        ("::", False),
        ("10.0.0.1", False),
        ("localhost.evil.example", False),
        ("127.0.0.1.evil.example", False),
        ("", False),
    ],
)
def test_is_loopback(host: str, expected: bool):
    assert is_loopback(host) is expected


def test_check_bind_host_only_lets_a_foreign_host_through_with_the_flag():
    check_bind_host("10.0.0.1", allow_non_loopback=True)
    check_bind_host("127.0.0.1", allow_non_loopback=False)
    with pytest.raises(ValidationError):
        check_bind_host("10.0.0.1", allow_non_loopback=False)
