from __future__ import annotations

import sys

from embed_papers import cli


def test_host_launches_streamlit(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda _: object())

    seen: dict[str, list[str]] = {}

    class _Completed:
        returncode = 0

    def _fake_run(command: list[str]) -> _Completed:
        seen["command"] = command
        return _Completed()

    monkeypatch.setattr(cli.subprocess, "run", _fake_run)

    exit_code = cli.main(["host", "--port", "9999"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out == ""
    assert seen["command"][:4] == [sys.executable, "-m", "streamlit", "run"]
    assert "--server.port" in seen["command"]
    assert "9999" in seen["command"]


def test_host_reports_missing_streamlit(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda _: None)

    exit_code = cli.main(["host"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert "embed-papers[viewer]" in captured.err
