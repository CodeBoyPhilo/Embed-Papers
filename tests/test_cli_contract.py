from __future__ import annotations

import json

from embed_papers import cli
from embed_papers.exceptions import InvalidPapersFileError, NoPapersFoundError


def test_crawl_outputs_strict_success_envelope(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "crawl_papers", lambda **_: [{"id": "p1"}, {"id": "p2"}])

    exit_code = cli.main(
        [
            "crawl",
            "--venue-id",
            "ICLR.cc/2026/Conference",
            "--output-file",
            "papers.json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload == {
        "ok": True,
        "schema_version": "1",
        "command": "crawl",
        "data": {
            "venue_id": "ICLR.cc/2026/Conference",
            "total": 2,
            "output_file": "papers.json",
        },
    }


def test_argument_errors_are_json(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "crawl_papers", lambda **_: [])

    exit_code = cli.main(["crawl", "--venue-id", "ICLR.cc/2026/Conference"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 2
    assert payload["ok"] is False
    assert payload["schema_version"] == "1"
    assert payload["command"] == "crawl"
    assert payload["error"]["type"] == "ArgumentError"


def test_runtime_errors_are_json(monkeypatch, capsys) -> None:
    def _raise_error(**_: object) -> list[dict[str, str]]:
        raise InvalidPapersFileError("broken papers file")

    monkeypatch.setattr(cli, "crawl_papers", _raise_error)

    exit_code = cli.main(
        [
            "crawl",
            "--venue-id",
            "ICLR.cc/2026/Conference",
            "--output-file",
            "papers.json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 1
    assert payload == {
        "ok": False,
        "schema_version": "1",
        "command": "crawl",
        "error": {
            "type": "InvalidPapersFileError",
            "message": "broken papers file",
        },
    }


def test_empty_crawl_returns_json_error(monkeypatch, capsys) -> None:
    def _raise_error(**_: object) -> list[dict[str, str]]:
        raise NoPapersFoundError("No papers found")

    monkeypatch.setattr(cli, "crawl_papers", _raise_error)

    exit_code = cli.main(
        [
            "crawl",
            "--venue-id",
            "Bad/Venue",
            "--output-file",
            "papers.json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["command"] == "crawl"
    assert payload["error"]["type"] == "NoPapersFoundError"


def test_allow_empty_flag_is_forwarded(monkeypatch, capsys) -> None:
    seen: dict[str, object] = {}

    def _crawl_papers(**kwargs: object) -> list[dict[str, str]]:
        seen.update(kwargs)
        return []

    monkeypatch.setattr(cli, "crawl_papers", _crawl_papers)

    exit_code = cli.main(
        [
            "crawl",
            "--venue-id",
            "Bad/Venue",
            "--output-file",
            "papers.json",
            "--allow-empty",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["ok"] is True
    assert seen["allow_empty"] is True
