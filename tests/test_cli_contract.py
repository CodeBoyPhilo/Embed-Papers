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

    exit_code = cli.main(["crawl"])

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


def test_crawl_defaults_output_file_to_cache_dir(monkeypatch, capsys, tmp_path) -> None:
    seen: dict[str, object] = {}
    expected_output = tmp_path / "iclr-cache.json"

    monkeypatch.setattr(
        cli,
        "default_papers_cache_file",
        lambda venue_id: expected_output,
    )

    def _crawl_papers(**kwargs: object) -> list[dict[str, str]]:
        seen.update(kwargs)
        return [{"id": "p1"}]

    monkeypatch.setattr(cli, "crawl_papers", _crawl_papers)

    exit_code = cli.main(["crawl", "--venue-id", "ICLR.cc/2026/Conference"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["data"]["output_file"] == str(expected_output)
    assert seen["output_file"] == str(expected_output)


def test_warm_cache_defaults_papers_file_from_venue(
    monkeypatch, capsys, tmp_path
) -> None:
    seen: dict[str, object] = {}
    expected_papers_file = tmp_path / "iclr-cache.json"

    monkeypatch.setattr(
        cli,
        "default_papers_cache_file",
        lambda venue_id: expected_papers_file,
    )

    class _FakeSearcher:
        def __init__(self, **kwargs: object) -> None:
            seen.update(kwargs)
            self.venue_id = str(kwargs.get("venue_id") or "ICLR.cc/2026/Conference")
            self.model_name = str(kwargs.get("model_name") or "text-embedding-3-large")
            self.cache_file = "cache.npy"
            self.cache_metadata_file = "cache.npy.meta.json"

        def ensure_embeddings(self, force: bool = False):
            seen["force"] = force

            class _Shape:
                shape = (2, 3)

            return _Shape()

    monkeypatch.setattr(cli, "PaperSearcher", _FakeSearcher)

    exit_code = cli.main(["warm-cache", "--venue-id", "ICLR.cc/2026/Conference"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["ok"] is True
    assert seen["papers_file"] == str(expected_papers_file)


def test_search_defaults_papers_file_from_venue(monkeypatch, capsys, tmp_path) -> None:
    seen: dict[str, object] = {}
    expected_papers_file = tmp_path / "iclr-cache.json"

    monkeypatch.setattr(
        cli,
        "default_papers_cache_file",
        lambda venue_id: expected_papers_file,
    )

    class _FakeSearcher:
        def __init__(self, **kwargs: object) -> None:
            seen.update(kwargs)
            self.venue_id = str(kwargs.get("venue_id") or "ICLR.cc/2026/Conference")
            self.model_name = str(kwargs.get("model_name") or "text-embedding-3-large")

        def search(self, **_: object) -> list[dict[str, object]]:
            return []

        def save(self, results: list[dict[str, object]], output_file: str) -> None:
            seen["saved_to"] = output_file

    monkeypatch.setattr(cli, "PaperSearcher", _FakeSearcher)

    exit_code = cli.main(
        [
            "search",
            "--venue-id",
            "ICLR.cc/2026/Conference",
            "--query",
            "test query",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["ok"] is True
    assert seen["papers_file"] == str(expected_papers_file)


def test_warm_cache_requires_papers_file_or_venue(capsys) -> None:
    exit_code = cli.main(["warm-cache"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["command"] == "warm-cache"
    assert payload["error"]["type"] == "ValueError"


def test_search_requires_papers_file_or_venue(capsys) -> None:
    exit_code = cli.main(["search", "--query", "test query"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["command"] == "search"
    assert payload["error"]["type"] == "ValueError"
