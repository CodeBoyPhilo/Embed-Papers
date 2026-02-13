from __future__ import annotations

import json
from pathlib import Path

from embed_papers import crawler
from embed_papers.exceptions import NoPapersFoundError


def test_crawl_raises_on_empty_results_by_default(tmp_path, monkeypatch) -> None:
    output_file = tmp_path / "papers.json"

    monkeypatch.setattr(crawler, "fetch_submissions", lambda **_: {"notes": []})

    try:
        crawler.crawl_papers(
            venue_id="Totally.Wrong/Venue",
            output_file=str(output_file),
        )
        raise AssertionError("Expected NoPapersFoundError")
    except NoPapersFoundError:
        pass

    assert not output_file.exists()


def test_crawl_allows_empty_results_with_flag(tmp_path, monkeypatch) -> None:
    output_file = tmp_path / "papers.json"

    monkeypatch.setattr(crawler, "fetch_submissions", lambda **_: {"notes": []})

    papers = crawler.crawl_papers(
        venue_id="Totally.Wrong/Venue",
        output_file=str(output_file),
        allow_empty=True,
    )

    assert papers == []
    payload = json.loads(Path(output_file).read_text(encoding="utf-8"))
    assert payload["venue_id"] == "Totally.Wrong/Venue"
    assert payload["total"] == 0
    assert payload["papers"] == []
