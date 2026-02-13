from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from embed_papers.searcher import PaperSearcher


def _write_papers_file(path: Path, first_title: str = "Paper A") -> None:
    payload = {
        "venue_id": "ICLR.cc/2026/Conference",
        "papers": [
            {
                "id": "p1",
                "title": first_title,
                "abstract": "Abstract A",
            },
            {
                "id": "p2",
                "title": "Paper B",
                "abstract": "Abstract B",
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fake_embed(self: PaperSearcher, texts: str | list[str]) -> np.ndarray:
    items = [texts] if isinstance(texts, str) else texts
    rows = [[float(index + 1), 0.0, -1.0] for index, _ in enumerate(items)]
    return np.array(rows, dtype=float)


def test_cache_metadata_written_and_reused(tmp_path, monkeypatch) -> None:
    papers_file = tmp_path / "papers.json"
    _write_papers_file(papers_file)
    monkeypatch.setattr(PaperSearcher, "_embed_openai", _fake_embed)

    searcher = PaperSearcher(
        papers_file=str(papers_file),
        cache_dir=str(tmp_path),
        require_api_key_on_cache_miss=False,
    )
    first_embeddings = searcher.ensure_embeddings()

    assert Path(searcher.cache_file).exists()
    assert Path(searcher.cache_metadata_file).exists()

    reloaded = PaperSearcher(
        papers_file=str(papers_file),
        cache_dir=str(tmp_path),
        require_api_key_on_cache_miss=False,
    )

    assert reloaded.has_cache
    assert reloaded.embeddings is not None
    np.testing.assert_array_equal(first_embeddings, reloaded.embeddings)


def test_cache_invalidates_when_papers_change(tmp_path, monkeypatch) -> None:
    papers_file = tmp_path / "papers.json"
    _write_papers_file(papers_file)
    monkeypatch.setattr(PaperSearcher, "_embed_openai", _fake_embed)

    searcher = PaperSearcher(
        papers_file=str(papers_file),
        cache_dir=str(tmp_path),
        require_api_key_on_cache_miss=False,
    )
    searcher.ensure_embeddings()

    _write_papers_file(papers_file, first_title="Paper A (revised)")

    reloaded = PaperSearcher(
        papers_file=str(papers_file),
        cache_dir=str(tmp_path),
        require_api_key_on_cache_miss=False,
    )

    assert not reloaded.has_cache
