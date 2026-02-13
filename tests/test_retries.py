from __future__ import annotations

import json

import numpy as np
import requests

from embed_papers import crawler, retry
from embed_papers.searcher import PaperSearcher


def test_openreview_request_retries_on_timeout(monkeypatch) -> None:
    calls = {"count": 0}

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"notes": []}

    def _fake_get(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise requests.Timeout("temporary timeout")
        return _FakeResponse()

    monkeypatch.setattr(crawler.requests, "get", _fake_get)
    monkeypatch.setattr(retry.time, "sleep", lambda *_: None)

    payload = crawler.fetch_submissions("ICLR.cc/2026/Conference")

    assert calls["count"] == 2
    assert payload == {"notes": []}


def test_embedding_request_retries_on_transient_error(tmp_path, monkeypatch) -> None:
    papers_file = tmp_path / "papers.json"
    papers_file.write_text(
        json.dumps(
            {
                "venue_id": "ICLR.cc/2026/Conference",
                "papers": [{"id": "p1", "title": "Title", "abstract": "Abstract"}],
            }
        ),
        encoding="utf-8",
    )

    class _FakeItem:
        def __init__(self, embedding: list[float]) -> None:
            self.embedding = embedding

    class _FakeEmbeddingResponse:
        data = [_FakeItem([0.1, 0.2])]

    class _FakeEmbeddingsApi:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, *, input, model):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("transient timeout")
            return _FakeEmbeddingResponse()

    class _FakeClient:
        def __init__(self) -> None:
            self.embeddings = _FakeEmbeddingsApi()

    searcher = PaperSearcher(
        papers_file=str(papers_file),
        require_api_key_on_cache_miss=False,
        cache_dir=str(tmp_path),
    )
    searcher._client = _FakeClient()
    monkeypatch.setattr(retry.time, "sleep", lambda *_: None)

    embeddings = searcher._embed_openai(["hello world"])

    assert searcher._client.embeddings.calls == 2
    np.testing.assert_array_equal(embeddings, np.array([[0.1, 0.2]]))
