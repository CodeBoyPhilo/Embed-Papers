from __future__ import annotations

import json
import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .exceptions import (
    CacheMissRequiresApiKeyError,
    EmbeddingRequestError,
    InvalidPapersFileError,
)
from .models import Paper
from .retry import call_with_retry


def _slugify(value: str) -> str:
    lowered = value.strip().lower()
    return re.sub(r"[^a-z0-9]+", "-", lowered).strip("-") or "unknown"


def _log(message: str) -> None:
    print(message, file=sys.stderr)


def _papers_fingerprint(papers: list[Paper]) -> str:
    hasher = hashlib.sha256()
    for paper in papers:
        payload = {
            "id": paper.id,
            "number": paper.number,
            "title": paper.title,
            "authors": paper.authors,
            "abstract": paper.abstract,
            "keywords": paper.keywords,
            "primary_area": paper.primary_area,
            "forum_url": paper.forum_url,
        }
        hasher.update(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        )
    return hasher.hexdigest()


def _cosine_similarity_scores(
    query_embedding: np.ndarray, embeddings: np.ndarray
) -> np.ndarray:
    if embeddings.size == 0:
        return np.array([])

    query = np.asarray(query_embedding, dtype=float).reshape(-1)
    matrix = np.asarray(embeddings, dtype=float)

    query_norm = np.linalg.norm(query)
    matrix_norm = np.linalg.norm(matrix, axis=1)

    if query_norm == 0.0:
        return np.zeros(len(matrix), dtype=float)

    safe_denom = matrix_norm * query_norm
    safe_denom[safe_denom == 0.0] = 1e-12

    return (matrix @ query) / safe_denom


def _is_retryable_embedding_error(exc: Exception) -> bool:
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True

    if type(exc).__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
    }:
        return True

    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and status_code in {429, 500, 502, 503, 504}:
        return True

    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    if isinstance(response_status, int) and response_status in {
        429,
        500,
        502,
        503,
        504,
    }:
        return True

    return False


def _load_papers_file(
    file_path: str, venue_id_override: str | None = None
) -> tuple[str, list[Paper]]:
    raw_payload = json.loads(Path(file_path).read_text(encoding="utf-8"))

    papers_payload: list[dict[str, Any]]
    venue_id: str | None = venue_id_override

    if isinstance(raw_payload, dict):
        papers_raw = raw_payload.get("papers")
        if not isinstance(papers_raw, list):
            raise InvalidPapersFileError(
                "Papers file object must contain a 'papers' list."
            )
        papers_payload = papers_raw
        if venue_id is None:
            raw_venue = raw_payload.get("venue_id")
            if raw_venue:
                venue_id = str(raw_venue)
    elif isinstance(raw_payload, list):
        papers_payload = raw_payload
    else:
        raise InvalidPapersFileError(
            "Papers file must be either a list or an object with a 'papers' field."
        )

    if not venue_id:
        raise InvalidPapersFileError(
            "Missing venue_id. Provide venue_id explicitly or use crawler output containing venue_id."
        )

    papers = [
        Paper.from_mapping(item) for item in papers_payload if isinstance(item, dict)
    ]
    if not papers:
        raise InvalidPapersFileError("No valid papers found in papers file.")

    return venue_id, papers


class PaperSearcher:
    CACHE_SCHEMA_VERSION = 1

    def __init__(
        self,
        papers_file: str,
        venue_id: str | None = None,
        model_name: str = "text-embedding-3-large",
        api_key: str | None = None,
        base_url: str | None = None,
        cache_dir: str | None = None,
        require_api_key_on_cache_miss: bool = True,
    ) -> None:
        self.papers_file = papers_file
        self.venue_id, self.papers = _load_papers_file(
            papers_file, venue_id_override=venue_id
        )
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.require_api_key_on_cache_miss = require_api_key_on_cache_miss

        cache_root = Path(cache_dir) if cache_dir else Path(papers_file).parent
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_name = f"cache_{_slugify(self.venue_id)}_{_slugify(self.model_name)}.npy"
        self.cache_file = str(cache_root / cache_name)
        self.cache_metadata_file = f"{self.cache_file}.meta.json"
        self.papers_fingerprint = _papers_fingerprint(self.papers)

        self.embeddings: np.ndarray | None = None
        self._client: Any | None = None
        self._load_cache()

    @property
    def resolved_api_key(self) -> str | None:
        return self.api_key or os.getenv("OPENAI_API_KEY")

    @property
    def has_cache(self) -> bool:
        return self.embeddings is not None

    def _load_cache(self) -> bool:
        if not os.path.exists(self.cache_file):
            return False

        try:
            loaded = np.load(self.cache_file)
        except Exception:
            _log(f"Invalid cache file, rebuilding: {self.cache_file}")
            self.embeddings = None
            return False

        if len(loaded) != len(self.papers):
            _log("Cache row count mismatch, rebuilding cache")
            self.embeddings = None
            return False

        metadata = self._load_cache_metadata()
        if metadata is None:
            _log("Missing or invalid cache metadata, rebuilding cache")
            self.embeddings = None
            return False

        if not self._is_cache_metadata_valid(metadata, loaded):
            _log("Cache metadata mismatch, rebuilding cache")
            self.embeddings = None
            return False

        self.embeddings = loaded
        _log(f"Loaded cache: {getattr(loaded, 'shape', None)}")
        return True

    def _load_cache_metadata(self) -> dict[str, Any] | None:
        metadata_path = Path(self.cache_metadata_file)
        if not metadata_path.exists():
            return None

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            return None

        if not isinstance(metadata, dict):
            return None
        return metadata

    def _is_cache_metadata_valid(
        self,
        metadata: dict[str, Any],
        embeddings: np.ndarray,
    ) -> bool:
        expected = {
            "schema_version": self.CACHE_SCHEMA_VERSION,
            "venue_id": self.venue_id,
            "model_name": self.model_name,
            "papers_count": len(self.papers),
            "papers_fingerprint": self.papers_fingerprint,
            "embedding_shape": list(embeddings.shape),
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                return False
        return True

    def _save_cache(self) -> None:
        if self.embeddings is None:
            return
        np.save(self.cache_file, self.embeddings)
        metadata = {
            "schema_version": self.CACHE_SCHEMA_VERSION,
            "venue_id": self.venue_id,
            "model_name": self.model_name,
            "papers_count": len(self.papers),
            "papers_fingerprint": self.papers_fingerprint,
            "embedding_shape": list(self.embeddings.shape),
        }
        Path(self.cache_metadata_file).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _log(f"Saved cache: {self.cache_file}")

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        key = self.resolved_api_key
        if not key:
            raise CacheMissRequiresApiKeyError(
                "Embeddings cache not found. Set OPENAI_API_KEY (or pass api_key) to create conference cache."
            )

        from openai import OpenAI

        self._client = OpenAI(api_key=key, base_url=self.base_url)
        return self._client

    def _embed_openai(self, texts: str | list[str]) -> np.ndarray:
        text_items = [texts] if isinstance(texts, str) else texts
        if not text_items:
            return np.empty((0, 0))

        client = self._get_client()
        embeddings: list[list[float]] = []
        batch_size = 100

        for start in range(0, len(text_items), batch_size):
            batch = text_items[start : start + batch_size]
            try:
                response = call_with_retry(
                    lambda: client.embeddings.create(
                        input=batch, model=self.model_name
                    ),
                    is_retryable=_is_retryable_embedding_error,
                    max_attempts=5,
                )
            except Exception as exc:
                raise EmbeddingRequestError(
                    f"Embedding request failed for model={self.model_name}, batch_start={start}."
                ) from exc
            embeddings.extend(item.embedding for item in response.data)

        return np.array(embeddings)

    def compute_embeddings(self, force: bool = False) -> np.ndarray:
        if self.embeddings is not None and not force:
            _log("Using cached embeddings")
            return self.embeddings

        if self.require_api_key_on_cache_miss and not self.resolved_api_key:
            raise CacheMissRequiresApiKeyError(
                "No cache for this venue/model and no API key found. Set OPENAI_API_KEY to build cache."
            )

        _log(f"Computing embeddings ({self.model_name})...")
        texts = [paper.to_embedding_text() for paper in self.papers]
        computed = self._embed_openai(texts)
        self.embeddings = computed
        _log(f"Computed: {computed.shape}")
        self._save_cache()
        return self.embeddings

    def ensure_embeddings(self, force: bool = False) -> np.ndarray:
        return self.compute_embeddings(force=force)

    def search(
        self,
        examples: list[dict[str, Any]] | None = None,
        query: str | None = None,
        top_k: int = 100,
    ) -> list[dict[str, Any]]:
        if self.embeddings is None:
            self.compute_embeddings()

        embeddings = self.embeddings
        if embeddings is None:
            raise RuntimeError("Embeddings were not computed.")

        if examples:
            texts: list[str] = []
            for example in examples:
                title = str(example.get("title") or "")
                abstract = str(example.get("abstract") or "")
                text = f"Title: {title}".strip()
                if abstract:
                    text += f" Abstract: {abstract}"
                texts.append(text)
            query_embedding = np.mean(self._embed_openai(texts), axis=0).reshape(1, -1)
        elif query:
            query_embedding = self._embed_openai(query).reshape(1, -1)
        else:
            raise ValueError("Provide either examples or query")

        similarities = _cosine_similarity_scores(query_embedding, embeddings)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [
            {
                "paper": self.papers[index].to_dict(),
                "similarity": float(similarities[index]),
            }
            for index in top_indices
        ]

    def display(
        self,
        results: list[dict[str, Any]],
        n: int = 10,
        show_abstract: bool = False,
        abstract_max_chars: int = 500,
    ) -> None:
        print(f"\n{'=' * 80}")
        print(f"Top {len(results)} Results (showing {min(n, len(results))})")
        print(f"{'=' * 80}\n")

        for idx, result in enumerate(results[:n], start=1):
            paper = result["paper"]
            similarity = result["similarity"]

            print(f"{idx}. [{similarity:.4f}] {paper['title']}")
            print(
                f"   #{paper.get('number', 'N/A')} | {paper.get('primary_area', 'N/A')}"
            )
            if show_abstract:
                abstract = " ".join(str(paper.get("abstract") or "").split())
                if abstract_max_chars > 0 and len(abstract) > abstract_max_chars:
                    abstract = f"{abstract[:abstract_max_chars].rstrip()}..."
                print(f"   Abstract: {abstract or 'N/A'}")
            print(f"   {paper.get('forum_url', '')}\n")

    def save(self, results: list[dict[str, Any]], output_file: str) -> None:
        payload = {
            "venue_id": self.venue_id,
            "model": self.model_name,
            "total": len(results),
            "results": results,
        }
        Path(output_file).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _log(f"Saved to {output_file}")
