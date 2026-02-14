from __future__ import annotations

import re
from pathlib import Path


def default_cache_root() -> Path:
    return Path.home() / ".cache" / "embed-papers"


def default_papers_cache_dir() -> Path:
    return default_cache_root() / "papers"


def default_embeddings_cache_dir() -> Path:
    return default_cache_root() / "embeddings"


def default_atlas_cache_dir() -> Path:
    return default_cache_root() / "atlas"


def _slugify(value: str) -> str:
    lowered = value.strip().lower()
    return re.sub(r"[^a-z0-9]+", "-", lowered).strip("-") or "unknown"


def default_papers_cache_file(venue_id: str) -> Path:
    return default_papers_cache_dir() / f"{_slugify(venue_id)}.json"
