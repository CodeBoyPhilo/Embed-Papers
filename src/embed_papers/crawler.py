from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import requests

from .exceptions import NoPapersFoundError, OpenReviewRequestError
from .models import Paper
from .retry import call_with_retry


def _log(message: str) -> None:
    print(message, file=sys.stderr)


def _is_retryable_request_error(exc: Exception) -> bool:
    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return True
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code in {429, 500, 502, 503, 504}
    return False


def fetch_submissions(
    venue_id: str,
    offset: int = 0,
    limit: int = 1000,
    timeout_seconds: float = 30.0,
    retry_attempts: int = 5,
) -> dict[str, Any]:
    url = "https://api2.openreview.net/notes"
    params = {
        "content.venueid": venue_id,
        "details": "replyCount,invitation",
        "limit": limit,
        "offset": offset,
        "sort": "number:desc",
    }
    headers = {"User-Agent": "embed-papers/0.3"}

    def _request() -> dict[str, Any]:
        response = requests.get(
            url, params=params, headers=headers, timeout=timeout_seconds
        )
        response.raise_for_status()
        return response.json()

    try:
        return call_with_retry(
            _request,
            is_retryable=_is_retryable_request_error,
            max_attempts=retry_attempts,
        )
    except Exception as exc:
        raise OpenReviewRequestError(
            f"OpenReview request failed for venue={venue_id}, offset={offset}, limit={limit}."
        ) from exc


def _note_to_paper(note: dict[str, Any]) -> Paper:
    content = note.get("content", {})
    return Paper.from_mapping(
        {
            "id": note.get("id"),
            "number": note.get("number"),
            "title": content.get("title", {}).get("value", ""),
            "authors": content.get("authors", {}).get("value", []),
            "abstract": content.get("abstract", {}).get("value", ""),
            "keywords": content.get("keywords", {}).get("value", []),
            "primary_area": content.get("primary_area", {}).get("value", ""),
            "forum_url": f"https://openreview.net/forum?id={note.get('id')}",
        }
    )


def crawl_papers(
    venue_id: str,
    output_file: str,
    limit: int = 1000,
    sleep_seconds: float = 0.5,
    timeout_seconds: float = 30.0,
    retry_attempts: int = 5,
    allow_empty: bool = False,
) -> list[dict[str, Any]]:
    all_papers: list[Paper] = []
    offset = 0

    _log(f"Fetching papers from {venue_id}...")
    while True:
        data = fetch_submissions(
            venue_id=venue_id,
            offset=offset,
            limit=limit,
            timeout_seconds=timeout_seconds,
            retry_attempts=retry_attempts,
        )
        notes = data.get("notes", [])
        if not notes:
            break

        for note in notes:
            all_papers.append(_note_to_paper(note))

        _log(f"Fetched {len(notes)} papers (total: {len(all_papers)})")
        if len(notes) < limit:
            break
        offset += limit
        time.sleep(sleep_seconds)

    total = len(all_papers)
    if total == 0 and not allow_empty:
        raise NoPapersFoundError(
            f"No papers found for venue_id={venue_id}. This may indicate an invalid venue id. "
            "Use --allow-empty to accept zero results."
        )

    payload = {
        "venue_id": venue_id,
        "total": total,
        "papers": [paper.to_dict() for paper in all_papers],
    }
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _log(f"Total: {total} papers")
    _log(f"Saved to {output_file}")
    return payload["papers"]
