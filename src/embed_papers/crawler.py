from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests

from .models import Paper


def fetch_submissions(
    venue_id: str, offset: int = 0, limit: int = 1000
) -> dict[str, Any]:
    url = "https://api2.openreview.net/notes"
    params = {
        "content.venueid": venue_id,
        "details": "replyCount,invitation",
        "limit": limit,
        "offset": offset,
        "sort": "number:desc",
    }
    headers = {"User-Agent": "embed-papers/0.2"}
    response = requests.get(url, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


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
) -> list[dict[str, Any]]:
    all_papers: list[Paper] = []
    offset = 0

    print(f"Fetching papers from {venue_id}...")
    while True:
        data = fetch_submissions(venue_id=venue_id, offset=offset, limit=limit)
        notes = data.get("notes", [])
        if not notes:
            break

        for note in notes:
            all_papers.append(_note_to_paper(note))

        print(f"Fetched {len(notes)} papers (total: {len(all_papers)})")
        if len(notes) < limit:
            break
        offset += limit
        time.sleep(sleep_seconds)

    payload = {
        "venue_id": venue_id,
        "total": len(all_papers),
        "papers": [paper.to_dict() for paper in all_papers],
    }
    output_path = Path(output_file)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nTotal: {len(all_papers)} papers")
    print(f"Saved to {output_file}")
    return payload["papers"]
