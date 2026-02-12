from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _to_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


@dataclass(frozen=True)
class Paper:
    id: str
    number: int | None
    title: str
    authors: list[str]
    abstract: str
    keywords: list[str]
    primary_area: str
    forum_url: str

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "Paper":
        paper_id = str(payload.get("id") or "")
        forum_url = payload.get("forum_url")
        if not forum_url and paper_id:
            forum_url = f"https://openreview.net/forum?id={paper_id}"

        number = payload.get("number")
        if number is not None:
            try:
                number = int(number)
            except (TypeError, ValueError):
                number = None

        return cls(
            id=paper_id,
            number=number,
            title=str(payload.get("title") or ""),
            authors=_to_string_list(payload.get("authors")),
            abstract=str(payload.get("abstract") or ""),
            keywords=_to_string_list(payload.get("keywords")),
            primary_area=str(payload.get("primary_area") or ""),
            forum_url=str(forum_url or ""),
        )

    def to_embedding_text(self) -> str:
        parts: list[str] = []
        if self.title:
            parts.append(f"Title: {self.title}")
        if self.abstract:
            parts.append(f"Abstract: {self.abstract}")
        if self.keywords:
            parts.append(f"Keywords: {', '.join(self.keywords)}")
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "number": self.number,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "keywords": self.keywords,
            "primary_area": self.primary_area,
            "forum_url": self.forum_url,
        }
