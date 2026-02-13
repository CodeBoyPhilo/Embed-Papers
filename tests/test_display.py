from __future__ import annotations

from embed_papers.searcher import PaperSearcher


def _sample_results() -> list[dict[str, object]]:
    return [
        {
            "paper": {
                "id": "p1",
                "number": 1,
                "title": "Sample Title",
                "authors": ["A"],
                "abstract": "This is a long abstract with enough words to test truncation output.",
                "keywords": [],
                "primary_area": "LLMs",
                "forum_url": "https://openreview.net/forum?id=p1",
            },
            "similarity": 0.9876,
        }
    ]


def test_display_hides_abstract_by_default(capsys) -> None:
    searcher = PaperSearcher.__new__(PaperSearcher)
    searcher.display(_sample_results(), n=1)

    captured = capsys.readouterr()
    assert "Abstract:" not in captured.out


def test_display_can_show_truncated_abstract(capsys) -> None:
    searcher = PaperSearcher.__new__(PaperSearcher)
    searcher.display(_sample_results(), n=1, show_abstract=True, abstract_max_chars=24)

    captured = capsys.readouterr()
    assert "Abstract:" in captured.out
    assert "This is a long abstract" in captured.out
    assert "..." in captured.out
