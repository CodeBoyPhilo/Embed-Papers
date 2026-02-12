from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .crawler import crawl_papers
from .exceptions import EmbedPapersError
from .searcher import PaperSearcher


def _read_examples(examples_file: str) -> list[dict[str, Any]]:
    loaded = json.loads(Path(examples_file).read_text(encoding="utf-8"))
    if not isinstance(loaded, list):
        raise ValueError("examples-file must contain a JSON list")
    examples: list[dict[str, Any]] = []
    for item in loaded:
        if isinstance(item, dict):
            examples.append(item)
    if not examples:
        raise ValueError("examples-file contains no valid example objects")
    return examples


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="embed-papers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    crawl_parser = subparsers.add_parser("crawl", help="Fetch papers from OpenReview")
    crawl_parser.add_argument("--venue-id", required=True)
    crawl_parser.add_argument("--output-file", required=True)
    crawl_parser.add_argument("--limit", type=int, default=1000)
    crawl_parser.add_argument("--sleep-seconds", type=float, default=0.5)

    warm_parser = subparsers.add_parser(
        "warm-cache", help="Compute cache for a conference/model"
    )
    warm_parser.add_argument("--papers-file", required=True)
    warm_parser.add_argument("--venue-id")
    warm_parser.add_argument("--model-name", default="text-embedding-3-large")
    warm_parser.add_argument("--api-key")
    warm_parser.add_argument("--base-url")
    warm_parser.add_argument("--cache-dir")
    warm_parser.add_argument("--force", action="store_true")

    search_parser = subparsers.add_parser("search", help="Run semantic paper search")
    search_parser.add_argument("--papers-file", required=True)
    search_parser.add_argument("--venue-id")
    search_parser.add_argument("--model-name", default="text-embedding-3-large")
    search_parser.add_argument("--api-key")
    search_parser.add_argument("--base-url")
    search_parser.add_argument("--cache-dir")
    search_parser.add_argument("--top-k", type=int, default=100)
    search_parser.add_argument("--output-file")
    group = search_parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query")
    group.add_argument("--examples-file")

    return parser


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "crawl":
        papers = crawl_papers(
            venue_id=args.venue_id,
            output_file=args.output_file,
            limit=args.limit,
            sleep_seconds=args.sleep_seconds,
        )
        return {
            "ok": True,
            "command": "crawl",
            "venue_id": args.venue_id,
            "total": len(papers),
            "output_file": args.output_file,
        }

    if args.command == "warm-cache":
        searcher = PaperSearcher(
            papers_file=args.papers_file,
            venue_id=args.venue_id,
            model_name=args.model_name,
            api_key=args.api_key,
            base_url=args.base_url,
            cache_dir=args.cache_dir,
        )
        shape = searcher.ensure_embeddings(force=args.force).shape
        return {
            "ok": True,
            "command": "warm-cache",
            "venue_id": searcher.venue_id,
            "model": searcher.model_name,
            "cache_file": searcher.cache_file,
            "embedding_shape": list(shape),
        }

    if args.command == "search":
        searcher = PaperSearcher(
            papers_file=args.papers_file,
            venue_id=args.venue_id,
            model_name=args.model_name,
            api_key=args.api_key,
            base_url=args.base_url,
            cache_dir=args.cache_dir,
        )
        examples = _read_examples(args.examples_file) if args.examples_file else None
        results = searcher.search(query=args.query, examples=examples, top_k=args.top_k)
        if args.output_file:
            searcher.save(results, args.output_file)
        return {
            "ok": True,
            "command": "search",
            "venue_id": searcher.venue_id,
            "model": searcher.model_name,
            "total": len(results),
            "results": results,
        }

    raise ValueError(f"Unknown command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _create_parser()
    args = parser.parse_args(argv)
    try:
        payload = _run(args)
    except EmbedPapersError as exc:
        print(
            json.dumps(
                {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
            ),
            file=sys.stderr,
        )
        return 1
    except Exception as exc:  # pragma: no cover
        print(
            json.dumps(
                {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
            ),
            file=sys.stderr,
        )
        return 1

    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
