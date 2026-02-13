from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, NoReturn, Sequence

from .cache_paths import default_papers_cache_file
from .crawler import crawl_papers
from .exceptions import EmbedPapersError
from .searcher import PaperSearcher

SCHEMA_VERSION = "1"
KNOWN_COMMANDS = {"crawl", "warm-cache", "search"}


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise ValueError(message)


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


def _guess_command(argv: Sequence[str] | None) -> str | None:
    if not argv:
        return None
    for token in argv:
        if token.startswith("-"):
            continue
        if token in KNOWN_COMMANDS:
            return token
        return token
    return None


def _success_payload(command: str, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": True,
        "schema_version": SCHEMA_VERSION,
        "command": command,
        "data": data,
    }


def _error_payload(
    command: str | None,
    error_type: str,
    message: str,
) -> dict[str, Any]:
    return {
        "ok": False,
        "schema_version": SCHEMA_VERSION,
        "command": command,
        "error": {
            "type": error_type,
            "message": message,
        },
    }


def _create_parser() -> argparse.ArgumentParser:
    parser = JsonArgumentParser(prog="embed-papers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    crawl_parser = subparsers.add_parser("crawl", help="Fetch papers from OpenReview")
    crawl_parser.add_argument("--venue-id", required=True)
    crawl_parser.add_argument("--output-file")
    crawl_parser.add_argument("--limit", type=int, default=1000)
    crawl_parser.add_argument("--sleep-seconds", type=float, default=0.5)
    crawl_parser.add_argument("--timeout-seconds", type=float, default=30.0)
    crawl_parser.add_argument("--retry-attempts", type=int, default=5)
    crawl_parser.add_argument("--allow-empty", action="store_true")

    warm_parser = subparsers.add_parser(
        "warm-cache", help="Compute cache for a conference/model"
    )
    warm_parser.add_argument("--papers-file")
    warm_parser.add_argument("--venue-id")
    warm_parser.add_argument("--model-name", default="text-embedding-3-large")
    warm_parser.add_argument("--api-key")
    warm_parser.add_argument("--base-url")
    warm_parser.add_argument("--cache-dir")
    warm_parser.add_argument("--force", action="store_true")

    search_parser = subparsers.add_parser("search", help="Run semantic paper search")
    search_parser.add_argument("--papers-file")
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


def _create_host_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="embed-papers host")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    return parser


def _run_host(raw_argv: Sequence[str]) -> int:
    if importlib.util.find_spec("streamlit") is None:
        print(
            'Streamlit is not installed. Install viewer extras with: pip install "embed-papers[viewer]"',
            file=sys.stderr,
        )
        return 1

    command_index = list(raw_argv).index("host")
    host_argv = raw_argv[command_index + 1 :]

    host_parser = _create_host_parser()
    try:
        host_args = host_parser.parse_args(host_argv)
    except SystemExit as exc:
        code = exc.code
        return code if isinstance(code, int) else 2

    app_path = Path(__file__).resolve().parent / "viewer" / "app.py"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        host_args.host,
        "--server.port",
        str(host_args.port),
        "--server.headless",
        "false",
    ]

    completed = subprocess.run(command)
    return completed.returncode


def _resolve_papers_file(papers_file: str | None, venue_id: str | None) -> str:
    if papers_file:
        return papers_file
    if venue_id:
        return str(default_papers_cache_file(venue_id))
    raise ValueError("Provide --papers-file or --venue-id.")


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "crawl":
        output_file = args.output_file or str(default_papers_cache_file(args.venue_id))
        papers = crawl_papers(
            venue_id=args.venue_id,
            output_file=output_file,
            limit=args.limit,
            sleep_seconds=args.sleep_seconds,
            timeout_seconds=args.timeout_seconds,
            retry_attempts=args.retry_attempts,
            allow_empty=args.allow_empty,
        )
        return {
            "venue_id": args.venue_id,
            "total": len(papers),
            "output_file": output_file,
        }

    if args.command == "warm-cache":
        papers_file = _resolve_papers_file(args.papers_file, args.venue_id)
        searcher = PaperSearcher(
            papers_file=papers_file,
            venue_id=args.venue_id,
            model_name=args.model_name,
            api_key=args.api_key,
            base_url=args.base_url,
            cache_dir=args.cache_dir,
        )
        shape = searcher.ensure_embeddings(force=args.force).shape
        return {
            "venue_id": searcher.venue_id,
            "model": searcher.model_name,
            "cache_file": searcher.cache_file,
            "cache_metadata_file": searcher.cache_metadata_file,
            "embedding_shape": list(shape),
        }

    if args.command == "search":
        papers_file = _resolve_papers_file(args.papers_file, args.venue_id)
        searcher = PaperSearcher(
            papers_file=papers_file,
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
            "venue_id": searcher.venue_id,
            "model": searcher.model_name,
            "total": len(results),
            "results": results,
            "output_file": args.output_file,
        }

    raise ValueError(f"Unknown command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv: Sequence[str] = argv if argv is not None else sys.argv[1:]
    if _guess_command(raw_argv) == "host":
        return _run_host(raw_argv)

    parser = _create_parser()
    command = _guess_command(raw_argv)

    try:
        args = parser.parse_args(argv)
    except ValueError as exc:
        print(
            json.dumps(
                _error_payload(command, "ArgumentError", str(exc)),
                ensure_ascii=False,
            )
        )
        return 2

    command = args.command

    try:
        payload = _success_payload(command, _run(args))
        exit_code = 0
    except EmbedPapersError as exc:
        print(
            json.dumps(
                _error_payload(command, type(exc).__name__, str(exc)),
                ensure_ascii=False,
            )
        )
        return 1
    except Exception as exc:  # pragma: no cover
        print(
            json.dumps(
                _error_payload(command, type(exc).__name__, str(exc)),
                ensure_ascii=False,
            )
        )
        return 1

    print(json.dumps(payload, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
