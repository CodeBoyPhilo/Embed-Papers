from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

import streamlit as st

from embed_papers import PaperSearcher, crawl_papers

CACHE_ROOT = Path.home() / ".cache" / "embed-papers"
PAPERS_DIR = CACHE_ROOT / "papers"
EMBEDDINGS_DIR = CACHE_ROOT / "embeddings"


def _build_venue_id(conference: str, year: int) -> str:
    normalized = "".join(conference.strip().split())
    return f"{normalized}.cc/{year}/Conference"


def _papers_file_path(conference: str, year: int) -> Path:
    normalized = "_".join(conference.strip().lower().split())
    return PAPERS_DIR / f"{normalized}_{year}_conference.json"


def _load_examples(raw_payload: bytes) -> list[dict[str, Any]]:
    loaded = json.loads(raw_payload.decode("utf-8"))
    if not isinstance(loaded, list):
        raise ValueError("examples-file must contain a JSON list")

    examples = [item for item in loaded if isinstance(item, dict)]
    if not examples:
        raise ValueError("examples-file contains no valid example objects")
    return examples


def _short_abstract(value: str, max_chars: int = 700) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[:max_chars].rstrip()}..."


def _normalize_text(value: str) -> str:
    return " ".join(value.split())


def _show_sidebar_header() -> None:
    st.title("Viewer")
    st.markdown(
        "Semantic search for conference papers via OpenReview API."
    )
    st.markdown(
        "Built on top of the fork from [gyj155/SearchPaperByEmbedding](https://github.com/gyj155/SearchPaperByEmbedding)"
    )
    st.warning(
        "Set OPENAI_API_KEY in your shell before running searches. "
        "The viewer assumes it is already configured."
    )
    with st.popover("How to use"):
        st.markdown(
            """
1. Enter conference abbreviation and year.
2. Choose `Query` or `Upload Examples`.
3. Set `top-k`, then click `Run search`.
4. Read results in card view.
"""
        )
        st.markdown("Examples JSON format:")
        st.code(
            """[
  {
    "title": "Paper title",
    "abstract": "Short abstract text"
  },
  {
    "title": "Another title",
    "abstract": "Another abstract"
  }
]""",
            language="json",
        )
        st.caption(
            "Use a JSON list of objects. `title` and `abstract` are recommended."
        )
        st.caption(
            "Cache: `~/.cache/embed-papers/papers` and `~/.cache/embed-papers/embeddings`."
        )


@st.dialog("Abstract")
def _show_abstract_dialog(title: str, abstract: str, forum_url: str) -> None:
    st.caption("Paper")
    st.write(title)
    if forum_url:
        st.markdown(f"[OpenReview thread]({forum_url})")
    st.divider()
    st.write(abstract)
    if st.button("Close", type="primary"):
        st.rerun()


def _render_results(results: list[dict[str, Any]], cards_per_row: int = 3) -> None:
    st.subheader(f"Results ({len(results)})")
    if not results:
        st.info("No results found.")
        return

    cards_per_row = max(1, cards_per_row)
    columns = st.columns([1] * cards_per_row, gap="medium")
    for result_index, result in enumerate(results):
        column = columns[result_index % cards_per_row]
        index = result_index + 1
        paper = result.get("paper", {})
        similarity = float(result.get("similarity", 0.0))

        with column:
            title_raw = str(paper.get("title") or "Untitled")
            title = escape(title_raw)
            forum_url = escape(str(paper.get("forum_url") or ""), quote=True)
            number = escape(str(paper.get("number", "N/A")))
            area = escape(str(paper.get("primary_area", "N/A")))
            raw_authors = paper.get("authors")
            normalized_authors: list[str] = []
            if isinstance(raw_authors, list):
                for author in raw_authors:
                    text = _normalize_text(str(author))
                    if text:
                        normalized_authors.append(text)
            authors_text = (
                ", ".join(normalized_authors) if normalized_authors else "N/A"
            )

            title_html = (
                f'<a href="{forum_url}" target="_blank" rel="noopener noreferrer">{title}</a>'
                if forum_url
                else title
            )
            abstract_text = str(paper.get("abstract") or "").strip() or "N/A"
            forum_url_plain = str(paper.get("forum_url") or "")

            with st.container(border=True):
                st.markdown(
                    f"""
<div class='paper-card-marker'></div>
<div class="paper-title">{index}. {title_html}</div>
<div class="paper-meta">Similarity: {similarity:.4f}</div>
<div class="paper-meta">Paper #: {number}</div>
<div class="paper-area"><strong>Area</strong>: {area}</div>
<div class="paper-authors"><strong>Authors</strong>: {escape(authors_text)}</div>
""",
                    unsafe_allow_html=True,
                )
                if st.button("Abstract", key=f"paper-abstract-{index}"):
                    _show_abstract_dialog(
                        title=title_raw,
                        abstract=abstract_text,
                        forum_url=forum_url_plain,
                    )


def _inject_styles() -> None:
    st.markdown(
        """
<style>
div[data-testid="stVerticalBlockBorderWrapper"]:has(.paper-card-marker) {
  border: 1px solid rgba(128, 128, 128, 0.35) !important;
  border-radius: 0.6rem !important;
  padding: 0.8rem !important;
  margin-bottom: 0.9rem;
  box-shadow: 0 6px 18px rgba(0, 0, 0, 0.08);
  background: transparent;
}

div[data-testid="stVerticalBlockBorderWrapper"]:has(.paper-card-marker) > div {
  gap: 0.35rem;
}

.paper-card-marker {
  display: none;
}

.paper-title {
  font-weight: 750;
  font-size: 1.1rem;
  line-height: 1.4;
  min-height: 4.6rem;
  max-height: 4.6rem;
  overflow: hidden;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  text-overflow: ellipsis;
  word-break: break-word;
}

.paper-title a {
  font-weight: 750;
  text-decoration-thickness: 1px;
}

.paper-meta {
  font-size: 0.88rem;
  color: rgb(110, 110, 110);
  line-height: 1.3;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.paper-area {
  font-size: 0.92rem;
  color: rgb(110, 110, 110);
  line-height: 1.35;
  white-space: normal;
  overflow-wrap: anywhere;
}

.paper-authors {
  font-size: 0.92rem;
  line-height: 1.35;
  white-space: normal;
  overflow-wrap: anywhere;
  margin-bottom: 0.35rem;
}

div[data-testid="stVerticalBlockBorderWrapper"]:has(.paper-card-marker) .stButton > button {
  font-size: 0.88rem;
  font-weight: 600;
  color: inherit;
  border: 1px solid rgba(128, 128, 128, 0.45);
  border-radius: 0.45rem;
  padding: 0.2rem 0.6rem;
  background: rgba(240, 240, 240, 0.65);
  min-height: 0;
}

div[data-testid="stVerticalBlockBorderWrapper"]:has(.paper-card-marker) .stButton > button:hover {
  background: rgba(220, 220, 220, 0.75);
}
</style>
""",
        unsafe_allow_html=True,
    )


def _run_pipeline(
    conference: str,
    year: int,
    venue_id: str,
    top_k: int,
    query: str | None,
    examples: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    papers_file = _papers_file_path(conference, year)

    with st.status("Running embed-papers pipeline", expanded=True) as status:
        status.write(f"Venue: {venue_id}")
        status.write(f"Papers cache path: {papers_file}")

        if not papers_file.exists():
            status.write("No papers file found. Crawling OpenReview...")
            crawl_papers(venue_id=venue_id, output_file=str(papers_file))
        else:
            status.write("Papers file found. Skipping crawl.")

        searcher = PaperSearcher(
            papers_file=str(papers_file),
            venue_id=venue_id,
            cache_dir=str(EMBEDDINGS_DIR),
        )
        if searcher.has_cache:
            status.write("Embedding cache found. Reusing cache.")
        else:
            status.write("Embedding cache missing. Computing embeddings...")
            searcher.ensure_embeddings()

        status.write("Running semantic search...")
        results = searcher.search(query=query, examples=examples, top_k=top_k)
        status.update(label="Search completed", state="complete")

    return results


def _render_sidebar_form() -> dict[str, Any] | None:
    with st.sidebar:
        _show_sidebar_header()

        conference = st.text_input("Conference abbreviation", value="ICLR")
        year = st.number_input(
            "Year", min_value=2013, max_value=2100, value=2026, step=1
        )
        venue_id = _build_venue_id(conference, int(year))
        st.caption(f"Venue ID: {venue_id}")

        mode = st.radio(
            "Search mode",
            options=["Search", "Upload Examples"],
            horizontal=True,
        )
        top_k = st.number_input(
            "Top-k results", min_value=1, max_value=500, value=20, step=1
        )

        query: str | None = None
        examples: list[dict[str, Any]] | None = None

        if mode == "Search":
            query_value = st.text_area(
                "Query", placeholder="e.g. language models for code generation"
            )
            query = query_value.strip() or None
        else:
            uploaded = st.file_uploader("Upload examples JSON", type=["json"])
            if uploaded is not None:
                try:
                    examples = _load_examples(uploaded.getvalue())
                    st.success(f"Loaded {len(examples)} examples")
                except Exception as exc:
                    st.error(f"Invalid examples file: {exc}")

        submitted = st.button("Run search", type="primary")

        if not submitted:
            return None

        if not conference.strip():
            st.error("Conference abbreviation is required.")
            return None

        if mode == "Direct query" and not query:
            st.error("Enter a query before running search.")
            return None

        if mode == "Examples upload" and not examples:
            st.error("Upload a valid examples JSON file before running search.")
            return None

    return {
        "conference": conference,
        "year": int(year),
        "venue_id": venue_id,
        "top_k": int(top_k),
        "query": query,
        "examples": examples,
    }


def _render_empty_state() -> None:
    st.markdown("## No results yet")

def main() -> None:
    st.set_page_config(page_title="embed-papers viewer", layout="wide")
    _inject_styles()

    if "results" not in st.session_state:
        st.session_state["results"] = None

    request = _render_sidebar_form()

    if request is not None:
        try:
            st.session_state["results"] = _run_pipeline(
                conference=request["conference"],
                year=request["year"],
                venue_id=request["venue_id"],
                top_k=request["top_k"],
                query=request["query"],
                examples=request["examples"],
            )
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")

    results = st.session_state.get("results")
    if isinstance(results, list):
        _render_results(results, cards_per_row=3)
    else:
        _render_empty_state()


if __name__ == "__main__":
    main()
