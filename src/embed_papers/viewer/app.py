from __future__ import annotations

import hashlib
import json
import os
import re
from html import escape
from pathlib import Path
from typing import Any

import streamlit as st

from embed_papers import PaperSearcher, crawl_papers
from embed_papers.cache_paths import (
    default_papers_cache_file,
    default_embeddings_cache_dir,
    default_atlas_cache_dir,
)

EMBEDDINGS_DIR = default_embeddings_cache_dir()
ATLAS_CACHE_DIR = default_atlas_cache_dir()


def _build_venue_id(conference: str, year: int) -> str:
    normalized = "".join(conference.strip().split())
    return f"{normalized}.cc/{year}/Conference"


def _papers_file_path(venue_id: str) -> Path:
    return default_papers_cache_file(venue_id)


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


def _slugify(value: str) -> str:
    lowered = value.strip().lower()
    return re.sub(r"[^a-z0-9]+", "-", lowered).strip("-") or "unknown"


def _hash_payload(payload: Any) -> str:
    serialized = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, default=str
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _atlas_request_key(request: dict[str, Any]) -> str:
    payload = {
        "conference": request.get("conference"),
        "year": request.get("year"),
        "venue_id": request.get("venue_id"),
        "top_k": request.get("top_k"),
        "query": request.get("query"),
        "examples": request.get("examples"),
    }
    return _hash_payload(payload)


def _show_sidebar_header() -> None:
    st.title("Viewer")
    st.markdown("Semantic search for conference papers via OpenReview API.")
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
            "Cache: `~/.cache/embed-papers/papers`, `~/.cache/embed-papers/embeddings`, and "
            "`~/.cache/embed-papers/atlas`."
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


def _atlas_cache_paths(venue_id: str, model_name: str) -> dict[str, Path]:
    root = ATLAS_CACHE_DIR
    stem = f"atlas_{_slugify(venue_id)}_{_slugify(model_name)}"
    return {
        "root": root,
        "projection": root / f"{stem}.npz",
        "umap": root / f"{stem}.umap.joblib",
        "meta": root / f"{stem}.json",
    }


def _atlas_cache_missing(cache_paths: dict[str, str] | None) -> bool:
    if not cache_paths:
        return True
    required = ("projection", "umap", "meta")
    for name in required:
        path_value = cache_paths.get(name)
        if not path_value:
            return True
        if not Path(path_value).exists():
            return True
    return False


def _load_atlas_cache(
    paths: dict[str, Path], expected_meta: dict[str, Any]
) -> tuple[Any, Any, Any, Any, Any] | None:
    projection_path = paths["projection"]
    umap_path = paths["umap"]
    meta_path = paths["meta"]

    if not projection_path.exists() or not umap_path.exists() or not meta_path.exists():
        return None

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    for key, value in expected_meta.items():
        if meta.get(key) != value:
            return None

    try:
        import numpy as np
        from joblib import load as joblib_load

        payload = np.load(projection_path, allow_pickle=False)
        projection_x = payload["x"]
        projection_y = payload["y"]
        knn_indices = payload["knn_indices"]
        knn_distances = payload["knn_distances"]
        umap_model = joblib_load(umap_path)
    except Exception:
        return None

    return projection_x, projection_y, knn_indices, knn_distances, umap_model


def _save_atlas_cache(
    paths: dict[str, Path],
    meta: dict[str, Any],
    projection_x: Any,
    projection_y: Any,
    knn_indices: Any,
    knn_distances: Any,
    umap_model: Any,
) -> None:
    ATLAS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    projection_path = paths["projection"]
    umap_path = paths["umap"]
    meta_path = paths["meta"]

    import numpy as np
    from joblib import dump as joblib_dump

    np.savez_compressed(
        projection_path,
        x=projection_x,
        y=projection_y,
        knn_indices=knn_indices,
        knn_distances=knn_distances,
    )
    joblib_dump(umap_model, umap_path)
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _compute_neighbors(embeddings: Any, neighbors_k: int) -> tuple[Any, Any, Any]:
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    count = len(embeddings)
    if count == 0:
        empty_indices = np.empty((0, 0), dtype=int)
        empty_distances = np.empty((0, 0), dtype=float)
        return empty_indices, empty_distances, None

    neighbors_k = max(0, neighbors_k)
    n_neighbors = min(neighbors_k + 1, count)
    nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    nn_model.fit(embeddings)
    distances, indices = nn_model.kneighbors(embeddings, return_distance=True)

    if indices.shape[1] > 1:
        indices = indices[:, 1:]
        distances = distances[:, 1:]
    else:
        indices = np.empty((count, 0), dtype=int)
        distances = np.empty((count, 0), dtype=float)

    return indices, distances, nn_model


def _ensure_atlas_projection(
    searcher: PaperSearcher, embeddings: Any, neighbors_k: int
) -> tuple[Any, Any, Any, Any, Any, bool, dict[str, Path]]:
    umap_params = {
        "n_neighbors": 15,
        "min_dist": 0.1,
        "metric": "cosine",
        "random_state": 42,
    }

    meta = {
        "schema_version": 1,
        "venue_id": searcher.venue_id,
        "model_name": searcher.model_name,
        "papers_fingerprint": searcher.papers_fingerprint,
        "embedding_shape": list(getattr(embeddings, "shape", ())),
        "neighbors_k": neighbors_k,
        "umap_params": umap_params,
    }

    paths = _atlas_cache_paths(searcher.venue_id, searcher.model_name)
    cached = _load_atlas_cache(paths, meta)
    if cached is not None:
        (
            projection_x,
            projection_y,
            knn_indices,
            knn_distances,
            umap_model,
        ) = cached
        return (
            projection_x,
            projection_y,
            knn_indices,
            knn_distances,
            umap_model,
            True,
            paths,
        )

    import umap

    umap_model = umap.UMAP(**umap_params)
    projection = umap_model.fit_transform(embeddings)
    projection_x = projection[:, 0]
    projection_y = projection[:, 1]

    knn_indices, knn_distances, _ = _compute_neighbors(embeddings, neighbors_k)
    _save_atlas_cache(
        paths,
        meta,
        projection_x,
        projection_y,
        knn_indices,
        knn_distances,
        umap_model,
    )
    return (
        projection_x,
        projection_y,
        knn_indices,
        knn_distances,
        umap_model,
        False,
        paths,
    )


def _example_to_text(example: dict[str, Any]) -> str:
    title = str(example.get("title") or "")
    abstract = str(example.get("abstract") or "")
    text = f"Title: {title}".strip()
    if abstract:
        text += f" Abstract: {abstract}"
    return text


def _build_atlas_dataframe(
    venue_id: str,
    query: str | None,
    examples: list[dict[str, Any]] | None,
    results: list[dict[str, Any]] | None,
) -> tuple[Any, list[dict[str, Any]], bool, dict[str, str]]:
    import numpy as np
    import pandas as pd

    papers_file = _papers_file_path(venue_id)
    searcher = PaperSearcher(
        papers_file=str(papers_file),
        venue_id=venue_id,
        cache_dir=str(EMBEDDINGS_DIR),
    )
    embeddings = searcher.ensure_embeddings()
    neighbors_k = 15

    (
        projection_x,
        projection_y,
        knn_indices,
        knn_distances,
        umap_model,
        used_cache,
        cache_paths,
    ) = _ensure_atlas_projection(searcher, embeddings, neighbors_k)

    result_lookup: dict[str, dict[str, Any]] = {}
    if isinstance(results, list):
        for index, result in enumerate(results, start=1):
            paper = result.get("paper", {})
            paper_id = str(paper.get("id") or "")
            if not paper_id:
                continue
            result_lookup[paper_id] = {
                "rank": index,
                "similarity": float(result.get("similarity", 0.0)),
            }

    rows: list[dict[str, Any]] = []
    for idx, paper in enumerate(searcher.papers):
        paper_id = paper.id or str(idx)
        result_info = result_lookup.get(paper_id)
        neighbors_ids = knn_indices[idx].tolist() if len(knn_indices) > 0 else []
        neighbors_distances = (
            knn_distances[idx].tolist() if len(knn_distances) > 0 else []
        )
        rows.append(
            {
                "row_id": idx,
                "paper_id": paper_id,
                "title": paper.title,
                "abstract": paper.abstract,
                "abstract_short": _short_abstract(paper.abstract),
                "authors": ", ".join(paper.authors),
                "primary_area": paper.primary_area,
                "keywords": ", ".join(paper.keywords),
                "forum_url": paper.forum_url,
                "text": paper.to_embedding_text(),
                "source": "paper",
                "in_results": result_info is not None,
                "result_rank": result_info["rank"] if result_info else None,
                "similarity": result_info["similarity"] if result_info else None,
                "projection_x": float(projection_x[idx]),
                "projection_y": float(projection_y[idx]),
                "neighbors": {
                    "ids": [int(value) for value in neighbors_ids],
                    "distances": [float(value) for value in neighbors_distances],
                },
            }
        )

    labels: list[dict[str, Any]] = []

    nn_model = None
    if len(embeddings) > 0:
        from sklearn.neighbors import NearestNeighbors

        n_neighbors = min(neighbors_k, len(embeddings))
        nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        nn_model.fit(embeddings)

    def add_highlight(
        label: str,
        embedding: Any,
        title: str,
        abstract: str,
        source: str,
        primary_area: str,
    ) -> None:
        nonlocal rows, labels

        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        projected = umap_model.transform(embedding)
        x_value = float(projected[0, 0])
        y_value = float(projected[0, 1])

        neighbors_payload = {"ids": [], "distances": []}
        if nn_model is not None and len(embeddings) > 0:
            distances, indices = nn_model.kneighbors(
                embedding, n_neighbors=min(neighbors_k, len(embeddings))
            )
            neighbors_payload = {
                "ids": [int(value) for value in indices[0].tolist()],
                "distances": [float(value) for value in distances[0].tolist()],
            }

        row_id = len(rows)
        rows.append(
            {
                "row_id": row_id,
                "paper_id": f"highlight-{row_id}",
                "title": title,
                "abstract": abstract,
                "abstract_short": _short_abstract(abstract),
                "authors": "",
                "primary_area": primary_area,
                "keywords": "",
                "forum_url": "",
                "text": abstract or title,
                "source": source,
                "in_results": False,
                "result_rank": None,
                "similarity": None,
                "projection_x": x_value,
                "projection_y": y_value,
                "neighbors": neighbors_payload,
            }
        )
        labels.append(
            {
                "x": x_value,
                "y": y_value,
                "text": label,
                "priority": 100,
                "level": 0,
            }
        )

    if query:
        query_embedding = searcher._embed_openai(query)
        add_highlight(
            label="Query",
            embedding=query_embedding,
            title="Query",
            abstract=query,
            source="query",
            primary_area="Query",
        )

    if examples:
        example_texts = [_example_to_text(example) for example in examples]
        if example_texts:
            example_embeddings = searcher._embed_openai(example_texts)
            if example_embeddings.ndim == 1:
                example_embeddings = example_embeddings.reshape(1, -1)
            for index, example in enumerate(examples, start=1):
                title = str(example.get("title") or f"Example {index}")
                abstract = str(example.get("abstract") or "")
                label = f"Example {index}: {title}" if title else f"Example {index}"
                embedding = np.array(example_embeddings[index - 1])
                add_highlight(
                    label=label,
                    embedding=embedding,
                    title=title,
                    abstract=abstract,
                    source="example",
                    primary_area=f"Example {index}",
                )

    return (
        pd.DataFrame(rows),
        labels,
        used_cache,
        {name: str(path) for name, path in cache_paths.items()},
    )


def _render_position_tab(
    request: dict[str, Any] | None, results: list[dict[str, Any]] | None
) -> None:
    st.subheader("Position your work")
    if not request:
        st.info("Run a search to populate the embedding atlas.")
        return

    venue_id = request.get("venue_id")
    if not venue_id:
        st.info("Venue is missing. Run a search to load papers.")
        return

    request_key = _atlas_request_key(request)
    if "atlas_ready" not in st.session_state:
        st.session_state["atlas_ready"] = False
    if "atlas_rendered_once" not in st.session_state:
        st.session_state["atlas_rendered_once"] = False
    if "atlas_phase" not in st.session_state:
        st.session_state["atlas_phase"] = None
    if "last_predicate" not in st.session_state:
        st.session_state["last_predicate"] = None

    status_placeholder = st.empty()
    payload = st.session_state.get("atlas_payload")
    cache_paths = None
    cache_missing = True
    if isinstance(payload, dict):
        cache_paths = payload.get("cache_paths")
        cache_missing = _atlas_cache_missing(cache_paths)
        if payload.get("df") is None:
            cache_missing = True

    needs_rebuild = (
        not isinstance(payload, dict)
        or payload.get("key") != request_key
        or cache_missing
    )

    if needs_rebuild:
        st.session_state["atlas_ready"] = False
        st.session_state["atlas_rendered_once"] = False
        st.session_state["atlas_phase"] = "preparing"
        st.session_state["last_predicate"] = None
        status_placeholder.info("Preparing data...")
        try:
            df, labels, used_cache, cache_paths = _build_atlas_dataframe(
                venue_id=venue_id,
                query=request.get("query"),
                examples=request.get("examples"),
                results=results,
            )
        except Exception as exc:
            status_placeholder.empty()
            st.error(f"Atlas failed: {exc}")
            return

        payload = {
            "key": request_key,
            "df": df,
            "labels": labels,
            "used_cache": used_cache,
            "cache_paths": {
                name: path
                for name, path in (cache_paths or {}).items()
                if name in {"projection", "umap", "meta"}
            },
        }
        st.session_state["atlas_payload"] = payload
        st.session_state["atlas_ready"] = True
        st.session_state["atlas_phase"] = "rendering"
        status_placeholder.info("Rendering atlas...")
    else:
        if not isinstance(payload, dict):
            st.info("Atlas data is not available. Run a search to refresh.")
            return
        df = payload.get("df")
        labels = payload.get("labels") or []
        used_cache = bool(payload.get("used_cache"))
        if df is None:
            st.info("Atlas data is not available. Run a search to refresh.")
            return
        if not st.session_state.get("atlas_rendered_once"):
            st.session_state["atlas_phase"] = "rendering"

    if not st.session_state.get("atlas_rendered_once"):
        phase = st.session_state.get("atlas_phase")
        if phase == "preparing":
            status_placeholder.info("Preparing data...")
        elif phase == "rendering":
            status_placeholder.info("Rendering atlas...")
    else:
        status_placeholder.empty()

    if used_cache:
        st.caption("Using cached atlas projection.")

    st.caption(
        "Tip: color by `source` to spotlight your query/examples, or filter "
        "`in_results` to highlight top-k results."
    )

    preset = st.radio(
        "Atlas preset",
        options=["Default", "Among top-k"],
        horizontal=True,
        key="atlas_preset",
    )

    try:
        from embedding_atlas.options import make_embedding_atlas_props
        from embedding_atlas.streamlit import _embedding_atlas
    except Exception as exc:
        st.error(f"Embedding Atlas is not available: {exc}")
        return

    df_view = df
    neighbors_column = "neighbors"
    if preset == "Among top-k":
        neighbors_column = None
        if "in_results" in df.columns and "source" in df.columns:
            mask = df["in_results"].fillna(False) | df["source"].isin(
                ["query", "example"]
            )
            df_view = df.loc[mask]
        if df_view.empty:
            st.info("No results or highlights to display for this preset.")
            return

    props = make_embedding_atlas_props(
        row_id="row_id",
        x="projection_x",
        y="projection_y",
        text="text",
        neighbors=neighbors_column,
        labels=labels,
        point_size=2.5,
        show_table=False,
        show_charts=False,
        show_embedding=True,
    )
    props["defaultChartsConfig"] = {
        "embedding": {
            "data": {
                "x": "projection_x",
                "y": "projection_y",
                "text": "text",
                "category": "primary_area",
            }
        }
    }

    atlas_key = f"atlas-{request_key}-{preset.lower().replace(' ', '-')}"
    value = _embedding_atlas(data_frame=df_view, props=props, key=atlas_key, default={})

    if not st.session_state.get("atlas_rendered_once"):
        st.session_state["atlas_rendered_once"] = True
        st.session_state["atlas_phase"] = None

    predicate = value.get("predicate") if isinstance(value, dict) else None
    last_predicate = st.session_state.get("last_predicate")
    if predicate and st.session_state.get("atlas_rendered_once"):
        if predicate == last_predicate:
            return
        st.session_state["last_predicate"] = predicate
        import duckdb

        selection = duckdb.query_df(
            df_view, "dataframe", "SELECT * FROM dataframe WHERE " + predicate
        ).df()
        selection_columns = [
            "title",
            "primary_area",
            "source",
            "result_rank",
            "similarity",
            "forum_url",
        ]
        selection_columns = [
            column for column in selection_columns if column in selection.columns
        ]
        if selection_columns:
            selection = selection[selection_columns]
        st.markdown("### Selection")
        st.dataframe(selection, use_container_width=True, hide_index=True)


def _run_pipeline(
    conference: str,
    year: int,
    venue_id: str,
    top_k: int,
    query: str | None,
    examples: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    papers_file = _papers_file_path(venue_id)

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

        if not os.getenv("OPENAI_API_KEY"):
            st.error(
                "OPENAI_API_KEY is required to embed queries/examples. Set it in your shell and restart the app."
            )
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
    if "last_request" not in st.session_state:
        st.session_state["last_request"] = None

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
            st.session_state["last_request"] = request
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")

    results = st.session_state.get("results")
    tabs = st.tabs(["Search results", "Position your work"])
    with tabs[0]:
        if isinstance(results, list):
            _render_results(results, cards_per_row=3)
        else:
            _render_empty_state()

    with tabs[1]:
        _render_position_tab(
            st.session_state.get("last_request"),
            results if isinstance(results, list) else None,
        )


if __name__ == "__main__":
    main()
