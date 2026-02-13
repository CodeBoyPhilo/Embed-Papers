# embed-papers

`embed-papers` crawls OpenReview submissions and runs semantic search with OpenAI embeddings.

This is a helper package for my agentic research workflow.
Originally forked from [gyj155/SearchPaperByEmbedding](https://github.com/gyj155/SearchPaperByEmbedding).

# For Agents

## CLI contract

- stdout always prints one JSON object
- stderr is reserved for logs/progress
- non-zero exit codes still emit JSON on stdout

Success envelope:

```json
{
  "ok": true,
  "schema_version": "1",
  "command": "search",
  "data": {}
}
```

Error envelope:

```json
{
  "ok": false,
  "schema_version": "1",
  "command": "search",
  "error": {
    "type": "InvalidPapersFileError",
    "message": "..."
  }
}
```

## CLI usage

### Crawl

```bash
embed-papers crawl --venue-id "ICLR.cc/2026/Conference" --skip-if-exists
```

By default, crawl fails when zero papers are found (to catch wrong venue ids early).

Use `--skip-if-exists` to reuse an existing output file and skip calling OpenReview.

If `--output-file` is omitted, crawl defaults to:

- `~/.cache/embed-papers/papers/<venue-id-slug>.json`

### Warm cache

```bash
export OPENAI_API_KEY="<your-key>"
embed-papers warm-cache \
  --papers-file iclr2026_papers.json \
  --venue-id "ICLR.cc/2026/Conference"
```

`--papers-file` is optional if `--venue-id` is provided.
In that case, it defaults to `~/.cache/embed-papers/papers/<venue-id-slug>.json`.

If `--cache-dir` is omitted, embeddings default to:

- `~/.cache/embed-papers/embeddings`

### Search

```bash
embed-papers search \
  --papers-file iclr2026_papers.json \
  --venue-id "ICLR.cc/2026/Conference" \
  --query "foundation models for planning" \
  --top-k 20
```

`--papers-file` is optional if `--venue-id` is provided.
In that case, it defaults to `~/.cache/embed-papers/papers/<venue-id-slug>.json`.

`search` uses the same default embeddings cache dir (`~/.cache/embed-papers/embeddings`) unless `--cache-dir` is provided.

# For Human

Make sure you have set an `OPENAI_API_KEY` in your shell.
In the command line, run:

```bash
embed-papers host
```

This launches a local Streamlit UI in your browser for interactive use.

Viewer flow:

- enter conference abbreviation + year (auto-builds venue id)
- choose direct query or examples upload
- set top-k and run search
- auto-crawl papers if missing
- auto-build embeddings cache if missing

Cache directories used by viewer:

- `~/.cache/embed-papers/papers`
- `~/.cache/embed-papers/embeddings`

## Python API

### 1) Crawl conference papers

```python
from embed_papers import crawl_papers

_ = crawl_papers(
    venue_id="ICLR.cc/2026/Conference",
    output_file="iclr2026_papers.json",
)
```

### 2) Warm cache / search

```python
from embed_papers import PaperSearcher

searcher = PaperSearcher(
    papers_file="iclr2026_papers.json",
    venue_id="ICLR.cc/2026/Conference",
    model_name="text-embedding-3-large",
)

searcher.ensure_embeddings()
results = searcher.search(query="robotics planning language model", top_k=100)
searcher.display(results, n=10, show_abstract=True, abstract_max_chars=500)
searcher.save(results, "results.json")
```
