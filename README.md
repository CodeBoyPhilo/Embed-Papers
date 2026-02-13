# embed-papers

`embed-papers` crawls OpenReview submissions and runs semantic search with OpenAI embeddings.

This is a helper package for my agentic research workflow.
Originally forked from [gyj155/SearchPaperByEmbedding](https://github.com/gyj155/SearchPaperByEmbedding).

## CLI contract (for agents)

- stdout always prints one JSON object
- stderr is reserved for logs/progress
- non-zero exit codes still emit JSON on stdout

Success envelope:

```json
{
  "ok": true,
  "schema_version": "1",
  "command": "search",
  "data": {...}
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
embed-papers crawl \
  --venue-id "ICLR.cc/2026/Conference" \
  --output-file iclr2026_papers.json
```

By default, crawl fails when zero papers are found (to catch wrong venue ids early).
Use `--allow-empty` to explicitly accept empty results.

Optional reliability flags:

```bash
embed-papers crawl \
  --venue-id "ICLR.cc/2026/Conference" \
  --output-file iclr2026_papers.json \
  --timeout-seconds 30 \
  --retry-attempts 5

embed-papers crawl \
  --venue-id "Some/Venue" \
  --output-file empty-ok.json \
  --allow-empty
```

### Warm cache

```bash
export OPENAI_API_KEY="<your-key>"
embed-papers warm-cache \
  --papers-file iclr2026_papers.json \
  --venue-id "ICLR.cc/2026/Conference"
```

### Search

```bash
embed-papers search \
  --papers-file iclr2026_papers.json \
  --venue-id "ICLR.cc/2026/Conference" \
  --query "foundation models for planning" \
  --top-k 20
```

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
searcher.save(results, "results.json")
```
