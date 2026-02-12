# embed-papers

`embed-papers` crawls OpenReview submissions and performs semantic search using OpenAI embeddings.

This is a helper package for my agentic research workflow.
Originally forked from [gyj155/SearchPaperByEmbedding](https://github.com/gyj155/SearchPaperByEmbedding).

## Python API

### 1) Crawl conference papers

```python
from embed_papers import crawl_papers

crawl_papers(
    venue_id="ICLR.cc/2026/Conference",
    output_file="iclr2026_papers.json",
)
```

The crawler writes a JSON object with this shape:

```json
{
  "venue_id": "ICLR.cc/2026/Conference/Submission",
  "total": 1234,
  "papers": [
    {
      "id": "...",
      "title": "...",
      "abstract": "..."
    }
  ]
}
```

### 2) Warm cache / search

```python
from embed_papers import PaperSearcher

searcher = PaperSearcher(
    papers_file="iclr2026_papers.json",
    venue_id="ICLR.cc/2026/Conference",
    model_name="text-embedding-3-large",
)

# If cache exists, this works without OPENAI_API_KEY.
# If cache does not exist, OPENAI_API_KEY is required.
searcher.ensure_embeddings()

results = searcher.search(query="robotics planning language model", top_k=100)
searcher.display(results, n=10)
searcher.save(results, "results.json")
```

## CLI

### Crawl

```bash
embed-papers crawl \
  --venue-id "ICLR.cc/2026/Conference" \
  --output-file iclr2026_papers.json
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

All CLI commands print machine-readable JSON, suitable for LLM agents.

