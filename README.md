# DualGraphRAG

Implementation of [DualGraphRAG: A Dual-View Graph-Enhanced Retrieval-Augmented Generation Framework](https://doi.org/10.3390/app16052221) (Li & Qin, Applied Sciences 2026).

## Architecture

1. **KG Construction** — Chunks documents, extracts (head, relation, tail) triples via LLM (Groq llama-3.3-70b-versatile), stores in Neo4j, embeds node names in ChromaDB via `all-MiniLM-L6-v2`.
2. **Query Enhancement** — Extracts explicit/implied entities from the question via NER, aligns them to KG nodes via embedding similarity, generates implicit (intermediate) nodes for multi-hop paths.
3. **Dual-View Retrieval** — Retrieves 1-hop triples (local context) and shortest paths between relevant nodes (global connectivity).
4. **QA** — Feeds retrieved triples + paths to LLM for answer generation.

## Requirements

- Docker (for Neo4j)
- Groq API key (set in `.env`)

## Setup

```bash
# Start Neo4j
docker compose up -d

# Create .env with your key
echo "GROQ_API_KEY=gsk_..." > .env

# Build the knowledge graph from the demo corpus
uv run python src/main.py build

# Ask questions
uv run python src/main.py ask "What city is the birthplace of the person who discovered radium?"
```

## Project Structure

```
src/
  config.py           — Configuration (models, paths, thresholds)
  kg_construction.py  — Chunking, triple extraction, Neo4j + ChromaDB
  query_enhancer.py   — NER, embedding alignment, implicit node generation
  retriever.py        — 1-hop + shortest path retrieval
  qa_pipeline.py      — Prompt assembly and answer generation
  main.py             — CLI entry point (build / ask)
data/
  corpus.txt          — Demo corpus (scientists, places, discoveries)
  chroma_db/          — Persistent ChromaDB storage
```
