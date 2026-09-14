# ragkit

Reusable RAG retrieve / ingest / generate library extracted from the LangGraph playground.

Apps own their documents, prompts, and indexes. This package owns chunking into
Postgres/pgvector and Elasticsearch, Voyage embeddings, optional Cohere rerank,
and optional OpenAI generation.

## Install

From another project:

```toml
[project]
dependencies = ["ragkit[postgres,voyage]"]

[tool.uv.sources]
ragkit = { path = "../ragkit", editable = true }
```

Or from git once this repo has a remote:

```toml
ragkit = { git = "ssh://git@github.com/you/ragkit.git", extras = ["postgres", "voyage"] }
```

Extras:

| Extra | When you need it |
|---|---|
| `postgres` | dense vector ingest and retrieve |
| `elasticsearch` | BM25 + kNN hybrid retrieve |
| `voyage` | default Voyage-3.5 embeddings |
| `openai` | `RagPipeline.generate` |
| `rerank` | Cohere rerank |
| `tracing` | Braintrust `@traced` / `setup_braintrust` |
| `all` | everything above |

## Usage

```python
from langchain_core.documents import Document
from ragkit import RagConfig, RagPipeline

pipeline = RagPipeline(
    RagConfig(
        index_name="support_docs",
        chunk_size=512,
        strategy="vector",
        system_prompt="Answer only from the provided support articles.",
    )
)
pipeline.ingest(
    [Document(page_content="...", metadata={"source": "faq.md", "page": 1})],
    targets=("vector",),
)
hits = pipeline.retrieve("How do I reset my password?")
answer = pipeline.generate("How do I reset my password?")
```

Inject embeddings or URLs instead of relying on process env:

```python
from ragkit import RagConfig, RagPipeline, RagkitSettings
from ragkit.embeddings import voyage_embeddings

settings = RagkitSettings()
pipeline = RagPipeline(
    RagConfig(index_name="policies", strategy="hybrid"),
    embeddings=voyage_embeddings(),
)
```

`RagkitSettings` reads `DATABASE_URL`, `ELASTICSEARCH_URL`, and provider keys
from the environment. The package does **not** call `load_dotenv()` or start
Braintrust on import. If you want traces:

```python
from ragkit import setup_braintrust

setup_braintrust()
```

## What this is not

Document loading, eval goldens, FastAPI, and LangGraph graphs stay in the
calling app. Pass `list[Document]`; get `RetrievalResult` or a generate payload back.
