# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A multimodal Retrieval-Augmented Generation (RAG) prototype for technical PDF documents (engineering specs, maintenance manuals, wiring schematics). Unlike text-only RAG, it retrieves and reasons over four content modalities from a PDF: text chunks, tables, extracted embedded images, and full-page renders — the last of these lets it answer questions about diagrams/drawings that have no extractable text. This is a research/prototyping codebase, not a production service: there is no test suite, no CI, and no packaging — it runs as scripts invoked directly against local data files.

## Environment setup

There are two virtual environments, both currently configured for Windows (`pyvenv.cfg` points at Windows Python installs and `Scripts/` executables) and will not work as-is on macOS/Linux — they need to be recreated for the current platform before running anything:

- `.venv` — main app environment (Python 3.14 originally), install with `pip install -r requirements.txt`
- `.venv-mineru` — separate environment for the MinerU CLI, **must be Python 3.12** (MinerU pins this). `app/config.py` (`MINERU_EXE`) shells out to `.venv-mineru/Scripts/mineru.exe` (or the equivalent `bin/mineru` on macOS/Linux) as a subprocess — MinerU is not imported as a library.

AWS credentials (Bedrock access) are read via `python-dotenv` from `.env`. Required: `AWS_REGION`, plus standard AWS credential env vars/profile. All embedding and generation calls go through `boto3` `bedrock-runtime`.

## Running the pipeline

There is no test framework (no pytest). `scripts/test_*.py` are manual smoke-test scripts that hit real AWS Bedrock APIs — run them directly with the venv's Python, not via a test runner:

```bash
python -m scripts.test_extraction   # runs PDF extraction only, prints item counts/samples
python -m scripts.test_titan        # calls Titan embedding API, sanity-checks response
python -m scripts.test_nova         # calls Claude (Nova/Bedrock) generation with a mock context item
python -m scripts.ingest_pdf        # full pipeline: extract -> embed -> index -> retrieve -> generate, prints Q&A for hardcoded queries
python -m scripts.ingest_pdf --force          # bypass MinerU output cache, re-run MinerU CLI
python -m scripts.ingest_pdf --rebuild-index  # bypass FAISS/embedding cache, re-embed everything
python -m scripts.ingest_pdf --multi          # build one combined FAISS index over app.config.PDF_PATHS instead of app.config.FILEPATH
```

`scripts/ask_question.py` is a stub — not implemented yet.

There's no lint/format tooling configured (no ruff/black/flake8 config) — match existing style in the file you're editing.

## Configuration

All tunables live in `app/config.py` (no env-based overrides beyond AWS creds) — this is the first file to check/edit when changing behavior:

- `FILEPATH` / `PDF_PATHS` — single-document vs. multi-document (`--multi`) input selection
- `PARSER` — `"mineru"` (OCR + layout-aware, default) or `"pymupdf"` (simpler, faster, used by `pdf_processor.process_pdf_with_pymupdf`)
- `MINERU_BACKEND` / `MINERU_METHOD` / `MINERU_LANG` — passed straight through to the MinerU CLI invocation
- `MINERU_FORCE_REPARSE` / `MINERU_ADD_PAGE_RENDERS` — cache-busting and page-render toggle
- `CHUNK_SIZE` / `CHUNK_OVERLAP` — passed to `RecursiveCharacterTextSplitter` for text chunking
- `EMBEDDING_PROVIDER` — `"cohere"` (Cohere Embed v4 via Bedrock, current default — empirically much better on visual/diagram content per README) or `"titan"` (Amazon Titan Multimodal)
- `EMBEDDING_VECTOR_DIMENSION`, `TOP_K` — retrieval tuning

## Architecture

Two-stage pipeline: **ingestion** (`app/ingestion/`) turns a PDF into a flat list of typed "items", then **indexing/retrieval/generation** (`scripts/ingest_pdf.py`, `app/vector_store/`, `app/rag/`) embeds, indexes, retrieves, and answers.

### The `items` list — the pipeline's central data structure

Every stage (extraction, embedding, indexing, retrieval, prompting) passes around a plain `list[dict]`. Each item always has `page`, `type` (`"text"` | `"table"` | `"image"` | `"page"`), `path`, and a `metadata` dict (built by `app/utils/metadata.py:build_item_metadata`, containing `doc_id`, `source_file`, `chunk_id`, etc. — `chunk_id` must be unique across the corpus, checked by `_validate_chunk_ids` in multi-doc mode). Text/table items additionally carry `text`; image/page items carry `image` (base64). After embedding, an `embedding` key is added — it's the one key stripped out before anything is persisted (FAISS cache, retrieval results) since it's redundant with the FAISS index itself.

### Ingestion (`app/ingestion/`)

`pdf_processor.process_pdf()` dispatches on `config.PARSER`:

- **MinerU path** (`mineru_processor.py`, default): shells out to the MinerU CLI as a subprocess, which OCRs and layout-parses the PDF into a `content_list_v2.json` (blocks per page, typed `paragraph`/`title`/`table`/`image`/etc. under `data/mineru/<doc_id>/ocr/`). `parse_content_list_v2` walks these blocks, converting each into one or more `items` — text blocks get chunked via `RecursiveCharacterTextSplitter`, tables get flattened from HTML to pipe-delimited text (raw HTML kept in `metadata["table_html"]`), images get copied out and base64-encoded. MinerU output is cached on disk keyed by PDF mtime+size (`is_mineru_cache_valid` / `write_source_meta`) — a re-run with the same PDF reuses the cached parse unless `--force`.
- **pymupdf path** (`pdf_processor.py` + `table_extractor.py`/`text_chunker.py`/`image_extractor.py`): simpler per-page extraction — `tabula-py` for tables, `page.get_text()` for text, `page.get_images()` for embedded images. No OCR, no layout awareness.
- **Page rendering** (`page_renderer.py`, both paths, gated by `MINERU_ADD_PAGE_RENDERS`): rasterizes each full page to PNG via pymupdf regardless of parser — these `"page"`-type items are what let the system answer questions about diagrams/schematics that have no machine-readable text.

Extracted files are written under `data/{text,tables,images,page_images}/`; MinerU's raw parse output lives separately under `data/mineru/`.

### Indexing & caching (`app/vector_store/`)

`faiss_store.build_index` builds a flat `IndexFlatL2` (brute-force, no approximate search — fine at prototype scale per README limitations). `faiss_cache.py` is the persistence layer: it derives a cache key/filename from a `cache_metadata` dict (parser, embedding provider+dimension, chunk size/overlap, page-render flag, plus source PDF mtime/size for single-doc or a list of docs for `--multi`), and `is_faiss_cache_valid` does an exact dict comparison against what's stored in `data/indexes/<key>_metadata.json` to decide whether to reuse the cached `.faiss` index or re-embed from scratch. Changing *any* config value that feeds `cache_metadata` invalidates the cache automatically.

### Embeddings (`app/embeddings/`)

Two interchangeable backends, both called through Bedrock `invoke_model`, both selected via `config.EMBEDDING_PROVIDER` and dispatched from call sites (`scripts/ingest_pdf.py:_embed_item`, `app/rag/retriever.py`) — there's no shared interface/ABC, just two modules with matching `generate_multimodal_embeddings(prompt=None, image=None, ...)` signatures. Cohere Embed v4 requires text *or* image per call, never both; Titan accepts either or both in one call.

### RAG (`app/rag/`)

- `retriever.py:retrieve` — embeds the query (same provider as the index was built with), does a FAISS `k`-NN search, returns matched items (embedding stripped).
- `prompt_builder.py` — assigns `SOURCE N` citation IDs to retrieved items and formats per-source headers (`doc_id`, page, type) used both in console output and in the model prompt, so answers can cite `[SOURCE N]`.
- `generator.py:invoke_nova_multimodal` — builds a Bedrock Converse API message: text/table items go in as `{"text": ...}` blocks, image/page items are decoded, resized if needed (`MAX_IMAGE_SIDE`, Bedrock's multi-image limit), and inserted as `{"image": ...}` blocks. Despite the function name, the model invoked is Claude (`global.anthropic.claude-sonnet-4-6`) via Bedrock's Converse API, not a Nova model — the naming is a holdover.

### Data flow summary

```
PDF → process_pdf() → items[] (text/table/image/page, unembedded)
    → _embed_items() → items[] (+ embedding key)
    → build_index() → FAISS IndexFlatL2  ─┐
    → save_faiss_cache()                  ├─ cached under data/indexes/
    → retrieve(query, index, items)  ← ───┘
    → assign_citation_ids()
    → invoke_nova_multimodal() → answer text
```
