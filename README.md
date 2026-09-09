# Multimodal Technical Document RAG

## Overview

This is a multimodal Retrieval-Augmented Generation (RAG) system designed for technical PDF documents, engineering specifications, maintenance procedures, diagrams, and wiring schematics.

Unlike traditional text-only RAG systems, this RAG system processes and retrieves information from:

* Text
* Tables
* Extracted images
* Full-page rendered document images

This enables question answering over visual content that is not directly extractable from PDFs, including engineering drawings, wiring diagrams, dimensional layouts, and scanned documentation.

The system currently serves as a research and prototyping platform for multimodal technical-document retrieval and question answering.

---

## Current Architecture

```text
PDF
│
├── MinerU OCR + Layout Analysis
│
├── Text Blocks
├── Tables
├── Extracted Images
└── Full Page Renders
        │
        ▼
Multimodal Embeddings
(Cohere Embed v4 or Amazon Titan)
        │
        ▼
FAISS Vector Index
        │
        ▼
Top-K Retrieval
        │
        ▼
Prompt Construction
        │
        ▼
Claude via Amazon Bedrock
        │
        ▼
Answer Generation
```

---

## Features

### Document Processing

* MinerU-based PDF parsing
* OCR support for scanned and image-heavy PDFs
* Layout-aware extraction
* Metadata preservation
* Automatic parsing cache

### Text Retrieval

* Semantic chunking
* Multimodal embeddings
* Similarity search via FAISS
* Citation-aware retrieval

### Table Retrieval

* Table extraction
* Table flattening for retrieval
* Raw HTML preservation
* Table-aware question answering

### Image Retrieval

* Embedded image extraction
* Multimodal image embeddings
* Visual retrieval
* Diagram and illustration understanding

### Page-Level Retrieval

* Full-page rendering
* Retrieval of complete engineering drawings
* Retrieval of non-extractable visual content
* Improved reasoning over technical diagrams

### Embedding Framework

Supports A/B testing between:

* Cohere Embed v4
* Amazon Titan Multimodal Embeddings

Current evaluation indicates Cohere significantly improves retrieval of visual content and engineering diagrams.

### Question Answering

* Retrieval-Augmented Generation
* Multimodal context assembly
* Claude via Amazon Bedrock
* Support for text, tables, images, and page-level context

---

## Project Structure

```text
app/
├── embeddings/
│   ├── titan_embedder.py
│   └── cohere_embedder.py
│
├── ingestion/
│   ├── mineru_processor.py
│   ├── pdf_processor.py
│   ├── text_chunker.py
│   ├── table_extractor.py
│   ├── image_extractor.py
│   └── page_renderer.py
│
├── rag/
│   ├── retriever.py
│   ├── prompt_builder.py
│   └── generator.py
│
├── vector_store/
│   └── faiss_store.py
│
├── utils/
│   └── file_utils.py
│
└── config.py

scripts/
├── ingest_pdf.py
├── ask_question.py
├── test_extraction.py
├── test_nova.py
└── test_titan.py
```

---

## Current Strengths

* Strong text retrieval
* Strong table retrieval
* Multimodal image retrieval
* Full-page diagram retrieval
* Engineering drawing understanding
* Visual reasoning over technical documentation
* Cohere-based retrieval significantly improves visual-content ranking
* Supports OCR-heavy and scanned PDFs through MinerU

---


## Current Roadmap

### Retrieval Quality

* Retrieval reranking
* Vision summaries
* Contextual retrieval
* Metadata-aware retrieval

### Scalability

* Save and reuse FAISS indexes
* Large-PDF evaluation
* Multi-PDF retrieval
* OpenSearch Serverless migration
* S3 artifact storage

### Advanced Retrieval

* Hybrid search
* Query expansion
* Agentic retrieval workflows
* Production-scale technical-document retrieval

---

## Configuration

Example:

```env
AWS_REGION=...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

Embedding provider selection:

```python
EMBEDDING_PROVIDER = "cohere"
# or
EMBEDDING_PROVIDER = "titan"
```

---

## Running the Pipeline

Process a document:

```bash
python -m scripts.ingest_pdf
```

Run extraction tests:

```bash
python -m scripts.test_extraction
```

Ask a question:

```bash
python -m scripts.ask_question
```

---

## Technologies

* Python
* Amazon Bedrock
* Anthropic Claude
* Cohere Embed v4
* Amazon Titan Multimodal Embeddings
* MinerU
* FAISS
* NumPy
* Pillow
* BeautifulSoup




