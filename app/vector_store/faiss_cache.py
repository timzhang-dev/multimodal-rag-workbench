import json
import os

import faiss

from app.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COHERE_EMBED_MODEL_ID,
    EMBEDDING_PROVIDER,
    EMBEDDING_VECTOR_DIMENSION,
    MINERU_ADD_PAGE_RENDERS,
    PARSER,
)

TITAN_EMBED_MODEL_ID = "amazon.titan-embed-image-v1"
INDEXES_DIR = os.path.join("data", "indexes")


def _embed_model_id() -> str:
    return (
        COHERE_EMBED_MODEL_ID
        if EMBEDDING_PROVIDER == "cohere"
        else TITAN_EMBED_MODEL_ID
    )


def _shared_cache_fields() -> dict:
    return {
        "parser": PARSER,
        "embedding_provider": EMBEDDING_PROVIDER,
        "embedding_dimension": EMBEDDING_VECTOR_DIMENSION,
        "embed_model_id": _embed_model_id(),
        "mineru_add_page_renders": MINERU_ADD_PAGE_RENDERS,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }


def build_faiss_cache_metadata(pdf_path: str, doc_id: str) -> dict:
    stat = os.stat(pdf_path)
    return {
        "mode": "single",
        "doc_id": doc_id,
        "source_file": os.path.normpath(pdf_path),
        "pdf_mtime": stat.st_mtime,
        "pdf_size": stat.st_size,
        **_shared_cache_fields(),
    }


def build_multi_faiss_cache_metadata(
    pdf_paths: list[str], cache_id: str = "multi"
) -> dict:
    documents = []
    for pdf_path in pdf_paths:
        stat = os.stat(pdf_path)
        documents.append(
            {
                "doc_id": os.path.splitext(os.path.basename(pdf_path))[0],
                "source_file": os.path.normpath(pdf_path),
                "pdf_mtime": stat.st_mtime,
                "pdf_size": stat.st_size,
            }
        )
    return {
        "mode": "multi",
        "cache_id": cache_id,
        "documents": documents,
        **_shared_cache_fields(),
    }


def get_faiss_cache_paths(doc_id: str, cache_metadata: dict) -> tuple[str, str]:
    pages_flag = "on" if cache_metadata["mineru_add_page_renders"] else "off"
    base = (
        f"{doc_id}_{cache_metadata['parser']}_"
        f"{cache_metadata['embedding_provider']}_"
        f"{cache_metadata['embedding_dimension']}_pages-{pages_flag}"
    )
    index_path = os.path.join(INDEXES_DIR, f"{base}.faiss")
    metadata_path = os.path.join(INDEXES_DIR, f"{base}_metadata.json")
    return index_path, metadata_path


def get_multi_faiss_cache_paths(cache_metadata: dict) -> tuple[str, str]:
    pages_flag = "on" if cache_metadata["mineru_add_page_renders"] else "off"
    base = (
        f"multi_{cache_metadata['parser']}_"
        f"{cache_metadata['embedding_provider']}_"
        f"{cache_metadata['embedding_dimension']}_pages-{pages_flag}"
    )
    index_path = os.path.join(INDEXES_DIR, f"{base}.faiss")
    metadata_path = os.path.join(INDEXES_DIR, f"{base}_metadata.json")
    return index_path, metadata_path


def is_faiss_cache_valid(
    metadata_path: str, index_path: str, current_metadata: dict
) -> bool:
    if not os.path.isfile(metadata_path) or not os.path.isfile(index_path):
        return False

    with open(metadata_path, encoding="utf-8") as f:
        data = json.load(f)

    stored_metadata = data.get("cache_metadata")
    if stored_metadata is None:
        return False

    return stored_metadata == current_metadata


def save_faiss_cache(
    index, items, metadata_path: str, index_path: str, cache_metadata: dict
) -> None:
    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    items_without_embeddings = [
        {k: v for k, v in item.items() if k != "embedding"} for item in items
    ]

    faiss.write_index(index, index_path)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "cache_metadata": cache_metadata,
                "items": items_without_embeddings,
            },
            f,
            indent=2,
        )


def load_faiss_cache(index_path: str, metadata_path: str) -> tuple:
    index = faiss.read_index(index_path)

    with open(metadata_path, encoding="utf-8") as f:
        data = json.load(f)

    items = data["items"]

    if index.ntotal != len(items):
        raise ValueError(
            f"FAISS index size ({index.ntotal}) does not match "
            f"metadata items count ({len(items)})"
        )

    return index, items
