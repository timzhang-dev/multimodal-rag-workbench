import argparse
import logging
import os
from collections import Counter

from tqdm import tqdm

from app.config import EMBEDDING_PROVIDER, EMBEDDING_VECTOR_DIMENSION, FILEPATH, PDF_PATHS
from app.ingestion.pdf_processor import process_pdf
from app.embeddings import cohere_embedder
from app.embeddings.titan_embedder import generate_multimodal_embeddings as titan_generate
from app.rag.retriever import retrieve
from app.rag.generator import invoke_nova_multimodal
from app.rag.prompt_builder import assign_citation_ids, format_retrieved_sources_line
from app.vector_store.faiss_cache import (
    build_faiss_cache_metadata,
    build_multi_faiss_cache_metadata,
    get_faiss_cache_paths,
    get_multi_faiss_cache_paths,
    is_faiss_cache_valid,
    load_faiss_cache,
    save_faiss_cache,
)
from app.vector_store.faiss_store import build_index

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _embed_item(prompt=None, image=None):
    if EMBEDDING_PROVIDER == "cohere":
        return cohere_embedder.generate_multimodal_embeddings(
            prompt=prompt,
            image=image,
            output_embedding_length=EMBEDDING_VECTOR_DIMENSION,
            input_type="search_document",
        )
    return titan_generate(
        prompt=prompt,
        image=image,
        output_embedding_length=EMBEDDING_VECTOR_DIMENSION,
    )


def _log_item_counts_by_doc_id(items):
    counts = Counter(
        item.get("metadata", {}).get("doc_id", "unknown") for item in items
    )
    logger.info("Item counts by document:")
    for doc_id, count in sorted(counts.items()):
        logger.info("  %s: %s", doc_id, count)


def _validate_chunk_ids(items):
    chunk_ids = [
        item["metadata"]["chunk_id"]
        for item in items
        if item.get("metadata", {}).get("chunk_id")
    ]
    seen = set()
    duplicates = []
    for chunk_id in chunk_ids:
        if chunk_id in seen:
            duplicates.append(chunk_id)
        seen.add(chunk_id)
    if duplicates:
        unique_examples = list(dict.fromkeys(duplicates))[:5]
        raise RuntimeError(
            f"Found {len(duplicates)} duplicate chunk_id values. "
            f"Examples: {unique_examples}"
        )


def _embed_items(items):
    item_counts = {
        "text": sum(1 for item in items if item["type"] == "text"),
        "table": sum(1 for item in items if item["type"] == "table"),
        "image": sum(1 for item in items if item["type"] == "image"),
        "page": sum(1 for item in items if item["type"] == "page"),
    }
    print("\nItem counts by type:")
    for item_type, count in item_counts.items():
        print(f"  {item_type}: {count}")
    print(f"  total: {len(items)}")

    counters = dict.fromkeys(item_counts.keys(), 0)
    skipped_items = []

    logger.info("Embedding provider: %s", EMBEDDING_PROVIDER)
    first_embedding_logged = False

    with tqdm(
        total=len(items),
        desc="Generating embeddings",
        bar_format=(
            "{l_bar}{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        ),
    ) as pbar:
        for item in items:
            item_type = item["type"]
            counters[item_type] += 1

            try:
                if item_type in ["text", "table"]:
                    text = item.get("text")

                    if not text or not text.strip():
                        skipped_items.append(item)
                        pbar.update(1)
                        continue

                    item["embedding"] = _embed_item(prompt=text)

                else:
                    image = item.get("image")

                    if not image:
                        skipped_items.append(item)
                        pbar.update(1)
                        continue

                    item["embedding"] = _embed_item(image=image)

                if not first_embedding_logged and item.get("embedding"):
                    logger.info(
                        "First embedding dimension: %s", len(item["embedding"])
                    )
                    first_embedding_logged = True

            except Exception as e:
                raise RuntimeError(
                    f"Embedding failed for item type={item_type}, "
                    f"page={item.get('page')}, path={item.get('path')}. "
                    f"Original error: {e}"
                )

            pbar.set_postfix_str(
                f"Text: {counters.get('text', 0)}/{item_counts.get('text', 0)}, "
                f"Table: {counters.get('table', 0)}/{item_counts.get('table', 0)}, "
                f"Image: {counters.get('image', 0)}/{item_counts.get('image', 0)}"
            )
            pbar.update(1)

    if skipped_items:
        print("\nSkipped empty items:")
        for item in skipped_items:
            print(
                "type:",
                item.get("type"),
                "page:",
                item.get("page"),
                "path:",
                item.get("path"),
            )

    failed_items = [
        item
        for item in items
        if item not in skipped_items and item.get("embedding") is None
    ]
    if failed_items:
        item = failed_items[0]
        raise RuntimeError(
            f"Embedding failed for item type={item.get('type')}, "
            f"page={item.get('page')}, path={item.get('path')}"
        )

    return [item for item in items if item.get("embedding") is not None]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run MinerU CLI even if cached output exists",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Ignore FAISS cache and rebuild embeddings",
    )
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Build/retrieve from a combined FAISS index over PDF_PATHS",
    )
    args = parser.parse_args()

    if args.multi:
        pdf_paths = PDF_PATHS
        if not pdf_paths:
            raise ValueError(
                "PDF_PATHS is empty. Add PDFs to app/config.py before using --multi."
            )
        for pdf_path in pdf_paths:
            if not os.path.isfile(pdf_path):
                raise FileNotFoundError(f"PDF not found: {pdf_path}")
        current_metadata = build_multi_faiss_cache_metadata(pdf_paths)
        index_path, metadata_path = get_multi_faiss_cache_paths(current_metadata)
    else:
        pdf_paths = [FILEPATH]
        doc_id = os.path.splitext(os.path.basename(FILEPATH))[0]
        current_metadata = build_faiss_cache_metadata(FILEPATH, doc_id)
        index_path, metadata_path = get_faiss_cache_paths(doc_id, current_metadata)

    if not args.rebuild_index and is_faiss_cache_valid(
        metadata_path, index_path, current_metadata
    ):
        logger.info("Using cached FAISS index: %s", index_path)
        index, items = load_faiss_cache(index_path, metadata_path)
        logger.info("Loaded %s items (index.ntotal=%s)", len(items), index.ntotal)
        if args.multi:
            _log_item_counts_by_doc_id(items)
            _validate_chunk_ids(items)
    else:
        logger.info("FAISS cache missing or invalid; rebuilding embeddings...")
        if args.multi:
            all_items = []
            for pdf_path in pdf_paths:
                logger.info("Processing PDF: %s", pdf_path)
                doc_items = process_pdf(pdf_path=pdf_path, force=args.force)
                logger.info("Extracted %s items from %s", len(doc_items), pdf_path)
                all_items.extend(doc_items)
            items = all_items
            _log_item_counts_by_doc_id(items)
            _validate_chunk_ids(items)
        else:
            items = process_pdf(pdf_path=FILEPATH, force=args.force)

        items = _embed_items(items)
        index = build_index(items)
        save_faiss_cache(index, items, metadata_path, index_path, current_metadata)
        logger.info("Saved FAISS cache: %s", index_path)

    queries = [
        "What is the difference between droop speed control and isochronous speed control, and when is each mode typically used?",
        "How is overspeed protection implemented, and what are the two independent electronic circuits responsible for shutting off fuel during an overspeed event?",
        "According to the turbine speed pulse rate definition, what speed pickup output voltage and pickup gap are expected at full speed and at crank speed?",
        "Explain how the Constant Settable Droop Speed/Load Control system works, including the roles of the inner speed control loop and the outer megawatt control loop.",
        "Describe the signal flow shown in the Constant Settable Droop Speed/Load control diagram, including the roles of TNR, TNH, DWDROOP, DWATT, FSRN, and FSR.",
        "What is the purpose of the Constant Settable Droop Anti-Windup logic, and under what operating conditions does it become active?",
        "During a Trip to Island mode event, what sequence of events occurs when the tie-line breaker opens, and how does the gas turbine respond to maintain operation?",
    ]

    for query in queries:
        print("\n" + "=" * 80)
        print("QUESTION:")
        print(query)

        matched_items = retrieve(query, index, items)
        assign_citation_ids(matched_items)

        print("\nRETRIEVED SOURCES:\n")
        for item in matched_items:
            print(format_retrieved_sources_line(item))

        response = invoke_nova_multimodal(query, matched_items)

        print("\nANSWER:")
        print(response)


if __name__ == "__main__":
    main()
