import os


def assign_citation_ids(matched_items):
    """Assign SOURCE N IDs to retrieved items (1-based)."""
    for i, item in enumerate(matched_items, start=1):
        item["citation_id"] = f"SOURCE {i}"


def _document_page_type(item):
    metadata = item.get("metadata") or {}
    source_file = metadata.get("source_file", "")
    doc_id = metadata.get("doc_id") or (
        os.path.basename(source_file) if source_file else "unknown"
    )
    document = os.path.basename(source_file) if source_file else "unknown"
    page = metadata.get("page", item.get("page", "?"))
    item_type = metadata.get("type", item.get("type", "?"))
    return doc_id, document, page, item_type


def format_source_header(item) -> str:
    """Source block header for model context."""
    citation_id = item.get("citation_id", "SOURCE ?")
    doc_id, document, page, item_type = _document_page_type(item)
    return (
        f"[{citation_id}]\n"
        f"Document: {doc_id} | {document}\n"
        f"Page: {page}\n"
        f"Type: {item_type}"
    )


def format_retrieved_sources_line(item) -> str:
    """One-line source summary for console output."""
    citation_id = item.get("citation_id", "SOURCE ?")
    doc_id, document, page, item_type = _document_page_type(item)
    return f"[{citation_id}] {doc_id} | {document} | page {page} | {item_type}"
