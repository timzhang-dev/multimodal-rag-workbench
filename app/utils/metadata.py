def build_item_metadata(
    doc_id: str,
    source_file: str,
    page_num: int,
    item_type: str,
    path: str,
    chunk_index: int,
) -> dict:
    """Build standardized metadata for a retrievable corpus item."""
    chunk_id = f"{doc_id}_{item_type}_{page_num}_{chunk_index}"
    return {
        "doc_id": doc_id,
        "source_file": source_file,
        "page": page_num,
        "type": item_type,
        "path": path,
        "chunk_id": chunk_id,
        "chunk_index": chunk_index,
    }
