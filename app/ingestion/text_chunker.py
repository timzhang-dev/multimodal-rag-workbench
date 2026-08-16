import os

from app.config import FILEPATH
from app.utils.metadata import build_item_metadata


def process_text_chunks(
    text,
    text_splitter,
    page_num,
    base_dir,
    items,
    doc_id,
    source_file,
    pdf_path=FILEPATH,
):
    chunks = text_splitter.split_text(text)
    for i, chunk in enumerate(chunks):
        text_file_name = (
            f"{base_dir}/text/{os.path.basename(pdf_path)}_text_{page_num}_{i}.txt"
        )
        with open(text_file_name, "w", encoding="utf-8") as f:
            f.write(chunk)
        # Metadata attached here for retrieval corpus
        items.append(
            {
                "page": page_num,
                "type": "text",
                "text": chunk,
                "path": text_file_name,
                "metadata": build_item_metadata(
                    doc_id, source_file, page_num, "text", text_file_name, i
                ),
            }
        )
