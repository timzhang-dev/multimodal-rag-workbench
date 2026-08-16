import os

import tabula

from app.config import FILEPATH
from app.utils.metadata import build_item_metadata


def process_tables(
    doc, page_num, base_dir, items, doc_id, source_file, pdf_path=FILEPATH
):
    try:
        tables = tabula.read_pdf(pdf_path, pages=page_num + 1, multiple_tables=True)
        if not tables:
            return
        for table_idx, table in enumerate(tables):
            table_text = "\n".join(
                [" | ".join(map(str, row)) for row in table.values]
            )
            table_file_name = (
                f"{base_dir}/tables/{os.path.basename(pdf_path)}_table_{page_num}_{table_idx}.txt"
            )
            with open(table_file_name, "w", encoding="utf-8") as f:
                f.write(table_text)
            # Metadata attached here for retrieval corpus
            items.append(
                {
                    "page": page_num,
                    "type": "table",
                    "text": table_text,
                    "path": table_file_name,
                    "metadata": build_item_metadata(
                        doc_id,
                        source_file,
                        page_num,
                        "table",
                        table_file_name,
                        table_idx,
                    ),
                }
            )
    except Exception as e:
        print(f"Error extracting tables from page {page_num}: {str(e)}")
