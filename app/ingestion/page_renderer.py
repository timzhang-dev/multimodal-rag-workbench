import os
import base64

from app.utils.metadata import build_item_metadata


def process_page_images(page, page_num, base_dir, items, doc_id, source_file):
    pix = page.get_pixmap()
    page_path = os.path.join(base_dir, f"page_images/page_{page_num:03d}.png")
    pix.save(page_path)
    with open(page_path, "rb") as f:
        page_image = base64.b64encode(f.read()).decode("utf8")
    # Metadata attached here for retrieval corpus
    items.append(
        {
            "page": page_num,
            "type": "page",
            "path": page_path,
            "image": page_image,
            "metadata": build_item_metadata(
                doc_id, source_file, page_num, "page", page_path, 0
            ),
        }
    )
