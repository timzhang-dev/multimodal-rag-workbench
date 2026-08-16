import os
import base64

import pymupdf

from app.config import FILEPATH
from app.utils.metadata import build_item_metadata


def process_images(
    doc, page, page_num, base_dir, items, doc_id, source_file, pdf_path=FILEPATH
):
    images = page.get_images()
    for idx, image in enumerate(images):
        xref = image[0]
        pix = pymupdf.Pixmap(doc, xref)
        image_name = (
            f"{base_dir}/images/{os.path.basename(pdf_path)}_image_{page_num}_{idx}_{xref}.png"
        )
        pix.save(image_name)
        with open(image_name, "rb") as f:
            encoded_image = base64.b64encode(f.read()).decode("utf8")
        # Metadata attached here for retrieval corpus
        items.append(
            {
                "page": page_num,
                "type": "image",
                "path": image_name,
                "image": encoded_image,
                "metadata": build_item_metadata(
                    doc_id, source_file, page_num, "image", image_name, idx
                ),
            }
        )
