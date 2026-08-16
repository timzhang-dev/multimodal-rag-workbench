import base64
import json
import logging
import os
import subprocess

import pymupdf
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from app.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    FILEPATH,
    MINERU_ADD_PAGE_RENDERS,
    MINERU_BACKEND,
    MINERU_EXE,
    MINERU_FORCE_REPARSE,
    MINERU_LANG,
    MINERU_METHOD,
    MINERU_OUTPUT_ROOT,
)
from app.ingestion.page_renderer import process_page_images
from app.utils.file_utils import create_directories
from app.utils.metadata import build_item_metadata

logger = logging.getLogger(__name__)

SKIP_MINERU_TYPES = frozenset({"page_number", "page_footer", "page_header"})
TEXT_MINERU_TYPES = frozenset({"paragraph", "title", "algorithm", "list", "index"})


def _source_meta_path(doc_id: str) -> str:
    return os.path.join(MINERU_OUTPUT_ROOT, doc_id, ".source_meta.json")


def _content_list_path(doc_id: str) -> str:
    return os.path.join(
        MINERU_OUTPUT_ROOT, doc_id, "ocr", f"{doc_id}_content_list_v2.json"
    )


def write_source_meta(pdf_path: str, doc_id: str) -> None:
    stat = os.stat(pdf_path)
    meta = {
        "source_file": pdf_path,
        "pdf_mtime": stat.st_mtime,
        "pdf_size": stat.st_size,
    }
    meta_path = _source_meta_path(doc_id)
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def is_mineru_cache_valid(pdf_path: str, doc_id: str, force: bool) -> bool:
    if force or MINERU_FORCE_REPARSE:
        return False

    json_path = _content_list_path(doc_id)
    if not os.path.isfile(json_path):
        return False

    meta_path = _source_meta_path(doc_id)
    if not os.path.isfile(meta_path):
        write_source_meta(pdf_path, doc_id)
        return True

    stat = os.stat(pdf_path)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    return (
        meta.get("pdf_mtime") == stat.st_mtime
        and meta.get("pdf_size") == stat.st_size
    )


def run_mineru(pdf_path: str, output_root: str) -> None:
    if not os.path.isfile(MINERU_EXE):
        raise FileNotFoundError(
            f"MinerU executable not found at {MINERU_EXE}. "
            "Install MinerU in .venv-mineru (Python 3.12)."
        )

    pdf_path = os.path.abspath(pdf_path)
    output_root = os.path.abspath(output_root)

    result = subprocess.run(
        [
            MINERU_EXE,
            "-p",
            pdf_path,
            "-o",
            output_root,
            "-b",
            MINERU_BACKEND,
            "-m",
            MINERU_METHOD,
            "-l",
            MINERU_LANG,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"MinerU CLI failed (exit {result.returncode}).\n"
            f"stderr: {result.stderr}\nstdout: {result.stdout}"
        )


def flatten_table_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for tr in soup.find_all("tr"):
        cells = [cell.get_text(strip=True) for cell in tr.find_all(["td", "th"])]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _extract_text_nodes(nodes) -> str:
    if not nodes:
        return ""

    parts = []
    for node in nodes:
        if isinstance(node, str):
            text = node.strip()
            if text:
                parts.append(text)
            continue
        if not isinstance(node, dict):
            continue
        if node.get("type") == "text":
            text = str(node.get("content", "")).strip()
            if text:
                parts.append(text)
        else:
            nested = _extract_text_nodes([node])
            if nested:
                parts.append(nested)
    return " ".join(parts)


def _extract_block_text(block: dict) -> str:
    block_type = block.get("type")
    content = block.get("content") or {}

    if block_type == "paragraph":
        return _extract_text_nodes(content.get("paragraph_content", []))

    if block_type == "title":
        return _extract_text_nodes(content.get("title_content", []))

    if block_type == "algorithm":
        return _extract_text_nodes(content.get("algorithm_content", []))

    if block_type in ("list", "index"):
        list_items = content.get("list_items", [])
        lines = []
        prefix = "- " if content.get("attribute") == "unordered" else ""
        for item in list_items:
            line = _extract_text_nodes(item.get("item_content", []))
            if line:
                lines.append(f"{prefix}{line}")
        return "\n".join(lines)

    return ""


def _extract_image_caption(content: dict) -> str:
    caption_parts = []
    for key in ("image_caption", "image_footnote"):
        caption_parts.append(_extract_text_nodes(content.get(key, [])))
    return " ".join(part for part in caption_parts if part).strip()


def parse_content_list_v2(
    json_path: str,
    ocr_dir: str,
    doc_id: str,
    source_file: str,
    base_dir: str,
    text_splitter: RecursiveCharacterTextSplitter,
) -> list:
    with open(json_path, encoding="utf-8") as f:
        pages = json.load(f)

    basename = os.path.basename(source_file)
    items = []

    for page_num, blocks in enumerate(pages):
        text_chunk_index = 0
        table_index = 0
        image_index = 0

        for block in blocks:
            block_type = block.get("type")

            if block_type in SKIP_MINERU_TYPES:
                continue

            if block_type in TEXT_MINERU_TYPES:
                text = _extract_block_text(block)
                if not text or not text.strip():
                    continue

                chunks = text_splitter.split_text(text)
                for chunk in chunks:
                    if not chunk.strip():
                        continue
                    text_path = os.path.join(
                        base_dir,
                        "text",
                        f"{basename}_text_{page_num}_{text_chunk_index}.txt",
                    )
                    with open(text_path, "w", encoding="utf-8") as f:
                        f.write(chunk)

                    metadata = build_item_metadata(
                        doc_id,
                        source_file,
                        page_num,
                        "text",
                        text_path,
                        text_chunk_index,
                    )
                    metadata["mineru_type"] = block_type

                    items.append(
                        {
                            "page": page_num,
                            "type": "text",
                            "text": chunk,
                            "path": text_path,
                            "metadata": metadata,
                        }
                    )
                    text_chunk_index += 1
                continue

            if block_type == "table":
                content = block.get("content") or {}
                html = content.get("html", "")
                flattened = flatten_table_html(html)
                if not flattened.strip():
                    continue

                txt_path = os.path.join(
                    base_dir,
                    "tables",
                    f"{basename}_table_{page_num}_{table_index}.txt",
                )
                html_path = os.path.join(
                    base_dir,
                    "tables",
                    f"{basename}_table_{page_num}_{table_index}.html",
                )
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(flattened)
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(html)

                metadata = build_item_metadata(
                    doc_id,
                    source_file,
                    page_num,
                    "table",
                    txt_path,
                    table_index,
                )
                metadata["mineru_type"] = "table"
                metadata["table_html"] = html

                items.append(
                    {
                        "page": page_num,
                        "type": "table",
                        "text": flattened,
                        "path": txt_path,
                        "metadata": metadata,
                    }
                )
                table_index += 1
                continue

            if block_type == "image":
                content = block.get("content") or {}
                image_source = content.get("image_source") or {}
                rel_path = image_source.get("path", "")
                if not rel_path:
                    continue

                src_image_path = os.path.normpath(
                    os.path.join(ocr_dir, rel_path.replace("/", os.sep))
                )
                if not os.path.isfile(src_image_path):
                    logger.warning(
                        "MinerU image not found: %s (page %s)", src_image_path, page_num
                    )
                    continue

                image_path = os.path.join(
                    base_dir,
                    "images",
                    f"{basename}_image_{page_num}_{image_index}.jpg",
                )
                with open(src_image_path, "rb") as src_f:
                    image_bytes = src_f.read()
                with open(image_path, "wb") as dst_f:
                    dst_f.write(image_bytes)

                encoded_image = base64.b64encode(image_bytes).decode("utf-8")
                caption = _extract_image_caption(content)

                metadata = build_item_metadata(
                    doc_id,
                    source_file,
                    page_num,
                    "image",
                    image_path,
                    image_index,
                )
                metadata["mineru_type"] = "image"
                if caption:
                    metadata["caption"] = caption

                items.append(
                    {
                        "page": page_num,
                        "type": "image",
                        "image": encoded_image,
                        "path": image_path,
                        "metadata": metadata,
                    }
                )
                image_index += 1
                continue

            logger.warning(
                "Skipping unknown MinerU block type '%s' on page %s",
                block_type,
                page_num,
            )

    return items


def process_pdf_with_mineru(pdf_path=FILEPATH, force: bool = False) -> list:
    doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
    source_file = pdf_path
    base_dir = "data"
    ocr_dir = os.path.join(MINERU_OUTPUT_ROOT, doc_id, "ocr")
    json_path = _content_list_path(doc_id)

    create_directories(base_dir)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    if is_mineru_cache_valid(pdf_path, doc_id, force):
        logger.info("Using cached MinerU output for %s", doc_id)
    else:
        logger.info("Running MinerU CLI for %s", doc_id)
        run_mineru(pdf_path, MINERU_OUTPUT_ROOT)
        write_source_meta(pdf_path, doc_id)

    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"MinerU did not produce {json_path}")

    items = parse_content_list_v2(
        json_path, ocr_dir, doc_id, source_file, base_dir, text_splitter
    )

    if MINERU_ADD_PAGE_RENDERS:
        doc = pymupdf.open(pdf_path)
        for page_num in tqdm(range(len(doc)), desc="Rendering page images"):
            process_page_images(
                doc[page_num], page_num, base_dir, items, doc_id, source_file
            )

    return items
