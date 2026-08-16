import os

import pymupdf
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_OVERLAP, CHUNK_SIZE, FILEPATH, PARSER
from app.utils.file_utils import create_directories
from app.ingestion.table_extractor import process_tables
from app.ingestion.text_chunker import process_text_chunks
from app.ingestion.image_extractor import process_images
from app.ingestion.page_renderer import process_page_images
from app.ingestion.mineru_processor import process_pdf_with_mineru


def process_pdf_with_pymupdf(pdf_path=FILEPATH):
    doc = pymupdf.open(pdf_path)
    num_pages = len(doc)
    base_dir = "data"

    doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
    source_file = pdf_path

    create_directories(base_dir)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len
    )
    items = []

    for page_num in tqdm(range(num_pages), desc="Processing PDF pages"):
        page = doc[page_num]
        text = page.get_text()
        process_tables(
            doc, page_num, base_dir, items, doc_id, source_file, pdf_path=pdf_path
        )
        process_text_chunks(
            text,
            text_splitter,
            page_num,
            base_dir,
            items,
            doc_id,
            source_file,
            pdf_path=pdf_path,
        )
        process_images(
            doc, page, page_num, base_dir, items, doc_id, source_file, pdf_path=pdf_path
        )
        process_page_images(page, page_num, base_dir, items, doc_id, source_file)

    return items


def process_pdf(pdf_path=FILEPATH, force: bool = False):
    if PARSER == "mineru":
        return process_pdf_with_mineru(pdf_path=pdf_path, force=force)
    return process_pdf_with_pymupdf(pdf_path=pdf_path)
