import os

BASE_DIR = "data"
PDF_FILENAME = "365a2992-240-270.pdf"
FILEPATH = os.path.join(BASE_DIR, "raw", PDF_FILENAME)

PDF_PATHS = [
    os.path.join(BASE_DIR, "raw", "turbine_manual_1.pdf"),
    os.path.join(BASE_DIR, "raw", "GE7FA.pdf"),
    os.path.join(BASE_DIR, "raw", "365a2992-240-270.pdf"),
]

PARSER = "mineru"  # "pymupdf" | "mineru"

MINERU_EXE = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", ".venv-mineru", "Scripts", "mineru.exe")
)
MINERU_OUTPUT_ROOT = os.path.join(BASE_DIR, "mineru")
MINERU_BACKEND = "pipeline"
MINERU_METHOD = "ocr"
MINERU_LANG = "en"

MINERU_FORCE_REPARSE = False
MINERU_ADD_PAGE_RENDERS = True

CHUNK_SIZE = 700
CHUNK_OVERLAP = 200

AWS_REGION = "us-east-1"

EMBEDDING_PROVIDER = "cohere"  # "titan" | "cohere"
COHERE_EMBED_MODEL_ID = "cohere.embed-v4:0"

EMBEDDING_VECTOR_DIMENSION = 1024
TOP_K = 15