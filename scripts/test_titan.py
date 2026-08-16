import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import EMBEDDING_VECTOR_DIMENSION
from app.embeddings.titan_embedder import generate_multimodal_embeddings


def main():
    embedding = generate_multimodal_embeddings(
        prompt="This is a test embedding.",
        output_embedding_length=EMBEDDING_VECTOR_DIMENSION,
    )

    print("Titan embedding test successful.")
    print("Embedding type:", type(embedding))
    print("Embedding length:", len(embedding))
    print("First 5 values:", embedding[:5])


if __name__ == "__main__":
    main()