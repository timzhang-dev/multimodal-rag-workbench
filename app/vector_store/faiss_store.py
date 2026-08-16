import faiss

import numpy as np

from app.config import EMBEDDING_VECTOR_DIMENSION





def build_index(items, dimension=EMBEDDING_VECTOR_DIMENSION):

    """Create a FAISS index and add embeddings from items."""

    all_embeddings = np.array([item["embedding"] for item in items])



    index = faiss.IndexFlatL2(dimension)

    index.reset()

    index.add(np.array(all_embeddings, dtype=np.float32))



    return index


