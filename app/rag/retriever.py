import logging

import numpy as np

from app.embeddings import cohere_embedder
from app.embeddings.titan_embedder import generate_multimodal_embeddings as titan_generate

from app.config import EMBEDDING_PROVIDER, EMBEDDING_VECTOR_DIMENSION, TOP_K

logger = logging.getLogger(__name__)



def retrieve(query, index, items):

    # Generate embeddings for the query

    logger.info("Embedding provider: %s", EMBEDDING_PROVIDER)
    if EMBEDDING_PROVIDER == "cohere":
        query_embedding = cohere_embedder.generate_multimodal_embeddings(
            prompt=query,
            output_embedding_length=EMBEDDING_VECTOR_DIMENSION,
            input_type="search_query",
        )
    else:
        query_embedding = titan_generate(
            prompt=query, output_embedding_length=EMBEDDING_VECTOR_DIMENSION
        )
    if query_embedding:
        logger.info("Query embedding dimension: %s", len(query_embedding))



    # Search for the nearest neighbors in the vector database

    distances, result = index.search(np.array(query_embedding, dtype=np.float32).reshape(1,-1), k=TOP_K)

    

    # Retrieve the matched items

    matched_items = [{k: v for k, v in items[index].items() if k != 'embedding'} for index in result.flatten()]



    return matched_items
