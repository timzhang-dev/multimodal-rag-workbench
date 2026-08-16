import base64
import json

import boto3
from botocore.exceptions import ClientError

from app.config import AWS_REGION, COHERE_EMBED_MODEL_ID, EMBEDDING_VECTOR_DIMENSION


def _to_data_uri(image_base64: str) -> str:
    raw = base64.b64decode(image_base64)
    if raw.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    elif raw.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    else:
        mime = "image/jpeg"
    return f"data:{mime};base64,{image_base64}"


def _parse_embeddings(result: dict) -> list | None:
    embeddings = result.get("embeddings")
    if embeddings is None:
        return None
    if isinstance(embeddings, dict):
        float_embeddings = embeddings.get("float")
        if float_embeddings:
            return float_embeddings[0]
        return None
    if isinstance(embeddings, list) and embeddings:
        first = embeddings[0]
        if isinstance(first, list):
            return first
    return None


def generate_multimodal_embeddings(
    prompt=None,
    image=None,
    output_embedding_length=EMBEDDING_VECTOR_DIMENSION,
    input_type="search_document",
):
    if not prompt and not image:
        raise ValueError("Please provide either a text prompt or base64 image as input")
    if prompt and image:
        raise ValueError("Cohere embed v4 requires text or image per call, not both")

    client = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)

    body = {
        "input_type": input_type,
        "embedding_types": ["float"],
        "output_dimension": output_embedding_length,
    }

    if prompt:
        body["texts"] = [prompt]
    else:
        body["images"] = [_to_data_uri(image)]

    try:
        response = client.invoke_model(
            modelId=COHERE_EMBED_MODEL_ID,
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json",
        )
        result = json.loads(response.get("body").read())
        return _parse_embeddings(result)

    except ClientError as err:
        print(
            f"Couldn't invoke Cohere embedding model. "
            f"Error: {err.response['Error']['Message']}"
        )
        return None
