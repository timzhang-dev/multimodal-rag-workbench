import boto3
import json
from app.config import AWS_REGION, EMBEDDING_VECTOR_DIMENSION
from botocore.exceptions import ClientError

# Generating Multimodal Embeddings using Amazon Titan Multimodal Embeddings model
def generate_multimodal_embeddings(prompt=None, image=None, output_embedding_length=EMBEDDING_VECTOR_DIMENSION):
    """
    Invoke the Amazon Titan Multimodal Embeddings model using Amazon Bedrock runtime.

    Args:
        prompt (str): The text prompt to provide to the model.
        image (str): A base64-encoded image data.
    Returns:
        str: The model's response embedding.
    """
    if not prompt and not image:
        raise ValueError("Please provide either a text prompt, base64 image, or both as input")
    
    # Initialize the Amazon Bedrock runtime client
    client = boto3.client(
        service_name="bedrock-runtime",
        region_name=AWS_REGION,
    )
    model_id = "amazon.titan-embed-image-v1"
    
    body = {"embeddingConfig": {"outputEmbeddingLength": output_embedding_length}}
    
    if prompt:
        body["inputText"] = prompt
    if image:
        body["inputImage"] = image

    try:
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json"
        )

        # Process and return the response
        result = json.loads(response.get("body").read())
        return result.get("embedding")

    except ClientError as err:
        print(f"Couldn't invoke Titan embedding model. Error: {err.response['Error']['Message']}")
        return None