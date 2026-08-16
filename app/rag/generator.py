import base64
import io

import boto3
from PIL import Image

from app.config import AWS_REGION
from app.rag.prompt_builder import format_source_header

MAX_IMAGE_SIDE = 2000

SYSTEM_PROMPT = """You are a helpful assistant for question answering.
The text context and images are retrieved source material labeled [SOURCE N].
- Answer only using the provided sources.
- Cite supporting sources using [SOURCE X].
- If the answer cannot be supported by the provided sources, say: I do not know based on the provided sources.
- Do not invent citations.
Answer concisely in no more than 5 sentences. Do not include long explanations unless asked."""


def _prepare_image_for_converse(image_base64: str) -> bytes:
    """Decode base64 image and return PNG bytes (JPEG, PNG, etc. in; PNG out).

    Resizes when max side exceeds Bedrock multi-image limit.
    """
    image_bytes = base64.b64decode(image_base64)
    with Image.open(io.BytesIO(image_bytes)) as img:
        if max(img.size) > MAX_IMAGE_SIDE:
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                img = img.convert("RGBA")
            else:
                img = img.convert("RGB")
            img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
        elif img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            img = img.convert("RGBA")
        else:
            img = img.convert("RGB")

        out = io.BytesIO()
        img.save(out, format="PNG")
        return out.getvalue()


def invoke_nova_multimodal(prompt, matched_items):
    """
    Invoke the Amazon Nova model via Bedrock Converse API.
    Expects citation_id on each item (assigned by caller before invoke).
    """
    system_msg = [{"text": SYSTEM_PROMPT}]

    message_content = []

    for item in matched_items:
        header = format_source_header(item)
        if item["type"] == "text" or item["type"] == "table":
            message_content.append(
                {"text": f"{header}\n\n[CONTENT]\n{item['text']}"}
            )
        else:
            message_content.append({"text": header})
            prepared_bytes = _prepare_image_for_converse(item["image"])
            message_content.append(
                {
                    "image": {
                        "format": "png",
                        "source": {"bytes": prepared_bytes},
                    }
                }
            )

    inf_params = {"maxTokens": 800, "temperature": 0.1}

    message_list = [{"role": "user", "content": message_content}]
    message_list.append({"role": "user", "content": [{"text": prompt}]})

    model_id = "global.anthropic.claude-sonnet-4-6"
    client = boto3.client("bedrock-runtime", region_name=AWS_REGION)

    response = client.converse(
        modelId=model_id,
        messages=message_list,
        system=system_msg,
        inferenceConfig=inf_params,
    )

    output_message = response["output"]["message"]
    text_blocks = [block["text"] for block in output_message["content"] if "text" in block]
    return "\n".join(text_blocks)
