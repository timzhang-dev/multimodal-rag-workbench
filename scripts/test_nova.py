import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.generator import invoke_nova_multimodal


def main():
    matched_items = [
        {
            "page": 0,
            "type": "text",
            "text": "The model was trained using the Adam optimizer.",
            "path": "mock_context.txt",
        }
    ]

    response = invoke_nova_multimodal(
        prompt="Which optimizer was used?",
        matched_items=matched_items,
    )

    print("Nova test successful.")
    print("Response:")
    print(response)


if __name__ == "__main__":
    main()