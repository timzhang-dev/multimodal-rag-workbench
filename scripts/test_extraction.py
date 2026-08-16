import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.pdf_processor import process_pdf


def main():
    items = process_pdf()

    print(f"Total items extracted: {len(items)}")

    item_counts = {
        "text": sum(1 for item in items if item["type"] == "text"),
        "table": sum(1 for item in items if item["type"] == "table"),
        "image": sum(1 for item in items if item["type"] == "image"),
        "page": sum(1 for item in items if item["type"] == "page"),
    }

    print("Item counts:")
    print(item_counts)

    for item_type in ["text", "table", "image", "page"]:
        examples = [item for item in items if item["type"] == item_type]

        if examples:
            print(f"\nFirst {item_type} item:")
            print({k: v for k, v in examples[0].items() if k != "image"})


if __name__ == "__main__":
    main()