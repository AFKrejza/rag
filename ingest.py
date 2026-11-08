"""Build the vector index: PDF -> text -> chunks -> embeddings -> Chroma.

Usage:
    python ingest.py [--pdf-dir data/pdfs] [--reset]
"""

import argparse
from pathlib import Path

from rag import config
from rag.chunker import chunk_pages
from rag.pdf_loader import load_folder
from rag.store import get_client, get_collection, reset_collection

BATCH_SIZE = 128


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf-dir", type=Path, default=config.PDF_DIR)
    parser.add_argument("--reset", action="store_true", help="delete the existing index first")
    args = parser.parse_args()

    if not args.pdf_dir.is_dir():
        raise SystemExit(f"PDF folder not found: {args.pdf_dir}")

    client = get_client()
    collection = reset_collection(client) if args.reset else get_collection(client)

    docs = load_folder(args.pdf_dir)
    if not docs:
        raise SystemExit(f"No PDFs in {args.pdf_dir}")

    for filename, pages in docs.items():
        chunks = chunk_pages(pages, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            collection.upsert(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[{"source": c.source, "page": c.page, "chunk_index": c.chunk_index} for c in batch],
            )
        print(f"{filename}: {len(pages)} pages with text, {len(chunks)} chunks")
        if not pages:
            print(f"  warning: no extractable text in {filename} (scanned PDF? needs OCR)")

    print(f"Index now holds {collection.count()} chunks in {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()
