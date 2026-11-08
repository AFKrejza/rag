"""Ask questions about the indexed PDFs.

Usage:
    python query.py "How do I start implementing a register allocator?"
    python query.py                      # interactive mode
    python query.py "..." --show-sources # also print retrieved chunks
"""

import argparse
import textwrap

import anthropic

from rag import config
from rag.generator import generate
from rag.retriever import retrieve
from rag.store import get_collection


def answer(collection, client, question: str, k: int, show_sources: bool) -> None:
    chunks = retrieve(collection, question, k=k)
    result = generate(client, question, chunks)
    print(f"\n{result.text}\n")
    if show_sources:
        print("Retrieved chunks:" if chunks else f"No chunks within distance {config.MAX_DISTANCE}.")
        for c in chunks:
            snippet = textwrap.shorten(c.text, width=160, placeholder=" ...")
            print(f"  {c.citation}  distance={c.distance:.3f}\n    {snippet}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="?", help="question to ask; omit for interactive mode")
    parser.add_argument("--k", type=int, default=config.TOP_K, help="number of chunks to retrieve")
    parser.add_argument("--show-sources", action="store_true", help="print retrieved chunks")
    args = parser.parse_args()

    collection = get_collection()
    if collection.count() == 0:
        raise SystemExit("Index is empty. Run `python ingest.py` first.")
    client = anthropic.Anthropic()

    if args.question:
        answer(collection, client, args.question, args.k, args.show_sources)
        return

    print("Ask a question (empty line or Ctrl+C to quit).")
    try:
        while question := input("> ").strip():
            answer(collection, client, question, args.k, args.show_sources)
    except (KeyboardInterrupt, EOFError):
        pass


if __name__ == "__main__":
    main()
