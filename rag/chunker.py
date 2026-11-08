"""Split page text into overlapping chunks.

Chunking is done per page so every chunk maps to exactly one page number,
which keeps citations precise.
"""

from dataclasses import dataclass

from .pdf_loader import Page

SEPARATORS = ["\n\n", ". ", " "]


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int
    chunk_index: int


def _split(text: str, size: int, separators: list[str]) -> list[str]:
    """Recursively split text into pieces no longer than `size`."""
    if len(text) <= size:
        return [text]
    if not separators:
        return [text[i : i + size] for i in range(0, len(text), size)]
    sep, rest = separators[0], separators[1:]
    parts = text.split(sep)
    pieces = []
    for i, part in enumerate(parts):
        # Keep the separator attached so sentences stay readable.
        piece = part + sep if i < len(parts) - 1 else part
        if len(piece) > size:
            pieces.extend(_split(piece, size, rest))
        elif piece:
            pieces.append(piece)
    return pieces


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Greedily merge small pieces into chunks of ~`size` chars with ~`overlap` carry-over."""
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for piece in _split(text, size, SEPARATORS):
        if length + len(piece) > size and current:
            chunks.append("".join(current).strip())
            # Drop pieces from the front until only ~overlap chars remain.
            while current and length > overlap:
                length -= len(current.pop(0))
        current.append(piece)
        length += len(piece)
    if current:
        chunks.append("".join(current).strip())
    return [c for c in chunks if c]


def chunk_pages(pages: list[Page], size: int, overlap: int) -> list[Chunk]:
    chunks = []
    for page in pages:
        for i, text in enumerate(split_text(page.text, size, overlap)):
            chunks.append(
                Chunk(
                    id=f"{page.source}:{page.page}:{i}",
                    text=text,
                    source=page.source,
                    page=page.page,
                    chunk_index=i,
                )
            )
    return chunks
