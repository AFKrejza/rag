"""Extract text from PDFs page by page."""

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Page:
    source: str  # filename, e.g. "ddcg.pdf"
    page: int  # 1-based physical page number (as shown in a PDF viewer)
    text: str


def clean_text(text: str) -> str:
    # Expand typographic ligatures (e.g. "eﬀect" -> "effect") and other compatibility characters.
    text = unicodedata.normalize("NFKC", text)
    # Drop stray control characters left by fonts with broken encodings.
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    # Re-join words hyphenated across line breaks: "regis-\nter" -> "register".
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Single newlines inside paragraphs become spaces; keep blank-line breaks.
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def is_navigation_page(text: str) -> bool:
    """Table-of-contents and back-of-book index pages.

    They match many queries by their headings/keywords but contain no content, so
    they are skipped. TOC pages are full of dot leaders ("Introduction . . . 21");
    index pages are full of "term, 104, 124" entries.
    """
    dot_leaders = text.count(". .")
    index_entries = len(re.findall(r"[a-z], \d+", text))
    return dot_leaders > 20 or index_entries > 30


def load_pdf(path: Path) -> list[Page]:
    reader = PdfReader(path)
    pages = []
    for i, pdf_page in enumerate(reader.pages, start=1):
        text = clean_text(pdf_page.extract_text() or "")
        if text and not is_navigation_page(text):
            pages.append(Page(source=path.name, page=i, text=text))
    return pages


def load_folder(folder: Path) -> dict[str, list[Page]]:
    return {p.name: load_pdf(p) for p in sorted(folder.glob("*.pdf"))}
