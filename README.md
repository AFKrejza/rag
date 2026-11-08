# Compiler Design RAG

Ask practical compiler-construction questions ("how do I start writing a register allocator?") and get answers grounded in a small library of compiler papers. Every claim is cited as `[filename, p. N]`.

**Pipeline:** PDF → text (`pypdf`) → per-page chunks with overlap → embeddings (`sentence-transformers/all-MiniLM-L6-v2`) → Chroma vector DB → top-k semantic search → Claude answers from the retrieved chunks only.

## Setup

You need [uv](https://docs.astral.sh/uv/getting-started/installation/) and an [Anthropic API key](https://console.anthropic.com/). uv downloads the right Python (3.12) by itself, so you don't need a matching Python installed.

```bash
# 1. Install uv (once)
#   Windows:     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
#   macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and install dependencies
git clone <this repo> && cd rag
uv sync

# 3. Add your API key
cp .env.example .env        # then edit .env and set ANTHROPIC_API_KEY=...

# 4. Build the index (first run downloads the ~90 MB embedding model)
uv run ingest.py
```

<details>
<summary>Without uv (plain venv)</summary>

Use a regular CPython 3.11 or 3.12 from python.org. MSYS2, Cygwin and Conda Pythons can't install the prebuilt `torch`/`onnxruntime` wheels.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python ingest.py
```
</details>

## Usage

```bash
uv run query.py "What is a control destination?"            # one question
uv run query.py "..." --show-sources                       # also show retrieved chunks + distances
uv run query.py                                            # interactive mode
uv run streamlit run app.py                                # web UI at http://localhost:8501
```

To add documents, drop PDFs into `data/pdfs/` and run `uv run ingest.py` again. Re-ingesting is idempotent (chunk IDs are `filename:page:index`). Use `--reset` to rebuild from scratch after removing files.

## Corpus

Four freely available compiler texts, chosen because they parse cleanly to text:

| File | Document | Source |
|------|----------|--------|
| `basics-of-compiler-construction.pdf` | T. Æ. Mogensen. *Basics of Compiler Design*, 2007 edition. Introductory textbook: what a compiler is, the phases, lexing, parsing, type checking, intermediate code, register allocation. | Author's site: https://hjemmesider.diku.dk/~torbenm/Basics/ (licence: "Permission to copy and print for personal use is granted") |
| `wirth-compiler-construction.pdf` | N. Wirth. *Compiler Construction*, Part 1 (chapters 1–8). Revised edition, 2005. Compiler phases, grammars, recursive-descent parsing. | Author's site: https://people.inf.ethz.ch/wirth/CompilerConstruction/CompilerConstruction1.pdf |
| `ghuloum2006.pdf` | A. Ghuloum. *An Incremental Approach to Compiler Construction*. Scheme and Functional Programming Workshop, 2006. Building a compiler in small, tested steps. | http://scheme2006.cs.uchicago.edu/11-ghuloum.pdf |
| `ddcg.pdf` | R. K. Dybvig, R. Hieb, T. Butler. *Destination-Driven Code Generation*. Indiana University TR #302, 1990. Code generation from ASTs. | https://bernsteinbear.com/assets/img/ddcg.pdf |

## Sample questions and expected citations

Page numbers are **physical PDF pages** (what your PDF viewer shows), not the page numbers printed in the document. For example, the DDCG report has a cover page, so its printed page 7 is PDF page 8.

All citations below come from actual runs. Answers usually cite the main page(s) listed, sometimes along with a few related pages.

| Question | Expected citation(s) |
|----------|----------------------|
| What is a compiler? | `[basics-of-compiler-construction.pdf, p. 13]` |
| What are the phases of a compiler? | `[basics-of-compiler-construction.pdf, p. 14]`, `[basics-of-compiler-construction.pdf, p. 15]` |
| What is the difference between a compiler and an interpreter? | `[basics-of-compiler-construction.pdf, p. 15]`, `[basics-of-compiler-construction.pdf, p. 16]` |
| What does lexical analysis do? | `[basics-of-compiler-construction.pdf, p. 14]`, `[basics-of-compiler-construction.pdf, p. 21]`, `[wirth-compiler-construction.pdf, p. 6]` |
| What is a context-free grammar? | `[basics-of-compiler-construction.pdf, p. 64]`, `[wirth-compiler-construction.pdf, p. 14]` |
| What is recursive descent parsing? | `[basics-of-compiler-construction.pdf, p. 86]`, `[wirth-compiler-construction.pdf, p. 19]` |
| What is type checking? | `[basics-of-compiler-construction.pdf, p. 123]`, `[basics-of-compiler-construction.pdf, p. 126]` |
| What is register allocation? | `[basics-of-compiler-construction.pdf, p. 177]`, `[basics-of-compiler-construction.pdf, p. 178]` |
| Why build a compiler incrementally? | `[ghuloum2006.pdf, p. 1]`, `[ghuloum2006.pdf, p. 2]` |
| What is a control destination? | `[ddcg.pdf, p. 3]`, `[ddcg.pdf, p. 8]` |
| What is the capital of France? | *I don't know based on the provided documents.* (no chunks pass the distance threshold, so the LLM is never called) |
| How does LLVM's register allocator work? | *I don't know based on the provided documents.* (related chunks are retrieved, but none is about LLVM, so Claude declines) |

## Limitations

- **English only.** The embedding model (`all-MiniLM-L6-v2`) is English-only, so questions in other languages retrieve nothing and get "I don't know". A multilingual model such as `paraphrase-multilingual-MiniLM-L12-v2` (set `RAG_EMBEDDING_MODEL`, then re-ingest) would fix this.
- **Generic words dominate retrieval.** Broad questions containing "compiler" tend to pull in general introductions. For example, "What optimisations does a compiler perform?" gets "I don't know", while the more specific "What optimisations can be done during machine code generation?" retrieves Mogensen's section on optimisations.
- **Coverage.** The corpus is introductory. Constant folding, dead code elimination and specific production compilers (e.g. LLVM) are not explained in it, so those questions correctly get "I don't know".

## Design notes

- **Per-page chunking.** Each page is split on its own (~1000 chars, ~200 overlap, splitting at paragraphs, then sentences, then words), so every chunk belongs to exactly one page and every citation points to a single page.
- **Neighbouring chunks.** For each of the top-5 hits, the chunks just before and after it on the same page are also passed to the LLM. Lists such as "the phases of a compiler" often straddle a chunk boundary, and the short tail chunk would otherwise rank too low to be retrieved.
- **Contents and index pages are skipped.** Pages full of dot leaders (`Introduction . . . . 21`) or index entries (`abstract syntax, 104, 124`) match many questions by their keywords but contain no content.
- **"I don't know" in two layers.**
  1. Retrieval drops chunks whose cosine distance is above `RAG_MAX_DISTANCE` (default 0.75). If nothing is left, the app answers "I don't know" without calling the LLM. The best match for an on-topic question is usually 0.2–0.6 away; for off-topic questions, even the best match is 0.8 or more.
  2. If some chunks pass but don't actually answer the question, the system prompt tells Claude to reply with the exact "I don't know" sentence.
- **Grounding.** Each excerpt in the prompt carries its `[filename, p. N]` label, and Claude is told to cite only those labels and to use no outside knowledge.
- **Config.** Everything in `rag/config.py` can be overridden with environment variables or in `.env`: model, top-k, chunk size, threshold.

## Project layout

```
data/pdfs/       sample PDFs
rag/             library code: pdf_loader, chunker, store, retriever, generator, config
ingest.py        build the index
query.py         CLI
app.py           Streamlit UI
```
