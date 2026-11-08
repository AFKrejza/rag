"""Answer generation with Claude, grounded in retrieved chunks."""

from dataclasses import dataclass, field

import anthropic

from . import config
from .retriever import RetrievedChunk

SYSTEM_PROMPT = f"""You answer questions about compiler design using ONLY the document excerpts provided in the user message.

Rules:
- Base every statement on the excerpts. Do not use outside knowledge, even if you know the answer.
- Cite the excerpt(s) supporting each claim inline, using exactly the label shown above that excerpt, e.g. [ddcg.pdf, p. 9]. Use one bracket per source; for two sources write [a.pdf, p. 1] [b.pdf, p. 2], never combine them in one bracket. Never invent a filename or page.
- If the excerpts do not contain enough information to answer, reply with exactly this sentence and nothing else: {config.IDK_ANSWER}
- If the excerpts answer only part of the question, answer that part with citations and say what the documents do not cover.
- Be concise and practical: a short explanation, then key steps or points if useful."""


@dataclass
class Answer:
    text: str
    sources: list[RetrievedChunk] = field(default_factory=list)


def build_user_message(question: str, chunks: list[RetrievedChunk]) -> str:
    excerpts = "\n\n".join(f"<excerpt label=\"{c.citation}\">\n{c.text}\n</excerpt>" for c in chunks)
    return f"<excerpts>\n{excerpts}\n</excerpts>\n\nQuestion: {question}"


def generate(client: anthropic.Anthropic, question: str, chunks: list[RetrievedChunk]) -> Answer:
    if not chunks:
        return Answer(text=config.IDK_ANSWER)

    response = client.beta.messages.create(
        model=config.LLM_MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_message(question, chunks)}],
        output_config={"effort": config.LLM_EFFORT},
        # On a safety-classifier refusal, the API re-runs the request on a fallback model.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )

    if response.stop_reason == "refusal":
        return Answer(text="The model declined to answer this question.", sources=chunks)

    text = "".join(block.text for block in response.content if block.type == "text").strip()
    return Answer(text=text or config.IDK_ANSWER, sources=chunks)
