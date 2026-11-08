"""Minimal Streamlit UI. Run with: streamlit run app.py"""

import anthropic
import streamlit as st

from rag import config
from rag.generator import generate
from rag.retriever import retrieve
from rag.store import get_collection


@st.cache_resource
def load_collection():
    return get_collection()


@st.cache_resource
def load_client():
    return anthropic.Anthropic()


st.set_page_config(page_title="Compiler Design RAG", page_icon="📚")
st.title("Compiler Design RAG")
st.caption("Answers are grounded in the PDFs in data/pdfs and cite [filename, p. N].")

collection = load_collection()
if collection.count() == 0:
    st.error("Index is empty. Run `python ingest.py` first.")
    st.stop()

k = st.sidebar.slider("Chunks to retrieve (top-k)", 1, 15, config.TOP_K)
st.sidebar.write(f"Indexed chunks: {collection.count()}")

with st.form("ask"):
    question = st.text_input("Question", placeholder="How do I start implementing a register allocator?")
    submitted = st.form_submit_button("Ask")

if submitted and question.strip():
    with st.spinner("Searching and generating..."):
        chunks = retrieve(collection, question, k=k)
        result = generate(load_client(), question, chunks)
    st.markdown(result.text)

    st.subheader("Retrieved sources")
    if not chunks:
        st.write(f"No chunks within cosine distance {config.MAX_DISTANCE}.")
    for c in chunks:
        with st.expander(f"{c.citation} (distance {c.distance:.3f})"):
            st.write(c.text)
