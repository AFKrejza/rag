"""Chroma vector store with a sentence-transformers embedding function."""

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from . import config


def get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def get_collection(client: chromadb.ClientAPI | None = None):
    client = client or get_client()
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        embedding_function=SentenceTransformerEmbeddingFunction(model_name=config.EMBEDDING_MODEL),
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection(client: chromadb.ClientAPI | None = None):
    client = client or get_client()
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass  # collection did not exist yet
    return get_collection(client)
