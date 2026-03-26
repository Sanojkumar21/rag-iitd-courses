import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

VECTORSTORE_DIR = "vectorstore"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def build_vectorstore(chunks: dict) -> tuple[Chroma, list[Document]]:
    search_chunks = chunks["search_chunks"]

    embeddings = get_embeddings()
    print(f"building vectorstore with {len(search_chunks)} chunks...")

    vs = Chroma.from_documents(
        documents=search_chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_DIR
    )
    print("done — vectorstore saved to disk")

    # returning chunks alongside vs so chain.py can build BM25 on the same data
    return vs, search_chunks


def load_vectorstore() -> Chroma:
    embeddings = get_embeddings()
    vs = Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embeddings
    )
    print("vectorstore loaded from disk")
    return vs


def vectorstore_exists() -> bool:
    return os.path.exists(VECTORSTORE_DIR) and len(os.listdir(VECTORSTORE_DIR)) > 0