from pathlib import Path
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embeddings(model_name: str) -> HuggingFaceEmbeddings:

    return HuggingFaceEmbeddings(model_name=model_name)


def build_vectorstore(documents: List, model_name: str) -> FAISS:

    embeddings = get_embeddings(model_name)
    return FAISS.from_documents(documents, embeddings)


def load_vectorstore(path: str, model_name: str) -> FAISS:

    index_dir = Path(path)
    if not index_dir.exists():
        raise FileNotFoundError(f"FAISS-индекс не найден по пути: {path}")
    embeddings = get_embeddings(model_name)
    return FAISS.load_local(
        folder_path=str(index_dir),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )
