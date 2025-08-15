import os
import yaml
from pathlib import Path
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from .vectorstore import load_vectorstore

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
_cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def build_rag_chain() -> RetrievalQA:

    vs = load_vectorstore(
        path=_cfg["vectorstore"]["path"], model_name=_cfg["vectorstore"]["model"]
    )
    retriever = vs.as_retriever(search_kwargs={"k": _cfg["retrieval"]["k"]})
    api_key = os.getenv(_cfg["llm"]["api_key_env"])
    if not api_key:
        raise EnvironmentError(
            f"Не задана переменная окружения {_cfg['llm']['api_key_env']}"
        )
    llm = ChatGroq(model=_cfg["llm"]["model"], groq_api_key=api_key)
    return RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, return_source_documents=True
    )
