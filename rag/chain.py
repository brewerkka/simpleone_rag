import os
import logging
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from .vectorstore import load_vectorstore
from .config_validator import load_and_validate_config

logger = logging.getLogger(__name__)


def build_rag_chain() -> RetrievalQA:
    """Создает RAG цепочку с валидированной конфигурацией."""
    cfg = load_and_validate_config()

    logger.info("Загрузка векторного хранилища...")
    vs = load_vectorstore(
        path=cfg["vectorstore"]["path"], model_name=cfg["vectorstore"]["model"]
    )
    
    retriever = vs.as_retriever(search_kwargs={"k": cfg["retrieval"]["k"]})
    
    api_key = os.getenv(cfg["llm"]["api_key_env"])
    if not api_key:
        raise EnvironmentError(
            f"Не задана переменная окружения {cfg['llm']['api_key_env']}"
        )
    
    logger.info(f"Инициализация LLM: {cfg['llm']['model']}")
    llm = ChatGroq(model=cfg["llm"]["model"], groq_api_key=api_key)
    
    return RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, return_source_documents=True
    )
