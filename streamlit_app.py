from dotenv import load_dotenv

load_dotenv()

import os
import logging
import streamlit as st
from datetime import datetime
from rag.chain import build_rag_chain
from rag.config_validator import load_and_validate_config

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("rag_app.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Semantic Search по SimpleOne", layout="centered")

# Валидация конфигурации при запуске
@st.cache_resource
def validate_configuration():
    """Валидирует конфигурацию при запуске приложения."""
    try:
        config = load_and_validate_config()
        logger.info("Конфигурация успешно валидирована")
        return True, None
    except Exception as e:
        logger.error(f"Ошибка валидации конфигурации: {e}")
        return False, str(e)


@st.cache_resource(show_spinner="Загрузка RAG цепочки...")
def get_chain():
    """Создает и кэширует RAG цепочку."""
    try:
        chain = build_rag_chain()
        logger.info("RAG цепочка успешно создана")
        return chain
    except Exception as e:
        logger.error(f"Ошибка при создании RAG цепочки: {e}")
        raise


def extract_answer(response):
    """Извлекает ответ из ответа RAG цепочки."""
    if isinstance(response, dict):
        # Пробуем различные ключи, которые могут содержать ответ
        for key in ["answer", "result", "output"]:
            if key in response:
                return response[key]
        # Если не нашли стандартный ключ, ищем первую строку в значениях
        for value in response.values():
            if isinstance(value, str) and value.strip():
                return value
        return "Ответ не найден в ответе модели"
    return str(response)


def extract_source_documents(response):
    """Извлекает исходные документы из ответа RAG цепочки."""
    if isinstance(response, dict):
        return response.get("context", []) or response.get("source_documents", [])
    return []


def format_document(doc):
    """Форматирует документ для отображения."""
    if hasattr(doc, "metadata"):
        metadata = doc.metadata
        content = doc.page_content if hasattr(doc, "page_content") else str(doc)
    else:
        metadata = {}
        content = str(doc)

    return {
        "title": metadata.get("title", "Без названия"),
        "heading": metadata.get("heading", ""),
        "content": content,
    }


# Проверка конфигурации
config_valid, config_error = validate_configuration()
if not config_valid:
    st.error(f"❌ Ошибка конфигурации: {config_error}")
    st.stop()

st.title("🔍 Поиск по SimpleOne")
st.markdown("Задайте вопрос о SimpleOne API, и система найдет релевантную информацию.")

query = st.text_input("Введите вопрос:", placeholder="Например: Как использовать s_i18n.getMessage?")

if query:
    try:
        rag_chain = get_chain()
        
        # Логирование запроса
        logger.info(f"Получен запрос: {query}")
        start_time = datetime.now()
        
        # Выполнение запроса
        with st.spinner("Поиск информации..."):
            if hasattr(rag_chain, "invoke"):
                response = rag_chain.invoke({"input": query})
            else:
                response = rag_chain({"query": query})
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Запрос обработан за {elapsed_time:.2f} секунд")
        
        # Извлечение ответа и источников
        answer = extract_answer(response)
        source_docs = extract_source_documents(response)
        
        # Логирование ответа
        logger.info(f"Ответ получен, найдено источников: {len(source_docs)}")
        
        # Отображение ответа
        st.subheader("📝 Ответ:")
        st.write(answer)
        
        # Отображение источников
        if source_docs:
            st.subheader(f"📚 Источники ({len(source_docs)}):")
            for idx, doc in enumerate(source_docs, 1):
                doc_info = format_document(doc)
                content_snippet = doc_info["content"][:300].replace("\n", " ")
                
                with st.expander(f"{idx}. {doc_info['title']} — {doc_info['heading']}"):
                    st.markdown(f"**Заголовок:** {doc_info['heading']}")
                    st.markdown(f"**Содержание:**\n\n{doc_info['content']}")
        else:
            st.info("ℹ️ Источники не найдены")
            
    except Exception as e:
        error_msg = f"Ошибка при обработке запроса: {str(e)}"
        logger.error(error_msg, exc_info=True)
        st.error(f"❌ {error_msg}")
        st.info("Проверьте логи для получения дополнительной информации.")
