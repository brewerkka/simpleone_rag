from dotenv import load_dotenv

load_dotenv()

import os
import streamlit as st
from rag.chain import build_rag_chain

st.set_page_config(page_title=" Semantic Search по SimpleOne", layout="centered")


@st.cache_resource(show_spinner=False)
def get_chain():
    return build_rag_chain()


st.title(" Поиск по SimpleOne")
query = st.text_input("Введите вопрос:")

if query:
    rag_chain = get_chain()
    try:
        if hasattr(rag_chain, "invoke"):
            response = rag_chain.invoke({"input": query})
        else:
            response = rag_chain({"query": query})
    except Exception as e:
        st.error(f"Ошибка при запросе: {e}")
    else:
        if isinstance(response, dict):
            if "answer" in response:
                answer = response["answer"]
            elif "result" in response:
                answer = response["result"]
            elif "output" in response:
                answer = response["output"]
            else:
                answer = next(
                    (v for v in response.values() if isinstance(v, str)),
                    "Ответ не найден",
                )

            source_docs = response.get("context", []) or response.get(
                "source_documents", []
            )
        else:
            answer = str(response)
            source_docs = []

        st.subheader("Ответ:")
        st.write(answer)

        if source_docs:
            st.subheader("Источники:")
            for doc in source_docs:
                if hasattr(doc, "metadata"):
                    metadata = doc.metadata
                    content = (
                        doc.page_content if hasattr(doc, "page_content") else str(doc)
                    )
                else:
                    metadata = {}
                    content = str(doc)

                title = metadata.get("title", "Без названия")
                heading = metadata.get("heading", "")
                content_snippet = content[:300].replace("\n", " ")
                st.markdown(f"**{title}** — `{heading}`")
                st.markdown(f"> {content_snippet} ...")
        else:
            st.info("Источники не найдены")
