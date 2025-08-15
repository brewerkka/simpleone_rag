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
        response = rag_chain.invoke({"query": query})
    except Exception as e:
        st.error(f"Ошибка при запросе: {e}")
    else:
        st.subheader("Ответ:")
        st.write(response["result"])

        st.subheader("Источники:")
        for doc in response["source_documents"]:
            title = doc.metadata.get("title", "Без названия")
            heading = doc.metadata.get("heading", "")
            content_snippet = doc.page_content[:300].replace("\n", " ")
            st.markdown(f"**{title}** — `{heading}`")
            st.markdown(f"> {content_snippet} ...")
