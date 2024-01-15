import streamlit as st
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥"
)

st.title("SiteGPT")

html2text_transformer = Html2TextTransformer()

with st.sidebar:
    url = st.text_input("Write down a URL", placeholder="https://example.com")


if url:
    # async chroimium loader
    # browser를 실행 중이기 때문에 많은 url로 동작할 때 느려질 수 있음
    # 코드와 text를 얻었지만 브라우저를 볼 수 없었던 이유 : playwrite를 headless mode로 실행했기 때문
    # headless: 브라우저 프로세스가 로컬로부터 시작되었다는 것 -> 느려질 수 있음
    loader = AsyncChromiumLoader(urls=[url])
    docs = loader.load()
    transformed = html2text_transformer.transform_documents(docs)
    st.write(transformed)

