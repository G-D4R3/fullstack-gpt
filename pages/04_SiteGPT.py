import streamlit as st
from langchain.document_loaders import SitemapLoader
from langchain.document_transformers import Html2TextTransformer

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥"
)

st.title("SiteGPT")

html2text_transformer = Html2TextTransformer()

@st.cache_data(show_spinner="Loading website")
def load_website(url):
    loader = SitemapLoader(url)
    loader.requests_per_second = 1
    docs = loader.load_and_split()
    return docs

with st.sidebar:
    url = st.text_input("Write down a URL", placeholder="https://example.com")


if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL")
    else:
        docs = load_website(url)
        st.write(docs)


