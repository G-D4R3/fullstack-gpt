import streamlit as st
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃"
)

st.title("DocumentGPT")

def embed_file(file):
    ''' 현재 data가 변경될 때마다 매번 embed file function을 실행
     사용자가 message를 보낼 때마다 function 실행 -> 불필요한 반복 실행'''
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, 'wb') as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator='\n',
        chunk_size=600,
        chunk_overlap=100,
    )

    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()

    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir
    )

    vectorstore = FAISS.from_documents(docs, cached_embeddings)

    retriever = vectorstore.as_retriever()
    return retriever


# 사용자의 파일 업로드 요청
st.markdown("""
Welcome!

Use this chatbot to ask questions to an AI about your files!
""")

file = st.file_uploader(
    "Upload a .txt, .pdf, .docx file",
    type=["pdf", "txt", "docx"]
)

if file:
    retriever = embed_file(file)
    s = retriever.invoke("winston")
    s