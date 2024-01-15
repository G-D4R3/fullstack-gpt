import streamlit as st
from langchain.document_loaders import UnstructuredFileLoader
from langchain.retrievers import WikipediaRetriever
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓"
)

st.title("QuizGPT")

@st.cache_data(show_spinner="Embedding file...")
def split_file(file):
    ''' 현재 data가 변경될 때마다 매번 embed file function을 실행
     사용자가 message를 보낼 때마다 function 실행 -> 불필요한 반복 실행'''
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, 'wb') as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/quiz_embeddings/{file.name}")

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator='\n',
        chunk_size=600,
        chunk_overlap=100,
    )

    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs

with st.sidebar:
    choice = st.selectbox(
        "Choose what you want to use.",
        ("File", "Wikipedia Article")
    )

    if choice == 'File':
        file = st.file_uploader("Upload a .docx, .txt or .pdf file", type=["pdf", "txt", "docx"])
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            retriever = WikipediaRetriever(top_k_results=1)
            with st.status("Searching Wikipedia..."):
                docs = retriever.get_relevant_documents(topic)

