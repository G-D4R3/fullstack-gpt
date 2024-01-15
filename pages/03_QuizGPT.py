import streamlit as st
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import WikipediaRetriever
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓"
)

st.title("QuizGPT")

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-1106",
    callbacks=[
        StreamingStdOutCallbackHandler()
    ]
)

@st.cache_data(show_spinner="Embedding file...")
def split_file(file):
    ''' 현재 data가 변경될 때마다 매번 embed file function을 실행
     사용자가 message를 보낼 때마다 function 실행 -> 불필요한 반복 실행'''
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, 'wb') as f:
        f.write(file_content)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator='\n',
        chunk_size=600,
        chunk_overlap=100,
    )

    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


with st.sidebar:
    docs = None
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

if not docs:
    st.markdown(
        """
        Welcome to QuizGPT
        
        I will make a quiz from Wikipedia articles or files you upload to test
        your knowledge and help you study.
        
        Get started by uploading a file or searching on Wikipedia in the sidebar.
        """
    )
else:
    st.write(docs)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a helpful assistant that is role playing as a teacher.
            
            Based ONLY on the following context make 10 questions to test the user's
            knowledge about the text.
            
            Each question should have 4 answers, three of them must be incorrect and
            one should me correct.
            
            Use (o) signal to correct answer.
            
            Question examples:
            
            Question: What is the color of the ocean?
            Answers: Red|Yellow|Green|Blue(o)
            
            Question: What is the capital of Georgia?
            Answers: Baku|Tbilisi(o)|Manila|Beirut
            
            Question: When was Avartar released?
            Answers: 2007|2001|2009(o)|1998
            
            Question: Who was Julius Caesar?
            Answers: A Roman Emperor(o)|Painter|Actor|Model
            
            Your turn!
            
            Context: {context}
            """
        ),
    ])

    chain = {
        "context": format_docs
    } | prompt | llm

    start = st.button("Generate Quiz")

    if start:
        a = chain.invoke(docs)
        st.write(a)
