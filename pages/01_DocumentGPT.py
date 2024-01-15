import time
import streamlit as st

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃"
)

st.title("DocumentGPT")

with st.chat_message("human"):
    st.write("Helloooooo")


with st.chat_message("ai"):
    st.write("how are you")

with st.status("Embedding file...", expanded=True) as status:
    time.sleep(2)
    st.write("Getting the file")
    time.sleep(2)
    st.write("Embedding the file")
    time.sleep(2)
    st.write("Caching the file")

    status.update(label="Error", state="error")

st.chat_input("Send an message to the ai")