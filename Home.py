import streamlit as st
from langchain.prompts import PromptTemplate

st.write(PromptTemplate)

p = PromptTemplate.from_template("Xxxx")
st.write(p)

a = [1, 2, 3, 4]

st.selectbox("Choose your model", ("GPT-3", "GPT-4"))