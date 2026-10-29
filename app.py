import streamlit as st
import weaviate
from openai import OpenAI
from answer import answer

st.title("RAG scientifique")

question = st.text_input("Question")

if question:
    client = weaviate.connect_to_local()
    openai_client = OpenAI()
    try:
        with st.spinner("Recherche dans les articles…"):
            reponse = answer(client, openai_client, question)
        st.write(reponse)
    finally:
        client.close()