"""
App Streamlit pour le RAG scientifique
"""


import streamlit as st
import weaviate
from openai import OpenAI
from answer import answer
import os
from weaviate.classes.init import Auth

st.title("RAG scientifique")

question = st.text_input("Question")

if question:
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY"))
    )
    openai_client = OpenAI()
    try:
        with st.spinner("Recherche dans les articles…"):
            reponse = answer(client, openai_client, question)
        st.write(reponse)
    finally:
        client.close()