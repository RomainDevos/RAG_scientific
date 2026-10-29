"""
search.py — Retrouve les chunks les plus pertinents pour une question.
"""

import weaviate
from openai import OpenAI
from dotenv import load_dotenv
import os 
from weaviate.classes.init import Auth

from chunking import Chunk
from indexer import COLLECTION_NAME, embeddings

load_dotenv()  # charge OPENAI_API_KEY depuis le fichier .env


def search(
    client: weaviate.WeaviateClient,
    openai_client: OpenAI,
    question: str,
    limit: int = 3,
    alpha: float = 0.5,
) -> list[Chunk]:
    """Recherche hybride : mots-clés (BM25) + sens (vecteurs).

    alpha = 0 -> 100 % mots-clés, alpha = 1 -> 100 % sens.
    """
    # La question est vectorisée avec le même modèle que les chunks
    question_vector = embeddings(openai_client, [question])[0]

    collection = client.collections.use(COLLECTION_NAME)
    results = collection.query.hybrid(
        query=question,
        vector=question_vector,
        alpha=alpha,
        limit=limit,
    )
    return [Chunk(**obj.properties) for obj in results.objects]


if __name__ == "__main__":
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=Auth.api_key(
            api_key=os.getenv("WEAVIATE_API_KEY")
        )
    )
    openai_client = OpenAI()
    try:
        question = "Quel matériel a servi à l'entraînement ?"
        for chunk in search(client, openai_client, question):
            print(f"\n[{chunk.section} | p.{chunk.page_start}-{chunk.page_end}]")
            print(chunk.text[:300], "...")
    finally:
        client.close()