"""
answer.py — Répond à une question à partir des articles indexés (le « G » de RAG).
"""

import weaviate
from openai import OpenAI
from dotenv import load_dotenv
import os
from weaviate.classes.init import Auth

from search import search

load_dotenv()  # charge OPENAI_API_KEY depuis le fichier .env

ANSWER_MODEL = "gpt-6-luna"  # le modèle le moins cher de la gamme actuelle d'OpenAI

INSTRUCTIONS = """Tu es un assistant qui répond à des questions sur des articles scientifiques.
Règles :
- Réponds UNIQUEMENT à partir des extraits fournis, jamais de tes connaissances générales.
- Cite tes sources après chaque information, au format (section, p. X).
- Si les extraits ne contiennent pas la réponse, dis simplement que tu ne la trouves pas dans les documents.
- Réponds dans la langue de la question."""


def answer(client, openai_client, question: str, limit: int = 3, alpha: float = 0.5) -> str:
    """Répond à une question en utilisant les chunks les plus pertinents."""
    chunks = search(client, openai_client, question, limit=limit, alpha=alpha)
    if not chunks:
        return "Aucun passage pertinent n'a été trouvé dans les documents."

    # pymupdf numérote les pages à partir de 0 : +1 pour citer la vraie page
    context = "\n\n".join(
        f"[{chunk.section} | p.{chunk.page_start + 1}-{chunk.page_end + 1}]\n{chunk.text}"
        for chunk in chunks
    )
    prompt = f"Extraits :\n{context}\n\nQuestion : {question}"

    response = openai_client.responses.create(
        model=ANSWER_MODEL,
        instructions=INSTRUCTIONS,
        input=prompt,
    )
    return response.output_text.strip()


if __name__ == "__main__":
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY"))
    )
    openai_client = OpenAI()
    try:
        question = "Quel matériel a servi à l'entraînement ?"
        print(answer(client, openai_client, question))
    finally:
        client.close()