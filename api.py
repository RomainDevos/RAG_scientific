"""
api.py — API REST du RAG scientifique.

Lancer :  uvicorn api:app --port 8000
Docs   :  http://localhost:8000/docs

Routes :
    GET  /health  -> vérifie que le service et Weaviate répondent
    POST /ask     -> pose une question, renvoie la réponse et ses sources
"""

import logging
import os
from contextlib import asynccontextmanager

import weaviate
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from openai import OpenAI
from pydantic import BaseModel, Field
from weaviate.classes.init import Auth

from answer import answer
from search import search

load_dotenv()
logger = logging.getLogger("rag-api")


# --- Modèles de requête / réponse ----------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="Question posée sur les articles")


class Source(BaseModel):
    document: str
    section: str
    pages: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


# --- Cycle de vie : les connexions sont ouvertes une seule fois ----------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.weaviate = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")),
    )
    app.state.openai = OpenAI()
    try:
        yield
    finally:
        app.state.weaviate.close()


app = FastAPI(
    title="RAG scientifique",
    description="Répond à des questions sur des articles scientifiques, avec citation des sources.",
    lifespan=lifespan,
)


# --- Routes ---------------------------------------------------------------------
@app.get("/health")
def health(request: Request) -> dict:
    """Le service est vivant et Weaviate est joignable."""
    if not request.app.state.weaviate.is_ready():
        raise HTTPException(status_code=503, detail="Weaviate n'est pas disponible")
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request) -> AskResponse:
    """Pose une question : renvoie la réponse du RAG et les passages utilisés."""
    client = request.app.state.weaviate
    openai_client = request.app.state.openai
    try:
        reply = answer(client, openai_client, body.question)
        chunks = search(client, openai_client, body.question)
    except Exception:
        # Le détail reste dans les logs : on ne renvoie pas d'information interne au client
        logger.exception("Erreur pendant le traitement de la question")
        raise HTTPException(status_code=502, detail="Erreur pendant le traitement de la question")

    sources = [
        Source(
            document=chunk.source,
            section=chunk.section,
            # PyMuPDF numérote les pages à partir de 0 : +1 pour la vraie page
            pages=f"{chunk.page_start + 1}-{chunk.page_end + 1}",
        )
        for chunk in chunks
    ]
    return AskResponse(answer=reply, sources=sources)
