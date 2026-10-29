"""
indexer.py — Envoie les chunks d'un PDF dans Weaviate.

Les chunks sont insérés sans vecteur pour l'instant : la recherche par
mots-clés (BM25) fonctionne déjà. La fonction `embeddings` permettra
d'ajouter les vecteurs plus tard.
"""

import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.data import DataObject
from weaviate.classes.query import Filter
from weaviate.util import generate_uuid5
from openai import OpenAI
import os
from weaviate.classes.init import Auth

from PDF import PDF
from chunking import Chunk, Chunker

from dotenv import load_dotenv
load_dotenv()

COLLECTION_NAME = "Chunk"
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100


# --- Embedding ---------------------------------------------------------------
def embeddings(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Renvoie les embeddings pour une liste de textes."""
    vectors = []
    for i in range(0, len(texts), BATCH_SIZE):
        texts_batch = texts[i:i + BATCH_SIZE]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts_batch)
        vectors.extend([item.embedding for item in response.data])
    return vectors


# --- Weaviate ----------------------------------------------------------------
def create_collection(client: weaviate.WeaviateClient) -> None:
    """Crée la collection si elle n'existe pas encore."""
    if client.collections.exists(COLLECTION_NAME):
        return
    client.collections.create(
        name=COLLECTION_NAME,
        vector_config=Configure.Vectors.self_provided(),
        properties=[
            Property(name="text", data_type=DataType.TEXT),
            Property(name="pdf_id", data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT),
            Property(name="section", data_type=DataType.TEXT),
            Property(name="section_id", data_type=DataType.INT),
            Property(name="chunk_index", data_type=DataType.INT),
            Property(name="page_start", data_type=DataType.INT),
            Property(name="page_end", data_type=DataType.INT),
            Property(name="n_tokens", data_type=DataType.INT),
        ],
    )


def chunk_to_properties(chunk: Chunk) -> dict:
    """Transforme un Chunk en dictionnaire de propriétés pour Weaviate."""
    return {
        "text": chunk.text,
        "pdf_id": chunk.pdf_id,
        "source": chunk.source,
        "section": chunk.section,
        "section_id": chunk.section_id,
        "chunk_index": chunk.chunk_index,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "n_tokens": chunk.n_tokens,
    }


def delete_pdf_chunks(collection, pdf_id: str) -> None:
    """Supprime tous les chunks d'un PDF déjà présents dans la collection."""
    collection.data.delete_many(where=Filter.by_property("pdf_id").equal(pdf_id))


def insert_chunks(collection, chunks: list[Chunk], vectors: list[list[float]] | None = None) -> None:
    """Insère les chunks, avec leurs vecteurs s'il y en a."""
    objects = []
    for index, elt in enumerate(chunks):
        properties = chunk_to_properties(elt)
        uuid = generate_uuid5(elt.chunk_id)
        vector = vectors[index] if vectors is not None else None
        objects.append(DataObject(
            properties=properties,
            uuid=uuid,
            vector=vector,
        ))

    # Envoi en une seule fois, APRÈS la boucle
    result = collection.data.ingest(objects)
    if result.errors:
        print(f"{len(result.errors)} chunks n'ont pas pu être insérés :")
        for pos, error in result.errors.items():
            print(f"  - position {pos} : {error.message}")


def count_chunks(collection) -> int:
    """Nombre total d'objets dans la collection."""
    return collection.aggregate.over_all(total_count=True).total_count


if __name__ == "__main__":
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=Auth.api_key(
            api_key=os.getenv("WEAVIATE_API_KEY")
        )
    )
    openai_client = OpenAI()
    try:
        print("Weaviate prêt :", client.is_ready())

        create_collection(client)
        collection = client.collections.use(COLLECTION_NAME)

        pdf = PDF.from_file("Attention_is_all_you_need.pdf")
        chunks = Chunker().chunk_pdf(pdf)
        print(len(chunks), "chunks produits")
        
        vectors = embeddings(openai_client, [chunk.text for chunk in chunks])

        delete_pdf_chunks(collection, pdf.pdf_id)
        insert_chunks(collection, chunks, vectors)  # sans vecteurs pour l'instant
        print(count_chunks(collection), "objets dans Weaviate")

        # Première recherche par mots-clés (BM25)
        results = collection.query.bm25(query="multi-head attention", limit=3)
        for obj in results.objects:
            p = obj.properties
            print(f"\n[{p['section']} | p.{p['page_start']}-{p['page_end']}]")
            print(p["text"][:200], "...")
    finally:
        client.close()