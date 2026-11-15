"""Tests de l'API REST. Weaviate, OpenAI et le RAG sont simulés : aucun appel réseau."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import api
from chunking import Chunk


class FakeWeaviate:
    def __init__(self, ready=True):
        self.ready = ready
        self.closed = False

    def is_ready(self):
        return self.ready

    def close(self):
        self.closed = True


def fake_chunk():
    return Chunk(
        text="Le modèle utilise 8 GPU.",
        pdf_id="abc",
        section_id=1,
        chunk_index=0,
        source="Attention_is_all_you_need.pdf",
        section="5.2 Hardware and Schedule",
        page_start=6,   # numérotation PyMuPDF (à partir de 0)
        page_end=7,
        n_tokens=8,
    )


@pytest.fixture
def fake_weaviate(monkeypatch):
    fake = FakeWeaviate()
    monkeypatch.setenv("WEAVIATE_URL", "https://exemple.weaviate.cloud")
    monkeypatch.setenv("WEAVIATE_API_KEY", "cle-de-test")
    monkeypatch.setenv("OPENAI_API_KEY", "cle-de-test")
    monkeypatch.setattr(api.weaviate, "connect_to_weaviate_cloud", lambda **kwargs: fake)
    monkeypatch.setattr(api, "OpenAI", lambda: SimpleNamespace())
    monkeypatch.setattr(api, "answer", lambda client, openai_client, question: f"Réponse à : {question}")
    monkeypatch.setattr(api, "search", lambda client, openai_client, question: [fake_chunk()])
    return fake


@pytest.fixture
def client(fake_weaviate):
    with TestClient(api.app) as test_client:
        yield test_client


# --- /health -----------------------------------------------------------------------
def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_503_si_weaviate_indisponible(client, fake_weaviate):
    fake_weaviate.ready = False
    assert client.get("/health").status_code == 503


# --- /ask --------------------------------------------------------------------------
def test_ask_renvoie_la_reponse_et_les_sources(client):
    response = client.post("/ask", json={"question": "Quel matériel ?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Réponse à : Quel matériel ?"
    assert body["sources"] == [
        {
            "document": "Attention_is_all_you_need.pdf",
            "section": "5.2 Hardware and Schedule",
            "pages": "7-8",  # +1 : PyMuPDF compte les pages à partir de 0
        }
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},            # vide
        {},                          # champ manquant
        {"question": "x" * 1001},    # trop long
        {"question": 42},            # mauvais type
    ],
)
def test_ask_rejette_les_requetes_invalides(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_ask_erreur_interne_ne_divulgue_pas_de_details(client, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("secret interne : cle=sk-123")

    monkeypatch.setattr(api, "answer", boom)
    response = client.post("/ask", json={"question": "Une question"})
    assert response.status_code == 502
    assert "secret" not in response.text
    assert "sk-123" not in response.text


# --- Cycle de vie ---------------------------------------------------------------------
def test_la_connexion_weaviate_est_fermee_a_l_arret(fake_weaviate):
    with TestClient(api.app):
        assert fake_weaviate.closed is False
    assert fake_weaviate.closed is True
