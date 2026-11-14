"""
benchmark.py — Fait tourner le RAG sur les questions du benchmark, puis l'évalue avec Ragas.
"""

import json
import os

import weaviate
from weaviate.classes.init import Auth
from dotenv import load_dotenv
from openai import OpenAI
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.llms import llm_factory
from ragas.metrics import FactualCorrectness, Faithfulness, LLMContextRecall

from search import search
from answer import answer

load_dotenv()

JUDGE_MODEL = "gpt-6-luna"

with open("Benchmarks/querry.json", "r", encoding="utf-8") as f:
    benchmark = json.load(f)

benchmark = benchmark[:]  

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.getenv("WEAVIATE_URL"),
    auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")),
)
openai_client = OpenAI()

# --- 1. Faire tourner le RAG sur chaque question ------------------------------
samples = []
results = []
try:
    for item in benchmark:
        question = item["querry"]

        chunks = search(client, openai_client, question)       # les chunks retrouvés
        response = answer(client, openai_client, question)     # la réponse du RAG

        samples.append(SingleTurnSample(
            user_input=question,
            retrieved_contexts=[chunk.text for chunk in chunks],
            response=response,
            reference=item["answers"],                         # la bonne réponse
        ))

        # On garde tout, pour pouvoir relire les résultats question par question
        results.append({
            "querry_id": item["querry_id"],
            "querry": question,
            "reference": item["answers"],
            "response": response,
            "expected_doc": item["doc_source"],
            "expected_section": item["section"],
            "retrieved": [
                {"source": chunk.source, "section": chunk.section, "text": chunk.text}
                for chunk in chunks
            ],
        })
        print(f"Question {item['querry_id']} traitée")
finally:
    client.close()

with open("Benchmarks/results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

# --- 2. Évaluation avec Ragas -------------------------------------------------
llm_judge = llm_factory(JUDGE_MODEL, client=openai_client)

result = evaluate(
    dataset=EvaluationDataset(samples=samples),
    metrics=[
        FactualCorrectness(),
        Faithfulness(),
        LLMContextRecall(),
    ],
    llm=llm_judge,
)
print(result)