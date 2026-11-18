# Image de l'API REST du RAG scientifique.
#   docker build -t rag-api .
#   docker run -p 8000:8000 --env-file .env rag-api
# Les clés (.env) ne sont JAMAIS copiées dans l'image : elles sont passées au lancement.

FROM python:3.10-slim

# Pas de fichiers .pyc, et des logs affichés immédiatement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1. Dépendances d'abord : cette couche est mise en cache tant que la liste ne change pas
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# 2. Uniquement le code nécessaire à l'API (pas de .env, de PDF ni de tests)
COPY api.py answer.py search.py indexer.py chunking.py PDF.py ./

# 3. Exécution avec un utilisateur sans privilèges
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
