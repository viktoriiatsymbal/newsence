import os
import json
import time
import requests
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

model = SentenceTransformer("all-MiniLM-L6-v2")

BASE_URL = "https://newsapi.org/v2/everything"

QUERY_TERMS = [
    "technology",
    "science",
    "world",
    "business",
    "health",
    "environment",
    "AI",
    "machine learning",
    "startup",
    "economy",
]

FROM_DATE = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
TO_DATE = datetime.utcnow().strftime("%Y-%m-%d")


def fetch_newsapi_query(query, max_pages=100):
    print(f"Fetching query: {query}")

    all_articles = []

    for page in range(1, max_pages + 1):
        params = {
            "apiKey": NEWS_API_KEY,
            "q": query,
            "from": FROM_DATE,
            "to": TO_DATE,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": 100,
            "page": page,
        }

        response = requests.get(BASE_URL, params=params)
        data = response.json()

        if data.get("status") != "ok":
            print("NewsAPI error:", data)
            break

        articles = data.get("articles", [])
        if not articles:
            break

        all_articles.extend(articles)

        if len(articles) < 100:
            break

        time.sleep(0.2)

    print(f"Collected {len(all_articles)} articles for '{query}'")
    return all_articles

def fetch_large_news_corpus():
    all_articles = {}


    for q in QUERY_TERMS:
        results = fetch_newsapi_query(q, max_pages=100)

        for a in results:
            url = a.get("url")
            if url:
                all_articles[url] = a

    deduped = list(all_articles.values())
    print(f"\nTotal deduped NewsAPI articles: {len(deduped)}")

    return deduped

def preprocess_articles(articles):
    docs = []
    metadata = []

    for a in articles:
        title = a.get("title") or ""
        desc = a.get("description") or ""
        content = a.get("content") or ""

        text = " ".join([title, desc, content]).strip()
        if not text:
            continue

        docs.append(text)
        metadata.append({
            "title": title,
            "description": desc,
            "content": content,
            "source": a.get("source", {}).get("name"),
            "url": a.get("url"),
            "publishedAt": a.get("publishedAt"),
            "text": text,
        })

    print(f"Preprocessed {len(docs)} documents.")
    return docs, metadata

def embed_documents(docs):
    return model.encode(docs, batch_size=32, show_progress_bar=True)


def build_faiss_index(embeddings, metadata, index_path, meta_path):
    embeddings = np.array(embeddings).astype("float32")
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, index_path)

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved FAISS index → {index_path}")
    print(f"Saved metadata → {meta_path}")

    return index, metadata


def load_faiss_index(index_path, meta_path):
    index = faiss.read_index(index_path)
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    return index, metadata

def build_large_newsapi_index(index_path="newsapi_index.faiss", meta_path="newsapi_metadata.json"):
    articles = fetch_large_news_corpus()
    docs, metadata = preprocess_articles(articles)
    embeddings = embed_documents(docs)

    return build_faiss_index(
        embeddings, metadata, index_path=index_path, meta_path=meta_path
    )
