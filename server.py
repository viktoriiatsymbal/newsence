import os
import json
from pathlib import Path

import faiss
from tqdm import tqdm
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer

from rag.dataset_loader import load_dataset_local
from rag.embedder import embed_texts
from rag.faiss_handler import build_faiss_index, load_faiss_index
from rag.generator import generate_answer
from rag.chat_manager import load_history, save_history, clear_history
from rag.generate_queries import generate_three_queries, build_query_index
from rag.news_api import (
    build_large_newsapi_index,
    load_faiss_index as load_faiss_index_api
)

ROOT = Path(__file__).resolve().parent
INDEX_DIR = ROOT / "index"
INDEX_DIR.mkdir(exist_ok=True)

NEWS_API_INDEX = INDEX_DIR / "news_api_index.faiss"
NEWS_API_META  = INDEX_DIR / "news_api_metadata.json"

NEWS_API_QUERY_INDEX = INDEX_DIR / "news_api_query_index.faiss"
NEWS_API_QUERY_META  = INDEX_DIR / "news_api_queries.json"

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

print("Initializing News RAG Agent (Flask backend)...")
model = SentenceTransformer("all-MiniLM-L6-v2")

clear_history()

if not NEWS_API_INDEX.exists():
    print("Building NewsAPI article FAISS index...")
    api_index, api_metadata = build_large_newsapi_index(
        index_path=str(NEWS_API_INDEX),
        meta_path=str(NEWS_API_META)
    )
else:
    api_index, api_metadata = load_faiss_index_api(
        index_path=str(NEWS_API_INDEX),
        meta_path=str(NEWS_API_META)
    )

if not NEWS_API_QUERY_INDEX.exists() or not NEWS_API_QUERY_META.exists():
    print("Generating NewsAPI query index...")

    original_texts = [item["text"] for item in api_metadata]
    queries_nested = []

    for txt in tqdm(original_texts, desc="Generating queries"):
        queries_nested.append(generate_three_queries(txt))

    build_query_index(queries_nested)

    enriched_api_metadata = []
    for article, qs in zip(api_metadata, queries_nested):
        enriched_api_metadata.append({
            "article": article,
            "queries": qs
        })

    with open(NEWS_API_QUERY_META, "w", encoding="utf-8") as f:
        json.dump(enriched_api_metadata, f, ensure_ascii=False, indent=2)

    q_api_index = faiss.read_index(str(NEWS_API_QUERY_INDEX))

else:
    print("Loading existing NewsAPI query index...")
    q_api_index = faiss.read_index(str(NEWS_API_QUERY_INDEX))

    with open(NEWS_API_QUERY_META, "r", encoding="utf-8") as f:
        enriched_api_metadata = json.load(f)


print("Backend ready at http://127.0.0.1:5000")

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(force=True)
    query = (data.get("message") or "").strip()

    if not query:
        return jsonify({"answer": "Please type a question about news."})

    chat_history = load_history()

    user_emb = model.encode([query]).astype("float32")
    D, I = q_api_index.search(user_emb, k=10)

    hit_articles = []
    for idx in I[0]:
        article_idx = idx // 3
        hit_articles.append(article_idx)

    seen = set()
    ranked_articles = []
    for idx in hit_articles:
        if idx not in seen:
            ranked_articles.append(idx)
            seen.add(idx)

    retrieved_texts = [
        enriched_api_metadata[i]["article"]["text"]
        for i in ranked_articles[:5]
    ]

    answer = generate_answer(query, retrieved_texts, chat_history)

    chat_history.append((query, answer))
    save_history(chat_history)

    return jsonify({"answer": answer})


@app.route("/api/clear", methods=["POST"])
def api_clear():
    clear_history()
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
