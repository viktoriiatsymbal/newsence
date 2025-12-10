import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from rag.dataset_loader import load_dataset_local
from rag.embedder import embed_texts
from rag.faiss_handler import build_faiss_index, load_faiss_index, search_faiss
from rag.generator import generate_answer
from rag.chat_manager import load_history, save_history, clear_history
from rag.generate_queries import (
            generate_three_queries,
            build_query_index
        )
from rag.news_api import (
    build_large_newsapi_index,
    load_faiss_index as load_faiss_index_api
)

import faiss
import json


def main():
    # print("Initializing News RAG Agent (AG News)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    clear_history()

    # if not os.path.exists("index/news_index.faiss"):
    #     print("Building index for original AG News articles...")
    #     dataset = load_dataset_local("ag_news")
    #     texts = dataset["text"]
    #     categories = dataset["category"]

    #     combined_texts = [f"[{cat}] {txt}" for cat, txt in zip(categories, texts)]
    #     embeddings = embed_texts(combined_texts)

    #     index, metadata = build_faiss_index(
    #         embeddings,
    #         combined_texts,
    #         save_path="index/news_index.faiss",
    #         metadata_path="index/metadata.json"
    #     )
    #     print("Original article index has been built.")
    # else:
    #     index, metadata = load_faiss_index(
    #         save_path="index/news_index.faiss",
    #         metadata_path="index/metadata.json"
    #     )
    #     print("Loaded existing original FAISS index.")

    NEWS_API_INDEX = "index/news_api_index.faiss"
    NEWS_API_META = "index/news_api_metadata.json"

    if not os.path.exists(NEWS_API_INDEX):
        print("Building NewsAPI FAISS index...")

        api_index, api_metadata = build_large_newsapi_index(
            index_path=NEWS_API_INDEX,
            meta_path=NEWS_API_META
        )

        print(f"NewsAPI FAISS index stored at: {NEWS_API_INDEX}")
        print(f"NewsAPI metadata stored at:   {NEWS_API_META}")
    else:
        print("Loading existing NewsAPI FAISS index...")
        api_index, api_metadata = load_faiss_index_api(
            index_path=NEWS_API_INDEX,
            meta_path=NEWS_API_META
        )


    NEWS_API_QUERY_INDEX = "index/news_api_query_index.faiss"
    NEWS_API_QUERY_META = "index/news_api_queries.json"

    if not os.path.exists(NEWS_API_QUERY_INDEX):
        print("Generating search queries for NewsAPI articles...")

        original_texts = [item["text"] for item in api_metadata]
        queries_nested = []

        for txt in tqdm(original_texts, desc="Generating queries for NewsAPI"):
            qs = generate_three_queries(txt)
            queries_nested.append(qs)

        build_query_index(queries_nested)

        enriched = []
        for article, qs in zip(api_metadata, queries_nested):
            enriched.append({
                "article": article,
                "queries": qs
            })

        with open(NEWS_API_QUERY_META, "w", encoding="utf-8") as f:
            json.dump(enriched, f, ensure_ascii=False, indent=2)

        print("NewsAPI query FAISS index created.")

    else:
        print("Loading existing NewsAPI query FAISS index...")
        q_api_index = faiss.read_index(NEWS_API_QUERY_INDEX)

        with open(NEWS_API_QUERY_META, "r", encoding="utf-8") as f:
            enriched_api_metadata = json.load(f)


    print("\nAsk me about news! (type 'exit' to quit)\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in {"exit", "quit"}:
            clear_history()
            print("Goodbye!")
            break

        chat_history = load_history()

        user_emb = model.encode([query]).astype("float32")

        D, I = q_api_index.search(user_emb, k=10) 

        hit_articles = []
        for idx in I[0]:
            article_idx = idx // 3
            hit_articles.append((article_idx, float(D[0][0])))

        seen = set()
        ranked_articles = []
        for article_idx, score in hit_articles:
            if article_idx not in seen:
                ranked_articles.append(article_idx)
                seen.add(article_idx)

        retrieved_texts = [
            enriched_api_metadata[i]["article"]["text"]
            for i in ranked_articles[:5]
        ]

        answer = generate_answer(query, retrieved_texts, chat_history)

        print(f"\nBot: {answer}\n")
        chat_history.append((query, answer))
        save_history(chat_history)


if __name__ == "__main__":
    main()
