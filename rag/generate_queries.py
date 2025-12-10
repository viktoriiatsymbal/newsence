import os
import json
import faiss
from tqdm import tqdm
from openai import OpenAI
from sentence_transformers import SentenceTransformer

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
OPENAI_MODEL = "gpt-4o-mini"

def generate_three_queries(article_text):
    prompt = f"""
Generate EXACTLY 3 short search queries for the following news article.
Rules:
- Each query must be 3–6 words.
- Must look like a realistic search query.
- No bullet points.
- No numbering.
- Output ONLY the queries separated by newlines.

Article:
{article_text}

Output:
"""
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=60,
    )

    text = response.choices[0].message.content.strip()
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    return lines[:3]

def build_query_index(queries_nested):
    os.makedirs("index", exist_ok=True)

    all_queries = [q for group in queries_nested for q in group]
    print(f"Total queries to embed: {len(all_queries)}")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(all_queries, show_progress_bar=True).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, "index/news_api_query_index.faiss")

def generate_queries_for_newsapi_metadata(api_metadata):
    print(f"Loaded {len(api_metadata)} NewsAPI metadata entries.")

    original_texts = [item["text"] for item in api_metadata]

    queries_nested = []

    for txt in tqdm(original_texts, desc="Generating query triples"):
        qs = generate_three_queries(txt)
        queries_nested.append(qs)

    build_query_index(queries_nested)

    enriched = []
    for meta, qs in zip(api_metadata, queries_nested):
        enriched.append({
            "article": meta,
            "queries": qs
        })

    with open("index/news_api_queries.json", "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    example = {
        "title": "Economic Digest: Nepal’s Business News in a Snap",
        "description": "KATHMANDU: Economic Digest offers ...",
        "content": "KATHMANDU: Economic Digest offers ... [+12217 chars]",
        "source": "Khabarhub.com",
        "url": "https://english.khabarhub.com/2025/11/511034/",
        "publishedAt": "2025-12-11T02:15:30Z",
        "text": (
            "Economic Digest: Nepal’s Business News in a Snap "
            "KATHMANDU: Economic Digest offers a concise yet comprehensive overview "
            "of significant business happenings in Nepal..."
        )
    }

    print("Testing OpenAI query generation:\n")
    qs = generate_three_queries(example["text"])
    print(qs)