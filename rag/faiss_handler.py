import faiss
import numpy as np
import json
import os

def build_faiss_index(embeddings, texts, save_path="index/news_index.faiss", metadata_path="index/metadata.json"):
    os.makedirs("index", exist_ok=True)
    embeddings = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, save_path)
    metadata = [{"text": t} for t in texts]
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return index, metadata

def load_faiss_index(save_path="index/news_index.faiss", metadata_path="index/metadata.json"):
    index = faiss.read_index(save_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata

def search_faiss(index, query, metadata, model, k=5):

    query_emb = model.encode([query]).astype("float32")
    distances, indices = index.search(query_emb, k)

    results = []
    for i, dist in zip(indices[0], distances[0]):
        score = 1 / (1 + dist)
        results.append({"text": metadata[i].get("text"), "score": float(score)})

    return results

