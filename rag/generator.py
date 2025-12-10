import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from config import SYSTEM_PROMPT
from pathlib import Path

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

client = InferenceClient(MODEL_ID, token=HF_TOKEN)

def generate_answer(query, context_docs, chat_history, max_new_tokens=200):
    context_text = "\n\n".join([str(d) for d in context_docs[:3]]) or "No context"
    history_text = "\n".join([f"User: {u}\nAssistant: {b}" for u,b in chat_history[-5:]])
    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context_text}\n\n{history_text}\nUser: {query}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        resp = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[Error contacting model API: {e}]"