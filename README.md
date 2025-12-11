# Newsence. RAG for News

Newsence is a local Retrieval-Augmented Generation (RAG) project for news search and question answering.  
It ingests recent articles from NewsAPI, generates 3 user-style search queries per article with GPT-4o-mini, retrieves relevant articles using FAISS and Sentence-Transformers, and produces concise answers with Mistral-7B-Instruct via Hugging Face Inference API. Also the project contains Flask backend and HTML UI

## How it works

1. News ingestion from NewsAPI: fetches recent English news articles
2. Preprocessing: merges `title`, `description`, and `content` into one `text` field
3. Query expansion: for every article, generates exactly 3 short search queries with GPT-4o-mini
4. Indexing: builds a FAISS index for both articles and generated queries
5. Retrieval and Generation:
   - User query is embedded
   - Nearest generated queries are retrieved and then mapped back to articles
   - Top articles are passed to the LLM as context
   - Answer is generated and saved to chat history

## Data format with examples

### `index/news_api_metadata.json` (per-article metadata)
Each entry contains fields `title`, `description`, `content`, `source`, `url`, `publishedAt`, and merged `text`

Example snippet:
```json
[
  {
    "title": "Slack CEO leaves Salesforce to become OpenAI\u2019s first revenue chief, tackle multibillion-dollar losses",
    "description": "Salesforce said in a statement that it was...",
    "content": "OpenAI said Tuesday it has picked Slack CEO Denise Dresser...",
    "source": "Fortune",
    "url": "https://fortune.com/...",
    "publishedAt": "2025-12-11T12:46:51Z",
    "text": "Slack CEO leaves Salesforce..."
  }
]
```

### `index/news_api_queries.json` (article and generated queries mapping)

Example snippet:
```json
[
  {
    "article": { "...": "..." },
    "queries": [
        "Denise Dresser joins OpenAI",
        "Slack CEO moves to OpenAI",
        "OpenAI revenue chief announcement"
    ]
  }
]
```

## Requirements

* Python 3.11+
* A Hugging Face token for model inference
* A NewsAPI key for collecting articles
* An OpenAI key for query generation (GPT-4o-mini)

## Installation / Setup

### 1) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> NOTE: FAISS on macOS can be sensitive to NumPy versions.
> If you get FAISS/NumPy errors, install NumPy < 2 and use faiss-cpu=1.7.4:

```bash
pip install faiss-cpu==1.7.4
pip install "numpy<2" --force-reinstall
```

### 3) Configure environment variables

Create a `.env` file in the project root (or copy `.env.example`) and set:

```env
HUGGINGFACE_TOKEN=hf_...
NEWS_API_KEY=...
OPENAI_API_KEY=...
```

## Running the project

### — Via CLI mode

Run the local interactive console chatbot:

```bash
python main.py
```

On first run, it will:
* fetch NewsAPI articles (free version: limited to recent content, newest available articles are at least 24 hours old)
* build `index/news_api_index.faiss` and `index/news_api_metadata.json`
* generate 3 queries per article and build `index/news_api_query_index.faiss`
* save mapping to `index/news_api_queries.json`

Then you can ask questions until you type `exit` or `quit`

### — Via Web UI

The web demo uses a Flask backend `server.py` and a local HTML UI `ui.html`

#### 0) Install demo-only dependencies
These are required for the Flask server and CORS:

```bash
python -m pip install flask flask-cors
```

#### 1) Start the backend

```bash
python server.py
```

Backend runs at:

* `http://127.0.0.1:5000`

#### 2) Open the UI

Open `ui.html` in your browser

The UI calls:

* `POST /api/chat` — send a message
* `POST /api/clear` — clear chat history

## External services / links

* NewsAPI (data source): [https://newsapi.org](https://newsapi.org)
* Hugging Face Inference API: [https://huggingface.co/docs/huggingface_hub/en/guides/inference](https://huggingface.co/docs/huggingface_hub/en/guides/inference)
* Sentence-Transformers: [https://www.sbert.net](https://www.sbert.net)
* FAISS: [https://github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)

## Authors / Contributors

* Mykhailo Ponomarenko
* Viktoriia Tsymbal
* Yaryna Hirniak