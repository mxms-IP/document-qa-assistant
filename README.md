---
title: SQA Assistant
emoji: 📄
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---


# SQA Assistant

> A retrieval-augmented generation (RAG) application for software quality assurance professionals. Upload your test plans, defect reports, and requirements specs — then ask questions in plain English.

**[Live Demo](https://mxms-ip.github.io/document-qa-assistant/)** 

<img width="1440" height="773" alt="image" src="https://github.com/user-attachments/assets/7f35f5d9-904f-41c8-9956-988e494f8876" />

---

## What it does

SQA professionals spend hours searching through PDFs for answers — "which test cases cover the login module?", "what's the severity of this known defect?", "does our test plan address performance requirements?". This assistant answers those questions instantly, with citations.

Upload multiple documents. Ask anything. Get answers with exact source and page number.

---git ls-files

## How it works

```
PDF documents
     │
     ▼
OCR cleanup + chunking (150 tokens, 30 overlap)
     │
     ▼
Sentence embeddings (all-MiniLM-L6-v2)
     │
     ▼
FAISS vector index (IndexFlatL2)
     │
  question ──► embed question ──► similarity search
                                        │
                                        ▼
                              top-k chunks + confidence score
                                        │
                                        ▼
                              prompt with strict context rules
                                        │
                                        ▼
                                   LLM answer + source citations
```

The LLM is **instructed to refuse** answering from outside the uploaded documents. If the context doesn't contain the answer, it says so — no hallucination.

---

## Tech stack

| Layer | Technology | Why |
|---|---|---|
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` | Fast, local, no API cost |
| Vector search | FAISS `IndexFlatL2` | In-memory, zero infrastructure |
| PDF loading | LangChain `PyPDFLoader` | Metadata (filename, page) attached automatically |
| Chunking | `RecursiveCharacterTextSplitter` with tiktoken | Semantic boundaries, token-aware overlap |
| API | FastAPI + Pydantic | Auto-validation, auto-docs at `/docs` |
| LLM | Google Gemini Flash | Fast, free tier, no local GPU needed |
| Frontend | Vanilla HTML/CSS/JS | Zero dependencies, instant load |

---

## Project structure

```
sqa-assistant/
├── src/
│   ├── rag.py              # Full RAG pipeline — load, chunk, embed, index, retrieve, prompt
│   ├── api.py              # FastAPI — /ask and /upload endpoints
│   └── frontend/
│       └── index.html      # Single-file UI — upload panel, chat interface, source citations
├── data/
│   ├── sample.pdf          # Example document for testing
│   └── uploads/            # Uploaded documents stored here
├── requirements.txt
└── README.md
```

---

## Getting started

### Prerequisites

- Python 3.10+
- Gemini API key — free at [aistudio.google.com](https://aistudio.google.com)

### Install

```bash
git clone https://github.com/yourusername/sqa-assistant.git
cd sqa-assistant

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Configure

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_key_here
```

### Run

```bash
uvicorn app.src.api:app --reload
```

Open `http://localhost:8000/home` in your browser.

API docs available at `http://localhost:8000/docs`.

---

## API reference

### `POST /upload`

Upload one or more PDF documents. Builds a FAISS index across all files.

```bash
curl -X POST http://localhost:8000/upload \
  -F "files=@testplan.pdf" \
  -F "files=@defects.pdf"
```

Response:
```json
{
  "documents": [
    { "filename": "testplan.pdf", "chunks": 43 },
    { "filename": "defects.pdf",  "chunks": 21 }
  ]
}
```

### `POST /ask`

Ask a question against the indexed documents.

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Which test cases cover the authentication module?"}'
```

Response:
```json
{
  "answer": "Test cases TC-014 through TC-019 cover the authentication module...",
  "metadata": [
    { "source": "testplan.pdf", "page": 4 },
    { "source": "testplan.pdf", "page": 7 }
  ]
}
```

---

## Docker

```bash
docker-compose up
```


## Author

**[mxms-IP]**
