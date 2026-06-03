# rag.py

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import re


# ── STEP 1: Load and clean ────────────────────────────────────────────────────

def clean_text(text):
    """Fix OCR artifacts from scanned PDFs."""
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    text = re.sub(r'\bth\s+(at|ose|e|eir|ere)\b', r'th\1', text)
    text = re.sub(r'(?<=[a-z])\s+\.\s+(?=[a-z])', ', ', text)
    text = re.sub(r'\s+\.\s+', '. ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def load_and_clean(pdf_path):
    """Load a PDF and return cleaned pages."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    for page in pages:
        page.page_content = clean_text(page.page_content)
    print(f"[load] Loaded {len(pages)} pages from {pdf_path}")
    return pages


# ── STEP 2: Chunk ─────────────────────────────────────────────────────────────

def chunk_pages(pages):
    """Split pages into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-3.5-turbo",
        chunk_size=150,
        chunk_overlap=30,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
    )
    chunks = splitter.split_documents(pages)
    print(f"[chunk] Created {len(chunks)} chunks")
    return chunks


# ── STEP 3: Embed ─────────────────────────────────────────────────────────────

def build_embeddings(chunks):
    """Embed all chunks. Returns the model and the embedding array."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [chunk.page_content for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    print(f"[embed] Shape: {embeddings.shape}")
    return model, embeddings, texts


# ── STEP 4: Index ─────────────────────────────────────────────────────────────

def build_index(embeddings):
    """Store embeddings in a FAISS index for fast search."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype("float32"))
    print(f"[index] Stored {index.ntotal} vectors")
    return index


# ── STEP 5: Retrieve ──────────────────────────────────────────────────────────

def retrieve(question, embed_model, index, chunks, k=3):
    """Find the k most relevant chunks for a question."""
    q_vector = embed_model.encode([question]).astype("float32")
    distances, indices_found = index.search(q_vector, k=k)

    results = []
    for rank, idx in enumerate(indices_found[0]):
        dist = distances[0][rank]
        text = chunks[idx].page_content

        if dist < 0.8:
            confidence = "HIGH"
        elif dist < 1.2:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        results.append({"text": text, "distance": dist, "confidence": confidence})

    return results


# ── STEP 6: Build prompt ──────────────────────────────────────────────────────

def build_prompt(question, retrieved):
    """Assemble the context + question into a prompt string."""
    context = "\n\n".join([r["text"] for r in retrieved])
    prompt = f"""You are a helpful assistant. You must follow these rules strictly:
RULE 1: Use ONLY the context below to answer. Do not add outside knowledge.
RULE 2: If the context does not contain enough information, respond with exactly: "The document does not answer this question."
RULE 3: Keep your answer to 2-3 sentences maximum.

Context:
{context}

Question: {question}
Answer (2-3 sentences, context only):"""
    return prompt


# ── STEP 7: Ask the LLM ───────────────────────────────────────────────────────

def ask_llm(prompt, model_name="llama3.2:3b"):
    """Send prompt to Ollama and return the answer string."""
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model_name, "prompt": prompt, "stream": False}
    )
    return response.json()["response"]


# ── MAIN: Run the full pipeline ───────────────────────────────────────────────

# This block only runs when you execute: python rag.py
# It does NOT run when another file does: import rag
# That distinction matters in Stage 3 when api.py imports from this file.

if __name__ == "__main__":
    PDF_PATH = "data/sample.pdf"
    QUESTION = "What makes kangaroos special?"

    pages = load_and_clean(PDF_PATH)
    chunks = chunk_pages(pages)
    embed_model, embeddings, texts = build_embeddings(chunks)
    index = build_index(embeddings)

    retrieved = retrieve(QUESTION, embed_model, index, chunks)
    
    print(f"\nQuestion: {QUESTION}\n")
    for r in retrieved:
        print(f"[{r['confidence']} | {r['distance']:.3f}] {r['text'][:80]}...")
    
    prompt = build_prompt(QUESTION, retrieved)
    answer = ask_llm(prompt)
    
    print(f"\nAnswer: {answer}")