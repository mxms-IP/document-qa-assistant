# api.py

from fastapi import FastAPI
from pydantic import BaseModel
import rag  

# create object
app = FastAPI()

# Defining request body
class AskRequest(BaseModel):
    question: str

# Defining response body
class AskResponse(BaseModel):
    answer: str

# RAG PIPELINE (PHASE 1)
print("[startup] Loading pipeline...")

pages = rag.load_and_clean("..\data\sample.pdf")
chunks = rag.chunk_pages(pages)
embed_model, embeddings, texts = rag.build_embeddings(chunks)
index = rag.build_index(embeddings)

print("[startup] Pipeline ready.")


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    retrieved = rag.retrieve(request.question, embed_model, index, chunks)
    prompt = rag.build_prompt(request.question, retrieved)
    answer = rag.ask_llm(prompt)
    return AskResponse(answer=answer)