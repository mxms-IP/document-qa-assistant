# api.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel
import sys
import os
sys.path.append(os.path.dirname(__file__))
import app.src.rag as rag  
from pathlib import Path
from contextlib import asynccontextmanager

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = str(BASE_DIR / "data" / "sample.pdf")
UPLOAD_DIR = BASE_DIR / "data" / "uploads"  

# create pipeline dictionary to store model, chunks and embeddings
pipeline = {}

#purpose: use of --reload with uvicorn starts two process watcher and worker, they cause running the rag pipeline twice that is unnecessary
@asynccontextmanager
async def lifespan(app: FastAPI):
  
    print("[startup] Loading embedding model...")

    pipeline["embed_model"] = rag.load_embed_model()
    
    print("[startup] Embedding model ready.")
    yield
    # Anything after yield runs on shutdown 


app = FastAPI(lifespan=lifespan)\

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Defining request body
class AskRequest(BaseModel):
    question: str

# Defining response body
class AskResponse(BaseModel):
    answer: str


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    retrieved = rag.retrieve(request.question, pipeline["embed_model"], pipeline["index"], pipeline["chunks"])
    prompt = rag.build_prompt(request.question, retrieved)
    answer = rag.ask_llm(prompt)
    return AskResponse(answer=answer)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    # 1. Only accept PDFs
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    
    # 2. Save the file to data/uploads/
    save_path = UPLOAD_DIR / file.filename
    contents = await file.read()
    with open(save_path, "wb") as f:
        f.write(contents)
    print(f"[upload] Saved {file.filename}")

    # 3. Run the pipeline on the new file
    #    Notice: embed_model is REUSED from startup, not reloaded
    pages = rag.load_and_clean(str(save_path))
    chunks = rag.chunk_pages(pages)
    embeddings = rag.build_embeddings_from_model(chunks, pipeline["embed_model"])
    new_index = rag.build_index(embeddings)

    # 4. Replace the active index — now /ask searches the new document
    pipeline["index"] = new_index
    pipeline["chunks"] = chunks
    print(f"[upload] Index updated. {len(chunks)} chunks from {file.filename}")

    return {"message": "File uploaded and indexed.", "filename": file.filename, "chunks": len(chunks)}
