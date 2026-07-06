import os
import io
import logging
import zipfile
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Hook directly into your modular core files
from document_parser import MultiModalDocumentParser
from database_manager import RAGDatabaseManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("RAGBackend")

app = FastAPI(title="Multi-Modal Context RAG Engine API")

# Shared storage path mapping inside Docker Container
STATIC_ASSET_DIR = "/app/processed_data"
os.makedirs(STATIC_ASSET_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_ASSET_DIR), name="static")

# Core Engine Layer Initialization
parser_engine = MultiModalDocumentParser(base_output_dir=STATIC_ASSET_DIR, batch_size=3)
db_engine = RAGDatabaseManager(db_path=os.path.join(STATIC_ASSET_DIR, "rag_storage.db"))

class QueryRequest(BaseModel):
    question: str
    limit: Optional[int] = 3

@app.post("/vault/ingest-mixed", status_code=status.HTTP_201_CREATED)
async def ingest_mixed_files(files: List[UploadFile] = File(...)):
    """Accepts individual files or zipped documents, parsing them sequentially."""
    supported_formats = (".pdf", ".docx", ".xlsx", ".pptx", ".csv", ".md", ".txt")
    processed_count = 0

    for file in files:
        file_ext = os.path.splitext(file.filename)[-1].lower()

        if file_ext == ".zip":
            try:
                zip_bytes = await file.read()
                with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
                    for arch_name in archive.namelist():
                        if arch_name.startswith("__MACOSX") or os.path.basename(arch_name) == "":
                            continue
                        
                        arch_ext = os.path.splitext(arch_name)[-1].lower()
                        if arch_ext in supported_formats:
                            with archive.open(arch_name) as extracted_file:
                                file_bytes = extracted_file.read()
                                chunks = parser_engine.parse_file(file_bytes, os.path.basename(arch_name))
                                if chunks:
                                    db_engine.insert_document_chunks(chunks, os.path.basename(arch_name))
                                    processed_count += 1
            except Exception as e:
                logger.error(f"Failed to process ZIP archive: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Failed to process ZIP archive: {str(e)}")

        elif file_ext in supported_formats:
            try:
                file_bytes = await file.read()
                chunks = parser_engine.parse_file(file_bytes, file.filename)
                if chunks:
                    db_engine.insert_document_chunks(chunks, file.filename)
                    processed_count += 1
            except Exception as e:
                logger.error(f"Failed to parse '{file.filename}': {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to parse '{file.filename}': {str(e)}")
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: '{file.filename}'")

    return {
        "status": "Success",
        "message": f"Successfully parsed and indexed {processed_count} files."
    }

@app.post("/vault/ask")
async def ask_rag(payload: QueryRequest):
    """Executes FTS5 sub-query context filtering combined with remote Ollama inference."""
    from langchain_ollama import ChatOllama
    
    # 1. Retrieve text nodes alongside page-level asset fallback structures
    matched_nodes = db_engine.search_bm25(payload.question, limit=payload.limit)
    
    if not matched_nodes:
        return {"answer": "No matching contextual elements found in knowledge matrix.", "matches": []}
        
    # 2. Build Llama-3 instruction context string blocks
    context_str = "\n\n".join([f"[Source: {m['filename']}]\n{m['text']}" for m in matched_nodes])
    
    llama_prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"Answer the user query strictly utilizing the text snippets provided below.\n\n"
        f"--- CONTEXT ---\n{context_str}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"{payload.question}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )
    
    try:
        ollama_endpoint = os.getenv("OLLAMA_BASE_URL", "http://ollama_service:11434")
        llm = ChatOllama(base_url=ollama_endpoint, model="llama3.2", temperature=0.1)
        response = llm.invoke(llama_prompt)
        answer = response.content.strip() if hasattr(response, 'content') else str(response).strip()
    except Exception as e:
        logger.error(f"Ollama network linkage connection failed: {str(e)}")
        answer = f"[Fallback: Context Loaded, but Ollama container unreachable at {ollama_endpoint}]: {str(e)}"

    return {
        "answer": answer,
        "matches": matched_nodes
    }

@app.post("/vault/clear")
async def clear_database():
    db_engine.wipe_all_data()
    return {"status": "Success", "message": "Database wiped successfully."}