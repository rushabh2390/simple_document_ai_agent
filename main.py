import os
import re
import io
import shutil
import zipfile
from typing import List, Dict
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()
import os


#  CORRECT DOCLING V2+ IMPORTS
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.chunking import HybridChunker
from docling.datamodel.base_models import DocumentStream, InputFormat

# Lexical Search & LLM Interfaces
from rank_bm25 import BM25Okapi
from langchain_huggingface import HuggingFaceEndpoint

app = FastAPI(title="Production Docling Multi-File & Visual ZIP RAG Engine")

# Create directories for physical asset preservation
STATIC_IMAGE_DIR = "./static_images"
os.makedirs(STATIC_IMAGE_DIR, exist_ok=True)

# Mount the static directory so the frontend/Streamlit can fetch images via HTTP URLs
app.mount("/static", StaticFiles(directory=STATIC_IMAGE_DIR), name="static")

# 1. Instantiate the new master pipeline config object
pipeline_options = PipelineOptions()

# 2. Modify properties directly under the PDF namespace 
pipeline_options.pdf_options.images_scale = 1.0  # Captures figures/charts natively from PDFs
pipeline_options.pdf_options.do_table_structure = True 

# 3. Associate your settings using the InputFormat mapping
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: {"pipeline_options": pipeline_options}
    }
)


import logging
# Suppress the verbose transformers/huggingface module warning streams
logging.getLogger("transformers").setLevel(logging.ERROR)

# 2. Configure a Custom Structural Chunker
CUSTOM_CHUNK_SIZE_TOKENS = 400
docling_chunker = HybridChunker(max_tokens=CUSTOM_CHUNK_SIZE_TOKENS)

# 3. Cloud LLM Setup
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN","")
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    temperature=0.1,
    max_new_tokens=512
)

# Global in-memory indexes
CHUNKS_POOL: List[Dict[str, any]] = []
BM25_INDEX: BM25Okapi = None

def tokenize_text(text: str) -> List[str]:
    return re.sub(r"[^\w\s]", "", text.lower()).split()


def process_and_index_docling_result(conversion_result, source_name: str):
    """
    Extracts text chunks, extracts images, maps placeholders,
    and hot-reloads the global BM25 matrix.
    """
    global CHUNKS_POOL, BM25_INDEX
    
    doc = conversion_result.document
    
    # --- PHASE A: Standard Smart Structural Text Chunking ---
    markdown_text = doc.export_to_markdown()
    doc_chunks = docling_chunker.chunk(markdown_text)
    
    for chunk in doc_chunks:
        CHUNKS_POOL.append({
            "text": chunk.text,
            "source": source_name,
            "is_image": False,
            "image_url": None
        })
        
    # --- PHASE B: Visual Artifact Mapping & Extraction ---
    # Iterate through layout objects to isolate diagrams, charts, and figures
    for element, level in doc.iterate_items():
        if element.label in ["Pictures", "Figure", "Table"]:
            # Check if an image payload is physically attached
            if hasattr(element, "image") and element.image is not None:
                image_filename = f"extracted_{element.id}.png"
                image_save_path = os.path.join(STATIC_IMAGE_DIR, image_filename)
                
                # Write extracted visual binary directly to disk
                with open(image_save_path, "wb") as f:
                    f.write(element.image.bytes)
                
                # Extract caption text if available, or generate a standard descriptive placeholder
                caption_text = getattr(element, "caption", "Data visualization chart, map, or corporate diagram.")
                if hasattr(caption_text, "text"):
                    caption_text = caption_text.text
                
                # Construct a text metadata anchor for the BM25 algorithm to index
                image_text_representation = f"IMAGE_REFERENCE: {image_filename} | Caption/Content: {caption_text}"
                
                CHUNKS_POOL.append({
                    "text": image_text_representation,
                    "source": source_name,
                    "is_image": True,
                    "image_url": f"/static/{image_filename}"  # API URL reference path
                })
                
    # --- PHASE C: Global Search Matrix Update ---
    if CHUNKS_POOL:
        tokenized_corpus = [tokenize_text(item["text"]) for item in CHUNKS_POOL]
        BM25_INDEX = BM25Okapi(tokenized_corpus)


# --- API ENDPOINT: ACCEPTS INDIVIDUAL FILES OR ZIP CONTAINERS ---
@app.post("/vault/ingest-mixed", status_code=status.HTTP_201_CREATED)
async def ingest_mixed_files(files: List[UploadFile] = File(...)):
    supported_formats = (".pdf", ".docx", ".xlsx", ".pptx", ".csv", ".md", ".txt")
    processed_count = 0

    for file in files:
        file_ext = os.path.splitext(file.filename)[-1].lower()

        # CASE A: ZIP Archives
        if file_ext == ".zip":
            try:
                zip_bytes = await file.read()
                with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
                    for arch_file_name in archive.namelist():
                        if arch_file_name.startswith("__MACOSX") or os.path.basename(arch_file_name) == "":
                            continue
                        
                        arch_ext = os.path.splitext(arch_file_name)[-1].lower()
                        if arch_ext in supported_formats:
                            with archive.open(arch_file_name) as extracted_file:
                                file_bytes = extracted_file.read()
                                
                                # Convert directly via Docling from raw memory streams
                                result = converter.convert(io.BytesIO(file_bytes))
                                process_and_index_docling_result(result, os.path.basename(arch_file_name))
                                processed_count += 1
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to process ZIP archive '{file.filename}': {str(e)}")

        # CASE B: Individual Files
        elif file_ext in supported_formats:
            try:
                file_bytes = await file.read()
                result = converter.convert(io.BytesIO(file_bytes))
                process_and_index_docling_result(result, file.filename)
                processed_count += 1
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to parse '{file.filename}': {str(e)}")
        
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported format found: '{file.filename}'"
            )

    return {
        "status": "Success",
        "message": f"Successfully parsed and indexed {processed_count} files (including structural images).",
        "total_active_chunks": len(CHUNKS_POOL)
    }


# --- RAG QUERY ENDPOINT ---
class QueryModel(BaseModel):
    question: str

@app.post("/vault/ask")
async def ask_rag(payload: QueryModel):
    global CHUNKS_POOL, BM25_INDEX
    if not BM25_INDEX:
        raise HTTPException(status_code=400, detail="The index is empty. Please upload files.")
        
    tokenized_query = tokenize_text(payload.question)
    matched_objects = BM25_INDEX.get_top_n(tokenized_query, CHUNKS_POOL, n=3)
    
    # Synthesize plain context blocks for the LLM
    context_str = "\n\n".join([f"[Source: {m['source']}]\n{m['text']}" for m in matched_objects])
    
    llama3_prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"Answer the user query strictly utilizing the text snippets provided below. If you reference images or data from diagrams, state it clearly.\n\n"
        f"--- CONTEXT ---\n{context_str}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"{payload.question}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )
    
    llm_response = llm.invoke(llama3_prompt)
    
    # Identify any active images that matched the user's keywords
    images_returned = [
        {"source": m["source"], "url": m["image_url"]} 
        for m in matched_objects if m["is_image"]
    ]
    
    return {
        "answer": llm_response.strip(),
        "sources_used": list(set([m['source'] for m in matched_objects])),
        "context_images": images_returned  # Contains the relative paths for UI image rendering
    }