import functools
import os
import re
import uuid
from pathlib import Path
from typing import Annotated, Any

from dotenv import load_dotenv
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from langchain_core.runnables import RunnableConfig
from sqlalchemy.orm import Session

from agent.agent_engine import agent_app
from config.config import settings
from database.database import get_db
from database.database_manager import RAGDatabaseManager

# Hook directly into your modular core files
from document.document_parser import MultiModalDocumentParser
from models.models import IngestionJob
from schemas.schemas import AgentChatRequest

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

from config.config import logger

app = FastAPI(title="LangGraph Multimodal AI Agent Hub API")

# Shared storage path mapping inside Docker Container


@functools.lru_cache
def initialize_rag_services():
    parser = MultiModalDocumentParser(
        base_output_dir=settings.STATIC_ASSET_DIR, batch_size=3
    )
    db_manager = RAGDatabaseManager()
    return parser, db_manager


parser_engine, db_engine = initialize_rag_services()


def clean_deepseek_response(text: str) -> str:
    """Removes the internal <think>...</think> monologue blocks from DeepSeek output."""
    if not text:
        return ""
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned_text.strip()


def extract_final_agent_response(final_state) -> str:
    """
    Safely iterates backward through LangGraph state messages to find the TRUE final answer,
    ignoring ToolMessages, HumanMessages, or intermediate empty AIMessages.
    """
    if not final_state or "messages" not in final_state:
        return "I couldn't process that request properly. Please try again."

    messages = final_state.get("messages", [])

    # Iterate backward through messages
    for msg in reversed(messages):
        # 1. Skip ToolMessages / SystemMessages / HumanMessages
        msg_type = getattr(msg, "type", None) or (
            msg.get("type") if isinstance(msg, dict) else None
        )
        if msg_type in ["tool", "human", "system"]:
            continue

        # 2. Extract content
        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content", "")

        if isinstance(content, str):
            cleaned = clean_deepseek_response(content)
            # Ensure it's not empty and not just tool call structure
            if cleaned and not cleaned.startswith("{") and len(cleaned) > 2:
                return cleaned

    return "The agent executed tools but did not return a final written summary."


def process_batch_ingestion(
    job_id: int,
    file_batch: list[dict],
    chunk_size: int,
    chunk_overlap: int,
    db: Annotated[Session, Depends(get_db)],
):
    # Open a fresh DB session for the background thread

    try:
        total_files = len(file_batch)

        for idx, file_info in enumerate(file_batch, start=1):
            file_path = file_info["path"]
            filename = file_info["filename"]
            try:
                with open(file_path, "rb") as f:
                    file_bytes = f.read()

                # Pass chunk parameters into your parser
                chunks = parser_engine.parse_file(
                    file_bytes,
                    filename,
                    db=db,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )

                if chunks:
                    # Atomic database insertion
                    db_engine.insert_document_chunks(chunks, filename)
                    logger.info(f"Finished background indexing for {filename}")

            except Exception as e:
                logger.error(f"Failed background task for {filename}: {e!s}")
                job = (
                    db.query(IngestionJob).filter(IngestionJob.job_id == job_id).first()
                )
                if job:
                    job.status = "failed"
                    job.error_message = str(e)
                    db.commit()

            finally:
                # Always clean up temporary file from disk when done
                if os.path.exists(file_path):
                    os.remove(file_path)

            # Dynamically calculate progress %
            progress_pct = int((idx / total_files) * 90)  # Reserve 100% for final step

            job = db.query(IngestionJob).filter(IngestionJob.job_id == job_id).first()
            if job:
                job.step = f"Processed {idx}/{total_files} files: {filename}"
                job.progress = progress_pct
                db.commit()

            # Clean up temp file
            if os.path.exists(file_path):
                os.remove(file_path)

        # ALL FILES PROCESSED SUCCESSFULLY -> Set to Completed
        job = db.query(IngestionJob).filter(IngestionJob.job_id == job_id).first()
        if job:
            job.status = "completed"
            job.step = "All files ingested successfully"
            job.progress = 100
            db.commit()

    except Exception as e:
        # db.rollback()
        job = db.query(IngestionJob).filter(IngestionJob.job_id == job_id).first()
        if job:
            job.status = "failed"
            job.error_message = str(e)
            db.commit()

    # finally:
    #     db.close()


# 2. Ingestion Endpoint
@app.post("/vault/ingest-mixed", status_code=status.HTTP_202_ACCEPTED)
async def ingest_mixed_files(
    request: Request,
    background_tasks: BackgroundTasks,
    db: Annotated[Session, Depends(get_db)],
    files: list[UploadFile] = File(...),
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    """
    Accepts large files or ZIP archives, saves them to disk while monitoring disconnections,
    and forwards chunk_size and chunk_overlap to the background parsing job.
    """
    supported_formats = (".pdf", ".docx", ".xlsx", ".pptx", ".csv", ".md", ".txt")
    queued_files = []
    file_names = ",".join([file.filename for file in files])
    job = IngestionJob(uploaded_files=file_names, status="submitted")
    db.add(job)
    db.commit()
    db.refresh(job)
    job_id = job.job_id
    file_batch = []
    try:
        for file in files:
            file_ext = os.path.splitext(file.filename)[-1].lower()
            if file_ext not in supported_formats and file_ext != ".zip":
                job.status = "failed"
                job.error_message = f"Unsupported file format: {file.filename}"
                db.commit()
                raise HTTPException(
                    status_code=400, detail=f"Unsupported format: '{file.filename}'"
                )
            temp_file_id = f"{uuid.uuid4()}_{file.filename}"
            temp_path = os.path.join(settings.TEMP_UPLOAD_DIR, temp_file_id)

            try:
                # Stream disk writes (Phase 1)
                with open(temp_path, "wb") as buffer:
                    while chunk := await file.read(1024 * 1024):  # 1MB blocks
                        if await request.is_disconnected():
                            logger.warning(
                                f"Client disconnected during upload of {file.filename}"
                            )
                            buffer.close()
                            if os.path.exists(temp_path):
                                os.remove(temp_path)

                            # Update job as failed in DB
                            job.status = "failed"
                            job.error_message = (
                                f"Client disconnected during {file.filename} upload."
                            )
                            db.commit()

                            return {
                                "status": "Aborted",
                                "message": f"Client disconnected during {file.filename} upload.",
                            }
                        buffer.write(chunk)
                file_batch.append({"path": temp_path, "filename": file.filename})
            except Exception as e:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

                # Update job as failed in DB
                job.status = "failed"
                job.error_message = str(e)
                db.commit()
                raise e

            # UPDATE JOB STATUS TO 'PROCESSING' ON FIRST VALID FILE

            job.status = "processing"
            job.step = f"Processing {len(file_batch)} file(s)"
            job.progress = 10
            db.commit()
            # Queue ONE background task for the entire batch
            background_tasks.add_task(
                process_batch_ingestion,
                job_id,
                file_batch,
                chunk_size,
                chunk_overlap,
                db,
            )
        return {
            "job_id": job_id,
            "status": "Processing",
            "message": f"{len(queued_files)} file(s) accepted for background ingestion.",
            "files": queued_files,
            "config": {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
        }
    except Exception as e:
        logger.error(f"Error during agent execution: {e!s}")
        raise HTTPException(status_code=500, detail=f"Agent Execution Failed: {e!s}")
    # finally:
    #     # Safely close DB connection at the very end of the endpoint handler
    #     db.close()


@app.post("/vault/chat")
async def chat_with_agent(payload: AgentChatRequest):
    """Executes the full LangGraph Multimodal Agent loop and returns response + inspected node assets."""
    langgraph_messages = []
    for m in payload.messages:
        if m.content and m.content.strip():
            role_tag = "user" if m.role == "user" else "assistant"
            langgraph_messages.append((role_tag, m.content))

    if not langgraph_messages:
        raise HTTPException(status_code=400, detail="No valid messages provided.")

    node_holder: list[dict[str, Any]] = []
    agent_config: RunnableConfig = {
        "configurable": {
            "db_manager": db_engine,
            "retrieval_limit": payload.retrieval_k,
            "ollama_base_url": ollama_base_url,
            "temperature": payload.temperature,
            "top_k": payload.top_k,
            "shared_node_container": node_holder,
        }
    }

    try:
        stream_generator = agent_app.stream(
            {"messages": langgraph_messages}, config=agent_config, stream_mode="values"
        )

        final_state = None
        for chunk in stream_generator:
            final_state = chunk

        final_answer = extract_final_agent_response(final_state)

        return {
            "role": "assistant",
            "content": final_answer,
            "inspected_nodes": node_holder,  # Visual tables/images retrieved by tools
        }

    except Exception as e:
        logger.error(f"Error during agent execution: {e!s}")
        raise HTTPException(status_code=500, detail=f"Agent Execution Failed: {e!s}")


@app.post("/vault/clear")
async def clear_database():
    """Wipes the SQLite database and flushes cached files."""
    try:
        db_engine.wipe_all_data()
        tabular_db_path = os.path.join(
            settings.STATIC_ASSET_DIR, "dynamic_tabular_data.db"
        )
        if os.path.exists(tabular_db_path):
            os.remove(tabular_db_path)
        return {
            "status": "Success",
            "message": "All database and tabular states cleared successfully.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear database: {e!s}")


@app.get("/job/status/id")
async def get_job_status_by_id(id: int):
    pass
