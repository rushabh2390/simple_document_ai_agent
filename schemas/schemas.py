from datetime import datetime

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class AgentChatRequest(BaseModel):
    messages: list[ChatMessage]
    retrieval_k: int | None = 3
    temperature: float | None = 0.1
    top_k: int | None = 40


# Schema for creating a job (request body)
class JobCreate(BaseModel):
    uploaded_files: list[str]  # e.g., ["file1.pdf", "file2.docx"]


# Schema for API responses
class JobResponse(BaseModel):
    job_id: int
    uploaded_files: str
    status: str
    step: str | None = None
    progress: int
    error_message: str | None = None
    uploaded_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True  # Allows Pydantic to parse ORM models directly
