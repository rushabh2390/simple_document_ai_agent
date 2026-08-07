import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text

from database.database import Base


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    chunk_id = Column(String, primary_key=True, index=True)
    filename = Column(String, nullable=True)
    raw_text = Column(Text, nullable=True)
    table_path = Column(String, nullable=True)
    image_path = Column(String, nullable=True)


class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    job_id = Column(Integer, primary_key=True, index=True)
    uploaded_files = Column(Text, nullable=False)
    status = Column(
        String(20), nullable=False, default="submitted"
    )  # submitted, processing, completed, failed
    step = Column(String(255), nullable=True)
    progress = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    uploaded_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(
        DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow
    )
