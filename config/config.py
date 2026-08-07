import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class Settings:
    STATIC_ASSET_DIR: str = os.getenv("STATIC_ASSET_DIR", "./processed_data")
    TEMP_UPLOAD_DIR: str = os.getenv("TEMP_UPLOAD_DIR", "./temp_upload_directory")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Raw database path from environment (e.g., ./processed_data/rag_storage.db)
    RAW_DB_PATH: str = os.getenv("DATABASE_URL", "./processed_data/rag_storage.db")

    @property
    def DATABASE_URL(self) -> str:
        """Ensures the database path is formatted properly for SQLAlchemy engines."""
        if self.RAW_DB_PATH.startswith("sqlite:///") or "://" in self.RAW_DB_PATH:
            return self.RAW_DB_PATH
        # Convert local relative/absolute file path to SQLAlchemy SQLite URI
        return f"sqlite:///{Path(self.RAW_DB_PATH).resolve().as_posix()}"


settings = Settings()

# Ensure required system directories exist on startup
os.makedirs(settings.STATIC_ASSET_DIR, exist_ok=True)
os.makedirs(settings.TEMP_UPLOAD_DIR, exist_ok=True)
Path(settings.RAW_DB_PATH).parent.mkdir(parents=True, exist_ok=True)


class ClientIPFilter(logging.Filter):
    """Injects a default 'clientip' into log records if missing."""

    def filter(self, record):
        if not hasattr(record, "clientip"):
            record.clientip = "N/A"
        return True


# 1. Create your handler
handler = logging.StreamHandler()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

logger = logging.getLogger("LangGraph Multimodal AI Agent Hub")
