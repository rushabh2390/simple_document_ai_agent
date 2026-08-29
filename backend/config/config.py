import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# 1. Resolve path hierarchy relative to config.py
CONFIG_DIR = Path(__file__).resolve().parent  # backend/config
BACKEND_DIR = CONFIG_DIR.parent  # backend
PROJECT_ROOT = BACKEND_DIR.parent  # my_project (Root)

# 2. Load .env files from project locations
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(BACKEND_DIR / ".env")


def resolve_system_path(env_var_name: str, default_relative: str) -> Path:
    """Resolves system paths cleanly.

    - If path is Absolute (e.g. Docker '/app/backend/processed_data'), returns as-is.
    - If path is Relative, anchors it directly inside BACKEND_DIR (backend/).
    """
    raw_val = os.getenv(env_var_name, default_relative)

    # Strip sqlite:/// prefix if passed inside DATABASE_URL
    if raw_val.startswith("sqlite:///"):
        raw_val = raw_val.replace("sqlite:///", "")

    path = Path(raw_val)

    if path.is_absolute():
        return path

    # Anchors relative paths strictly to backend/
    return (BACKEND_DIR / path).resolve()


class Settings:
    STATIC_ASSET_PATH: Path = resolve_system_path("STATIC_ASSET_DIR", "processed_data")
    TEMP_UPLOAD_PATH: Path = resolve_system_path(
        "TEMP_UPLOAD_DIR", "temp_upload_directory"
    )
    RAW_DB_PATH: Path = resolve_system_path(
        "DATABASE_URL", "processed_data/rag_storage.db"
    )

    # String exports for application usage
    STATIC_ASSET_DIR: str = str(STATIC_ASSET_PATH)
    TEMP_UPLOAD_DIR: str = str(TEMP_UPLOAD_PATH)
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
    LANGCHAIN_ENDPOINT = os.getenv(
        "LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"
    )
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "your-api-key")
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "document-ai-agent")

    @property
    def DATABASE_URL(self) -> str:
        """Formats absolute file path into a valid SQLAlchemy SQLite URI."""
        return f"sqlite:///{self.RAW_DB_PATH.as_posix()}"


settings = Settings()

# Ensure required system directories exist on startup inside backend/
os.makedirs(settings.STATIC_ASSET_DIR, exist_ok=True)
os.makedirs(settings.TEMP_UPLOAD_DIR, exist_ok=True)
settings.RAW_DB_PATH.parent.mkdir(parents=True, exist_ok=True)


handler = logging.StreamHandler()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

logger = logging.getLogger("LangGraph Multimodal AI Agent Hub")
