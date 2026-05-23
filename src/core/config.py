from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    """
    Application settings and environment variables for local-first operations.
    """
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Local Offline Connection Configurations (Ollama / LocalAI / LM Studio)
    LOCAL_LLM_BASE_URL: str = "http://localhost:11434/v1"
    LOCAL_EMBEDDING_BASE_URL: str = "http://localhost:11434/v1"

    # Swappable Model Identifiers
    MODEL_NAME: str = "qwen2.5"
    EMBEDDING_MODEL: str = "nomic-embed-text"
    
    # Vector Store Configuration
    PERSIST_DIRECTORY: str = "./chroma_db"
    COLLECTION_NAME: str = "customer_support_kb"
    VECTOR_DB_TYPE: str = "qdrant" # "qdrant" or "faiss"
    QDRANT_PATH: str = "./qdrant_db"
    QDRANT_URL: Optional[str] = None
    FAISS_INDEX_PATH: str = "./faiss_db"
    SQL_DATABASE_PATH: str = "data/support_records.db"

    # Logging
    LOG_LEVEL: str = "INFO"

settings = Settings()
