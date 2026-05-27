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

    # Relational Database Configuration
    DB_TYPE: str = "sqlite"  # "sqlite" or "oracle"
    SQLITE_DB_PATH: str = "data/customer_support.db"

    # Oracle 11g Relational Database Configuration
    DB_USER: str = "system"
    DB_PASSWORD: str = "oracle"
    DB_HOST: str = "localhost"
    DB_PORT: int = 1521
    DB_SERVICE_NAME: str = "xe"

    # Logging
    LOG_LEVEL: str = "INFO"

settings = Settings()

# Dynamic model and database configurations loading
import json
import os

# Create data directory if not exists
os.makedirs("data", exist_ok=True)

model_config_path = os.path.join("data", "model_config.json")
if os.path.exists(model_config_path):
    try:
        with open(model_config_path, "r", encoding="utf-8") as f:
            m_config = json.load(f)
            if "MODEL_NAME" in m_config:
                settings.MODEL_NAME = m_config["MODEL_NAME"]
            if "EMBEDDING_MODEL" in m_config:
                settings.EMBEDDING_MODEL = m_config["EMBEDDING_MODEL"]
            if "LOCAL_LLM_BASE_URL" in m_config:
                settings.LOCAL_LLM_BASE_URL = m_config["LOCAL_LLM_BASE_URL"]
            if "LOCAL_EMBEDDING_BASE_URL" in m_config:
                settings.LOCAL_EMBEDDING_BASE_URL = m_config["LOCAL_EMBEDDING_BASE_URL"]
            if "VECTOR_DB_TYPE" in m_config:
                settings.VECTOR_DB_TYPE = m_config["VECTOR_DB_TYPE"]
    except Exception:
        pass
