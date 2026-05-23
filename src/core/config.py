from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    """
    Application settings and environment variables.
    """
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = None
    
    # Provider Settings
    LLM_PROVIDER: str = "openai" # "openai" or "local"
    EMBEDDING_PROVIDER: str = "openai" # "openai" or "local"
    
    # Local Offline Model Configuration (Ollama / LocalAI / LM Studio)
    LOCAL_LLM_BASE_URL: str = "http://localhost:11434/v1"
    LOCAL_MODEL_NAME: str = "llama3"
    LOCAL_EMBEDDING_BASE_URL: str = "http://localhost:11434/v1"
    LOCAL_EMBEDDING_MODEL: str = "nomic-embed-text"
    
    # Vector Store Configuration
    PERSIST_DIRECTORY: str = "./chroma_db"
    COLLECTION_NAME: str = "customer_support_kb"
    VECTOR_DB_TYPE: str = "qdrant" # "qdrant" or "faiss"
    QDRANT_PATH: str = "./qdrant_db"
    QDRANT_URL: Optional[str] = None
    FAISS_INDEX_PATH: str = "./faiss_db"
    SQL_DATABASE_PATH: str = "data/support_records.db"

    # Model Configuration (OpenAI Defaults)
    MODEL_NAME: str = "gpt-4o"
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Logging
    LOG_LEVEL: str = "INFO"

settings = Settings()
