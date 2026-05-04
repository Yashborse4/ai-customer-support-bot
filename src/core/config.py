from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    """
    Application settings and environment variables.
    """
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # OpenAI Configuration
    OPENAI_API_KEY: str
    
    # Vector Store Configuration
    PERSIST_DIRECTORY: str = "./chroma_db"
    COLLECTION_NAME: str = "customer_support_kb"

    # Model Configuration
    MODEL_NAME: str = "gpt-4o"
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Logging
    LOG_LEVEL: str = "INFO"

settings = Settings()
