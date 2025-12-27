from pydantic_settings import BaseSettings
import os

class QdrantSettings(BaseSettings):
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    qdrant_collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "textbook_content")

    class Config:
        env_file = ".env"

qdrant_settings = QdrantSettings()