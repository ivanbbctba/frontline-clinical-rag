from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    GROK_API_KEY: str
    MODEL_NAME: str = "grok-beta"
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    TEMPERATURE: float = 0.0

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
