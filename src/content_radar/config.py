from urllib.parse import quote_plus, urlparse

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    SUPABASE_DATABASE_URL: str

    # --- Reddit ---
    reddit_client_id: str
    reddit_client_secret: str
    reddit_user_agent: str = "content_radar/0.1"

    # --- Embedding ---
    embedding_model: str = "all-MiniLM-L6-v2"

    # --- Pipeline ---
    fetch_lookback_hours: int = 24
    deep_summary_count: int = 10
    brief_summary_count: int = 20

    # --- LLM (Phase 2) ---
    llm_url: str = "http://localhost:11434/v1"
    llm_key: str = "ollama"
    llm_model: str


settings = Settings()
x = 0
