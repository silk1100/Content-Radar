from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    # --- Supabase ---
    supabase_url: str
    supabase_key: str

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
    model_key: str
    llm_model: str

settings = Settings()

