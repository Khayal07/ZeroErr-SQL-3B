from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="ZEROERR_", extra="ignore")

    app_name: str = "ZeroErr-SQL-3B"

    engine_backend: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "zeroerr:3b"

    sandbox_engine: str = "sqlite"
    sandbox_data_dir: str = "data/databases"
    sandbox_timeout_seconds: int = 5
    sandbox_max_rows: int = 100

    max_retry_rounds: int = 3
    results_dir: str = "eval/results"


settings = Settings()