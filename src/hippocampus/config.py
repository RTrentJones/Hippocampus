"""Configuration for Hippocampus."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Database
    database_url: str = "postgresql://postgres:postgres@localhost:5432/postgres"

    # Kafka (pg_kafka)
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_consumer_group: str = "hippocampus"
    kafka_topic_pattern: str = "decisions.*"

    # Embeddings
    embedding_provider: str = "openai"  # "openai" or "local"
    embedding_model: str = "text-embedding-3-small"  # OpenAI model
    embedding_dimensions: int = 1536
    openai_api_key: str = ""

    # Local embeddings (when embedding_provider = "local")
    local_model_name: str = "all-MiniLM-L6-v2"

    # Consumer
    batch_size: int = 100
    batch_timeout_ms: int = 500

    # MCP Server
    mcp_server_name: str = "hippocampus"


settings = Settings()
