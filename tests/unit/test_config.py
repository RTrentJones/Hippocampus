"""Unit tests for configuration module."""

import os
from unittest.mock import patch

import pytest


@pytest.mark.unit
class TestSettings:
    """Tests for Settings configuration."""

    def test_default_settings(self):
        """Test that default settings have sensible values."""
        from hippocampus.config import Settings

        # Create settings with no env vars
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)

        assert settings.database_url == "postgresql://postgres:postgres@localhost:5432/postgres"
        assert settings.kafka_bootstrap_servers == "localhost:9092"
        assert settings.kafka_consumer_group == "hippocampus"
        assert settings.kafka_topic_pattern == "decisions.*"
        assert settings.embedding_provider == "openai"
        assert settings.embedding_model == "text-embedding-3-small"
        assert settings.embedding_dimensions == 1536
        assert settings.batch_size == 100
        assert settings.batch_timeout_ms == 500
        assert settings.mcp_server_name == "hippocampus"

    def test_settings_from_env(self):
        """Test that environment variables override defaults."""
        from hippocampus.config import Settings

        env_vars = {
            "DATABASE_URL": "postgresql://test:test@testhost:5433/testdb",
            "KAFKA_BOOTSTRAP_SERVERS": "kafka:9093",
            "KAFKA_CONSUMER_GROUP": "test-group",
            "KAFKA_TOPIC_PATTERN": "test.*",
            "EMBEDDING_PROVIDER": "local",
            "EMBEDDING_MODEL": "text-embedding-3-large",
            "EMBEDDING_DIMENSIONS": "3072",
            "OPENAI_API_KEY": "sk-test-key",
            "LOCAL_MODEL_NAME": "paraphrase-MiniLM-L6-v2",
            "BATCH_SIZE": "50",
            "BATCH_TIMEOUT_MS": "250",
            "MCP_SERVER_NAME": "test-server",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings(_env_file=None)

        assert settings.database_url == "postgresql://test:test@testhost:5433/testdb"
        assert settings.kafka_bootstrap_servers == "kafka:9093"
        assert settings.kafka_consumer_group == "test-group"
        assert settings.kafka_topic_pattern == "test.*"
        assert settings.embedding_provider == "local"
        assert settings.embedding_model == "text-embedding-3-large"
        assert settings.embedding_dimensions == 3072
        assert settings.openai_api_key == "sk-test-key"
        assert settings.local_model_name == "paraphrase-MiniLM-L6-v2"
        assert settings.batch_size == 50
        assert settings.batch_timeout_ms == 250
        assert settings.mcp_server_name == "test-server"

    def test_settings_partial_override(self):
        """Test that partial env vars work correctly."""
        from hippocampus.config import Settings

        env_vars = {
            "DATABASE_URL": "postgresql://custom:custom@localhost:5432/custom",
            "EMBEDDING_PROVIDER": "mock",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings(_env_file=None)

        # Overridden
        assert settings.database_url == "postgresql://custom:custom@localhost:5432/custom"
        assert settings.embedding_provider == "mock"
        # Defaults
        assert settings.kafka_consumer_group == "hippocampus"
        assert settings.batch_size == 100

    def test_settings_numeric_conversion(self):
        """Test that numeric env vars are converted correctly."""
        from hippocampus.config import Settings

        env_vars = {
            "EMBEDDING_DIMENSIONS": "768",
            "BATCH_SIZE": "200",
            "BATCH_TIMEOUT_MS": "1000",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings(_env_file=None)

        assert settings.embedding_dimensions == 768
        assert isinstance(settings.embedding_dimensions, int)
        assert settings.batch_size == 200
        assert isinstance(settings.batch_size, int)
        assert settings.batch_timeout_ms == 1000
        assert isinstance(settings.batch_timeout_ms, int)


@pytest.mark.unit
class TestGlobalSettings:
    """Tests for the global settings singleton."""

    def test_global_settings_exists(self):
        """Test that global settings object is available."""
        from hippocampus.config import settings

        assert settings is not None
        assert hasattr(settings, "database_url")
        assert hasattr(settings, "embedding_provider")
