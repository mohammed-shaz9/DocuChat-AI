from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()


def _split_origins(origins: str) -> List[str]:
    if not origins:
        return []
    return [o.strip() for o in origins.split(",") if o.strip()]


class Config:
    """Centralized configuration for the application.

    Required keys:
      - OPENAI_API_KEY
      - FLASK_SECRET_KEY

    Optional keys with defaults:
      - FLASK_ENV (default: "production")
      - PORT (default: 5000)
      - CORS_ORIGINS (comma-separated list)
      - CHUNK_SIZE (default: 800)
      - CHUNK_OVERLAP (default: 100)
      - MAX_CONVERSATION_HISTORY (default: 5)
      - EMBEDDING_MODEL (default: "text-embedding-3-small")
      - CHAT_MODEL (default: "gpt-4o-mini")
      - CHROMA_PERSIST_DIR (default: "./chroma_db")
      - OPENAI_BASE_URL (default: "https://openrouter.ai/api/v1")
      - EMBEDDING_PROVIDER (default: "openai")
      - HF_EMBEDDING_MODEL (default: "all-MiniLM-L6-v2")
    """

    # Required
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    FLASK_SECRET_KEY: str = os.getenv("FLASK_SECRET_KEY", "")

    # Optional with defaults
    FLASK_ENV: str = os.getenv("FLASK_ENV", "production")
    PORT: int = int(os.getenv("PORT", "5000"))
    CORS_ORIGINS: List[str] = _split_origins(os.getenv("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    MAX_CONVERSATION_HISTORY: int = int(os.getenv("MAX_CONVERSATION_HISTORY", "5"))

    # Models
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    CHAT_MODEL: str = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "openai")
    HF_EMBEDDING_MODEL: str = os.getenv("HF_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Vector backend abstraction
    VECTOR_BACKEND: str = os.getenv("VECTOR_BACKEND", "chroma")

    # Storage
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

    # Base URL for OpenAI-compatible API (supports OpenRouter)
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

    @classmethod
    def validate(cls) -> None:
        """Validate required configuration keys.

        Raises:
            ValueError: if required keys are explicitly set in env but empty.
        """
        # Only enforce when the environment variable is present but empty.
        if ("OPENAI_API_KEY" in os.environ) and (not cls.OPENAI_API_KEY):
            raise ValueError("OPENAI_API_KEY is missing or empty. Please set it in your environment or .env file.")
        if ("FLASK_SECRET_KEY" in os.environ) and (not cls.FLASK_SECRET_KEY):
            raise ValueError("FLASK_SECRET_KEY is missing or empty. Please set it in your environment or .env file.")

    @classmethod
    def __repr__(cls) -> str:
        # Hide sensitive values in representation
        masked_key = (cls.OPENAI_API_KEY[:6] + "***") if cls.OPENAI_API_KEY else "<unset>"
        masked_secret = (cls.FLASK_SECRET_KEY[:6] + "***") if cls.FLASK_SECRET_KEY else "<unset>"
        return (
            "Config("
            f"OPENAI_API_KEY={masked_key}, "
            f"FLASK_SECRET_KEY={masked_secret}, "
            f"FLASK_ENV={cls.FLASK_ENV}, PORT={cls.PORT}, "
            f"CORS_ORIGINS={cls.CORS_ORIGINS}, CHUNK_SIZE={cls.CHUNK_SIZE}, CHUNK_OVERLAP={cls.CHUNK_OVERLAP}, "
            f"MAX_CONVERSATION_HISTORY={cls.MAX_CONVERSATION_HISTORY}, EMBEDDING_MODEL={cls.EMBEDDING_MODEL}, "
            f"CHAT_MODEL={cls.CHAT_MODEL}, CHROMA_PERSIST_DIR={cls.CHROMA_PERSIST_DIR}, "
            f"EMBEDDING_PROVIDER={cls.EMBEDDING_PROVIDER}, HF_EMBEDDING_MODEL={cls.HF_EMBEDDING_MODEL}"
            ")"
        )


# Create a config instance and export it; callers should invoke Config.validate() when appropriate
config = Config()
# Validate required keys at import time so misconfiguration fails fast
Config.validate()