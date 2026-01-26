from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv

# Load env once here
load_dotenv()


def _parse_origins(origins_str: str) -> List[str]:
    if not origins_str:
        return []
    return [o.strip() for o in origins_str.split(",") if o.strip()]


class Config:
    # Models
    MODEL_EMBEDDING = os.getenv("MODEL_EMBEDDING", "text-embedding-3-small")
    MODEL_CHAT = os.getenv("MODEL_CHAT", "gpt-4o-mini")

    # Basic env
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-secret")
    FLASK_ENV = os.getenv("FLASK_ENV", "development")
    PORT = int(os.getenv("PORT", "5000"))

    # RAG settings
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
    MAX_CONVERSATION_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", "5"))

    # CORS
    CORS_ORIGINS = _parse_origins(os.getenv("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"))

    @classmethod
    def validate(cls) -> None:
        # Minimal validation; can be extended
        if not cls.OPENAI_API_KEY:
            # Not raising to allow local testing of health; modules that require key will error when used
            pass

    @classmethod
    def as_dict(cls) -> dict:
        return {
            "MODEL_EMBEDDING": cls.MODEL_EMBEDDING,
            "MODEL_CHAT": cls.MODEL_CHAT,
            "FLASK_ENV": cls.FLASK_ENV,
            "PORT": cls.PORT,
            "CHUNK_SIZE": cls.CHUNK_SIZE,
            "CHUNK_OVERLAP": cls.CHUNK_OVERLAP,
            "MAX_CONVERSATION_HISTORY": cls.MAX_CONVERSATION_HISTORY,
            "CORS_ORIGINS": cls.CORS_ORIGINS,
        }