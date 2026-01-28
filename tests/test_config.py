import importlib
import sys
import pytest


def reload_config(monkeypatch, env=None):
    keys = [
        "OPENAI_API_KEY",
        "FLASK_SECRET_KEY",
        "FLASK_ENV",
        "PORT",
        "CORS_ORIGINS",
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "MAX_CONVERSATION_HISTORY",
        "EMBEDDING_MODEL",
        "CHAT_MODEL",
        "CHROMA_PERSIST_DIR",
    ]
    # Clear relevant env vars
    for k in keys:
        monkeypatch.delenv(k, raising=False)
    # Set provided env vars
    env = env or {}
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))

    # Reload module to re-evaluate env
    # Mock load_dotenv so it doesn't read the .env file and override our monkeypatch
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda **kwargs: None)

    sys.modules.pop("backend.config", None)
    import backend.config as config_module
    importlib.reload(config_module)
    return config_module


def test_config_defaults(monkeypatch):
    module = reload_config(monkeypatch, {"OPENAI_API_KEY": "sk-test", "FLASK_SECRET_KEY": "secret"})
    assert module.Config.FLASK_ENV == "production"
    assert module.Config.PORT == 5000
    assert isinstance(module.Config.CORS_ORIGINS, list)
    assert module.Config.CHUNK_SIZE == 800
    assert module.Config.CHUNK_OVERLAP == 100
    assert module.Config.MAX_CONVERSATION_HISTORY == 5
    assert module.Config.EMBEDDING_MODEL == "text-embedding-3-small"
    assert module.Config.CHAT_MODEL == "gpt-4o-mini"
    assert module.Config.CHROMA_PERSIST_DIR == "./chroma_db"


def test_config_custom_env(monkeypatch):
    env = {
        "OPENAI_API_KEY": "sk-custom",
        "FLASK_SECRET_KEY": "secret",
        "FLASK_ENV": "development",
        "PORT": "7000",
        "CORS_ORIGINS": "http://a.com, http://b.com",
        "CHUNK_SIZE": "900",
        "CHUNK_OVERLAP": "120",
        "MAX_CONVERSATION_HISTORY": "10",
        "EMBEDDING_MODEL": "text-embedding-3-large",
        "CHAT_MODEL": "gpt-4o",
        "CHROMA_PERSIST_DIR": "/tmp/chroma",
    }
    module = reload_config(monkeypatch, env)
    assert module.Config.FLASK_ENV == "development"
    assert module.Config.PORT == 7000
    assert module.Config.CORS_ORIGINS == ["http://a.com", "http://b.com"]
    assert module.Config.CHUNK_SIZE == 900
    assert module.Config.CHUNK_OVERLAP == 120
    assert module.Config.MAX_CONVERSATION_HISTORY == 10
    assert module.Config.EMBEDDING_MODEL == "text-embedding-3-large"
    assert module.Config.CHAT_MODEL == "gpt-4o"
    assert module.Config.CHROMA_PERSIST_DIR == "/tmp/chroma"


def test_config_validation_missing_openai(monkeypatch):
    with pytest.raises(ValueError):
        reload_config(monkeypatch, {"OPENAI_API_KEY": "", "FLASK_SECRET_KEY": "secret"})


def test_config_validation_missing_secret(monkeypatch):
    with pytest.raises(ValueError):
        reload_config(monkeypatch, {"OPENAI_API_KEY": "sk-test", "FLASK_SECRET_KEY": ""})


def test_validate_google_doc_url():
    from backend.utils.validators import validate_google_doc_url

    ok = validate_google_doc_url("https://docs.google.com/document/d/abc123/edit")
    assert ok["valid"] is True
    assert ok["doc_id"] == "abc123"
    assert ok["error"] is None

    bad = validate_google_doc_url("https://example.com/notgoogle")
    assert bad["valid"] is False
    assert bad["doc_id"] is None
    assert isinstance(bad["error"], str)