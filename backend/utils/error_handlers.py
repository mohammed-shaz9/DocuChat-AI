from __future__ import annotations

import logging
import traceback
from datetime import datetime
from typing import Any, Dict, Optional

from flask import jsonify


class DocumentFetchError(Exception):
    """Custom exception for document fetching issues.

    Attributes:
        message: Description of the error
        status_code: HTTP status code associated with the failure
    """

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class EmbeddingGenerationError(Exception):
    """Custom exception for embedding generation failures.

    Attributes:
        message: Description of the error
        retry_count: How many retries were attempted before failing
    """

    def __init__(self, message: str, retry_count: int = 0) -> None:
        super().__init__(message)
        self.message = message
        self.retry_count = retry_count


class ChatGenerationError(Exception):
    """Custom exception for chat/LLM generation failures.

    Attributes:
        message: Description of the error
        context: Additional context information for debugging
    """

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or {}


def handle_api_error(error: Exception) -> Dict[str, Any]:
    """Return standardized error dict: {"error": str, "type": str, "retryable": bool}.

    Attempts to parse common error types (OpenAI, requests, custom exceptions) and
    decide whether the error may be retryable.
    """
    type_str = "unknown_error"
    msg = str(error) if str(error) else error.__class__.__name__
    retryable = False

    # Try to detect requests-related errors
    try:
        import requests
        from requests.exceptions import HTTPError, ConnectionError, Timeout

        if isinstance(error, Timeout):
            type_str = "network_timeout"
            retryable = True
        elif isinstance(error, ConnectionError):
            type_str = "network_connection_error"
            retryable = True
        elif isinstance(error, HTTPError):
            status = getattr(error.response, "status_code", None)
            type_str = "http_error"
            retryable = status in {429, 503}
            if status:
                msg = f"HTTP {status}: " + (getattr(error.response, "text", msg) or msg)
    except Exception:
        # requests may not be installed; ignore detection
        pass

    # Detect OpenAI-style errors without importing the library
    cls_name = error.__class__.__name__.lower()
    mod_name = getattr(error.__class__, "__module__", "")
    if "openai" in mod_name or "openai" in cls_name:
        type_str = "openai_error"
        # Heuristic: rate limit, timeout, or api connection errors are retryable
        if any(s in cls_name for s in ["rate", "limit", "timeout", "connection", "server"]):
            retryable = True

    # Custom exceptions
    if isinstance(error, DocumentFetchError):
        type_str = "document_fetch_error"
        retryable = error.status_code in {429, 503}
        msg = error.message
    elif isinstance(error, EmbeddingGenerationError):
        type_str = "embedding_generation_error"
        retryable = error.retry_count > 0
        msg = error.message
    elif isinstance(error, ChatGenerationError):
        type_str = "chat_generation_error"
        retryable = False
        msg = error.message

    return {"error": msg, "type": type_str, "retryable": retryable}


def log_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
    """Print formatted error with timestamp and include context information.

    Uses logging with levels: ERROR (default), WARNING for retryable cases, INFO for context.
    """
    context = context or {}
    info = handle_api_error(error)

    # Basic logging configuration if not already configured
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
        )

    level = logging.ERROR
    if info.get("retryable"):
        level = logging.WARNING

    # Log the main error
    logging.log(level, f"{info['type']}: {info['error']}")

    # Log context information
    if context:
        logging.info(f"Context: {context}")

    # Log traceback for debugging
    tb = traceback.format_exc()
    if tb and "NoneType: None" not in tb:
        logging.debug(tb)


def register_error_handlers(app):
    """Register Flask error handlers that use our standardized error payloads."""

    @app.errorhandler(404)
    def handle_404(error):
        return jsonify({"error": "Not Found", "type": "not_found", "retryable": False}), 404

    @app.errorhandler(400)
    def handle_400(error):
        info = handle_api_error(error)
        info.setdefault("type", "bad_request")
        info["retryable"] = False
        return jsonify(info), 400

    @app.errorhandler(500)
    def handle_500(error):
        info = handle_api_error(error)
        info.setdefault("type", "internal_server_error")
        return jsonify(info), 500

    @app.errorhandler(Exception)
    def handle_generic_exception(error):
        # Catch-all handler to standardize unexpected exceptions
        log_error(error, context={"timestamp": datetime.utcnow().isoformat()})
        info = handle_api_error(error)
        return jsonify(info), 500