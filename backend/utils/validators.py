from __future__ import annotations

import re
import string
from typing import Dict, List, Optional


def validate_google_doc_url(url: Optional[str]) -> Dict[str, object]:
    """Validate Google Doc URL and extract document ID if present.

    Returns: {"valid": bool, "error": str or None, "doc_id": str or None}
    """
    if not url or not isinstance(url, str) or not url.strip():
        return {"valid": False, "error": "URL is required.", "doc_id": None}
    if "docs.google.com/document" not in url:
        return {"valid": False, "error": "URL must be a Google Doc.", "doc_id": None}
    m = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if not m:
        return {"valid": False, "error": "Document ID not found in URL.", "doc_id": None}
    return {"valid": True, "error": None, "doc_id": m.group(1)}


def validate_api_key(key: Optional[str]) -> bool:
    """Basic check for OpenAI-style keys (sk-...)."""
    if not key or not isinstance(key, str) or not key.strip():
        return False
    return key.strip().startswith("sk-")


def sanitize_text(text: Optional[str]) -> str:
    """Remove null bytes, excessive whitespace, and non-printable characters."""
    if text is None:
        return ""
    # Remove null bytes
    cleaned = text.replace("\x00", "")
    # Remove non-printable characters (except common whitespace)
    printable = set(string.printable)
    cleaned = "".join(ch for ch in cleaned if ch in printable)
    # Collapse whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def validate_chunk_size(size: int) -> bool:
    """Ensure chunk size is within reasonable bounds."""
    try:
        s = int(size)
    except Exception:
        return False
    return 200 <= s <= 2000


# Backwards compatibility exports if needed
def is_valid_google_doc_url(url: str) -> bool:
    res = validate_google_doc_url(url)
    return res.get("valid", False)


def require_json_fields(data: Dict[str, object], fields: List[str]) -> List[str]:
    missing = []
    for f in fields:
        v = data.get(f)
        if v is None or (isinstance(v, str) and not v.strip()):
            missing.append(f)
    return missing