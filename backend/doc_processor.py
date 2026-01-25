"""Google Doc fetching & chunking utilities (Phase 1 placeholder).

In later phases, this module will:
- Fetch publicly shared Google Doc content programmatically.
- Chunk text into semantic segments suitable for embeddings.
"""

from typing import List


def fetch_google_doc(doc_url: str) -> str:
    """Return raw text from a publicly shared Google Doc URL.
    Phase 1: placeholder implementation.
    """
    return ""


def chunk_text(text: str, max_tokens: int = 800) -> List[str]:
    """Split text into rough chunks.
    Phase 1: placeholder implementation.
    """
    return [text] if text else []