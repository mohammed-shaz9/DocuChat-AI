"""Google Docs processing utilities for RAG pipeline.

DocumentProcessor responsibilities:
- Fetch Google Docs content via export API
- Clean and normalize text
- Count tokens (using tiktoken when available)
- Chunk text into sentence-based segments with token budgets and overlap
- Provide validation for produced chunks
- Robust error handling that never crashes and returns structured error dicts
"""

from __future__ import annotations

import re
import io
from typing import Any, Dict, List, Optional

import requests

try:
    # Import lazily in constructor; keep here for type hints only
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover
    tiktoken = None  # Fallback if tiktoken is not installed

from backend.config import Config
from backend.utils import (
    validate_google_doc_url,
    sanitize_text,
    log_error,
    validate_chunk_size,
)


class DocumentProcessor:
    """Class-based processor for Google Docs content with error handling."""

    def __init__(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None) -> None:
        # Initialize encoding safely (lazy import)
        self.encoding = None
        try:
            if tiktoken:
                self.encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as e:  # pragma: no cover
            # If tiktoken isn't available, we will use a simple fallback method
            log_error(e, context={"where": "DocumentProcessor.__init__", "note": "tiktoken unavailable"})
            self.encoding = None

        # Load sizes from config if not provided
        self.chunk_size = int(chunk_size) if chunk_size is not None else int(getattr(Config, "CHUNK_SIZE", 800))
        self.chunk_overlap = int(chunk_overlap) if chunk_overlap is not None else int(getattr(Config, "CHUNK_OVERLAP", 100))

        # Validate and adjust sizes using utility
        size_is_valid = validate_chunk_size(self.chunk_size)
        if not size_is_valid:
            self.chunk_size = 800
        # Overlap must be non-negative and less than chunk_size; cap to 25% of chunk_size
        if self.chunk_overlap < 0:
            self.chunk_overlap = 0
        max_overlap = max(0, self.chunk_size // 4)
        if self.chunk_overlap > max_overlap:
            self.chunk_overlap = max_overlap

    # Backward-compatibility helper
    def validate_url(self, url: str) -> Dict[str, object]:
        """Validate a Google Doc URL using validators."""
        return validate_google_doc_url(url)

    def fetch_document(self, url: str) -> Dict[str, Any]:
        """Fetch plain text content from a Google Doc export endpoint.

        Returns a dict:
          {"success": bool, "error": str or None, "content": str or None, "doc_id": str or None}
        """
        try:
            v = validate_google_doc_url(url)
            if not v.get("valid"):
                return {"success": False, "error": v.get("error") or "Invalid Google Docs URL.", "content": None, "doc_id": None}

            doc_id = v.get("doc_id")
            export_url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
            headers = {"User-Agent": "Mozilla/5.0"}

            resp = requests.get(export_url, timeout=15, headers=headers)

            # HTTP error handling per spec
            status = resp.status_code
            if status == 403:
                return {"success": False, "error": "Document is private. Make it publicly viewable.", "content": None, "doc_id": doc_id}
            if status == 404:
                return {"success": False, "error": "Document not found. Check the URL.", "content": None, "doc_id": doc_id}
            if status == 429:
                return {"success": False, "error": "Rate limited. Please wait a moment.", "content": None, "doc_id": doc_id}
            if 500 <= status <= 599:
                return {"success": False, "error": "Server error. Try again later.", "content": None, "doc_id": doc_id}
            if status != 200:
                return {"success": False, "error": f"HTTP error {status}.", "content": None, "doc_id": doc_id}

            text = resp.text or ""
            if len(text.strip()) < 50:
                return {"success": False, "error": "Document too short or empty (min 50 chars).", "content": None, "doc_id": doc_id}

            return {"success": True, "error": None, "content": text, "doc_id": doc_id}
        except requests.Timeout as e:
            log_error(e, context={"where": "fetch_document", "url": url})
            return {"success": False, "error": "Network timeout while fetching document.", "content": None, "doc_id": None}
        except requests.ConnectionError as e:  # type: ignore[attr-defined]
            log_error(e, context={"where": "fetch_document", "url": url})
            return {"success": False, "error": "Network connection error while fetching document.", "content": None, "doc_id": None}
        except Exception as e:
            log_error(e, context={"where": "fetch_document", "url": url})
            return {"success": False, "error": str(e) or "Unexpected error fetching document.", "content": None, "doc_id": None}

    def clean_text(self, text: str) -> str:
        """Clean text using validators and regex rules.

        Steps:
        - Use sanitize_text to remove control characters
        - Remove URLs (http/https)
        - Collapse multiple spaces to single space
        - Collapse multiple newlines to double newline (preserve paragraphs)
        - Strip leading/trailing whitespace
        """
        try:
            if not text:
                return ""
            # Use sanitize_text first (removes non-printables, null bytes)
            base = sanitize_text(text)
            # Remove URLs
            base = re.sub(r"https?://\S+", "", base)
            # Normalize Windows/Mac newlines
            base = base.replace("\r\n", "\n").replace("\r", "\n")
            # Collapse multiple spaces but preserve newlines
            base = re.sub(r"[ \t]+", " ", base)
            # Reduce 3+ newlines to exactly 2 newlines (paragraph separation)
            base = re.sub(r"\n{3,}", "\n\n", base)
            # Trim
            base = base.strip()
            return base
        except Exception as e:
            log_error(e, context={"where": "clean_text"})
            return sanitize_text(text or "")

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken when available; fallback to whitespace split length."""
        try:
            if not text:
                return 0
            if self.encoding is not None:
                return len(self.encoding.encode(text))
            # Fallback heuristic
            return len(text.split())
        except Exception as e:
            log_error(e, context={"where": "count_tokens"})
            return 0

    def _encode(self, text: str) -> List[int]:
        if self.encoding is None:
            # Fallback: approximate tokens by words
            return text.split()
        return self.encoding.encode(text)

    def _decode(self, tokens: List[int]) -> str:
        if self.encoding is None:
            # Fallback: tokens are words
            return " ".join(tokens)
        return self.encoding.decode(tokens)

    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Chunk text into sentence-based segments with token budget and overlap.

        Uses self.chunk_size and self.chunk_overlap.
        """
        try:
            if not text:
                return []

            sentences = re.split(r"(?<=[.!?])\s+", text)

            chunks: List[Dict[str, Any]] = []
            current_chunk = ""
            current_tokens = 0
            chunk_id = 0

            for sentence in sentences:
                s = sentence.strip()
                if not s:
                    continue

                candidate = (current_chunk + (" " if current_chunk else "") + s)
                candidate_tokens = self.count_tokens(candidate)

                if current_chunk and candidate_tokens > self.chunk_size:
                    # save current chunk
                    token_count = self.count_tokens(current_chunk)
                    chunks.append(
                        {
                            "id": f"chunk_{chunk_id}",
                            "text": current_chunk,
                            "metadata": {
                                "section": f"Section {chunk_id + 1}",
                                "token_count": token_count,
                                "chunk_index": chunk_id,
                            },
                        }
                    )
                    chunk_id += 1

                    # build overlap by tokens if possible
                    if self.encoding is not None:
                        toks = self.encoding.encode(current_chunk)
                        overlap_tokens = toks[-self.chunk_overlap :] if self.chunk_overlap > 0 else []
                        overlap_text = self.encoding.decode(overlap_tokens) if overlap_tokens else ""
                    else:
                        words = current_chunk.split()
                        overlap_words = words[-self.chunk_overlap :] if self.chunk_overlap > 0 else []
                        overlap_text = " ".join(overlap_words)

                    current_chunk = (overlap_text + (" " if overlap_text else "") + s).strip()
                    current_tokens = self.count_tokens(current_chunk)
                else:
                    current_chunk = candidate
                    current_tokens = candidate_tokens

            # Save final chunk
            if current_chunk:
                token_count = self.count_tokens(current_chunk)
                chunks.append(
                    {
                        "id": f"chunk_{chunk_id}",
                        "text": current_chunk,
                        "metadata": {
                            "section": f"Section {chunk_id + 1}",
                            "token_count": token_count,
                            "chunk_index": chunk_id,
                        },
                    }
                )

            return chunks
        except Exception as e:
            log_error(e, context={"where": "chunk_text"})
            return []

    def _detect_file_type(self, url: str) -> Optional[str]:
        try:
            u = (url or "").strip().lower()
            u_no_query = u.split("?")[0]
            if u_no_query.endswith(".pdf"):
                return "pdf"
            if u_no_query.endswith(".docx"):
                return "docx"
            # Try HEAD for content-type
            try:
                resp = requests.head(url, timeout=10)
                ct = (resp.headers.get("Content-Type") or "").lower()
                if "application/pdf" in ct:
                    return "pdf"
                if "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in ct:
                    return "docx"
            except Exception:
                pass
            return None
        except Exception:
            return None

    def _fetch_file_bytes(self, url: str) -> Dict[str, Any]:
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, timeout=20, headers=headers)
            status = resp.status_code
            if status == 403:
                return {"success": False, "error": "File is private or access denied.", "content": None}
            if status == 404:
                return {"success": False, "error": "File not found. Check the URL.", "content": None}
            if status == 429:
                return {"success": False, "error": "Rate limited. Please wait a moment.", "content": None}
            if 500 <= status <= 599:
                return {"success": False, "error": "Server error. Try again later.", "content": None}
            if status != 200:
                return {"success": False, "error": f"HTTP error {status}.", "content": None}
            data = resp.content or b""
            if len(data) < 512:
                return {"success": False, "error": "Downloaded file is too small or empty.", "content": None}
            return {"success": True, "error": None, "content": data}
        except requests.Timeout as e:
            log_error(e, context={"where": "_fetch_file_bytes", "url": url})
            return {"success": False, "error": "Network timeout while fetching file.", "content": None}
        except requests.ConnectionError as e:  # type: ignore[attr-defined]
            log_error(e, context={"where": "_fetch_file_bytes", "url": url})
            return {"success": False, "error": "Network connection error while fetching file.", "content": None}
        except Exception as e:
            log_error(e, context={"where": "_fetch_file_bytes", "url": url})
            return {"success": False, "error": str(e) or "Unexpected error fetching file.", "content": None}

    def fetch_pdf_text(self, url: str) -> Dict[str, Any]:
        try:
            f = self._fetch_file_bytes(url)
            if not f.get("success"):
                return {"success": False, "error": f.get("error"), "content": None, "doc_id": None}
            data = f.get("content")
            if data is None:
                return {"success": False, "error": "No file content.", "content": None, "doc_id": None}
            try:
                from PyPDF2 import PdfReader  # lazy import
            except Exception as e:
                log_error(e, context={"where": "fetch_pdf_text", "note": "PyPDF2 not installed"})
                return {"success": False, "error": "PyPDF2 not available on server.", "content": None, "doc_id": None}
            reader = PdfReader(io.BytesIO(data))
            texts: List[str] = []
            for page in getattr(reader, "pages", []):
                try:
                    t = page.extract_text() or ""
                except Exception:
                    t = ""
                if t.strip():
                    texts.append(t)
            merged = "\n\n".join(texts).strip()
            if len(merged) < 50:
                return {"success": False, "error": "PDF content too short or empty (min 50 chars).", "content": None, "doc_id": None}
            return {"success": True, "error": None, "content": merged, "doc_id": None}
        except Exception as e:
            log_error(e, context={"where": "fetch_pdf_text", "url": url})
            return {"success": False, "error": str(e) or "Unexpected error reading PDF.", "content": None, "doc_id": None}

    def fetch_docx_text(self, url: str) -> Dict[str, Any]:
        try:
            f = self._fetch_file_bytes(url)
            if not f.get("success"):
                return {"success": False, "error": f.get("error"), "content": None, "doc_id": None}
            data = f.get("content")
            if data is None:
                return {"success": False, "error": "No file content.", "content": None, "doc_id": None}
            try:
                import docx  # python-docx lazy import
            except Exception as e:
                log_error(e, context={"where": "fetch_docx_text", "note": "python-docx not installed"})
                return {"success": False, "error": "python-docx not available on server.", "content": None, "doc_id": None}
            document = docx.Document(io.BytesIO(data))
            paras = [p.text for p in getattr(document, "paragraphs", []) if p.text and p.text.strip()]
            merged = "\n\n".join(paras).strip()
            if len(merged) < 50:
                return {"success": False, "error": "DOCX content too short or empty (min 50 chars).", "content": None, "doc_id": None}
            return {"success": True, "error": None, "content": merged, "doc_id": None}
        except Exception as e:
            log_error(e, context={"where": "fetch_docx_text", "url": url})
            return {"success": False, "error": str(e) or "Unexpected error reading DOCX.", "content": None, "doc_id": None}

    def process_document(self, url: str) -> Dict[str, Any]:
        """Complete pipeline: fetch, clean, chunk, and summarize results."""
        try:
            # Detect PDF/DOCX and fetch accordingly; otherwise treat as Google Doc
            file_type = self._detect_file_type(url)
            if file_type == "pdf":
                fetched = self.fetch_pdf_text(url)
            elif file_type == "docx":
                fetched = self.fetch_docx_text(url)
            else:
                fetched = self.fetch_document(url)
            if not fetched.get("success"):
                return {"success": False, "error": fetched.get("error"), "chunks": [], "total_chunks": 0, "total_tokens": 0, "doc_id": fetched.get("doc_id")}

            raw_text = fetched.get("content", "")
            doc_id = fetched.get("doc_id")
            cleaned = self.clean_text(raw_text)
            chunks = self.chunk_text(cleaned)

            # Validate chunks
            if not self.validate_chunks(chunks):
                return {"success": False, "error": "Generated chunks failed validation.", "chunks": [], "total_chunks": 0, "total_tokens": 0, "doc_id": doc_id}

            total_tokens = sum(int(c.get("metadata", {}).get("token_count", 0)) for c in chunks)

            return {
                "success": True,
                "chunks": chunks,
                "total_chunks": len(chunks),
                "total_tokens": total_tokens,
                "doc_id": doc_id,
            }
        except Exception as e:
            log_error(e, context={"where": "process_document", "url": url})
            return {"success": False, "error": str(e) or "Unexpected error processing document.", "chunks": [], "total_chunks": 0, "total_tokens": 0, "doc_id": None}

    def validate_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """Quality checks for chunks.

        - Each chunk has required fields (id, text, metadata)
        - No empty text
        - Token counts are reasonable (100-1500)
        """
        try:
            if not isinstance(chunks, list):
                return False
            for ch in chunks:
                if not isinstance(ch, dict):
                    return False
                if not ch.get("id") or not ch.get("text"):
                    return False
                meta = ch.get("metadata", {})
                if not isinstance(meta, dict):
                    return False
                tc = int(meta.get("token_count", 0))
                if tc < 100 or tc > 1500:
                    return False
            return True
        except Exception as e:
            log_error(e, context={"where": "validate_chunks"})
            return False