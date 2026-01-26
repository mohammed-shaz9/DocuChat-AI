"""Google Docs processing utilities for RAG pipeline.

Implements:
- URL validation
- Document fetching via Google Docs export (txt)
- Text cleanup
- Token counting using tiktoken (cl100k_base)
- Sentence-based chunking with token budget and overlap
- End-to-end process orchestration
"""

from __future__ import annotations

import re
from typing import Dict, List

import requests
import tiktoken


class DocumentProcessor:
    def __init__(self) -> None:
        # Initialize tiktoken encoding once for reuse
        self.encoding = tiktoken.get_encoding("cl100k_base")

    # 1. validate_url(url)
    def validate_url(self, url: str) -> Dict[str, object]:
        valid = isinstance(url, str) and "docs.google.com/document" in url
        return {"valid": bool(valid), "error": None if valid else "Invalid Google Docs URL."}

    # Helper: extract document ID
    def _extract_id(self, url: str) -> str | None:
        # Typical formats include /d/<ID>/edit or /d/<ID>/view
        match = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
        return match.group(1) if match else None

    # 2. fetch_document(url)
    def fetch_document(self, url: str) -> Dict[str, object]:
        doc_id = self._extract_id(url)
        if not doc_id:
            return {"success": False, "content": "", "error": "Unable to extract document ID."}

        export_url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
        try:
            resp = requests.get(export_url, timeout=10)
        except requests.RequestException as e:
            return {"success": False, "content": "", "error": f"Network error: {e}"}

        # HTTP Handling
        if resp.status_code == 403:
            return {"success": False, "content": "", "error": "Document is private or access is forbidden (403)."}
        if resp.status_code == 404:
            return {"success": False, "content": "", "error": "Document not found (404)."}
        if resp.status_code != 200:
            return {"success": False, "content": "", "error": f"HTTP error {resp.status_code}."}

        text = resp.text or ""
        if len(text.strip()) < 50:
            return {"success": False, "content": "", "error": "Empty or very short document (<50 chars)."}

        return {"success": True, "content": text, "error": None}

    # 3. clean_text(text)
    def clean_text(self, text: str) -> str:
        # Collapse all whitespace (spaces, tabs, newlines) into single spaces and trim
        cleaned = re.sub(r"\s+", " ", text or "").strip()
        return cleaned

    # 4. count_tokens(text)
    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text or ""))

    # 5. chunk_text(text, chunk_size=800)
    def chunk_text(self, text: str, chunk_size: int = 800) -> List[Dict[str, object]]:
        if not text:
            return []

        # Split by sentence boundaries: period, exclamation, question mark followed by whitespace
        sentences = re.split(r"(?<=[.!?])\s+", text)

        chunks: List[Dict[str, object]] = []
        current_text = ""
        current_tokens = 0

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            candidate = (current_text + " " + sent).strip() if current_text else sent
            candidate_tokens = self.count_tokens(candidate)

            if candidate_tokens <= chunk_size:
                current_text = candidate
                current_tokens = candidate_tokens
            else:
                # finalize current chunk if it has content
                if current_text:
                    idx = len(chunks) + 1
                    chunks.append(
                        {
                            "id": f"chunk-{idx}",
                            "text": current_text,
                            "metadata": {
                                "section": f"Section {idx}",
                                "token_count": current_tokens,
                            },
                        }
                    )

                    # Add overlap: last 20 words from previous chunk
                    last_words = " ".join(current_text.split()[-20:]) if current_text else ""
                    current_text = (last_words + " " + sent).strip() if last_words else sent
                    current_tokens = self.count_tokens(current_text)
                else:
                    # Edge case: a single sentence exceeds chunk_size; start new chunk with it
                    current_text = sent
                    current_tokens = self.count_tokens(current_text)

        # Append remaining text as final chunk
        if current_text:
            idx = len(chunks) + 1
            chunks.append(
                {
                    "id": f"chunk-{idx}",
                    "text": current_text,
                    "metadata": {
                        "section": f"Section {idx}",
                        "token_count": current_tokens,
                    },
                }
            )

        return chunks

    # 6. process_document(url)
    def process_document(self, url: str, chunk_size: int = 800) -> Dict[str, object]:
        validation = self.validate_url(url)
        if not validation["valid"]:
            return {"success": False, "chunks": [], "total_chunks": 0, "error": validation["error"]}

        fetched = self.fetch_document(url)
        if not fetched["success"]:
            return {"success": False, "chunks": [], "total_chunks": 0, "error": fetched["error"]}

        cleaned = self.clean_text(fetched["content"])
        chunks = self.chunk_text(cleaned, chunk_size=chunk_size)

        return {"success": True, "chunks": chunks, "total_chunks": len(chunks), "error": None}