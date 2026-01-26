"""LLM chat handler for generating answers with citations and history.

Implements ChatHandler:
- Conversation history management (last 5 exchanges)
- Prompt building with document chunk context and citations
- Answer generation using OpenAI chat completions (gpt-4o-mini)
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI


class ChatHandler:
    def __init__(self, max_history: int = 5) -> None:
        # OpenAI client reads API key from environment by default
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()

        # Conversations: {session_id: [ {role: "user"|"assistant", content: str}, ... ]}
        self.conversations: Dict[str, List[Dict[str, str]]] = {}
        # Keep only last N exchanges (2 messages per exchange)
        self.max_history: int = max_history

    def build_prompt(self, query: str, chunks: List[Dict[str, Any]], history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Construct messages list for the chat completion.

        Rules:
        - System instruction: answer ONLY using provided document sections with citations like [Section 2].
        - Context message: all chunks formatted as "[Section X]: chunk_text".
        - Include last 5 exchanges from history.
        - Append current user query.
        """
        messages: List[Dict[str, str]] = []

        system_msg = {
            "role": "system",
            "content": "You are a helpful assistant. Answer using ONLY the document sections. Cite sections like [Section 2].",
        }
        messages.append(system_msg)

        # Build context from chunks
        context_lines: List[str] = []
        for c in chunks or []:
            # Support both search result shape {text, section, score} and ingestion shape {id, text, metadata{section}}
            text = c.get("text") or c.get("document") or ""
            section = c.get("section")
            if not section and isinstance(c.get("metadata"), dict):
                section = c["metadata"].get("section")
            section = section or "Unknown"
            if text:
                context_lines.append(f"[{section}]: {text}")
        if context_lines:
            messages.append({"role": "system", "content": "Context:\n" + "\n\n".join(context_lines)})

        # Append recent history (last 5 exchanges => last 10 messages)
        if history:
            trimmed = history[-(self.max_history * 2) :]
            messages.extend(trimmed)

        # Current query
        messages.append({"role": "user", "content": query})
        return messages

    def generate_answer(self, query: str, chunks: List[Dict[str, Any]], session_id: str) -> Dict[str, Any]:
        """Generate an answer using GPT-4o-mini with citations.

        Fallback if no relevant chunks (empty or score < 0.5).
        Stores conversation history limited to the last 5 exchanges.
        Returns: {answer, sources, confidence}
        """
        # Confidence based on best score (if available)
        scores = [c.get("score") for c in (chunks or []) if isinstance(c.get("score"), (int, float))]
        max_score = max(scores) if scores else 0.0

        if not chunks or (scores and all(s < 0.5 for s in scores)):
            answer = "This information isn't in the document"
            # Update minimal history
            hist = self.conversations.setdefault(session_id, [])
            hist.append({"role": "user", "content": query})
            hist.append({"role": "assistant", "content": answer})
            # Trim
            if len(hist) > self.max_history * 2:
                self.conversations[session_id] = hist[-(self.max_history * 2) :]
            return {"answer": answer, "sources": [], "confidence": max_score}

        history = self.conversations.get(session_id, [])
        messages = self.build_prompt(query, chunks, history)

        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.3,
                max_tokens=500,
            )
        except Exception as e:
            # On error, return fallback
            answer = f"There was an error generating the answer: {e}"
            return {"answer": answer, "sources": [], "confidence": max_score}

        answer = (resp.choices[0].message.content or "").strip()

        # Extract citations like [Section 2]
        citations = re.findall(r"\[Section\s+\d+\]", answer)
        sources = sorted(set(citations))

        # Update conversation history
        hist = self.conversations.setdefault(session_id, [])
        hist.append({"role": "user", "content": query})
        hist.append({"role": "assistant", "content": answer})
        if len(hist) > self.max_history * 2:
            self.conversations[session_id] = hist[-(self.max_history * 2) :]

        return {"answer": answer, "sources": sources, "confidence": max_score}

    def clear_history(self, session_id: str) -> None:
        """Delete conversation history for a session."""
        if session_id in self.conversations:
            del self.conversations[session_id]

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Return the conversation history list for a session."""
        return list(self.conversations.get(session_id, []))