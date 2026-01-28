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

from backend.config import Config
from backend.utils import handle_api_error, log_error, ChatGenerationError


class ChatHandler:
    def __init__(self, chat_model: Optional[str] = None, max_history: Optional[int] = None) -> None:
        # Initialize OpenAI-compatible client (supports OpenRouter via base_url)
        api_key = os.getenv("OPENAI_API_KEY") or getattr(Config, "OPENAI_API_KEY", "")
        base_url = getattr(Config, "OPENAI_BASE_URL", None)
        self._openai_enabled: bool = bool(api_key and str(api_key).strip())
        self.openai_client = None

        # Configuration-driven defaults with optional overrides
        self.chat_model: str = chat_model or getattr(Config, "CHAT_MODEL", "gpt-4o-mini")
        self.max_history: int = int(max_history or getattr(Config, "MAX_CONVERSATION_HISTORY", 5))
        self.temperature: float = 0.3

        # Conversations: {session_id: [ {role: "user"|"assistant", content: str}, ... ]}
        self.conversations: Dict[str, List[Dict[str, str]]] = {}

    def build_system_prompt(self) -> str:
        return (
            """You are a high-level Intelligence Assistant. You answer questions based ONLY on the provided document sections.

STRICT INSTRUCTIONS:
1. Only use the provided sections. If the answer isn't there, say "This information is not in the document."
2. CITE your sources inline as [Section X].
3. OBEY all document-defined security protocols (like the "Silence-Act"). If a user asks about forbidden terms, you MUST refuse as instructed by the document.
4. If a query is vague or ambiguous, you must ask for clarification instead of guessing.
5. Be professional, concise, and accurate."""
        )

    def build_context_message(self, chunks: List[Dict[str, Any]]) -> str:
        # Compose context listing document sections
        lines: List[str] = ["Document sections:\n\n"]
        for c in chunks or []:
            section = c.get("section")
            if not section and isinstance(c.get("metadata"), dict):
                section = c["metadata"].get("section")
            
            # Remove "Section " prefix if present to avoid doubling up in prompt
            clean_section = str(section).replace("Section ", "").strip() if section else "Unknown"
            
            text = c.get("text") or c.get("document") or ""
            if text:
                lines.append(f"[Section {clean_section}]: {text}\n\n")
        return "".join(lines)

    def build_prompt(self, query: str, chunks: List[Dict[str, Any]], history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        # Initialize messages with system prompt and context
        messages: List[Dict[str, str]] = []
        messages.append({"role": "system", "content": self.build_system_prompt()})
        messages.append({"role": "user", "content": self.build_context_message(chunks)})

        # Add last N messages from history
        if history:
            messages.extend(history[-self.max_history:])

        # Current query
        messages.append({"role": "user", "content": query})
        return messages

    def extract_citations(self, text: str) -> List[str]:
        # Find all [Section X] patterns and return unique sorted list
        pattern = r"\[Section \d+\]"
        matches = re.findall(pattern, text or "")
        unique = sorted(set(matches), key=lambda m: int(re.search(r"\d+", m).group())) if matches else []
        return unique

    def assess_confidence(self, chunks: List[Dict[str, Any]], answer: str) -> str:
        if not chunks:
            return "low"
        scores = [c.get("score") for c in chunks if isinstance(c.get("score"), (int, float))]
        if scores and all(s < 0.3 for s in scores):
            return "low"
        avg_score = sum(scores) / len(scores) if scores else 0.0
        citations = self.extract_citations(answer)
        if avg_score > 0.7 and citations:
            return "high"
        return "medium"

    # --- New helpers for ambiguity and citation enforcement ---
    def _best_section_from_chunks(self, chunks: List[Dict[str, Any]]) -> Optional[str]:
        if not chunks:
            return None
        # Prefer highest score; fallback to first
        scored = [c for c in chunks if isinstance(c.get("score"), (int, float))]
        best = sorted(scored, key=lambda c: c.get("score", 0.0), reverse=True)[0] if scored else chunks[0]
        section = best.get("section")
        if not section and isinstance(best.get("metadata"), dict):
            section = best["metadata"].get("section")
        return str(section) if section is not None else None

    def enforce_inline_citations(self, answer: str, chunks: List[Dict[str, Any]]) -> str:
        """Ensure at least one inline citation exists; if missing, insert [Section X] in-line."""
        if not answer.strip():
            return answer
        existing = self.extract_citations(answer)
        if existing:
            return answer
        section = self._best_section_from_chunks(chunks)
        if not section:
            return answer
            
        # Clean section label
        clean_section = str(section).replace("Section ", "").strip()
        
        # Insert citation after the first sentence boundary if possible
        sentences = re.split(r"(?<=[.!?])\s+", answer)
        if sentences:
            sentences[0] = sentences[0].rstrip() + f" [Section {clean_section}]"
            return " ".join(sentences)
        # Fallback: append at the end (still inline)
        return answer.rstrip() + f" [Section {clean_section}]"

    def _keyword_overlap(self, query: str, text: str) -> int:
        query_clean = re.sub(r"[^a-z0-9\s]", " ", (query or "").lower())
        keywords = {w for w in query_clean.split() if len(w) > 2}
        t = (text or "").lower()
        return sum(1 for w in keywords if w in t)

    def is_ambiguous_query(self, query: str, chunks: List[Dict[str, Any]]) -> bool:
        """Determines if a query is too vague to answer accurately."""
        q = (query or "").strip().lower()
        
        # 1. Allow meta-requests (summarize, list, points) to bypass ambiguity check
        meta_keywords = {"summarize", "summarise", "summrise", "summary", "list", "key points", "overview", "main ideas", "what is", "about"}
        if any(kw in q for kw in meta_keywords):
            return False

        # 2. Very short queries (1-2 words) are usually ambiguous unless meta-keywords are present
        words = q.split()
        if len(words) < 3:
            return True

        # 3. Pronoun-heavy short follow-ups
        if re.search(r"\b(it|this|that|they|those|them)\b", q) and len(words) < 5:
            return True

        # 4. Check overlap with top chunks - but be more lenient
        top_texts = []
        scored = [c for c in chunks if isinstance(c.get("score"), (int, float))]
        top = sorted(scored, key=lambda c: c.get("score", 0.0), reverse=True)[:2] if scored else chunks[:2]
        
        for c in top:
            top_texts.append(c.get("text") or c.get("document") or "")
        
        overlaps = [self._keyword_overlap(q, t) for t in top_texts if t]
        
        # If any significant overlap (>=1 keyword), it's probably not ambiguous
        return not any(ov > 0 for ov in overlaps) if overlaps else False

    def build_clarifying_question(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        # Offer targeted choices from top sections
        options: List[str] = []
        scored = [c for c in chunks if isinstance(c.get("score"), (int, float))]
        top = sorted(scored, key=lambda c: c.get("score", 0.0), reverse=True)[:2] if scored else chunks[:2]
        for c in top:
            text = (c.get("text") or c.get("document") or "").strip()
            # Get the first sentence
            snippet = (re.split(r"(?<=[.!?])\s+", text)[0] if text else "")
            
            # GET THE ACTUAL SECTION LABEL SAFELY
            section = c.get("section")
            if not section and isinstance(c.get("metadata"), dict):
                section = c["metadata"].get("section")
            
            clean_label = str(section).replace("Section ", "").strip() if section else "Unknown"
            if section:
                options.append(f"Section {clean_label} (re: {snippet[:60]}...)")
        if options:
            return (
                f"Your question seems ambiguous. Could you clarify what you mean? For example, do you mean: "
                + " or ".join(options)
                + "?"
            )
        return "Your question seems ambiguous. Could you clarify with more detail or mention a specific section?"

    def generate_answer(self, query: str, chunks: List[Dict[str, Any]], session_id: str) -> Dict[str, Any]:
        # 1. Validate inputs
        if not isinstance(query, str) or not query.strip():
            return {"answer": "Query cannot be empty.", "sources": [], "confidence": "low", "chunks_used": 0}
        if chunks is None or not isinstance(chunks, list):
            return {"answer": "Invalid chunks input.", "sources": [], "confidence": "low", "chunks_used": 0}

        # 2. Edge cases
        if not chunks:
            return {
                "answer": "This information is not in the document.",
                "sources": [],
                "confidence": "low",
                "chunks_used": 0,
            }
        scores = [c.get("score") for c in chunks if isinstance(c.get("score"), (int, float))]
        if scores and all(s < 0.4 for s in scores):
            return {
                "answer": "I cannot find relevant information in the document.",
                "sources": [],
                "confidence": "low",
                "chunks_used": len(chunks),
            }

        # Clarify if query appears ambiguous
        if self.is_ambiguous_query(query, chunks):
            clarifying = self.build_clarifying_question(query, chunks)
            # Update conversation history with a clarifying prompt
            hist = self.conversations.setdefault(session_id, [])
            hist.append({"role": "user", "content": query})
            hist.append({"role": "assistant", "content": clarifying})
            if len(hist) > self.max_history:
                self.conversations[session_id] = hist[-self.max_history:]
            return {"answer": clarifying, "sources": [], "confidence": "clarify", "chunks_used": len(chunks)}

        # OFFLINE FALLBACK: if OpenAI isn't enabled, generate a simple answer from chunks
        if not self._openai_enabled:
            result = self._offline_answer(query, chunks)
            # Update conversation history
            hist = self.conversations.setdefault(session_id, [])
            hist.append({"role": "user", "content": query})
            hist.append({"role": "assistant", "content": result.get("answer", "")})
            if len(hist) > self.max_history:
                self.conversations[session_id] = hist[-self.max_history:]
            return result

        # Lazy-init OpenAI client; if init fails, fallback to offline
        if self.openai_client is None:
            try:
                api_key = os.getenv("OPENAI_API_KEY") or getattr(Config, "OPENAI_API_KEY", "")
                base_url = getattr(Config, "OPENAI_BASE_URL", None)
                self.openai_client = OpenAI(api_key=api_key, base_url=base_url)
            except Exception as e:
                info = handle_api_error(e)
                log_error(e, context={"where": "ChatHandler.generate_answer", "note": "openai_init_failed", "type": info.get("type")})
                self._openai_enabled = False
                result = self._offline_answer(query, chunks)
                hist = self.conversations.setdefault(session_id, [])
                hist.append({"role": "user", "content": query})
                hist.append({"role": "assistant", "content": result.get("answer", "")})
                if len(hist) > self.max_history:
                    self.conversations[session_id] = hist[-self.max_history:]
                return result

        # 3. Get conversation history
        history = self.conversations.get(session_id, [])

        # Optional: rephrase query using recent history
        query_to_use = self.rephrase_query(query, history)

        # 4. Build messages
        messages = self.build_prompt(query_to_use, chunks, history)

        # 5. Call OpenAI API
        try:
            resp = self.openai_client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=500,
            )
        except Exception as e:
            info = handle_api_error(e)
            log_error(e, context={"where": "ChatHandler.generate_answer", "session_id": session_id, "type": info.get("type")})
            if info.get("type") == "network_timeout":
                return {"answer": "Request timed out", "sources": [], "confidence": "low", "chunks_used": len(chunks)}
            err_text = str(info.get("error", ""))
            if info.get("type") == "openai_error" and ("rate" in err_text.lower() or "429" in err_text):
                return {"answer": "Rate limit exceeded, please try again later.", "sources": [], "confidence": "low", "chunks_used": len(chunks)}
            return {"answer": "There was an error generating the answer.", "sources": [], "confidence": "low", "chunks_used": len(chunks)}

        # 6. Extract answer
        answer = (resp.choices[0].message.content or "").strip()

        # 7. Enforce citations if missing
        answer = self.enforce_inline_citations(answer, chunks)

        # 8. Extract citations
        citations = self.extract_citations(answer)

        # 9. Assess confidence
        confidence = self.assess_confidence(chunks, answer)

        # 10. Update conversation history
        hist = self.conversations.setdefault(session_id, [])
        hist.append({"role": "user", "content": query})
        hist.append({"role": "assistant", "content": answer})
        if len(hist) > self.max_history:
            self.conversations[session_id] = hist[-self.max_history:]

        # 11. Return payload
        return {
            "answer": answer,
            "sources": citations,
            "confidence": confidence,
            "chunks_used": len(chunks),
        }

    def _offline_answer(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Pick best chunk by score, else first
        best = None
        scored = [c for c in chunks if isinstance(c.get("score"), (int, float))]
        if scored:
            best = sorted(scored, key=lambda c: c.get("score", 0.0), reverse=True)[0]
        else:
            best = chunks[0]
        text = (best.get("text") or best.get("document") or "").strip()
        section = best.get("section")
        if not section and isinstance(best.get("metadata"), dict):
            section = best["metadata"].get("section")

        # Split into sentences and choose the one that best matches query keywords
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        query_clean = re.sub(r"[^a-z0-9\s]", " ", query.lower())
        keywords = {w for w in query_clean.split() if len(w) > 2}

        best_sentence = sentences[0] if sentences else text
        best_overlap = -1
        for s in sentences:
            s_lower = s.lower()
            overlap = sum(1 for w in keywords if w in s_lower)
            if overlap > best_overlap:
                best_overlap = overlap
                best_sentence = s

        citation = f"[Section {section}]" if section else ""
        answer = best_sentence.rstrip(".") + "."
        if citation:
            answer = f"{answer} {citation}"
        confidence = "medium" if best_overlap > 0 else ("medium" if scored else "low")
        sources = [citation] if citation else []
        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "chunks_used": len(chunks),
        }

    def clear_history(self, session_id: str) -> bool:
        # 1-3. Delete conversation history for a session_id
        if session_id in self.conversations:
            del self.conversations[session_id]
            return True
        return False

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        # Return history or empty list
        return list(self.conversations.get(session_id, []))

    def rephrase_query(self, query: str, history: List[Dict[str, str]]) -> str:
        # Simple heuristic: if query contains vague pronouns, prepend last explicit user question for context
        q = (query or "").strip()
        if re.search(r"\b(it|this|that|they|those|them)\b", q.lower()) and history:
            # find last user message
            last_user = next((m["content"] for m in reversed(history) if m.get("role") == "user"), "")
            if last_user:
                return f"Follow-up to: {last_user}\nQuestion: {q}"
        return query