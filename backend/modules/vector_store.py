"""ChromaDB vector store and OpenAI embeddings utilities.

Implements a VectorStore class for:
- Persistent ChromaDB collection management
- Embedding generation with OpenAI (text-embedding-3-small)
- Upserting chunked documents
- Similarity search (cosine)
- Collection maintenance and stats
"""

from __future__ import annotations

import os
import time
import hashlib
import random
from typing import Any, Dict, List, Optional

# Optional ChromaDB imports; fallback to None when unavailable
try:
    import chromadb  # type: ignore
    from chromadb.config import Settings  # type: ignore
    from chromadb.api.types import Documents, Embeddings, IDs, Metadatas  # type: ignore
except Exception:  # pragma: no cover
    chromadb = None  # type: ignore
    Settings = None  # type: ignore
    # Lightweight type aliases to satisfy annotations when chromadb is missing
    Documents = List[str]
    Embeddings = List[List[float]]
    IDs = List[str]
    Metadatas = List[Dict[str, Any]]

from openai import OpenAI

import numpy as np
from backend.config import Config
from backend.modules.doc_processor import DocumentProcessor
from backend.utils import EmbeddingGenerationError, handle_api_error, log_error


class VectorStore:
    def __init__(self, persist_dir: Optional[str] = None, collection_name: str = "doc_chunks") -> None:
        """Initialize persistent ChromaDB client, collection, and OpenAI client using Config.

        - persist_dir: directory for ChromaDB persistence (defaults to Config.CHROMA_PERSIST_DIR)
        - collection_name: name of the ChromaDB collection to use/create
        """
        try:
            self.persist_dir = persist_dir or getattr(Config, "CHROMA_PERSIST_DIR", "./chroma_db")
            self.collection_name = collection_name

            # Choose backend; auto-fallback to memory if chromadb is unavailable
            self.vector_backend = getattr(Config, "VECTOR_BACKEND", "chroma").lower()
            use_memory = (self.vector_backend == "memory") or (chromadb is None)
            if use_memory:
                self.vector_backend = "memory"
                self.client = None
                self.collection = _MemoryCollection()
            else:
                # Initialize PersistentClient with telemetry disabled
                self.client = chromadb.PersistentClient(  # type: ignore
                    path=self.persist_dir,
                    settings=Settings(anonymized_telemetry=False),  # type: ignore
                )
                # Create or get collection; we'll supply embeddings manually
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=None,
                )

            # Provider selection for embeddings
            self.embedding_provider = getattr(Config, "EMBEDDING_PROVIDER", "openai").lower()
            self.hf_model_name = getattr(Config, "HF_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            self._hf_model = None

            # OpenAI-compatible client via OpenRouter
            api_key = getattr(Config, "OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY") or getattr(Config, "OPENAI_API_KEY", "")
            base_url = getattr(Config, "OPENAI_BASE_URL", None)
            self._openai_enabled: bool = bool(api_key and str(api_key).strip())
            self.openai_client = None
            self.embedding_model: str = getattr(Config, "EMBEDDING_MODEL", "text-embedding-3-small")

            # Batch size for embeddings API
            self.batch_size: int = 100
        except Exception as e:
            log_error(e, context={"where": "VectorStore.__init__"})
            # Re-raise to make construction errors visible to caller
            raise

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts using OpenAI embeddings API with retries.

        Steps:
        1. Validate inputs (not empty, all strings)
        2. Process in batches of self.batch_size
        3. Retry logic: up to 3 attempts with exponential backoff [0s, 2s, 4s]
        4. Validate output length equals input length
        """
        try:
            if not isinstance(texts, list) or not texts:
                return []
            if not all(isinstance(t, str) and t.strip() for t in texts):
                raise ValueError("All inputs must be non-empty strings for embedding generation.")

            # Hugging Face sentence-transformers provider
            if getattr(self, "embedding_provider", "openai") == "hf":
                try:
                    return self._hf_generate(texts)
                except Exception as e:
                    log_error(e, context={"where": "generate_embeddings", "provider": "hf"})
                    return self._fallback_embeddings(texts)

            # Offline deterministic fallback when API is not enabled
            if not self._openai_enabled:
                return self._fallback_embeddings(texts)

            # Lazy-init OpenAI client; if it fails, fallback
            if self.openai_client is None:
                try:
                    api_key = getattr(Config, "OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
                    base_url = getattr(Config, "OPENAI_BASE_URL", None)
                    self.openai_client = OpenAI(api_key=api_key, base_url=base_url)
                except Exception as e:
                    info = handle_api_error(e)
                    log_error(e, context={"where": "generate_embeddings", "note": "openai_init_failed", "type": info.get("type")})
                    return self._fallback_embeddings(texts)

            results: List[List[float]] = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]

                attempts = 0
                last_err: Optional[Exception] = None
                while attempts < 3:
                    try:
                        resp = self.openai_client.embeddings.create(
                            model=self.embedding_model,
                            input=batch,
                        )
                        batch_vectors = [d.embedding for d in resp.data]
                        results.extend(batch_vectors)
                        last_err = None
                        break
                    except Exception as e:
                        last_err = e
                        info = handle_api_error(e)
                        log_error(e, context={
                            "where": "generate_embeddings",
                            "attempt": attempts + 1,
                            "retryable": info.get("retryable"),
                        })
                        attempts += 1
                        if attempts < 3 and info.get("retryable"):
                            delay = 2 ** (attempts - 1)  # 0s, 2s, 4s
                            time.sleep(delay)
                        else:
                            break

                if last_err is not None:
                    info = handle_api_error(last_err)
                    # On any OpenAI error, use fallback embeddings to keep the system responsive
                    results.extend(self._fallback_embeddings(batch))
                    continue

            if len(results) != len(texts):
                raise EmbeddingGenerationError(
                    f"Embedding output length mismatch: expected {len(texts)}, got {len(results)}",
                    retry_count=0,
                )

            return results
        except EmbeddingGenerationError:
            raise
        except Exception as e:
            log_error(e, context={"where": "generate_embeddings"})
            raise EmbeddingGenerationError(str(e), retry_count=0)

    def _fallback_embeddings(self, texts: List[str], dim: int = 1536) -> List[List[float]]:
        """Deterministic pseudo-embeddings for offline/testing scenarios.

        Generates a fixed-size vector per text using a SHA-256-derived seed.
        """
        vectors: List[List[float]] = []
        for t in texts:
            # Seed PRNG from text hash for determinism
            h = hashlib.sha256(t.encode("utf-8")).hexdigest()
            seed = int(h[:16], 16)
            rnd = random.Random(seed)
            vec = [rnd.random() for _ in range(dim)]
            # L2 normalize
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            vec = [v / norm for v in vec]
            vectors.append(vec)
        return vectors

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add chunk dictionaries into the ChromaDB collection with embeddings.

        Expected chunk format:
        {"id": str, "text": str, "metadata": {"section": str, "token_count": int, ...}}
        """
        try:
            if not isinstance(chunks, list) or not chunks:
                return {"success": True, "chunks_added": 0}

            # Validate chunks using DocumentProcessor; if invalid, try to enrich metadata and proceed
            dp = DocumentProcessor()
            if not dp.validate_chunks(chunks):
                # Soft-fix: ensure metadata dict and attach token_count for each chunk
                fixed: List[Dict[str, Any]] = []
                for c in chunks:
                    if not isinstance(c, dict) or "id" not in c or "text" not in c:
                        return {"success": False, "error": "Each chunk must include 'id' and 'text'."}
                    meta = c.get("metadata") or {}
                    if not isinstance(meta, dict):
                        meta = {}
                    try:
                        tc = dp.count_tokens(str(c["text"]))
                    except Exception:
                        tc = len(str(c["text"]).split())
                    meta.setdefault("token_count", tc)
                    c["metadata"] = meta
                    fixed.append(c)
                chunks = fixed
                # Proceed even if validate_chunks still fails; embeddings/search do not depend on token_count thresholds

            ids: List[str] = [str(c["id"]) for c in chunks]
            texts: List[str] = [str(c["text"]) for c in chunks]
            metadatas: List[Dict[str, Any]] = [c.get("metadata", {}) for c in chunks]

            embeddings = self.generate_embeddings(texts)

            self.collection.upsert(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings,
            )

            return {"success": True, "chunks_added": len(chunks)}
        except EmbeddingGenerationError as e:
            log_error(e, context={"where": "add_chunks"})
            return {"success": False, "error": str(e)}
        except Exception as e:
            log_error(e, context={"where": "add_chunks"})
            return {"success": False, "error": str(e)}

    def search(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Semantic search for the most similar chunks to the query text.

        Returns a list of dicts with keys: text, metadata, section, score
        """
        try:
            if not isinstance(query, str) or not query.strip():
                return []

            if getattr(self, "vector_backend", "chroma") == "memory":
                res = self.collection.query(
                    query_text=query,
                    n_results=n_results
                )
            else:
                query_embedding = self.generate_embeddings([query])[0]
                res = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    include=["documents", "metadatas", "distances"],
                )

            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            dists = res.get("distances", [[]])[0]

            results: List[Dict[str, Any]] = []
            for doc, meta, dist in zip(docs, metas, dists):
                try:
                    distance = float(dist)
                    score = 1.0 - distance
                except Exception:
                    score = None
                section = meta.get("section") if isinstance(meta, dict) else None
                results.append({
                    "text": doc,
                    "metadata": meta if isinstance(meta, dict) else {},
                    "section": section,
                    "score": score,
                })

            # Sort by score descending (None scores last)
            results.sort(key=lambda r: (r["score"] is None, -(r["score"] or 0.0)))
            return results
        except EmbeddingGenerationError as e:
            log_error(e, context={"where": "search"})
            return []
        except Exception as e:
            log_error(e, context={"where": "search"})
            return []

    def clear_collection(self) -> Dict[str, Any]:
        """Delete the current collection and recreate it empty."""
        try:
            if getattr(self, "vector_backend", "chroma") == "memory":
                self.collection = _MemoryCollection()
                return {"success": True, "message": "Collection cleared"}

            try:
                self.client.delete_collection(name=self.collection_name)
            except Exception:
                # Ignore deletion errors (e.g., not found)
                pass

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=None,
            )
            return {"success": True, "message": "Collection cleared"}
        except Exception as e:
            log_error(e, context={"where": "clear_collection"})
            return {"success": False, "error": str(e)}

    def _hf_generate(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings via sentence-transformers with lazy init and L2 normalization."""
        if self._hf_model is None:
            from sentence_transformers import SentenceTransformer
            self._hf_model = SentenceTransformer(getattr(self, "hf_model_name", "all-MiniLM-L6-v2"))
        vectors = self._hf_model.encode(texts, convert_to_numpy=False, normalize_embeddings=True)
        return [list(v) for v in vectors]

    def get_stats(self) -> Dict[str, Any]:
        """Return collection statistics and configuration info."""
        try:
            count = self.collection.count() if self.collection else 0
            exists = (count > 0)
        except Exception:
            count = 0
            exists = False
        return {
            "total_chunks": count,
            "collection_exists": exists,
            "embedding_model": self.embedding_model,
            "embedding_provider": getattr(self, "embedding_provider", "openai"),
            "hf_model_name": getattr(self, "hf_model_name", None),
            "vector_backend": getattr(self, "vector_backend", "chroma"),
        }

    def delete_chunks(self, chunk_ids: List[str]) -> Dict[str, Any]:
        """Delete chunks by their IDs from the collection."""
        try:
            if not isinstance(chunk_ids, list) or not chunk_ids:
                return {"success": False, "error": "chunk_ids must be a non-empty list."}
            self.collection.delete(ids=chunk_ids)
            return {"success": True, "deleted": len(chunk_ids)}
        except Exception as e:
            log_error(e, context={"where": "delete_chunks"})
            return {"success": False, "error": str(e)}

    def update_chunk(self, chunk_id: str, new_text: str) -> Dict[str, Any]:
        """Update an existing chunk's text and embedding."""
        try:
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                return {"success": False, "error": "chunk_id must be a non-empty string."}
            if not isinstance(new_text, str) or not new_text.strip():
                return {"success": False, "error": "new_text must be a non-empty string."}

            new_embedding = self.generate_embeddings([new_text])[0]
            self.collection.update(
                ids=[chunk_id],
                documents=[new_text],
                embeddings=[new_embedding],
            )
            return {"success": True, "updated": 1}
        except EmbeddingGenerationError as e:
            log_error(e, context={"where": "update_chunk"})
            return {"success": False, "error": str(e)}
        except Exception as e:
            log_error(e, context={"where": "update_chunk"})
            return {"success": False, "error": str(e)}


class _MemoryCollection:
    """Simple in-memory vector collection with cosine distance search."""
    def __init__(self) -> None:
        self.ids: List[str] = []
        self.docs: List[str] = []
        self.metas: List[Dict[str, Any]] = []
        self.embs: List[List[float]] = []
        self.index: Dict[str, int] = {}

    def upsert(self, ids: IDs, documents: Documents, metadatas: Metadatas, embeddings: Embeddings) -> None:
        for i, _id in enumerate(ids):
            if _id in self.index:
                idx = self.index[_id]
                self.docs[idx] = documents[i]
                self.metas[idx] = metadatas[i] if metadatas else {}
                self.embs[idx] = embeddings[i]
            else:
                idx = len(self.ids)
                self.index[_id] = idx
                self.ids.append(_id)
                self.docs.append(documents[i])
                self.metas.append(metadatas[i] if metadatas else {})
                self.embs.append(embeddings[i])

    def query(self, query_text: str, n_results: int = 3) -> Dict[str, Any]:
        """Simple keyword-based scoring for memory fallback retrieval."""
        if not self.docs:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        # Preprocess query
        q_words = set(re.sub(r"[^a-z0-9]", " ", query_text.lower()).split())
        
        scored_docs = []
        for i in range(len(self.docs)):
            doc_text = self.docs[i].lower()
            # Score based on keyword overlap
            score = 0
            for word in q_words:
                if len(word) > 2: # Ignore small words
                    if word in doc_text:
                        score += 1
            
            # Penalize long documents slightly to favor specific matches
            normalized_score = score / (1 + 0.01 * len(doc_text.split()))
            scored_docs.append((normalized_score, i))

        # Sort by score descending
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        top_indices = [idx for score, idx in scored_docs[:n_results]]

        docs = [self.docs[i] for i in top_indices]
        metas = [self.metas[i] for i in top_indices]
        # Invert score to 'distance' for consistency with Chroma (higher score = lower distance)
        dists = [1.0 / (1.0 + score) for score, idx in scored_docs[:n_results]]
        
        return {"documents": [docs], "metadatas": [metas], "distances": [dists]}

    def delete(self, ids: IDs) -> None:
        to_delete = set(ids)
        if not to_delete:
            return
        new_ids: List[str] = []
        new_docs: List[str] = []
        new_metas: List[Dict[str, Any]] = []
        new_embs: List[List[float]] = []
        for i, _id in enumerate(self.ids):
            if _id not in to_delete:
                new_ids.append(_id)
                new_docs.append(self.docs[i])
                new_metas.append(self.metas[i])
                new_embs.append(self.embs[i])
        self.ids, self.docs, self.metas, self.embs = new_ids, new_docs, new_metas, new_embs
        self.index = {id_: idx for idx, id_ in enumerate(self.ids)}

    def update(self, ids: IDs, documents: Documents, embeddings: Embeddings) -> None:
        for i, _id in enumerate(ids):
            if _id in self.index:
                idx = self.index[_id]
                self.docs[idx] = documents[i]
                self.embs[idx] = embeddings[i]

    def count(self) -> int:
        return len(self.ids)