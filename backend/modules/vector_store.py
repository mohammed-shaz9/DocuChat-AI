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
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.api.types import Documents, Embeddings, IDs, Metadatas
from openai import OpenAI


class VectorStore:
    def __init__(self, persist_directory: str = ".chromadb") -> None:
        """Initialize persistent ChromaDB and OpenAI client.

        - Creates or retrieves a collection named "doc_chunks" using cosine similarity.
        - Reads OPENAI_API_KEY from environment (required for embedding calls).
        """
        self.persist_directory = persist_directory
        self.collection_name = "doc_chunks"

        # Persistent ChromaDB client and collection (cosine space)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # OpenAI API key & client
        self.api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        # The OpenAI client will also read from env if api_key is None
        self.openai = OpenAI(api_key=self.api_key) if self.api_key else OpenAI()

    # 2. generate_embeddings(texts)
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.

        - Batches requests in chunks of up to 100 inputs
        - Retries once after a 2-second delay on failure
        - Returns a list of 1536-d float vectors
        """
        if not texts:
            return []
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in environment.")

        embeddings: List[List[float]] = []
        batch_size = 100

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            def _embed(inputs: List[str]):
                return self.openai.embeddings.create(
                    model="text-embedding-3-small",
                    input=inputs,
                )

            try:
                resp = _embed(batch)
            except Exception:
                time.sleep(2)
                resp = _embed(batch)

            batch_vectors = [d.embedding for d in resp.data]
            embeddings.extend(batch_vectors)

        return embeddings

    # 3. add_chunks(chunks)
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Upsert chunk dictionaries into the ChromaDB collection.

        Expected chunk format:
        {"id": str, "text": str, "metadata": {"section": str, "token_count": int}}
        """
        if not chunks:
            return {"success": True, "chunks_added": 0}

        ids: IDs = [str(c["id"]) for c in chunks]
        documents: Documents = [str(c["text"]) for c in chunks]
        metadatas: Metadatas = [c.get("metadata", {}) for c in chunks]

        vectors: Embeddings = self.generate_embeddings(documents)

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=vectors,
        )

        return {"success": True, "chunks_added": len(ids)}

    # 4. search(query, n_results=3)
    def search(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Search for the most similar chunks to the query text.

        Returns a list of dicts: {"text", "section", "score"} where score is 1 - distance.
        """
        query_vec = self.generate_embeddings([query])[0]

        res = self.collection.query(
            query_embeddings=[query_vec],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        results: List[Dict[str, Any]] = []
        for doc, meta, dist in zip(docs, metas, dists):
            score = None
            try:
                score = 1.0 - float(dist)
            except Exception:
                score = None
            results.append({
                "text": doc,
                "section": meta.get("section") if isinstance(meta, dict) else None,
                "score": score,
            })
        return results

    # 5. clear_collection()
    def clear_collection(self) -> None:
        """Delete and recreate the collection."""
        # Delete if exists
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            # If deletion fails or collection does not exist, ignore
            pass
        # Recreate
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # 6. get_stats()
    def get_stats(self) -> Dict[str, Any]:
        """Return simple statistics about the collection."""
        try:
            count = self.collection.count() if self.collection else 0
            exists = self.collection is not None
        except Exception:
            count = 0
            exists = False
        return {"total_chunks": count, "collection_exists": exists}