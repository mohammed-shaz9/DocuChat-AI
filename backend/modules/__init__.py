"""
Backend modules for RAG Chatbot.

This package contains the core processing modules:
- doc_processor: Google Doc fetching and chunking
- vector_store: ChromaDB embeddings and semantic search  
- llm_handler: OpenAI chat completion and conversation management

Usage:
    from modules.doc_processor import DocumentProcessor
    from modules.vector_store import VectorStore
    from modules.llm_handler import ChatHandler
"""

from .doc_processor import DocumentProcessor
from .vector_store import VectorStore
from .llm_handler import ChatHandler

__all__ = [
    'DocumentProcessor',
    'VectorStore', 
    'ChatHandler'
]

__version__ = '1.0.0'