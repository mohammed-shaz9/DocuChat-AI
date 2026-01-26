import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv

from config import Config
from modules.doc_processor import DocumentProcessor
from modules.vector_store import VectorStore
from modules.llm_handler import ChatHandler
from utils.error_handlers import register_error_handlers

# Load environment variables from .env if present
load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "dev-secret")

# Enable CORS with configured origins
CORS(app, origins=Config.CORS_ORIGINS)
register_error_handlers(app)

# Initialize core components
processor = DocumentProcessor()
vector_store = VectorStore()
chat_handler = ChatHandler(max_history=Config.MAX_CONVERSATION_HISTORY)


@app.get("/health")
def health():
    """Health endpoint with vector store stats."""
    stats = vector_store.get_stats()
    total = int(stats.get("total_chunks", 0) or 0)
    exists = bool(stats.get("collection_exists", False))
    doc_loaded = exists and total > 0
    return jsonify({"status": "healthy", "doc_loaded": doc_loaded, "total_chunks": total})


@app.post("/ingest")
def ingest():
    """Ingest a Google Doc URL, process and store chunks in vector store."""
    data = request.get_json(silent=True) or {}
    doc_url = (data.get("doc_url") or "").strip()
    if not doc_url:
        return jsonify({"error": "doc_url is required"}), 400

    result = processor.process_document(doc_url, chunk_size=Config.CHUNK_SIZE)
    if not result.get("success"):
        return jsonify({"error": result.get("error") or "Failed to process document"}), 400

    chunks = result.get("chunks", [])
    try:
        vector_store.clear_collection()
        add_res = vector_store.add_chunks(chunks)
    except Exception as e:
        return jsonify({"error": f"Failed to ingest chunks: {e}"}), 500

    return jsonify({"status": "success", "chunks_created": int(add_res.get("chunks_added", 0))})


@app.post("/chat")
def chat():
    """Chat endpoint that searches relevant chunks and generates an answer with citations."""
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    session_id = (data.get("session_id") or "default").strip() or "default"

    if not message:
        return jsonify({"error": "message is required"}), 400

    stats = vector_store.get_stats()
    total = int(stats.get("total_chunks", 0) or 0)
    exists = bool(stats.get("collection_exists", False))
    if not (exists and total > 0):
        return jsonify({"error": "No document is loaded yet. Please ingest a document first."}), 400

    try:
        chunks = vector_store.search(message, n_results=3)
        result = chat_handler.generate_answer(message, chunks, session_id)
    except Exception as e:
        return jsonify({"error": f"Failed to generate answer: {e}"}), 500

    return jsonify({"answer": result.get("answer", ""), "sources": result.get("sources", [])})


@app.post("/clear")
def clear():
    """Clear conversation history for a session."""
    data = request.get_json(silent=True) or {}
    session_id = (data.get("session_id") or "default").strip() or "default"
    chat_handler.clear_history(session_id)
    return jsonify({"status": "success", "message": "Conversation history cleared."})




if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)