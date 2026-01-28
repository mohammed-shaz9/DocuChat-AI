import os
from datetime import datetime
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

# Ensure project root is on sys.path so 'backend' package imports work
import sys
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Support both package imports (when run from project root) and local imports (when run from backend/)
try:
    from backend.config import Config
    from backend.modules.doc_processor import DocumentProcessor
    from backend.modules.vector_store import VectorStore
    from backend.modules.llm_handler import ChatHandler
    from backend.utils.error_handlers import handle_api_error, log_error, register_error_handlers
except ImportError:
    from config import Config
    from modules.doc_processor import DocumentProcessor
    from modules.vector_store import VectorStore
    from modules.llm_handler import ChatHandler
    from utils.error_handlers import handle_api_error, log_error, register_error_handlers

# Load environment variables from .env if present
load_dotenv()

app = Flask(__name__)

# CORS configuration
CORS(
    app,
    origins=getattr(Config, "CORS_ORIGINS", ["*"]),
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

# Secret key for sessions
app.config["SECRET_KEY"] = getattr(Config, "FLASK_SECRET_KEY", os.getenv("FLASK_SECRET_KEY", "dev-secret"))

# Register standardized error handlers
register_error_handlers(app)

# Initialize modules
doc_processor = DocumentProcessor()
vector_store = VectorStore()
chat_handler = ChatHandler(max_history=Config.MAX_CONVERSATION_HISTORY)


@app.route("/health", methods=["GET"])
def health_check():
    """Health check and status endpoint."""
    try:
        stats = vector_store.get_stats()
        total_chunks = int(stats.get("total_chunks", 0) or 0)
        return jsonify({
            "status": "healthy",
            "doc_loaded": total_chunks > 0,
            "total_chunks": total_chunks,
            "embedding_model": stats.get("embedding_model"),
            "timestamp": datetime.now().isoformat()
        }), 200
    except Exception as e:
        log_error(e, {"endpoint": "/health"})
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


@app.route("/ingest", methods=["POST"])
def ingest_document():
    """Ingest Google Doc or raw text: validate request, process content, populate vector store."""
    try:
        # 1. Validate request
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400

        data = request.get_json() or {}
        doc_url = (data.get("doc_url") or "").strip()
        raw_text = (data.get("raw_text") or "").strip()
        if not doc_url and not raw_text:
            return jsonify({"error": "doc_url or raw_text is required"}), 400

        # 2. Process document or raw text
        if raw_text:
            cleaned = doc_processor.clean_text(raw_text)
            if len(cleaned) < 50:
                return jsonify({"error": "raw_text too short or empty (min 50 chars)."}), 400
            chunks = doc_processor.chunk_text(cleaned)
            result = {
                "success": True,
                "error": None,
                "chunks": chunks,
                "total_chunks": len(chunks),
                "total_tokens": doc_processor.count_tokens(cleaned),
            }
        else:
            result = doc_processor.process_document(doc_url)
            if not result.get("success"):
                return jsonify({"error": result.get("error")}), 400

        # 3. Clear existing vector store
        vector_store.clear_collection()

        # 4. Add chunks to vector store
        add_result = vector_store.add_chunks(result.get("chunks", []))
        if not add_result.get("success"):
            return jsonify({"error": add_result.get("error")}), 500

        # 5. Return success
        return jsonify({
            "status": "success",
            "chunks_created": int(result.get("total_chunks", 0) or 0),
            "total_tokens": int(result.get("total_tokens", 0) or 0),
            "message": "Content ingested successfully"
        }), 200
    except Exception as e:
        # Ensure doc_url is defined for context
        doc_url = locals().get("doc_url", "")
        log_error(e, {"endpoint": "/ingest", "doc_url": doc_url})
        error_response = handle_api_error(e)
        return jsonify(error_response), 500


@app.route("/chat", methods=["POST"])
def chat():
    """Process chat message using RAG: search chunks and generate an answer."""
    try:
        # 1. Validate request
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400

        data = request.get_json() or {}
        message = (data.get("message") or "").strip()
        session_id = (data.get("session_id") or "default").strip() or "default"
        if not message:
            return jsonify({"error": "message is required"}), 400

        # 2. Check document loaded
        stats = vector_store.get_stats()
        if int(stats.get("total_chunks", 0) or 0) == 0:
            return jsonify({
                "error": "No document loaded. Please ingest a document first."
            }), 400

        # 3. Search for relevant chunks
        chunks = vector_store.search(message, n_results=3)
        if not chunks:
            return jsonify({
                "answer": "I couldn't find relevant information in the document.",
                "sources": [],
                "confidence": "low"
            }), 200

        # 4. Generate answer
        response = chat_handler.generate_answer(message, chunks, session_id)

        # 5. Return response
        return jsonify({
            "answer": response.get("answer", ""),
            "sources": response.get("sources", []),
            "confidence": response.get("confidence", "medium"),
            "timestamp": datetime.now().isoformat()
        }), 200
    except Exception as e:
        log_error(e, {"endpoint": "/chat", "session_id": locals().get("session_id", "default")})
        error_response = handle_api_error(e)
        return jsonify(error_response), 500


@app.route("/clear", methods=["POST"])
def clear_conversation():
    """Clear conversation history for a session."""
    try:
        data = request.get_json() or {}
        session_id = (data.get("session_id") or "default").strip() or "default"

        chat_handler.clear_history(session_id)

        return jsonify({
            "status": "success",
            "message": "Conversation cleared"
        }), 200
    except Exception as e:
        log_error(e, {"endpoint": "/clear"})
        return jsonify({"error": str(e)}), 500


@app.route("/stats", methods=["GET"])
def get_stats():
    """Get detailed statistics for vector store and chat sessions."""
    try:
        vector_stats = vector_store.get_stats()
        return jsonify({
            "vector_store": vector_stats,
            "active_sessions": len(getattr(chat_handler, "conversations", {})),
            "config": {
                "chunk_size": getattr(Config, "CHUNK_SIZE", None),
                "max_history": getattr(Config, "MAX_CONVERSATION_HISTORY", None),
            },
            "timestamp": datetime.now().isoformat()
        }), 200
    except Exception as e:
        log_error(e, {"endpoint": "/stats"})
        return jsonify({"error": str(e)}), 500


@app.route('/')
def serve_index():
    """Serve the main HTML file"""
    return send_from_directory(os.path.join(PROJECT_ROOT, 'frontend'), 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    """Serve static files (CSS, JS, etc.)"""
    return send_from_directory(os.path.join(PROJECT_ROOT, 'frontend'), path)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)