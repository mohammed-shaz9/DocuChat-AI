from __future__ import annotations

from flask import jsonify


def register_error_handlers(app):
    @app.errorhandler(404)
    def handle_404(error):
        return jsonify({"error": "Not Found"}), 404

    @app.errorhandler(500)
    def handle_500(error):
        return jsonify({"error": "Internal Server Error"}), 500

    # Optional: 400 for bad requests
    @app.errorhandler(400)
    def handle_400(error):
        return jsonify({"error": "Bad Request"}), 400