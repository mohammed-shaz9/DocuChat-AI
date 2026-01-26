import os
import pytest

from backend.modules.vector_store import VectorStore

skip_no_key = pytest.mark.skipif(not os.getenv('OPENAI_API_KEY'), reason='OPENAI_API_KEY not set')

@skip_no_key
def test_add_and_search_chunks():
    vs = VectorStore()
    chunks = [
        {"id": "1", "text": "Python is a programming language.", "metadata": {"section": "Intro"}},
        {"id": "2", "text": "Flask is a micro web framework for Python.", "metadata": {"section": "Web"}},
    ]
    res = vs.add_chunks(chunks)
    assert int(res.get('chunks_added', 0)) >= 2

    hits = vs.search("What is Flask?", n_results=2)
    assert isinstance(hits, list)
    assert len(hits) > 0