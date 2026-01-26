from backend.modules.doc_processor import DocumentProcessor

def test_validate_url():
    dp = DocumentProcessor()
    ok = dp.validate_url("https://docs.google.com/document/d/abc123/edit")
    bad = dp.validate_url("https://example.com/foo")
    assert ok["valid"] is True
    assert bad["valid"] is False