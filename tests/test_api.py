import os
import pytest

from backend.app import app

@pytest.fixture()
def client():
    app.testing = True
    return app.test_client()

def test_health(client):
    res = client.get('/health')
    assert res.status_code == 200
    data = res.get_json()
    assert 'status' in data