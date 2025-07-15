import pytest
from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "embeddings_loaded" in data

@pytest.mark.xfail(reason="Depends on model and embeddings being loaded")
def test_rag_query_endpoint():
    response = client.post("/rag_query", json={"query": "Test question"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "context" in data

@pytest.mark.xfail(reason="Depends on model and embeddings being loaded")
def test_image_search_endpoint():
    # This test would require a real DICOM file upload; here we just check the endpoint exists
    response = client.post("/image_search", files={"file": ("fake.dcm", b"not a real dicom")})
    assert response.status_code == 200
    data = response.json()
    assert "results" in data 