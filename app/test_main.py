from fastapi.testclient import TestClient

from .main import app

client = TestClient(app)

def test_read_acc():
    response = client.get("/accuracy/6")
    assert response.status_code == 200

def test_read_acc_nodepth():
    response = client.get("/accuracy")
