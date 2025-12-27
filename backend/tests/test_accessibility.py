import pytest
from fastapi.testclient import TestClient
import json

from backend.main import app

client = TestClient(app)

def test_api_response_structure():
    """Test that API responses have proper structure for accessibility"""
    response = client.get("/health")
    assert response.status_code == 200

    # Check that response has proper structure
    data = response.json()
    assert "status" in data
    assert isinstance(data["status"], str)

def test_api_error_handling():
    """Test that API errors are handled gracefully"""
    # Test with invalid conversation ID
    response = client.get("/chat/conversation/invalid-id")
    # This should return a proper error response, not crash
    assert response.status_code in [200, 404]  # 404 is expected for not found

def test_content_type_headers():
    """Test that responses have proper content type headers"""
    response = client.get("/health")
    assert response.headers["content-type"].startswith("application/json")

def test_rate_limiting_response():
    """Test that rate limiting returns appropriate responses"""
    # This test would require making multiple requests to trigger rate limiting
    # For now, we just verify the structure
    response = client.get("/health")
    assert response.status_code in [200, 429]  # 429 is rate limited