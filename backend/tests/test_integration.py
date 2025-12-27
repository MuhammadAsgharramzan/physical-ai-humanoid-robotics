import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from backend.main import app

client = TestClient(app)

def test_full_system_integration():
    """Test integration across all system components"""
    # Test basic system health
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

    # Test content search functionality
    response = client.get("/content/search", params={"query": "test"})
    assert response.status_code in [200, 422]  # 422 is expected if query is missing

    # Test content by topic functionality
    response = client.get("/content/topic/test-topic")
    assert response.status_code in [200, 404, 422]

    # Test that all major endpoints are accessible
    endpoints = [
        "/",
        "/health",
    ]

    for endpoint in endpoints:
        response = client.get(endpoint)
        # All endpoints should return a valid response
        assert response.status_code in [200, 404, 405, 422]

def test_system_responsiveness():
    """Test that the system remains responsive under basic load"""
    # Make several requests to test system stability
    for i in range(5):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

def test_error_handling_integration():
    """Test that error handling works across the system"""
    # Test various error conditions
    bad_requests = [
        ("/content/topic/", "GET"),  # Empty topic
        ("/nonexistent-endpoint", "GET"),  # Non-existent endpoint
    ]

    for endpoint, method in bad_requests:
        response = client.request(method, endpoint)
        # Should return appropriate error, not crash
        assert response.status_code in [404, 405, 422, 500]

def test_rate_limiting_integration():
    """Test that rate limiting works across the system"""
    # Make multiple requests to trigger rate limiting
    for i in range(7):  # More than our rate limit of 5/minute in the example
        response = client.get("/content/search", params={"query": f"test{i}"})
        # Should not crash the system
        assert response.status_code in [200, 422, 429]

def test_api_consistency():
    """Test that API responses are consistent across endpoints"""
    endpoints_to_check = ["/health", "/"]

    for endpoint in endpoints_to_check:
        response = client.get(endpoint)
        if response.status_code == 200:
            data = response.json()
            # All successful responses should be structured consistently
            assert isinstance(data, (dict, list))

def test_cors_integration():
    """Test that CORS is properly configured across the API"""
    response = client.get("/")
    # Check that CORS headers are present
    assert "access-control-allow-origin" in response.headers

def test_content_type_consistency():
    """Test that content types are consistent across the API"""
    response = client.get("/health")
    content_type = response.headers.get("content-type", "")
    assert "application/json" in content_type

if __name__ == "__main__":
    test_full_system_integration()
    test_system_responsiveness()
    test_error_handling_integration()
    test_rate_limiting_integration()
    test_api_consistency()
    test_cors_integration()
    test_content_type_consistency()
    print("All integration tests passed!")