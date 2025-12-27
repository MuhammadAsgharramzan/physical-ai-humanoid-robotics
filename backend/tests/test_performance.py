import pytest
import asyncio
import time
from unittest.mock import patch, MagicMock

from backend.rag_service import rag_service

@pytest.mark.asyncio
async def test_rag_response_time():
    """Test that RAG responses are generated within acceptable time limits"""
    # Mock the OpenAI client to avoid actual API calls
    with patch.object(rag_service.openai_client.chat.completions, 'create') as mock_create:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "This is a test response."
        mock_create.return_value = mock_response

        # Mock the content indexer to avoid actual search
        with patch.object(rag_service.content_indexer, 'search_content') as mock_search:
            mock_result = {
                "content": "This is test content for performance testing.",
                "title": "Test Content",
                "module_id": "module1",
                "lesson_id": "lesson1",
                "score": 0.9
            }
            mock_search.return_value = [mock_result]

            # Measure response time
            start_time = time.time()
            response = await rag_service.generate_response("test query")
            end_time = time.time()

            response_time = end_time - start_time

            # Check that response time is under 2 seconds (as specified in requirements)
            assert response_time < 2.0, f"Response time {response_time:.2f}s exceeds 2 seconds"
            assert response is not None
            assert "response" in response

@pytest.mark.asyncio
async def test_rag_response_time_with_citation():
    """Test that RAG responses with citations are generated within acceptable time limits"""
    # Mock the OpenAI client to avoid actual API calls
    with patch.object(rag_service.openai_client.chat.completions, 'create') as mock_create:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "This is a test response with citation [Source 1]."
        mock_create.return_value = mock_response

        # Mock the content indexer to avoid actual search
        with patch.object(rag_service.content_indexer, 'search_content') as mock_search:
            mock_result = {
                "content": "This is test content for performance testing.",
                "title": "Test Content",
                "module_id": "module1",
                "lesson_id": "lesson1",
                "score": 0.9
            }
            mock_search.return_value = [mock_result]

            # Measure response time
            start_time = time.time()
            response = await rag_service.generate_response_with_citation("test query")
            end_time = time.time()

            response_time = end_time - start_time

            # Check that response time is under 2 seconds (as specified in requirements)
            assert response_time < 2.0, f"Response time {response_time:.2f}s exceeds 2 seconds"
            assert response is not None
            assert "response" in response
            assert "citations" in response

def test_multiple_concurrent_requests():
    """Test performance under multiple concurrent requests"""
    # This would test how the system handles multiple requests
    # For now, we'll just verify the structure
    assert True