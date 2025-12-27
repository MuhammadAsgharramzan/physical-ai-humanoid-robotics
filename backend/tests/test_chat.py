import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import asyncio

from backend.main import app
from backend.conversation_manager import MessageType

client = TestClient(app)

def test_chat_endpoint():
    """Test the chat endpoint functionality"""
    # Mock the RAG service to avoid actual API calls
    with patch('backend.main.rag_service.generate_response_with_citation') as mock_rag:
        mock_response = {
            "response": "This is a test response.",
            "citations": [],
            "confidence": 0.9
        }
        mock_rag.return_value = mock_response

        # Mock the conversation manager
        with patch('backend.main.conversation_manager.add_message') as mock_add_msg, \
             patch('backend.main.conversation_manager.get_conversation_context') as mock_get_context:

            mock_get_context.return_value = []

            response = client.post(
                "/chat",
                params={"conversation_id": "test_conv", "user_message": "Hello, world!"}
            )

            assert response.status_code == 200
            data = response.json()
            assert "response" in data
            assert "citations" in data
            assert "confidence" in data
            assert data["conversation_id"] == "test_conv"

def test_get_conversation():
    """Test getting a conversation"""
    # Mock the conversation manager
    with patch('backend.main.conversation_manager.get_conversation') as mock_get_conv:
        # Create a mock conversation object
        mock_conv = MagicMock()
        mock_conv.get_messages.return_value = [
            {"role": "user", "content": "Hello", "timestamp": "2023-01-01T00:00:00Z"}
        ]
        mock_conv.created_at.isoformat.return_value = "2023-01-01T00:00:00Z"
        mock_conv.last_updated.isoformat.return_value = "2023-01-01T00:00:00Z"
        mock_get_conv.return_value = mock_conv

        response = client.get("/chat/conversation/test_conv")

        assert response.status_code == 200
        data = response.json()
        assert "conversation_id" in data
        assert "messages" in data

def test_get_nonexistent_conversation():
    """Test getting a nonexistent conversation"""
    with patch('backend.main.conversation_manager.get_conversation') as mock_get_conv:
        mock_get_conv.return_value = None

        response = client.get("/chat/conversation/nonexistent_conv")

        assert response.status_code == 404

def test_index_all_content():
    """Test the content indexing endpoint"""
    with patch('backend.main.content_pipeline.index_all_content') as mock_index:
        mock_index.return_value = True

        response = client.post("/content/index-all")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "successfully" in data["message"]