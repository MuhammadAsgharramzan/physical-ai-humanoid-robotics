import pytest
import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set dummy API key for testing
os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["QDRANT_MODE"] = "memory"

from backend.content_indexer import ContentIndexer

@pytest.fixture
def content_indexer():
    """Create a content indexer instance for testing"""
    indexer = ContentIndexer()
    return indexer

@pytest.mark.asyncio
async def test_split_content():
    """Test content splitting functionality"""
    indexer = ContentIndexer()

    # Test with a short text
    text = "This is a test sentence. " * 10  # Repeat to make it longer
    chunks = indexer._split_content(text, max_tokens=20)

    assert len(chunks) > 0
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert all(len(chunk) > 0 for chunk in chunks)

@pytest.mark.asyncio
async def test_index_content():
    """Test content indexing functionality"""
    indexer = ContentIndexer()

    # Mock the OpenAI embedding call
    with patch.object(indexer.openai_client.embeddings, 'create') as mock_create:
        mock_response = MagicMock()
        mock_response.data = [MagicMock()]
        mock_response.data[0].embedding = [0.1] * 1536  # Mock embedding
        mock_create.return_value = mock_response

        # Mock the Qdrant client
        with patch.object(indexer.qdrant_client, 'upsert') as mock_upsert:
            mock_upsert.return_value = True

            result = await indexer.index_content(
                content_id="test123",
                title="Test Content",
                content="This is test content for indexing.",
                module_id="module1",
                lesson_id="lesson1"
            )

            assert result is True
            mock_upsert.assert_called_once()

@pytest.mark.asyncio
async def test_search_content():
    """Test content search functionality"""
    indexer = ContentIndexer()

    # Mock the OpenAI client
    mock_openai = MagicMock()
    indexer.openai_client = mock_openai
    
    mock_response = MagicMock()
    mock_response.data = [MagicMock()]
    mock_response.data[0].embedding = [0.1] * 1536
    mock_openai.embeddings.create.return_value = mock_response

    # Mock the Qdrant client
    mock_qdrant = MagicMock()
    indexer.qdrant_client = mock_qdrant
    
    mock_result = MagicMock()
    mock_result.id = "test_id"
    mock_result.payload = {
        "title": "Test Title",
        "content": "Test content",
        "module_id": "module1",
        "lesson_id": "lesson1"
    }
    mock_result.score = 0.9
    mock_qdrant.search.return_value = [mock_result]

    results = await indexer.search_content("test query")

    assert len(results) == 1
    assert results[0]["title"] == "Test Title"
    assert results[0]["content"] == "Test content"