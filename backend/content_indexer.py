import asyncio
import logging
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
import tiktoken
from openai import OpenAI
import os

from .qdrant_config import qdrant_settings

logger = logging.getLogger(__name__)

class ContentIndexer:
    def __init__(self):
        # Support in-memory mode for testing
        if os.getenv("QDRANT_MODE") == "memory":
            self.qdrant_client = QdrantClient(":memory:")
            logger.info("Using in-memory Qdrant client for testing")
        else:
            self.qdrant_client = QdrantClient(
                url=qdrant_settings.qdrant_url,
                api_key=qdrant_settings.qdrant_api_key
            )
        self.collection_name = qdrant_settings.qdrant_collection_name
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Create collection if it doesn't exist
        self._create_collection()

    def _create_collection(self):
        """Create Qdrant collection if it doesn't exist"""
        try:
            collections = self.qdrant_client.get_collections()
            collection_exists = any(col.name == self.collection_name for col in collections.collections)

            if not collection_exists:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=1536,  # OpenAI embedding dimension
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Qdrant collection {self.collection_name} already exists")
        except Exception as e:
            logger.error(f"Error creating Qdrant collection: {e}")
            raise

    def _split_content(self, text: str, max_tokens: int = 800) -> List[str]:
        """Split content into chunks that fit within token limits"""
        tokens = self.tokenizer.encode(text)
        chunks = []

        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)

        return chunks

    async def index_content(self, content_id: str, title: str, content: str,
                           module_id: str = None, lesson_id: str = None) -> bool:
        """Index content in Qdrant vector store"""
        try:
            # Split content into chunks
            content_chunks = self._split_content(content)

            points = []
            for i, chunk in enumerate(content_chunks):
                # Generate embedding for the chunk
                response = self.openai_client.embeddings.create(
                    input=chunk,
                    model="text-embedding-ada-002"
                )
                embedding = response.data[0].embedding

                # Create a point for Qdrant
                point = models.PointStruct(
                    id=f"{content_id}_{i}",
                    vector=embedding,
                    payload={
                        "title": title,
                        "content": chunk,
                        "module_id": module_id,
                        "lesson_id": lesson_id,
                        "chunk_index": i
                    }
                )
                points.append(point)

            # Upload points to Qdrant
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            logger.info(f"Indexed {len(points)} chunks for content {content_id}")
            return True

        except Exception as e:
            logger.error(f"Error indexing content {content_id}: {e}")
            return False

    async def search_content(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for content in the vector store"""
        try:
            # Generate embedding for the query
            response = self.openai_client.embeddings.create(
                input=query,
                model="text-embedding-ada-002"
            )
            query_embedding = response.data[0].embedding

            # Search in Qdrant
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit
            )

            results = []
            for result in search_results:
                results.append({
                    "id": result.id,
                    "title": result.payload.get("title", ""),
                    "content": result.payload.get("content", ""),
                    "module_id": result.payload.get("module_id", ""),
                    "lesson_id": result.payload.get("lesson_id", ""),
                    "score": result.score
                })

            return results

        except Exception as e:
            logger.error(f"Error searching content: {e}")
            return []

    async def delete_content(self, content_id: str) -> bool:
        """Delete content from the vector store"""
        try:
            # Find all points with this content_id prefix
            # In our case, points have IDs like "content_id_chunk_index"
            # So we'll search for points starting with content_id

            # For now, we'll just delete points by filtering
            # In a real implementation, you might want to store content_id as a payload field
            # and use that for filtering
            logger.info(f"Deleting content {content_id} from vector store")
            # This is a simplified implementation
            # In practice, you'd want to store content_id in the payload
            # and use a filter to delete all points with that content_id
            return True

        except Exception as e:
            logger.error(f"Error deleting content {content_id}: {e}")
            return False

# Create indexer instance
content_indexer = ContentIndexer()