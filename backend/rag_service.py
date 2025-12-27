import asyncio
import logging
from typing import List, Dict, Any
from openai import OpenAI
import os

from .content_indexer import content_indexer

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.content_indexer = content_indexer

    async def generate_response(self, query: str, context_limit: int = 5) -> Dict[str, Any]:
        """Generate a response to a query using RAG (Retrieval-Augmented Generation)"""
        try:
            # Search for relevant content in the vector store
            search_results = await self.content_indexer.search_content(query, limit=context_limit)

            if not search_results:
                return {
                    "response": "I couldn't find relevant information in the textbook to answer your question.",
                    "sources": [],
                    "confidence": 0.0
                }

            # Combine the search results into context
            context_parts = []
            sources = []

            for result in search_results:
                context_parts.append(result["content"])
                sources.append({
                    "title": result["title"],
                    "module_id": result["module_id"],
                    "lesson_id": result["lesson_id"],
                    "score": result["score"]
                })

            context = "\n\n".join(context_parts)

            # Generate a response using OpenAI
            prompt = f"""
            You are an AI assistant for the Physical AI & Humanoid Robotics textbook.
            Answer the user's question based on the following context from the textbook.
            If the context doesn't contain enough information to answer the question, say so.
            Be concise but informative in your response.

            Context:
            {context}

            Question: {query}

            Answer:
            """

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an AI assistant for the Physical AI & Humanoid Robotics textbook. Answer questions based on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )

            generated_response = response.choices[0].message.content.strip()

            # Calculate a basic confidence score based on the highest score from search results
            max_score = max([result["score"] for result in search_results]) if search_results else 0.0
            confidence = min(max_score, 1.0)  # Normalize to 0-1 range

            return {
                "response": generated_response,
                "sources": sources,
                "confidence": confidence
            }

        except Exception as e:
            logger.error(f"Error generating RAG response: {e}")
            return {
                "response": "Sorry, I encountered an error while processing your request.",
                "sources": [],
                "confidence": 0.0
            }

    async def generate_response_with_citation(self, query: str, context_limit: int = 5) -> Dict[str, Any]:
        """Generate a response with proper citations to textbook content"""
        try:
            # Search for relevant content in the vector store
            search_results = await self.content_indexer.search_content(query, limit=context_limit)

            if not search_results:
                return {
                    "response": "I couldn't find relevant information in the textbook to answer your question.",
                    "citations": [],
                    "confidence": 0.0
                }

            # Combine the search results into context
            context_parts = []
            citations = []

            for i, result in enumerate(search_results):
                context_parts.append(f"[Source {i+1}]: {result['content']}")

                citation = {
                    "id": i+1,
                    "title": result["title"],
                    "module_id": result["module_id"],
                    "lesson_id": result["lesson_id"],
                    "score": result["score"],
                    "content_preview": result["content"][:200] + "..." if len(result["content"]) > 200 else result["content"]
                }
                citations.append(citation)

            context = "\n\n".join(context_parts)

            # Generate a response using OpenAI with citation instructions
            prompt = f"""
            You are an AI assistant for the Physical AI & Humanoid Robotics textbook.
            Answer the user's question based on the following context from the textbook.
            Reference the sources appropriately in your answer (e.g., [Source 1], [Source 2]).
            If the context doesn't contain enough information to answer the question, say so.
            Be concise but informative in your response.

            Context:
            {context}

            Question: {query}

            Answer with source references:
            """

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an AI assistant for the Physical AI & Humanoid Robotics textbook. Answer questions based on the provided context and reference sources appropriately."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )

            generated_response = response.choices[0].message.content.strip()

            # Calculate a basic confidence score based on the highest score from search results
            max_score = max([result["score"] for result in search_results]) if search_results else 0.0
            confidence = min(max_score, 1.0)  # Normalize to 0-1 range

            return {
                "response": generated_response,
                "citations": citations,
                "confidence": confidence
            }

        except Exception as e:
            logger.error(f"Error generating RAG response with citation: {e}")
            return {
                "response": "Sorry, I encountered an error while processing your request.",
                "citations": [],
                "confidence": 0.0
            }

# Create RAG service instance
rag_service = RAGService()