import logging
from typing import Dict, Any, List
from agents import Agent, Runner

from .base_rag_agent import BaseRAGAgent

logger = logging.getLogger(__name__)

class FullBookRAGAgent(BaseRAGAgent):
    def __init__(self):
        instructions = """
        You are an AI assistant for the Physical AI & Humanoid Robotics textbook.
        Your role is to answer questions based on the entire textbook content provided in the context.
        Use the provided context to answer questions comprehensively.
        Always reference the specific modules and lessons where information is found.
        Be thorough but concise in your responses.
        """
        super().__init__("FullBookRAGAgent", instructions)

    async def generate_response_with_citation(self, query: str, context_limit: int = 5) -> Dict[str, Any]:
        """Generate a response with citations from the full book"""
        try:
            # Search for relevant content in the vector store
            search_results = await self.search_content(query, limit=context_limit)

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

            # Generate response using the agent
            result = Runner.run_sync(self.agent, f"""
            Context:
            {context}

            Question: {query}

            Please provide a comprehensive answer with source references (e.g., [Source 1], [Source 2]).
            Reference the specific modules and lessons where information is found.
            If the context doesn't contain enough information to answer the question, say so.
            Be thorough but concise in your response.
            """)

            return {
                "response": result.final_output,
                "citations": citations,
                "confidence": min(max([result["score"] for result in search_results]), 1.0) if search_results else 0.0
            }

        except Exception as e:
            logger.error(f"Error generating full-book RAG response: {e}")
            return {
                "response": "Sorry, I encountered an error while processing your request.",
                "citations": [],
                "confidence": 0.0
            }