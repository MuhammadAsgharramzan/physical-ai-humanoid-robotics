import logging
from typing import Dict, Any, List
from agents import Agent, Runner
from openai import OpenAI
import os

from ..content_indexer import content_indexer

logger = logging.getLogger(__name__)

class BaseRAGAgent:
    def __init__(self, name: str, instructions: str):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.content_indexer = content_indexer
        self.agent = Agent(
            name=name,
            instructions=instructions,
            model="gpt-3.5-turbo"  # Using the standard model for now
        )

    async def search_content(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant content in the vector store"""
        return await self.content_indexer.search_content(query, limit)

    async def generate_response(self, query: str, context: str = None) -> Dict[str, Any]:
        """Generate a response using the agent"""
        try:
            # Prepare the input for the agent
            if context:
                input_text = f"""
                Context: {context}

                Question: {query}

                Please provide a comprehensive answer based on the provided context.
                If the context doesn't contain enough information to answer the question, say so.
                """
            else:
                input_text = f"Question: {query}\n\nPlease provide an answer."

            # Run the agent
            result = Runner.run_sync(self.agent, input_text)

            return {
                "response": result.final_output,
                "sources": [],
                "confidence": 0.8  # Default confidence for agent responses
            }
        except Exception as e:
            logger.error(f"Error generating response with agent: {e}")
            return {
                "response": "Sorry, I encountered an error while processing your request.",
                "sources": [],
                "confidence": 0.0
            }