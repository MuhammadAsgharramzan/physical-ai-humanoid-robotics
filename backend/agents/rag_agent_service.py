import logging
from typing import Dict, Any
from enum import Enum

from .full_book_rag_agent import FullBookRAGAgent
from .selected_text_rag_agent import SelectedTextRAGAgent

logger = logging.getLogger(__name__)

class RAGMode(Enum):
    FULL_BOOK = "full_book"
    SELECTED_TEXT = "selected_text"

class RAGAgentService:
    def __init__(self):
        self.full_book_agent = FullBookRAGAgent()
        self.selected_text_agent = SelectedTextRAGAgent()

    async def generate_response(
        self,
        query: str,
        mode: RAGMode = RAGMode.FULL_BOOK,
        selected_text: str = None,
        context_limit: int = 5
    ) -> Dict[str, Any]:
        """Generate response based on the specified mode"""
        try:
            if mode == RAGMode.FULL_BOOK:
                return await self.full_book_agent.generate_response_with_citation(query, context_limit)
            elif mode == RAGMode.SELECTED_TEXT:
                return await self.selected_text_agent.generate_response_with_selected_text(query, selected_text or "")
            else:
                raise ValueError(f"Invalid mode: {mode}")
        except Exception as e:
            logger.error(f"Error in RAG agent service: {e}")
            return {
                "response": "Sorry, I encountered an error while processing your request.",
                "citations": [],
                "confidence": 0.0
            }

# Create RAG agent service instance
rag_agent_service = RAGAgentService()