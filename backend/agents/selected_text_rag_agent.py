import logging
from typing import Dict, Any, List
from agents import Agent, Runner

from .base_rag_agent import BaseRAGAgent

logger = logging.getLogger(__name__)

class SelectedTextRAGAgent(BaseRAGAgent):
    def __init__(self):
        instructions = """
        You are an AI assistant for the Physical AI & Humanoid Robotics textbook.
        Your role is to answer questions based only on the specific selected text provided in the context.
        Do not use any external knowledge or information beyond what's provided in the context.
        If the selected text doesn't contain enough information to answer the question, say so.
        Be precise and stick strictly to the provided content.
        """
        super().__init__("SelectedTextRAGAgent", instructions)

    async def generate_response_with_selected_text(self, query: str, selected_text: str) -> Dict[str, Any]:
        """Generate a response based only on the selected text"""
        try:
            if not selected_text.strip():
                return {
                    "response": "No selected text provided. Please provide the text you'd like me to answer questions about.",
                    "citations": [],
                    "confidence": 0.0
                }

            # Generate response using the agent with only the selected text
            result = Runner.run_sync(self.agent, f"""
            Selected Text:
            {selected_text}

            Question: {query}

            Please answer the question based ONLY on the provided selected text.
            Do not use any external knowledge or information beyond what's in the selected text.
            If the selected text doesn't contain enough information to answer the question, say so.
            Be precise and stick strictly to the provided content.
            """)

            return {
                "response": result.final_output,
                "citations": [{"id": 1, "title": "Selected Text", "content_preview": selected_text[:200] + "..." if len(selected_text) > 200 else selected_text}],
                "confidence": 0.9  # Higher confidence for selected text mode since it's more focused
            }

        except Exception as e:
            logger.error(f"Error generating selected-text RAG response: {e}")
            return {
                "response": "Sorry, I encountered an error while processing your request.",
                "citations": [],
                "confidence": 0.0
            }