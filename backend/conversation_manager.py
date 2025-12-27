import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class MessageType(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

@dataclass
class Message:
    role: MessageType
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

class Conversation:
    def __init__(self, conversation_id: str, user_id: Optional[str] = None):
        self.conversation_id = conversation_id
        self.user_id = user_id
        self.messages: List[Message] = []
        self.created_at = datetime.utcnow()
        self.last_updated = datetime.utcnow()
        self.is_active = True

    def add_message(self, role: MessageType, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a message to the conversation"""
        message = Message(
            role=role,
            content=content,
            timestamp=datetime.utcnow(),
            metadata=metadata
        )
        self.messages.append(message)
        self.last_updated = datetime.utcnow()

    def get_context(self, max_messages: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent messages as context for the AI"""
        recent_messages = self.messages[-max_messages:] if len(self.messages) > max_messages else self.messages
        return [
            {
                "role": message.role.value,
                "content": message.content,
                "timestamp": message.timestamp.isoformat()
            }
            for message in recent_messages
        ]

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all messages in the conversation"""
        return [
            {
                "role": message.role.value,
                "content": message.content,
                "timestamp": message.timestamp.isoformat(),
                "metadata": message.metadata
            }
            for message in self.messages
        ]

class ConversationManager:
    def __init__(self, max_conversations: int = 1000, max_messages_per_conversation: int = 50,
                 conversation_ttl_hours: int = 24):
        self.conversations: Dict[str, Conversation] = {}
        self.max_conversations = max_conversations
        self.max_messages_per_conversation = max_messages_per_conversation
        self.conversation_ttl = timedelta(hours=conversation_ttl_hours)
        self.lock = asyncio.Lock()

    async def create_conversation(self, conversation_id: str, user_id: Optional[str] = None) -> Conversation:
        """Create a new conversation"""
        async with self.lock:
            if len(self.conversations) >= self.max_conversations:
                # Remove oldest conversation if we're at max capacity
                oldest_id = min(
                    self.conversations.keys(),
                    key=lambda x: self.conversations[x].created_at
                )
                del self.conversations[oldest_id]

            conversation = Conversation(conversation_id, user_id)
            self.conversations[conversation_id] = conversation
            return conversation

    async def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get an existing conversation"""
        async with self.lock:
            conversation = self.conversations.get(conversation_id)

            # Check if conversation has expired
            if conversation and datetime.utcnow() - conversation.last_updated > self.conversation_ttl:
                await self.delete_conversation(conversation_id)
                return None

            return conversation

    async def add_message(self, conversation_id: str, role: MessageType, content: str,
                         metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add a message to a conversation"""
        conversation = await self.get_conversation(conversation_id)
        if not conversation:
            return False

        async with self.lock:
            conversation.add_message(role, content, metadata)

            # Trim messages if we exceed the limit
            if len(conversation.messages) > self.max_messages_per_conversation:
                conversation.messages = conversation.messages[-self.max_messages_per_conversation:]

        return True

    async def get_conversation_context(self, conversation_id: str, max_messages: int = 10) -> List[Dict[str, Any]]:
        """Get conversation context for AI"""
        conversation = await self.get_conversation(conversation_id)
        if not conversation:
            return []

        return conversation.get_context(max_messages)

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation"""
        async with self.lock:
            if conversation_id in self.conversations:
                del self.conversations[conversation_id]
                return True
            return False

    async def cleanup_expired_conversations(self):
        """Remove expired conversations"""
        async with self.lock:
            expired_ids = [
                cid for cid, conv in self.conversations.items()
                if datetime.utcnow() - conv.last_updated > self.conversation_ttl
            ]

            for cid in expired_ids:
                del self.conversations[cid]

            logger.info(f"Cleaned up {len(expired_ids)} expired conversations")

# Create conversation manager instance
conversation_manager = ConversationManager()