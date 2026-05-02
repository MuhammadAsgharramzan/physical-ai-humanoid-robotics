from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
import uvicorn
import os
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler

# Load environment variables
load_dotenv()
from slowapi.util import get_remote_address
from typing import List, Optional

from .database import get_db
from .content_service import content_service
from .models import ContentChunk
from .logging_config import logger
from .rag_service import rag_service
from .conversation_manager import conversation_manager, MessageType
from .user_service import user_service
from .profile_service import profile_service
from .adaptive_content import adaptive_content_service
from .learning_path_service import learning_path_service
from .recommendation_engine import recommendation_engine
from .agents.rag_agent_service import rag_agent_service, RAGMode
from .content_pipeline import content_pipeline

# Initialize rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    enabled=os.getenv("RATE_LIMIT_ENABLED", "True").lower() == "true"
)

app = FastAPI(
    title="Physical AI & Humanoid Robotics API",
    description="Backend API for the Physical AI & Humanoid Robotics textbook project",
    version="1.0.0"
)

from slowapi.middleware import SlowAPIMiddleware

# Add CORS middleware
origins = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add SlowAPI middleware
app.add_middleware(SlowAPIMiddleware)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

@app.get("/")
@limiter.limit("10/minute")
async def root(request: Request):
    logger.info("Root endpoint accessed")
    return {"message": "Physical AI & Humanoid Robotics API is running"}

@app.get("/health")
@limiter.limit("100/minute")
async def health_check(request: Request):
    logger.info("Health check endpoint accessed")
    return {"status": "healthy"}

@app.get("/content/topic/{topic}")
@limiter.limit("5/minute")
async def get_content_by_topic(request: Request, topic: str, db: AsyncSession = Depends(get_db)):
    """Retrieve content chunks related to a specific topic"""
    logger.info(f"Retrieving content for topic: {topic}")
    content_chunks = await content_service.get_content_by_topic(topic, db)
    return {"topic": topic, "content_chunks": content_chunks}

@app.get("/content/search")
@limiter.limit("5/minute")
async def search_content(request: Request, query: str, db: AsyncSession = Depends(get_db)):
    """Search for content based on a query string"""
    logger.info(f"Searching content for query: {query}")
    content_chunks = await content_service.search_content(query, db)
    return {"query": query, "content_chunks": content_chunks}

@app.post("/chat")
@limiter.limit("10/minute")
async def chat(
    request: Request,
    conversation_id: str,
    user_message: str,
    mode: str = "full_book",  # Default to full-book mode
    selected_text: Optional[str] = None  # For selected-text mode
):
    """Chat endpoint for interacting with the RAG chatbot with two modes"""
    logger.info(f"Chat message received for conversation {conversation_id} in mode: {mode}")

    # Add user message to conversation
    await conversation_manager.add_message(conversation_id, MessageType.USER, user_message)

    # Get conversation context
    context = await conversation_manager.get_conversation_context(conversation_id, max_messages=5)

    # Determine the RAG mode
    try:
        rag_mode = RAGMode(mode.lower())
    except ValueError:
        rag_mode = RAGMode.FULL_BOOK  # Default to full-book if invalid mode

    # Generate response using the appropriate RAG mode
    if rag_mode == RAGMode.FULL_BOOK:
        response_data = await rag_agent_service.generate_response(
            query=user_message,
            mode=rag_mode
        )
    elif rag_mode == RAGMode.SELECTED_TEXT:
        response_data = await rag_agent_service.generate_response(
            query=user_message,
            mode=rag_mode,
            selected_text=selected_text
        )

    # Add assistant response to conversation
    await conversation_manager.add_message(
        conversation_id,
        MessageType.ASSISTANT,
        response_data["response"],
        {"sources": response_data["citations"], "confidence": response_data["confidence"]}
    )

    return {
        "response": response_data["response"],
        "citations": response_data["citations"],
        "confidence": response_data["confidence"],
        "conversation_id": conversation_id,
        "mode": rag_mode.value
    }

@app.get("/chat/conversation/{conversation_id}")
@limiter.limit("20/minute")
async def get_conversation(request: Request, conversation_id: str):
    """Get the full conversation history"""
    logger.info(f"Retrieving conversation {conversation_id}")

    conversation = await conversation_manager.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {
        "conversation_id": conversation_id,
        "messages": conversation.get_messages(),
        "created_at": conversation.created_at.isoformat(),
        "last_updated": conversation.last_updated.isoformat()
    }

@app.post("/content/index-all")
@limiter.limit("2/minute")
async def index_all_content(request: Request):
    """Index all content from the Docusaurus docs directory"""

    logger.info("Starting content indexing process")
    success = await content_pipeline.index_all_content()

    if success:
        return {"message": "Content indexing completed successfully"}
    else:
        return {"message": "Content indexing completed with some failures"}

# User Authentication Endpoints
@app.post("/auth/register")
@limiter.limit("5/hour")
async def register_user(
    request: Request,
    username: str,
    email: str,
    password: str,
    db: AsyncSession = Depends(get_db)
):
    """Register a new user"""
    logger.info(f"Registering new user: {username}")
    user = await user_service.create_user(db, username, email, password)
    return user

@app.post("/auth/login")
@limiter.limit("10/minute")
async def login_user(
    request: Request,
    username: str,
    password: str,
    db: AsyncSession = Depends(get_db)
):
    """Authenticate user and return token"""
    logger.info(f"Login attempt for user: {username}")
    user = await user_service.authenticate_user(db, username, password)

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password"
        )

    # Generate access token
    from .auth import auth_handler
    token_data = {"sub": user["username"], "email": user["email"]}
    access_token = auth_handler.create_access_token(data=token_data)

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user
    }

# Learning Progress Endpoints
@app.get("/progress/user/{user_id}")
@limiter.limit("20/minute")
async def get_user_progress(request: Request, user_id: int, db: AsyncSession = Depends(get_db)):
    """Get all learning progress for a user"""
    logger.info(f"Retrieving progress for user: {user_id}")
    progress = await profile_service.get_user_progress(db, user_id)
    return {"user_id": user_id, "progress": progress}

@app.get("/progress/user/{user_id}/module/{module_id}")
@limiter.limit("20/minute")
async def get_module_progress(request: Request, user_id: int, module_id: str, db: AsyncSession = Depends(get_db)):
    """Get progress for a specific module"""
    logger.info(f"Retrieving progress for user: {user_id}, module: {module_id}")
    progress = await profile_service.get_module_progress(db, user_id, module_id)
    return progress

@app.post("/progress/update")
@limiter.limit("30/minute")
async def update_lesson_progress(
    request: Request,
    user_id: int,
    module_id: str,
    lesson_id: str,
    progress_percentage: int,
    time_spent: int,
    db: AsyncSession = Depends(get_db)
):
    """Update progress for a specific lesson"""
    logger.info(f"Updating progress for user: {user_id}, module: {module_id}, lesson: {lesson_id}")
    result = await profile_service.update_lesson_progress(
        db, user_id, module_id, lesson_id, progress_percentage, time_spent
    )
    return result

# Adaptive Content Endpoints
@app.get("/content/adaptive/{user_id}/{module_id}/{lesson_id}")
@limiter.limit("20/minute")
async def get_adaptive_content(
    request: Request,
    user_id: int,
    module_id: str,
    lesson_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get content adapted to the user's difficulty level"""
    logger.info(f"Getting adaptive content for user: {user_id}, module: {module_id}, lesson: {lesson_id}")
    content = await adaptive_content_service.get_adaptive_content(db, user_id, module_id, lesson_id)
    return content

# Learning Path Endpoints
@app.get("/learning-path/available")
@limiter.limit("10/minute")
async def get_available_learning_paths(request: Request):
    """Get all available learning paths"""
    paths = learning_path_service.get_available_paths()
    return {"paths": paths}

@app.get("/learning-path/user/{user_id}")
@limiter.limit("10/minute")
async def get_user_learning_path(request: Request, user_id: int, db: AsyncSession = Depends(get_db)):
    """Get the current learning path for a user"""
    path = await learning_path_service.get_user_learning_path(db, user_id)
    return path

@app.post("/learning-path/user/{user_id}")
@limiter.limit("5/minute")
async def create_user_learning_path(
    request: Request,
    user_id: int,
    path_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Create a learning path for a user"""
    path = await learning_path_service.create_user_learning_path(db, user_id, path_id)
    return path

# Recommendation Endpoints
@app.get("/recommendations/user/{user_id}")
@limiter.limit("10/minute")
async def get_user_recommendations(request: Request, user_id: int, db: AsyncSession = Depends(get_db)):
    """Get personalized recommendations for a user"""
    recommendations = await recommendation_engine.get_personalized_recommendations(db, user_id)
    return {"recommendations": recommendations}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)