import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.pool import StaticPool

from backend.main import app
from backend.database import get_db
from backend.models import Base

# Create test database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables
Base.metadata.create_all(bind=engine)

# Override dependency
def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Physical AI & Humanoid Robotics API is running"}

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_content_search():
    # Test search endpoint with a query parameter
    response = client.get("/content/search?query=test")
    assert response.status_code == 200
    # The response will be empty since we don't have content in the test database
    assert "query" in response.json()
    assert "content_chunks" in response.json()

def test_content_by_topic():
    # Test content by topic endpoint
    response = client.get("/content/topic/test-topic")
    assert response.status_code == 200
    # The response will be empty since we don't have content in the test database
    assert "topic" in response.json()
    assert "content_chunks" in response.json()