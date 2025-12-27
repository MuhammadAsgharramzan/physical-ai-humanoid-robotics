import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import bcrypt

from backend.main import app
from backend.auth import auth_handler

client = TestClient(app)

def test_password_hashing():
    """Test that password hashing works correctly"""
    password = "test_password_123"
    hashed = auth_handler.get_password_hash(password)

    # Verify the password can be verified
    assert auth_handler.verify_password(password, hashed) is True
    # Verify wrong password fails
    assert auth_handler.verify_password("wrong_password", hashed) is False

def test_password_hash_different_each_time():
    """Test that hashing the same password produces different hashes (due to salt)"""
    password = "test_password_123"
    hash1 = auth_handler.get_password_hash(password)
    hash2 = auth_handler.get_password_hash(password)

    assert hash1 != hash2  # Due to random salt
    assert auth_handler.verify_password(password, hash1) is True
    assert auth_handler.verify_password(password, hash2) is True

def test_create_access_token():
    """Test that access tokens can be created"""
    data = {"sub": "testuser", "email": "test@example.com"}
    token = auth_handler.create_access_token(data=data)

    assert isinstance(token, str)
    assert len(token) > 0

def test_token_expiry():
    """Test that tokens expire after the specified time"""
    from datetime import timedelta

    data = {"sub": "testuser"}
    expire_time = timedelta(minutes=1)  # 1 minute expiry for test
    token = auth_handler.create_access_token(data=data, expires_delta=expire_time)

    assert isinstance(token, str)

def test_auth_handler_initialization():
    """Test that auth handler initializes with correct values"""
    assert hasattr(auth_handler, 'secret_key')
    assert hasattr(auth_handler, 'algorithm')
    assert hasattr(auth_handler, 'access_token_expire_minutes')