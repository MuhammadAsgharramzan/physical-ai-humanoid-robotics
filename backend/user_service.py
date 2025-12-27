import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from fastapi import HTTPException, status

from .models import User
from .auth import auth_handler
from .database import get_db

logger = logging.getLogger(__name__)

class UserService:
    def __init__(self):
        self.auth_handler = auth_handler

    async def create_user(self, db: AsyncSession, username: str, email: str, password: str) -> Dict[str, Any]:
        """Create a new user"""
        try:
            # Check if user already exists
            existing_user = await db.execute(
                select(User).where((User.username == username) | (User.email == email))
            )
            existing_user = existing_user.scalars().first()

            if existing_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username or email already registered"
                )

            # Hash the password
            hashed_password = self.auth_handler.get_password_hash(password)

            # Create new user
            user = User(
                username=username,
                email=email,
                hashed_password=hashed_password
            )

            db.add(user)
            await db.commit()
            await db.refresh(user)

            logger.info(f"Created new user: {username}")

            return {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "created_at": user.created_at.isoformat()
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error creating user {username}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error creating user"
            )

    async def authenticate_user(self, db: AsyncSession, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate a user and return user data if successful"""
        try:
            # Find user by username
            result = await db.execute(
                select(User).where(User.username == username)
            )
            user = result.scalars().first()

            if not user or not self.auth_handler.verify_password(password, user.hashed_password):
                return None

            return {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "is_active": user.is_active
            }

        except Exception as e:
            logger.error(f"Error authenticating user {username}: {e}")
            return None

    async def get_user_by_id(self, db: AsyncSession, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        try:
            result = await db.execute(
                select(User).where(User.id == user_id)
            )
            user = result.scalars().first()

            if not user:
                return None

            return {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "is_active": user.is_active,
                "created_at": user.created_at.isoformat(),
                "updated_at": user.updated_at.isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting user by ID {user_id}: {e}")
            return None

    async def update_user(self, db: AsyncSession, user_id: int, **kwargs) -> Optional[Dict[str, Any]]:
        """Update user information"""
        try:
            # Prepare update data, excluding protected fields
            update_data = {}
            allowed_fields = {"username", "email"}
            for field, value in kwargs.items():
                if field in allowed_fields:
                    update_data[field] = value

            if not update_data:
                return await self.get_user_by_id(db, user_id)

            # Perform update
            await db.execute(
                update(User)
                .where(User.id == user_id)
                .values(**update_data)
            )
            await db.commit()

            return await self.get_user_by_id(db, user_id)

        except Exception as e:
            logger.error(f"Error updating user {user_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error updating user"
            )

    async def change_password(self, db: AsyncSession, user_id: int, current_password: str, new_password: str) -> bool:
        """Change user password after verifying current password"""
        try:
            # Get user
            result = await db.execute(
                select(User).where(User.id == user_id)
            )
            user = result.scalars().first()

            if not user:
                return False

            # Verify current password
            if not self.auth_handler.verify_password(current_password, user.hashed_password):
                return False

            # Update with new password
            new_hashed_password = self.auth_handler.get_password_hash(new_password)
            await db.execute(
                update(User)
                .where(User.id == user_id)
                .values(hashed_password=new_hashed_password)
            )
            await db.commit()

            logger.info(f"Password changed for user ID: {user_id}")
            return True

        except Exception as e:
            logger.error(f"Error changing password for user {user_id}: {e}")
            return False

# Create user service instance
user_service = UserService()