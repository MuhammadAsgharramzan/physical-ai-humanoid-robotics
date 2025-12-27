import asyncio
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_
from fastapi import HTTPException, status

from .models import LearningProgress
from .database import get_db

logger = logging.getLogger(__name__)

class ProfileService:
    def __init__(self):
        pass

    async def get_user_progress(self, db: AsyncSession, user_id: int) -> List[Dict[str, Any]]:
        """Get all learning progress for a user"""
        try:
            result = await db.execute(
                select(LearningProgress)
                .where(LearningProgress.user_id == user_id)
            )
            progress_records = result.scalars().all()

            return [
                {
                    "id": record.id,
                    "user_id": record.user_id,
                    "module_id": record.module_id,
                    "lesson_id": record.lesson_id,
                    "completed": record.completed,
                    "progress_percentage": record.progress_percentage,
                    "time_spent": record.time_spent,
                    "created_at": record.created_at.isoformat(),
                    "updated_at": record.updated_at.isoformat()
                }
                for record in progress_records
            ]

        except Exception as e:
            logger.error(f"Error getting user progress for user {user_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error retrieving user progress"
            )

    async def get_module_progress(self, db: AsyncSession, user_id: int, module_id: str) -> Optional[Dict[str, Any]]:
        """Get progress for a specific module"""
        try:
            result = await db.execute(
                select(LearningProgress)
                .where(
                    and_(
                        LearningProgress.user_id == user_id,
                        LearningProgress.module_id == module_id
                    )
                )
            )
            progress_records = result.scalars().all()

            if not progress_records:
                return None

            module_progress = {
                "module_id": module_id,
                "total_lessons": len(progress_records),
                "completed_lessons": sum(1 for record in progress_records if record.completed),
                "overall_progress": sum(record.progress_percentage for record in progress_records) / len(progress_records),
                "total_time_spent": sum(record.time_spent for record in progress_records),
                "lessons": [
                    {
                        "lesson_id": record.lesson_id,
                        "completed": record.completed,
                        "progress_percentage": record.progress_percentage,
                        "time_spent": record.time_spent
                    }
                    for record in progress_records
                ]
            }

            return module_progress

        except Exception as e:
            logger.error(f"Error getting module progress for user {user_id}, module {module_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error retrieving module progress"
            )

    async def update_lesson_progress(self, db: AsyncSession, user_id: int, module_id: str, lesson_id: str,
                                    progress_percentage: int, time_spent: int, completed: bool = False) -> Dict[str, Any]:
        """Update progress for a specific lesson"""
        try:
            # Check if record already exists
            result = await db.execute(
                select(LearningProgress)
                .where(
                    and_(
                        LearningProgress.user_id == user_id,
                        LearningProgress.module_id == module_id,
                        LearningProgress.lesson_id == lesson_id
                    )
                )
            )
            existing_record = result.scalars().first()

            if existing_record:
                # Update existing record
                await db.execute(
                    update(LearningProgress)
                    .where(LearningProgress.id == existing_record.id)
                    .values(
                        progress_percentage=progress_percentage,
                        time_spent=time_spent,
                        completed=completed,
                        updated_at=datetime.utcnow()
                    )
                )
                await db.commit()

                # Fetch the updated record
                result = await db.execute(
                    select(LearningProgress).where(LearningProgress.id == existing_record.id)
                )
                updated_record = result.scalars().first()
            else:
                # Create new record
                new_record = LearningProgress(
                    user_id=user_id,
                    module_id=module_id,
                    lesson_id=lesson_id,
                    progress_percentage=progress_percentage,
                    time_spent=time_spent,
                    completed=completed
                )
                db.add(new_record)
                await db.commit()
                await db.refresh(new_record)
                updated_record = new_record

            return {
                "id": updated_record.id,
                "user_id": updated_record.user_id,
                "module_id": updated_record.module_id,
                "lesson_id": updated_record.lesson_id,
                "completed": updated_record.completed,
                "progress_percentage": updated_record.progress_percentage,
                "time_spent": updated_record.time_spent,
                "created_at": updated_record.created_at.isoformat(),
                "updated_at": updated_record.updated_at.isoformat()
            }

        except Exception as e:
            logger.error(f"Error updating lesson progress for user {user_id}, module {module_id}, lesson {lesson_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error updating lesson progress"
            )

    async def get_user_lesson_status(self, db: AsyncSession, user_id: int, module_id: str, lesson_id: str) -> Optional[Dict[str, Any]]:
        """Get progress status for a specific lesson"""
        try:
            result = await db.execute(
                select(LearningProgress)
                .where(
                    and_(
                        LearningProgress.user_id == user_id,
                        LearningProgress.module_id == module_id,
                        LearningProgress.lesson_id == lesson_id
                    )
                )
            )
            record = result.scalars().first()

            if not record:
                return None

            return {
                "id": record.id,
                "user_id": record.user_id,
                "module_id": record.module_id,
                "lesson_id": record.lesson_id,
                "completed": record.completed,
                "progress_percentage": record.progress_percentage,
                "time_spent": record.time_spent,
                "created_at": record.created_at.isoformat(),
                "updated_at": record.updated_at.isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting lesson status for user {user_id}, module {module_id}, lesson {lesson_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error retrieving lesson status"
            )

    async def get_user_overall_progress(self, db: AsyncSession, user_id: int) -> Dict[str, Any]:
        """Get overall progress summary for a user"""
        try:
            result = await db.execute(
                select(LearningProgress)
                .where(LearningProgress.user_id == user_id)
            )
            all_records = result.scalars().all()

            if not all_records:
                return {
                    "user_id": user_id,
                    "total_lessons": 0,
                    "completed_lessons": 0,
                    "overall_progress": 0,
                    "total_time_spent": 0,
                    "modules_completed": 0
                }

            # Calculate overall statistics
            total_lessons = len(all_records)
            completed_lessons = sum(1 for record in all_records if record.completed)
            overall_progress = sum(record.progress_percentage for record in all_records) / total_lessons if total_lessons > 0 else 0
            total_time_spent = sum(record.time_spent for record in all_records)

            # Calculate modules completed (modules where all lessons are completed)
            module_lessons = {}
            for record in all_records:
                if record.module_id not in module_lessons:
                    module_lessons[record.module_id] = []
                module_lessons[record.module_id].append(record)

            modules_completed = sum(
                1 for lessons in module_lessons.values()
                if all(lesson.completed for lesson in lessons)
            )

            return {
                "user_id": user_id,
                "total_lessons": total_lessons,
                "completed_lessons": completed_lessons,
                "overall_progress": round(overall_progress, 2),
                "total_time_spent": total_time_spent,
                "modules_completed": modules_completed
            }

        except Exception as e:
            logger.error(f"Error getting overall progress for user {user_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error retrieving overall progress"
            )

# Create profile service instance
profile_service = ProfileService()