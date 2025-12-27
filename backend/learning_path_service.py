import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_
from fastapi import HTTPException, status

from .models import User
from .database import get_db

logger = logging.getLogger(__name__)

class LearningPathService:
    def __init__(self):
        # Define standard learning paths
        self.standard_paths = {
            "beginner": {
                "name": "Beginner Path",
                "description": "Start with fundamentals and build up gradually",
                "modules_order": [
                    "module-1", "module-2", "module-3",
                    "module-4", "module-5", "module-6"
                ],
                "prerequisites": {}
            },
            "intermediate": {
                "name": "Intermediate Path",
                "description": "For users with some background knowledge",
                "modules_order": [
                    "module-2", "module-3", "module-1",
                    "module-4", "module-5", "module-6"
                ],
                "prerequisites": {
                    "module-1": ["module-2"]  # Module 1 can be started after Module 2
                }
            },
            "advanced": {
                "name": "Advanced Path",
                "description": "For experienced users, focuses on advanced topics",
                "modules_order": [
                    "module-3", "module-5", "module-1",
                    "module-2", "module-4", "module-6"
                ],
                "prerequisites": {
                    "module-1": ["module-3"],  # Module 1 requires Module 3 as prerequisite
                    "module-2": ["module-3"]
                }
            },
            "hri_specialist": {
                "name": "Human-Robot Interaction Specialist",
                "description": "Focus on Module 4 (HRI) and related topics",
                "modules_order": [
                    "module-1", "module-4", "module-2",
                    "module-3", "module-5", "module-6"
                ],
                "prerequisites": {}
            },
            "control_systems": {
                "name": "Control Systems Focus",
                "description": "Focus on Module 5 (Advanced Control) and related topics",
                "modules_order": [
                    "module-1", "module-2", "module-5",
                    "module-3", "module-4", "module-6"
                ],
                "prerequisites": {}
            }
        }

    def get_available_paths(self) -> List[Dict[str, Any]]:
        """Get all available learning paths"""
        return [
            {
                "id": path_id,
                "name": path_data["name"],
                "description": path_data["description"],
                "modules_count": len(path_data["modules_order"])
            }
            for path_id, path_data in self.standard_paths.items()
        ]

    def get_path_details(self, path_id: str) -> Optional[Dict[str, Any]]:
        """Get details for a specific learning path"""
        if path_id not in self.standard_paths:
            return None

        path_data = self.standard_paths[path_id]
        return {
            "id": path_id,
            "name": path_data["name"],
            "description": path_data["description"],
            "modules_order": path_data["modules_order"],
            "prerequisites": path_data["prerequisites"]
        }

    def get_default_path_for_user(self, user_level: str = "beginner") -> Dict[str, Any]:
        """Get the default learning path based on user level"""
        if user_level in self.standard_paths:
            return self.get_path_details(user_level)
        return self.get_path_details("beginner")  # Default to beginner

    async def create_user_learning_path(self, db: AsyncSession, user_id: int, path_id: str,
                                      customizations: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a learning path for a specific user"""
        try:
            # Verify user exists
            user_result = await db.execute(
                select(User).where(User.id == user_id)
            )
            user = user_result.scalars().first()

            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )

            # Get the base path
            base_path = self.get_path_details(path_id)
            if not base_path:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Learning path '{path_id}' not found"
                )

            # Apply customizations if provided
            user_path = {
                "user_id": user_id,
                "path_id": path_id,
                "name": base_path["name"],
                "description": base_path["description"],
                "modules_order": base_path["modules_order"][:],  # Copy the list
                "prerequisites": base_path["prerequisites"].copy(),
                "created_at": datetime.utcnow().isoformat(),
                "customizations": customizations or {}
            }

            # Apply customizations
            if customizations:
                if "modules_order" in customizations:
                    # Validate that all required modules are included
                    provided_modules = set(customizations["modules_order"])
                    required_modules = set(base_path["modules_order"])

                    if not required_modules.issubset(provided_modules):
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Custom modules order must include all required modules"
                        )

                    user_path["modules_order"] = customizations["modules_order"]

                if "skipped_modules" in customizations:
                    skipped = set(customizations["skipped_modules"])
                    user_path["modules_order"] = [
                        mod for mod in user_path["modules_order"]
                        if mod not in skipped
                    ]

                if "focus_areas" in customizations:
                    # Reorder modules to prioritize focus areas
                    focus_modules = customizations["focus_areas"]
                    remaining_modules = [mod for mod in user_path["modules_order"] if mod not in focus_modules]
                    user_path["modules_order"] = focus_modules + remaining_modules

            logger.info(f"Created learning path for user {user_id}: {path_id}")
            return user_path

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error creating learning path for user {user_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error creating learning path"
            )

    async def get_user_learning_path(self, db: AsyncSession, user_id: int) -> Optional[Dict[str, Any]]:
        """Get the current learning path for a user"""
        # In a real implementation, this would fetch from a dedicated learning path table
        # For now, we'll return a default path based on their progress
        try:
            # This is a simplified implementation - in a real system, you'd have
            # a dedicated table to store user learning paths
            from .profile_service import profile_service

            # Get user's overall progress to determine appropriate path
            progress_summary = await profile_service.get_user_overall_progress(db, user_id)

            # Determine path based on progress
            if progress_summary["overall_progress"] < 0.3:
                user_level = "beginner"
            elif progress_summary["overall_progress"] < 0.7:
                user_level = "intermediate"
            else:
                user_level = "advanced"

            default_path = self.get_default_path_for_user(user_level)

            return {
                "user_id": user_id,
                "path_id": user_level,
                "name": default_path["name"],
                "description": default_path["description"],
                "modules_order": default_path["modules_order"],
                "current_module_index": self._get_current_module_index(
                    progress_summary, default_path["modules_order"]
                ),
                "completed_modules": self._get_completed_modules(
                    progress_summary, default_path["modules_order"]
                )
            }
        except Exception as e:
            logger.error(f"Error getting learning path for user {user_id}: {e}")
            return None

    def _get_current_module_index(self, progress_summary: Dict[str, Any], modules_order: List[str]) -> int:
        """Determine the current module index based on progress"""
        # Simplified logic - in reality this would be more sophisticated
        if progress_summary["overall_progress"] < 0.2:
            return 0
        elif progress_summary["overall_progress"] < 0.4:
            return 1
        elif progress_summary["overall_progress"] < 0.6:
            return 2
        elif progress_summary["overall_progress"] < 0.8:
            return 3
        else:
            return min(len(modules_order) - 1, 4)  # Don't exceed available modules

    def _get_completed_modules(self, progress_summary: Dict[str, Any], modules_order: List[str]) -> List[str]:
        """Get list of completed modules based on progress"""
        # Simplified logic - in reality this would check actual module completion
        completed_count = int(progress_summary["overall_progress"] * len(modules_order))
        return modules_order[:completed_count]

    async def update_user_learning_path(self, db: AsyncSession, user_id: int,
                                      path_id: Optional[str] = None,
                                      customizations: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Update a user's learning path"""
        try:
            # Get current path to merge changes
            current_path = await self.get_user_learning_path(db, user_id)
            if not current_path:
                # If no current path, create a new one
                if path_id:
                    return await self.create_user_learning_path(db, user_id, path_id, customizations)
                else:
                    # Use default beginner path
                    return await self.create_user_learning_path(db, user_id, "beginner", customizations)

            # Apply updates
            updated_path = current_path.copy()

            if path_id and path_id != current_path["path_id"]:
                # Change to a different path
                new_path = self.get_path_details(path_id)
                if new_path:
                    updated_path.update({
                        "path_id": path_id,
                        "name": new_path["name"],
                        "description": new_path["description"],
                        "modules_order": new_path["modules_order"],
                        "prerequisites": new_path["prerequisites"]
                    })

            if customizations:
                updated_path["customizations"] = customizations

            updated_path["updated_at"] = datetime.utcnow().isoformat()

            logger.info(f"Updated learning path for user {user_id}")
            return updated_path

        except Exception as e:
            logger.error(f"Error updating learning path for user {user_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error updating learning path"
            )

    def get_next_module(self, user_path: Dict[str, Any], current_module: str) -> Optional[str]:
        """Get the next module in the learning path"""
        modules_order = user_path.get("modules_order", [])
        try:
            current_index = modules_order.index(current_module)
            if current_index + 1 < len(modules_order):
                return modules_order[current_index + 1]
        except ValueError:
            # Current module not in path
            pass

        # If current module not found or at end, return first module
        return modules_order[0] if modules_order else None

    def get_path_recommendation(self, user_performance: Dict[str, Any]) -> str:
        """Get path recommendation based on user performance"""
        # Simplified recommendation logic
        performance_score = user_performance.get("overall_performance", 0.5)

        if performance_score < 0.4:
            return "beginner"
        elif performance_score < 0.7:
            return "intermediate"
        else:
            # For high performers, suggest specialized paths based on interests
            if user_performance.get("interests", {}).get("hri", False):
                return "hri_specialist"
            elif user_performance.get("interests", {}).get("control", False):
                return "control_systems"
            else:
                return "advanced"

# Create learning path service instance
learning_path_service = LearningPathService()