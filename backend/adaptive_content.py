import asyncio
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import statistics

from .profile_service import profile_service
from .database import get_db

logger = logging.getLogger(__name__)

class DifficultyLevel(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

@dataclass
class UserPerformance:
    user_id: int
    module_id: str
    lesson_id: str
    completion_rate: float
    time_efficiency: float
    accuracy: float
    engagement_level: float
    last_accessed: datetime

class AdaptiveContentService:
    def __init__(self):
        self.difficulty_thresholds = {
            DifficultyLevel.BEGINNER: (0, 0.4),
            DifficultyLevel.INTERMEDIATE: (0.4, 0.7),
            DifficultyLevel.ADVANCED: (0.7, 1.0)
        }
        self.performance_weights = {
            "completion_rate": 0.4,
            "time_efficiency": 0.2,
            "accuracy": 0.3,
            "engagement": 0.1
        }

    def calculate_overall_performance(self, user_performance: UserPerformance) -> float:
        """Calculate overall performance score based on multiple metrics"""
        weighted_score = (
            user_performance.completion_rate * self.performance_weights["completion_rate"] +
            user_performance.time_efficiency * self.performance_weights["time_efficiency"] +
            user_performance.accuracy * self.performance_weights["accuracy"] +
            user_performance.engagement_level * self.performance_weights["engagement"]
        )
        return min(max(weighted_score, 0.0), 1.0)  # Clamp between 0 and 1

    def determine_difficulty_level(self, performance_score: float) -> DifficultyLevel:
        """Determine the appropriate difficulty level based on performance score"""
        for level, (min_score, max_score) in self.difficulty_thresholds.items():
            if min_score <= performance_score < max_score:
                return level
        return DifficultyLevel.BEGINNER  # Default to beginner

    async def get_user_performance_metrics(self, db, user_id: int, module_id: str, lesson_id: str) -> Optional[UserPerformance]:
        """Get user performance metrics for a specific lesson"""
        try:
            # Get user progress for the lesson
            lesson_status = await profile_service.get_user_lesson_status(db, user_id, module_id, lesson_id)

            if not lesson_status:
                # If no progress exists, return default metrics
                return UserPerformance(
                    user_id=user_id,
                    module_id=module_id,
                    lesson_id=lesson_id,
                    completion_rate=0.0,
                    time_efficiency=0.5,  # Neutral time efficiency
                    accuracy=0.5,  # Neutral accuracy
                    engagement_level=0.5,  # Neutral engagement
                    last_accessed=datetime.utcnow()
                )

            # Calculate metrics based on progress data
            completion_rate = lesson_status["progress_percentage"] / 100.0

            # Calculate time efficiency (simplified - in real app this would be more complex)
            time_spent = lesson_status["time_spent"]
            # Assume an ideal time of 30 minutes for a lesson (1800 seconds)
            ideal_time = 1800
            if time_spent == 0:
                time_efficiency = 1.0  # Perfect efficiency if no time recorded yet
            else:
                # Higher efficiency for closer to ideal time
                efficiency_ratio = min(time_spent, ideal_time) / max(time_spent, ideal_time)
                time_efficiency = efficiency_ratio

            # Accuracy would come from quiz/exercise results (simplified here)
            accuracy = completion_rate  # Using completion rate as proxy for now

            # Engagement level (simplified - would come from interaction data)
            engagement_level = completion_rate

            return UserPerformance(
                user_id=user_id,
                module_id=module_id,
                lesson_id=lesson_id,
                completion_rate=completion_rate,
                time_efficiency=time_efficiency,
                accuracy=accuracy,
                engagement_level=engagement_level,
                last_accessed=datetime.utcnow()
            )

        except Exception as e:
            logger.error(f"Error calculating performance metrics for user {user_id}, module {module_id}, lesson {lesson_id}: {e}")
            return None

    async def get_adaptive_content(self, db, user_id: int, module_id: str, lesson_id: str) -> Dict[str, Any]:
        """Get content adapted to the user's difficulty level"""
        try:
            # Get user performance metrics
            user_performance = await self.get_user_performance_metrics(db, user_id, module_id, lesson_id)

            if not user_performance:
                # Default to intermediate if we can't get metrics
                difficulty_level = DifficultyLevel.INTERMEDIATE
            else:
                # Calculate overall performance
                overall_performance = self.calculate_overall_performance(user_performance)
                difficulty_level = self.determine_difficulty_level(overall_performance)

            # Return content configuration based on difficulty
            content_config = self._get_content_config_for_difficulty(difficulty_level)

            return {
                "user_id": user_id,
                "module_id": module_id,
                "lesson_id": lesson_id,
                "difficulty_level": difficulty_level.value,
                "content_config": content_config,
                "performance_score": user_performance.overall_performance if user_performance else 0.5,
                "suggested_next_steps": self._get_suggested_next_steps(difficulty_level)
            }

        except Exception as e:
            logger.error(f"Error getting adaptive content for user {user_id}, module {module_id}, lesson {lesson_id}: {e}")
            # Return default content if there's an error
            return {
                "user_id": user_id,
                "module_id": module_id,
                "lesson_id": lesson_id,
                "difficulty_level": DifficultyLevel.INTERMEDIATE.value,
                "content_config": self._get_content_config_for_difficulty(DifficultyLevel.INTERMEDIATE),
                "performance_score": 0.5,
                "suggested_next_steps": self._get_suggested_next_steps(DifficultyLevel.INTERMEDIATE)
            }

    def _get_content_config_for_difficulty(self, difficulty_level: DifficultyLevel) -> Dict[str, Any]:
        """Get content configuration for a specific difficulty level"""
        config = {
            "text_complexity": "standard",
            "example_count": 2,
            "exercise_count": 2,
            "support_resources": ["hints"],
            "challenge_level": "moderate",
            "explanation_depth": "standard"
        }

        if difficulty_level == DifficultyLevel.BEGINNER:
            config.update({
                "text_complexity": "simple",
                "example_count": 3,
                "exercise_count": 1,
                "support_resources": ["hints", "step-by-step", "visual_aids"],
                "challenge_level": "low",
                "explanation_depth": "detailed"
            })
        elif difficulty_level == DifficultyLevel.ADVANCED:
            config.update({
                "text_complexity": "complex",
                "example_count": 1,
                "exercise_count": 3,
                "support_resources": ["minimal_hints"],
                "challenge_level": "high",
                "explanation_depth": "concise"
            })

        return config

    def _get_suggested_next_steps(self, difficulty_level: DifficultyLevel) -> List[str]:
        """Get suggested next steps based on difficulty level"""
        if difficulty_level == DifficultyLevel.BEGINNER:
            return [
                "Review foundational concepts",
                "Practice with additional examples",
                "Seek additional support resources"
            ]
        elif difficulty_level == DifficultyLevel.INTERMEDIATE:
            return [
                "Continue with current module",
                "Attempt more challenging exercises",
                "Explore related concepts"
            ]
        else:  # ADVANCED
            return [
                "Move to next module",
                "Attempt advanced challenges",
                "Help others with difficult concepts"
            ]

    async def update_user_performance(self, db, user_id: int, module_id: str, lesson_id: str,
                                   completion_rate: float, time_spent: int,
                                   exercise_scores: Optional[List[float]] = None) -> Dict[str, Any]:
        """Update user performance and return adjusted difficulty"""
        try:
            # Calculate accuracy from exercise scores if provided
            accuracy = 0.5  # Default
            if exercise_scores:
                accuracy = sum(exercise_scores) / len(exercise_scores) if exercise_scores else 0.5

            # Calculate time efficiency (simplified)
            ideal_time = 1800  # 30 minutes in seconds
            if time_spent == 0:
                time_efficiency = 1.0
            else:
                efficiency_ratio = min(time_spent, ideal_time) / max(time_spent, ideal_time)
                time_efficiency = efficiency_ratio

            # Engagement level based on completion
            engagement_level = completion_rate

            # Create a temporary UserPerformance object
            temp_performance = UserPerformance(
                user_id=user_id,
                module_id=module_id,
                lesson_id=lesson_id,
                completion_rate=completion_rate,
                time_efficiency=time_efficiency,
                accuracy=accuracy,
                engagement_level=engagement_level,
                last_accessed=datetime.utcnow()
            )

            # Calculate overall performance
            overall_performance = self.calculate_overall_performance(temp_performance)
            difficulty_level = self.determine_difficulty_level(overall_performance)

            return {
                "difficulty_level": difficulty_level.value,
                "performance_score": overall_performance,
                "suggested_content_config": self._get_content_config_for_difficulty(difficulty_level)
            }

        except Exception as e:
            logger.error(f"Error updating user performance for user {user_id}: {e}")
            return {
                "difficulty_level": DifficultyLevel.INTERMEDIATE.value,
                "performance_score": 0.5,
                "suggested_content_config": self._get_content_config_for_difficulty(DifficultyLevel.INTERMEDIATE)
            }

# Create adaptive content service instance
adaptive_content_service = AdaptiveContentService()