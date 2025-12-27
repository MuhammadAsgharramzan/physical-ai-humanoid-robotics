import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import random
from dataclasses import dataclass

from .profile_service import profile_service
from .adaptive_content import adaptive_content_service
from .learning_path_service import learning_path_service

logger = logging.getLogger(__name__)

@dataclass
class Recommendation:
    id: str
    title: str
    description: str
    type: str  # 'module', 'lesson', 'resource', 'practice', 'review'
    priority: int  # 1-5 scale
    reason: str
    metadata: Dict[str, Any]

class RecommendationEngine:
    def __init__(self):
        self.content_similarity = {
            "module-1": ["module-2", "module-3"],  # Physical AI fundamentals connects to embodied intelligence and AI techniques
            "module-2": ["module-1", "module-3", "module-4"],  # Embodied intelligence connects to fundamentals, AI techniques, and HRI
            "module-3": ["module-1", "module-2", "module-5"],  # AI techniques connect to fundamentals, embodied intelligence, and control systems
            "module-4": ["module-2", "module-3"],  # HRI connects to embodied intelligence and AI techniques
            "module-5": ["module-3", "module-6"],  # Control systems connect to AI techniques and deployment
            "module-6": ["module-5"]  # Deployment connects to control systems
        }

    async def get_personalized_recommendations(self, db, user_id: int) -> List[Recommendation]:
        """Get personalized recommendations for a user based on their progress and performance"""
        try:
            # Get user's progress and performance
            progress_summary = await profile_service.get_user_overall_progress(db, user_id)
            if not progress_summary:
                # If no progress, return basic recommendations
                return self._get_default_recommendations()

            # Get current learning path
            user_path = await learning_path_service.get_user_learning_path(db, user_id)
            if not user_path:
                user_path = learning_path_service.get_default_path_for_user("beginner")

            recommendations = []

            # 1. Recommend next module in the path
            current_module_idx = user_path.get("current_module_index", 0)
            modules_order = user_path.get("modules_order", [])
            if current_module_idx < len(modules_order):
                next_module = modules_order[current_module_idx]
                if next_module not in user_path.get("completed_modules", []):
                    recommendations.append(Recommendation(
                        id=f"next_module_{next_module}",
                        title=f"Continue with {next_module}",
                        description=f"Continue your learning path with {next_module}",
                        type="module",
                        priority=5,
                        reason="Part of your learning path",
                        metadata={"module_id": next_module}
                    ))

            # 2. Recommend based on performance gaps
            performance_based_recs = await self._get_performance_based_recommendations(db, user_id, progress_summary)
            recommendations.extend(performance_based_recs)

            # 3. Recommend related content based on what they've completed
            related_recs = await self._get_related_content_recommendations(db, user_id, progress_summary)
            recommendations.extend(related_recs)

            # 4. Recommend practice/review based on time since last access
            review_recs = await self._get_review_recommendations(db, user_id)
            recommendations.extend(review_recs)

            # Sort by priority and limit to top 10
            recommendations.sort(key=lambda r: r.priority, reverse=True)
            return recommendations[:10]

        except Exception as e:
            logger.error(f"Error generating recommendations for user {user_id}: {e}")
            # Return safe fallback recommendations
            return self._get_default_recommendations()

    async def _get_performance_based_recommendations(self, db, user_id: int, progress_summary: Dict[str, Any]) -> List[Recommendation]:
        """Get recommendations based on user performance gaps"""
        recommendations = []

        # If overall progress is low, recommend starting modules
        if progress_summary["overall_progress"] < 0.3:
            recommendations.append(Recommendation(
                id="start_modules",
                title="Start with Module 1",
                description="Begin your journey with the fundamentals",
                type="module",
                priority=5,
                reason="Low overall progress",
                metadata={"module_id": "module-1"}
            ))

        # If user has spent little time, recommend engagement
        if progress_summary["total_time_spent"] < 3600:  # Less than 1 hour
            recommendations.append(Recommendation(
                id="increase_engagement",
                title="Spend more time learning",
                description="Try to spend at least 30 minutes per day on learning",
                type="advice",
                priority=3,
                reason="Low engagement",
                metadata={}
            ))

        return recommendations

    async def _get_related_content_recommendations(self, db, user_id: int, progress_summary: Dict[str, Any]) -> List[Recommendation]:
        """Get recommendations for related content based on completed modules"""
        recommendations = []
        completed_modules = progress_summary.get("completed_modules", [])

        for module in completed_modules:
            # Find related modules
            related_modules = self.content_similarity.get(module, [])
            for related_module in related_modules:
                if related_module not in completed_modules:
                    # Check if user has attempted this module
                    module_progress = await profile_service.get_module_progress(db, user_id, related_module)
                    if not module_progress or module_progress["overall_progress"] < 0.3:
                        recommendations.append(Recommendation(
                            id=f"related_{related_module}",
                            title=f"Explore related topic: {related_module}",
                            description=f"Since you completed {module}, you might find {related_module} interesting",
                            type="module",
                            priority=4,
                            reason=f"Related to {module}",
                            metadata={"module_id": related_module, "related_to": module}
                        ))

        return recommendations

    async def _get_review_recommendations(self, db, user_id: int) -> List[Recommendation]:
        """Get recommendations for content review based on time since last access"""
        recommendations = []

        # For simplicity, we'll recommend reviewing the first module if it's been a while
        # In a real system, this would track individual lesson access times
        time_since_start = datetime.utcnow() - timedelta(days=7)  # 1 week threshold

        # Simple recommendation for review
        if random.random() > 0.7:  # 30% chance of review recommendation
            recommendations.append(Recommendation(
                id="review_content",
                title="Review Previous Content",
                description="Revisit previously learned concepts to strengthen your understanding",
                type="review",
                priority=3,
                reason="Regular review helps with retention",
                metadata={"focus_area": "recently_learned"}
            ))

        return recommendations

    def _get_default_recommendations(self) -> List[Recommendation]:
        """Get default recommendations when no user data is available"""
        return [
            Recommendation(
                id="start_here",
                title="Start with Module 1: Introduction",
                description="Begin your journey with the fundamentals of Physical AI",
                type="module",
                priority=5,
                reason="Starting point for beginners",
                metadata={"module_id": "module-1"}
            ),
            Recommendation(
                id="chat_with_assistant",
                title="Chat with our AI Assistant",
                description="Ask questions about the textbook content",
                type="resource",
                priority=4,
                reason="Get personalized help",
                metadata={"resource_type": "chatbot"}
            )
        ]

    async def get_content_recommendations(self, db, user_id: int, content_id: str) -> List[Recommendation]:
        """Get recommendations for content to explore after consuming specific content"""
        try:
            recommendations = []

            # If content_id is a module, recommend next module in sequence
            if content_id.startswith("module-"):
                # Find the next module in the user's learning path
                user_path = await learning_path_service.get_user_learning_path(db, user_id)
                if user_path:
                    modules_order = user_path.get("modules_order", [])
                    try:
                        current_idx = modules_order.index(content_id)
                        if current_idx + 1 < len(modules_order):
                            next_module = modules_order[current_idx + 1]
                            recommendations.append(Recommendation(
                                id=f"next_after_{content_id}",
                                title=f"Continue with {next_module}",
                                description=f"Continue your learning path with {next_module}",
                                type="module",
                                priority=5,
                                reason="Next in sequence",
                                metadata={"module_id": next_module}
                            ))
                    except ValueError:
                        # content_id not in user's path
                        pass

            # Add related content recommendations
            if content_id in self.content_similarity:
                related_modules = self.content_similarity[content_id]
                for related_module in related_modules[:2]:  # Limit to 2 related items
                    recommendations.append(Recommendation(
                        id=f"related_to_{content_id}_{related_module}",
                        title=f"Related: {related_module}",
                        description=f"Content related to {content_id}",
                        type="module",
                        priority=4,
                        reason="Content similarity",
                        metadata={"module_id": related_module}
                    ))

            # Add practice recommendation
            recommendations.append(Recommendation(
                id=f"practice_after_{content_id}",
                title="Practice what you learned",
                description="Test your knowledge with exercises",
                type="practice",
                priority=3,
                reason="Reinforce learning",
                metadata={"target_content": content_id}
            ))

            return recommendations

        except Exception as e:
            logger.error(f"Error getting content recommendations for user {user_id}, content {content_id}: {e}")
            return []

    async def get_learning_style_recommendations(self, db, user_id: int, learning_style: str) -> List[Recommendation]:
        """Get recommendations tailored to user's learning style"""
        try:
            recommendations = []

            if learning_style == "visual":
                recommendations.append(Recommendation(
                    id="visual_learning",
                    title="Use Visual Resources",
                    description="Try diagrams, charts, and visual aids to understand concepts",
                    type="resource",
                    priority=4,
                    reason="Matched to visual learning style",
                    metadata={"resource_type": "visual", "learning_style": "visual"}
                ))
            elif learning_style == "hands_on":
                recommendations.append(Recommendation(
                    id="hands_on_practice",
                    title="Try Hands-On Labs",
                    description="Engage with the ROS2, Gazebo, and Isaac Sim labs",
                    type="practice",
                    priority=5,
                    reason="Matched to hands-on learning style",
                    metadata={"resource_type": "lab", "learning_style": "hands_on"}
                ))
            elif learning_style == "theoretical":
                recommendations.append(Recommendation(
                    id="deep_dive",
                    title="Deep Dive into Theory",
                    description="Focus on mathematical foundations and concepts",
                    type="module",
                    priority=4,
                    reason="Matched to theoretical learning style",
                    metadata={"resource_type": "theory", "learning_style": "theoretical"}
                ))

            return recommendations

        except Exception as e:
            logger.error(f"Error getting learning style recommendations for user {user_id}: {e}")
            return []

# Create recommendation engine instance
recommendation_engine = RecommendationEngine()