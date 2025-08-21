# components/relationship_manager.py
import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging
import json

from core.database import DatabaseManager, AgentState, MemoryItem
from core.llm_client import llm_client

logger = logging.getLogger(__name__)

class RelationshipManagerComponent:
    """
    Relationship Manager (Context & Social Layer)
    - Maintains relational and contextual continuity across interactions
    - Tracks ongoing conversations, tone, and rapport
    - Maintains interaction history per user/agent
    - Adjusts personality and emotional state dynamically
    """
    
    def __init__(self):
        self.name = "relationship_manager"
        self.status = "idle"
        self.active_conversations = {}
        self.user_profiles = {}
        self.interaction_count = 0
        self.context_updates = 0
        self.persona_adjustments = 0
        self.default_persona = {
            "tone": "professional",
            "formality": "medium",
            "empathy_level": "high",
            "response_style": "helpful",
            "personality_traits": ["curious", "supportive", "analytical"]
        }
        
    async def start(self):
        """Start the relationship manager component"""
        self.status = "active"
        await self._update_state()
        logger.info("Relationship Manager component started")
        
        # Start background maintenance
        asyncio.create_task(self._relationship_maintenance_loop())
    
    async def stop(self):
        """Stop the relationship manager component"""
        self.status = "stopped"
        await self._update_state()
        logger.info("Relationship Manager component stopped")
    
    async def track_interaction(self, user_id: str, interaction_data: Dict[str, Any]) -> str:
        """
        Track a new interaction with a user
        
        Args:
            user_id: Identifier for the user
            interaction_data: Data about the interaction
            
        Returns:
            Interaction ID
        """
        try:
            interaction_id = str(uuid.uuid4())
            
            # Initialize user profile if new
            if user_id not in self.user_profiles:
                await self._initialize_user_profile(user_id)
            
            # Update conversation context
            conversation_id = await self._get_or_create_conversation(user_id)
            
            # Analyze interaction for context updates
            context_analysis = await self._analyze_interaction_context(interaction_data)
            
            # Update user profile based on interaction
            await self._update_user_profile(user_id, interaction_data, context_analysis)
            
            # Store interaction in conversation history
            await self._store_interaction(conversation_id, interaction_id, interaction_data, context_analysis)
            
            # Adjust persona if needed
            await self._adjust_persona(user_id, context_analysis)
            
            self.interaction_count += 1
            await self._update_state()
            
            logger.info(f"Tracked interaction {interaction_id} for user {user_id}")
            return interaction_id
            
        except Exception as e:
            logger.error(f"Error tracking interaction: {e}")
            raise
    
    async def get_context(self, user_id: str) -> Dict[str, Any]:
        """
        Get current context for a user
        
        Args:
            user_id: User identifier
            
        Returns:
            Context information
        """
        try:
            if user_id not in self.user_profiles:
                await self._initialize_user_profile(user_id)
            
            profile = self.user_profiles[user_id]
            conversation_id = profile.get("current_conversation")
            
            # Get recent conversation history
            conversation_history = []
            if conversation_id and conversation_id in self.active_conversations:
                conversation = self.active_conversations[conversation_id]
                conversation_history = conversation.get("interactions", [])[-10:]  # Last 10 interactions
            
            context = {
                "user_id": user_id,
                "profile": profile,
                "conversation_history": conversation_history,
                "current_persona": profile.get("persona", self.default_persona),
                "relationship_stage": profile.get("relationship_stage", "new"),
                "preferences": profile.get("preferences", {}),
                "context_summary": await self._generate_context_summary(user_id)
            }
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting context for user {user_id}: {e}")
            return {"user_id": user_id, "error": str(e)}
    
    async def update_persona(self, user_id: str, persona_updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update persona for a specific user
        
        Args:
            user_id: User identifier
            persona_updates: Updates to apply to persona
            
        Returns:
            Updated persona
        """
        try:
            if user_id not in self.user_profiles:
                await self._initialize_user_profile(user_id)
            
            profile = self.user_profiles[user_id]
            current_persona = profile.get("persona", self.default_persona.copy())
            
            # Apply updates
            updated_persona = {**current_persona, **persona_updates}
            
            # Validate persona
            validated_persona = await self._validate_persona(updated_persona)
            
            # Update user profile
            self.user_profiles[user_id]["persona"] = validated_persona
            self.user_profiles[user_id]["last_updated"] = datetime.now()
            
            # Store updated profile
            await self._store_user_profile(user_id)
            
            self.persona_adjustments += 1
            await self._update_state()
            
            logger.info(f"Updated persona for user {user_id}")
            return validated_persona
            
        except Exception as e:
            logger.error(f"Error updating persona for user {user_id}: {e}")
            raise
    
    async def assess_rapport(self, user_id: str) -> Dict[str, Any]:
        """
        Assess current rapport level with user
        
        Args:
            user_id: User identifier
            
        Returns:
            Rapport assessment
        """
        try:
            if user_id not in self.user_profiles:
                return {"rapport_level": "unknown", "confidence": 0.0}
            
            profile = self.user_profiles[user_id]
            
            # Calculate rapport based on various factors
            interaction_count = profile.get("interaction_count", 0)
            positive_feedback = profile.get("positive_feedback", 0)
            negative_feedback = profile.get("negative_feedback", 0)
            conversation_length = profile.get("avg_conversation_length", 0)
            
            # Simple rapport calculation
            rapport_score = 0.0
            
            # Interaction frequency factor
            if interaction_count > 0:
                rapport_score += min(interaction_count / 20, 0.3)  # Max 0.3 for interactions
            
            # Feedback factor
            total_feedback = positive_feedback + negative_feedback
            if total_feedback > 0:
                feedback_ratio = positive_feedback / total_feedback
                rapport_score += feedback_ratio * 0.4  # Max 0.4 for feedback
            
            # Conversation engagement factor
            if conversation_length > 0:
                engagement_score = min(conversation_length / 100, 0.3)  # Max 0.3 for engagement
                rapport_score += engagement_score
            
            # Determine rapport level
            if rapport_score >= 0.8:
                rapport_level = "excellent"
            elif rapport_score >= 0.6:
                rapport_level = "good"
            elif rapport_score >= 0.4:
                rapport_level = "developing"
            elif rapport_score >= 0.2:
                rapport_level = "basic"
            else:
                rapport_level = "new"
            
            return {
                "rapport_level": rapport_level,
                "rapport_score": round(rapport_score, 3),
                "confidence": min(interaction_count / 10, 1.0),
                "factors": {
                    "interaction_count": interaction_count,
                    "positive_feedback": positive_feedback,
                    "negative_feedback": negative_feedback,
                    "avg_conversation_length": conversation_length
                }
            }
            
        except Exception as e:
            logger.error(f"Error assessing rapport for user {user_id}: {e}")
            return {"rapport_level": "error", "confidence": 0.0, "error": str(e)}
    
    async def _initialize_user_profile(self, user_id: str):
        """Initialize a new user profile"""
        profile = {
            "user_id": user_id,
            "created_at": datetime.now(),
            "last_updated": datetime.now(),
            "interaction_count": 0,
            "persona": self.default_persona.copy(),
            "preferences": {},
            "relationship_stage": "new",
            "positive_feedback": 0,
            "negative_feedback": 0,
            "avg_conversation_length": 0,
            "total_conversation_time": 0,
            "current_conversation": None,
            "conversation_history": []
        }
        
        self.user_profiles[user_id] = profile
        await self._store_user_profile(user_id)
        
        logger.info(f"Initialized user profile for {user_id}")
    
    async def _get_or_create_conversation(self, user_id: str) -> str:
        """Get existing conversation or create new one"""
        profile = self.user_profiles[user_id]
        current_conversation = profile.get("current_conversation")
        
        # Check if current conversation is still active (within last hour)
        if current_conversation and current_conversation in self.active_conversations:
            conversation = self.active_conversations[current_conversation]
            last_activity = datetime.fromisoformat(conversation["last_activity"])
            if (datetime.now() - last_activity).total_seconds() < 3600:  # 1 hour
                return current_conversation
        
        # Create new conversation
        conversation_id = str(uuid.uuid4())
        self.active_conversations[conversation_id] = {
            "id": conversation_id,
            "user_id": user_id,
            "started_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
            "interactions": [],
            "status": "active"
        }
        
        # Update user profile
        self.user_profiles[user_id]["current_conversation"] = conversation_id
        
        return conversation_id
    
    async def _analyze_interaction_context(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze interaction for contextual information"""
        try:
            # Extract key context elements
            context_analysis = {
                "timestamp": datetime.now().isoformat(),
                "sentiment": "neutral",
                "intent": "unknown",
                "topics": [],
                "urgency": "low",
                "formality": "medium",
                "emotional_state": "neutral"
            }
            
            # Analyze text content if available
            if "content" in interaction_data and isinstance(interaction_data["content"], str):
                content = interaction_data["content"]
                
                # Simple sentiment analysis
                positive_words = ["good", "great", "excellent", "thank", "pleased", "happy"]
                negative_words = ["bad", "terrible", "awful", "angry", "frustrated", "upset"]
                
                positive_count = sum(1 for word in positive_words if word in content.lower())
                negative_count = sum(1 for word in negative_words if word in content.lower())
                
                if positive_count > negative_count:
                    context_analysis["sentiment"] = "positive"
                elif negative_count > positive_count:
                    context_analysis["sentiment"] = "negative"
                
                # Urgency detection
                urgent_indicators = ["urgent", "asap", "immediately", "emergency", "critical"]
                if any(indicator in content.lower() for indicator in urgent_indicators):
                    context_analysis["urgency"] = "high"
                
                # Formality detection
                formal_indicators = ["please", "would you", "could you", "thank you"]
                informal_indicators = ["hey", "hi", "gonna", "wanna", "thanks"]
                
                formal_count = sum(1 for indicator in formal_indicators if indicator in content.lower())
                informal_count = sum(1 for indicator in informal_indicators if indicator in content.lower())
                
                if formal_count > informal_count:
                    context_analysis["formality"] = "high"
                elif informal_count > formal_count:
                    context_analysis["formality"] = "low"
            
            return context_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing interaction context: {e}")
            return {"timestamp": datetime.now().isoformat(), "error": str(e)}
    
    async def _update_user_profile(self, user_id: str, interaction_data: Dict[str, Any], context_analysis: Dict[str, Any]):
        """Update user profile based on interaction"""
        profile = self.user_profiles[user_id]
        
        # Increment interaction count
        profile["interaction_count"] += 1
        profile["last_updated"] = datetime.now()
        
        # Update preferences based on context
        preferences = profile.get("preferences", {})
        
        # Update communication style preferences
        if "formality" in context_analysis:
            preferences["preferred_formality"] = context_analysis["formality"]
        
        # Track sentiment patterns
        sentiment_history = preferences.get("sentiment_history", [])
        if "sentiment" in context_analysis:
            sentiment_history.append(context_analysis["sentiment"])
            # Keep last 10 sentiments
            preferences["sentiment_history"] = sentiment_history[-10:]
        
        profile["preferences"] = preferences
        
        # Update relationship stage
        interaction_count = profile["interaction_count"]
        if interaction_count == 1:
            profile["relationship_stage"] = "new"
        elif interaction_count <= 5:
            profile["relationship_stage"] = "getting_acquainted"
        elif interaction_count <= 20:
            profile["relationship_stage"] = "familiar"
        else:
            profile["relationship_stage"] = "established"
        
        await self._store_user_profile(user_id)
    
    async def _store_interaction(self, conversation_id: str, interaction_id: str, interaction_data: Dict[str, Any], context_analysis: Dict[str, Any]):
        """Store interaction in conversation history"""
        if conversation_id in self.active_conversations:
            conversation = self.active_conversations[conversation_id]
            
            interaction_record = {
                "id": interaction_id,
                "timestamp": datetime.now().isoformat(),
                "data": interaction_data,
                "context": context_analysis
            }
            
            conversation["interactions"].append(interaction_record)
            conversation["last_activity"] = datetime.now().isoformat()
            
            # Keep conversation size manageable
            if len(conversation["interactions"]) > 50:
                conversation["interactions"] = conversation["interactions"][-30:]  # Keep last 30
    
    async def _adjust_persona(self, user_id: str, context_analysis: Dict[str, Any]):
        """Adjust persona based on context analysis"""
        profile = self.user_profiles[user_id]
        current_persona = profile.get("persona", self.default_persona.copy())
        
        # Adjust based on user's communication style
        if "formality" in context_analysis:
            if context_analysis["formality"] == "high":
                current_persona["tone"] = "formal"
                current_persona["formality"] = "high"
            elif context_analysis["formality"] == "low":
                current_persona["tone"] = "casual"
                current_persona["formality"] = "low"
        
        # Adjust empathy based on sentiment
        if "sentiment" in context_analysis:
            if context_analysis["sentiment"] == "negative":
                current_persona["empathy_level"] = "very_high"
                current_persona["response_style"] = "supportive"
        
        profile["persona"] = current_persona
        self.persona_adjustments += 1
    
    async def _validate_persona(self, persona: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize persona settings"""
        valid_persona = self.default_persona.copy()
        
        # Validate each field
        if "tone" in persona and persona["tone"] in ["formal", "professional", "casual", "friendly"]:
            valid_persona["tone"] = persona["tone"]
        
        if "formality" in persona and persona["formality"] in ["low", "medium", "high"]:
            valid_persona["formality"] = persona["formality"]
        
        if "empathy_level" in persona and persona["empathy_level"] in ["low", "medium", "high", "very_high"]:
            valid_persona["empathy_level"] = persona["empathy_level"]
        
        if "response_style" in persona and persona["response_style"] in ["helpful", "supportive", "analytical", "creative"]:
            valid_persona["response_style"] = persona["response_style"]
        
        return valid_persona
    
    async def _generate_context_summary(self, user_id: str) -> str:
        """Generate a summary of current context for the user"""
        if user_id not in self.user_profiles:
            return "New user, no prior context available."
        
        profile = self.user_profiles[user_id]
        interaction_count = profile.get("interaction_count", 0)
        relationship_stage = profile.get("relationship_stage", "new")
        
        # Get recent sentiment
        preferences = profile.get("preferences", {})
        sentiment_history = preferences.get("sentiment_history", [])
        recent_sentiment = sentiment_history[-1] if sentiment_history else "neutral"
        
        summary = f"User has {interaction_count} previous interactions. "
        summary += f"Relationship stage: {relationship_stage}. "
        summary += f"Recent sentiment: {recent_sentiment}. "
        
        # Add persona information
        persona = profile.get("persona", {})
        if persona.get("tone"):
            summary += f"Preferred communication tone: {persona['tone']}. "
        
        return summary
    
    async def _store_user_profile(self, user_id: str):
        """Store user profile to persistent storage"""
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            
            memory_item = MemoryItem(
                id=f"user_profile_{user_id}",
                scope="global",
                data=profile,
                created_at=datetime.fromisoformat(profile["created_at"].isoformat() if isinstance(profile["created_at"], datetime) else profile["created_at"]),
                updated_at=datetime.now()
            )
            
            await DatabaseManager.save_memory_item(memory_item)
    
    async def _relationship_maintenance_loop(self):
        """Background maintenance for relationships"""
        while self.status == "active":
            try:
                # Clean up old conversations
                await self._cleanup_old_conversations()
                
                # Update relationship stages
                await self._update_relationship_stages()
                
                # Clean up inactive user profiles
                await self._cleanup_inactive_profiles()
                
                self.context_updates += 1
                await self._update_state()
                
                await asyncio.sleep(600)  # 10 minutes
                
            except Exception as e:
                logger.error(f"Error in relationship maintenance loop: {e}")
                await asyncio.sleep(600)
    
    async def _cleanup_old_conversations(self):
        """Clean up conversations older than 24 hours"""
        current_time = datetime.now()
        to_remove = []
        
        for conv_id, conversation in self.active_conversations.items():
            last_activity = datetime.fromisoformat(conversation["last_activity"])
            if (current_time - last_activity).total_seconds() > 86400:  # 24 hours
                to_remove.append(conv_id)
        
        for conv_id in to_remove:
            del self.active_conversations[conv_id]
    
    async def _update_relationship_stages(self):
        """Update relationship stages for all users"""
        for user_id, profile in self.user_profiles.items():
            old_stage = profile.get("relationship_stage", "new")
            interaction_count = profile.get("interaction_count", 0)
            
            # Determine new stage
            if interaction_count >= 50:
                new_stage = "established"
            elif interaction_count >= 20:
                new_stage = "familiar"
            elif interaction_count >= 5:
                new_stage = "getting_acquainted"
            else:
                new_stage = "new"
            
            if new_stage != old_stage:
                profile["relationship_stage"] = new_stage
                await self._store_user_profile(user_id)
    
    async def _cleanup_inactive_profiles(self):
        """Remove profiles that haven't been active for a long time"""
        current_time = datetime.now()
        to_remove = []
        
        for user_id, profile in self.user_profiles.items():
            last_updated = profile.get("last_updated")
            if isinstance(last_updated, datetime):
                if (current_time - last_updated).days > 30:  # 30 days
                    to_remove.append(user_id)
        
        for user_id in to_remove:
            del self.user_profiles[user_id]
    
    async def _update_state(self):
        """Update component state in database"""
        state = AgentState(
            agent_name=self.name,
            status=self.status,
            last_activity=datetime.now(),
            metrics={
                "interaction_count": float(self.interaction_count),
                "context_updates": float(self.context_updates),
                "persona_adjustments": float(self.persona_adjustments),
                "active_conversations": float(len(self.active_conversations)),
                "user_profiles": float(len(self.user_profiles))
            },
            config={
                "default_persona": self.default_persona
            }
        )
        await DatabaseManager.update_agent_state(state)
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current component status"""
        return {
            "name": self.name,
            "status": self.status,
            "interaction_count": self.interaction_count,
            "context_updates": self.context_updates,
            "persona_adjustments": self.persona_adjustments,
            "active_conversations": len(self.active_conversations),
            "user_profiles": len(self.user_profiles),
            "default_persona": self.default_persona
        }

