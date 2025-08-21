# components/unconscious_agent.py
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging
import random

from core.database import DatabaseManager, AgentState
from core.llm_client import llm_client

logger = logging.getLogger(__name__)

class UnconsciousAgentComponent:
    """
    Unconscious Agent (Automatic Processor / System 1)
    - Handles background learning + reflexive responses  
    - Updates long-term memory and knowledge base silently
    - Generates fast, heuristic-based responses
    - Detects patterns, triggers reflexes
    """
    
    def __init__(self):
        self.name = "unconscious_agent"
        self.status = "idle"
        self.reflex_responses = 0
        self.background_updates = 0
        self.pattern_detections = 0
        self.reflex_threshold = 0.8
        self.pattern_cache = {}
        self.last_cleanup = datetime.now()
        
    async def start(self):
        """Start the unconscious agent component"""
        self.status = "active"
        await self._update_state()
        logger.info("Unconscious Agent component started")
        
        # Start background processing loops
        asyncio.create_task(self._background_learning_loop())
        asyncio.create_task(self._pattern_detection_loop())
        asyncio.create_task(self._reflex_response_loop())
    
    async def stop(self):
        """Stop the unconscious agent component"""
        self.status = "stopped"
        await self._update_state()
        logger.info("Unconscious Agent component stopped")
    
    async def process_reflex(self, event_id: str) -> Optional[Dict[str, Any]]:
        """
        Generate a fast reflex response if appropriate
        
        Args:
            event_id: ID of the event to process
            
        Returns:
            Reflex response if generated, None otherwise
        """
        try:
            # Get event from database
            events = await DatabaseManager.get_events(limit=1000)
            event = next((e for e in events if e.id == event_id), None)
            
            if not event:
                return None
            
            # Quick intent analysis
            intent_confidence = await self._quick_intent_analysis(event)
            
            # Generate reflex if confidence is high and risk is low
            if intent_confidence >= self.reflex_threshold and await self._is_low_risk(event):
                reflex_response = await self._generate_reflex_response(event)
                self.reflex_responses += 1
                await self._update_state()
                
                logger.info(f"Reflex response generated for event {event_id}")
                return reflex_response
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing reflex for event {event_id}: {e}")
            return None
    
    async def detect_patterns(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect patterns in provided data
        
        Args:
            data: List of data items to analyze
            
        Returns:
            List of detected patterns
        """
        try:
            patterns = []
            
            # Frequency patterns
            frequency_patterns = await self._detect_frequency_patterns(data)
            patterns.extend(frequency_patterns)
            
            # Temporal patterns
            temporal_patterns = await self._detect_temporal_patterns(data)
            patterns.extend(temporal_patterns)
            
            # Content patterns
            content_patterns = await self._detect_content_patterns(data)
            patterns.extend(content_patterns)
            
            self.pattern_detections += len(patterns)
            await self._update_state()
            
            logger.info(f"Detected {len(patterns)} patterns in {len(data)} data items")
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
            return []
    
    async def update_heuristics(self, feedback: Dict[str, Any]):
        """
        Update heuristic rules based on feedback
        
        Args:
            feedback: Feedback data for heuristic adjustment
        """
        try:
            # Update reflex threshold based on success rate
            if "reflex_success_rate" in feedback:
                success_rate = feedback["reflex_success_rate"]
                if success_rate > 0.9:
                    self.reflex_threshold = max(0.5, self.reflex_threshold - 0.05)
                elif success_rate < 0.7:
                    self.reflex_threshold = min(0.95, self.reflex_threshold + 0.05)
            
            # Update pattern cache with new learnings
            if "patterns" in feedback:
                for pattern in feedback["patterns"]:
                    pattern_key = pattern.get("type", "unknown")
                    if pattern_key not in self.pattern_cache:
                        self.pattern_cache[pattern_key] = []
                    self.pattern_cache[pattern_key].append(pattern)
            
            await self._update_state()
            logger.info("Heuristics updated based on feedback")
            
        except Exception as e:
            logger.error(f"Error updating heuristics: {e}")
    
    async def _quick_intent_analysis(self, event) -> float:
        """Perform quick intent analysis without deep reasoning"""
        # Simple heuristic-based analysis
        confidence = 0.5  # Base confidence
        
        # Check for clear intent indicators
        if event.modality == "text" and "content" in event.payload:
            content = event.payload["content"].lower()
            
            # High confidence patterns
            if any(word in content for word in ["hello", "hi", "help", "please"]):
                confidence += 0.3
            
            # Question patterns
            if content.endswith("?") or content.startswith(("what", "how", "when", "where", "why")):
                confidence += 0.2
            
            # Length-based confidence
            if 5 <= len(content.split()) <= 20:
                confidence += 0.1
        
        # Check tags for intent hints
        if hasattr(event, 'tags') and event.tags:
            if any("intent:" in tag for tag in event.tags):
                confidence += 0.2
        
        return min(confidence, 1.0)
    
    async def _is_low_risk(self, event) -> bool:
        """Determine if event is low risk for reflex response"""
        # Simple risk assessment
        if event.modality != "text":
            return False
        
        if "content" in event.payload:
            content = event.payload["content"].lower()
            
            # High-risk indicators
            risk_words = ["delete", "remove", "cancel", "transfer", "pay", "buy", "urgent"]
            if any(word in content for word in risk_words):
                return False
        
        return True
    
    async def _generate_reflex_response(self, event) -> Dict[str, Any]:
        """Generate a quick reflex response"""
        if event.modality == "text" and "content" in event.payload:
            content = event.payload["content"]
            
            # Simple pattern matching for common queries
            content_lower = content.lower()
            
            if any(greeting in content_lower for greeting in ["hello", "hi", "hey"]):
                response_text = "Hello! How can I help you today?"
            elif content_lower.endswith("?"):
                response_text = "That's an interesting question. Let me think about that..."
            elif "help" in content_lower:
                response_text = "I'm here to help! Could you tell me more about what you need?"
            else:
                response_text = "I understand. Let me process that for you."
        else:
            response_text = f"I've received your {event.modality} input and I'm processing it."
        
        return {
            "type": "reflex_response",
            "content": response_text,
            "confidence": 0.8,
            "event_id": event.id,
            "generated_at": datetime.now().isoformat(),
            "agent": self.name
        }
    
    async def _detect_frequency_patterns(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect frequency-based patterns"""
        patterns = []
        
        # Count occurrences of various attributes
        frequencies = {}
        for item in data:
            for key, value in item.items():
                if isinstance(value, (str, int, float)):
                    freq_key = f"{key}:{value}"
                    frequencies[freq_key] = frequencies.get(freq_key, 0) + 1
        
        # Identify high-frequency patterns
        total_items = len(data)
        for pattern, count in frequencies.items():
            if count >= max(3, total_items * 0.3):  # At least 3 occurrences or 30%
                patterns.append({
                    "type": "frequency",
                    "pattern": pattern,
                    "count": count,
                    "percentage": round((count / total_items) * 100, 2),
                    "confidence": min(count / 10, 1.0)
                })
        
        return patterns
    
    async def _detect_temporal_patterns(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect time-based patterns"""
        patterns = []
        
        # Look for timestamps in data
        timestamps = []
        for item in data:
            for key, value in item.items():
                if isinstance(value, str) and any(char.isdigit() for char in value):
                    try:
                        # Try to parse as datetime
                        timestamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
                        timestamps.append(timestamp)
                    except:
                        continue
        
        if len(timestamps) >= 3:
            # Analyze temporal distribution
            time_diffs = []
            for i in range(1, len(timestamps)):
                diff = (timestamps[i] - timestamps[i-1]).total_seconds()
                time_diffs.append(diff)
            
            if time_diffs:
                avg_diff = sum(time_diffs) / len(time_diffs)
                
                patterns.append({
                    "type": "temporal",
                    "pattern": "regular_intervals",
                    "average_interval_seconds": round(avg_diff, 2),
                    "sample_count": len(timestamps),
                    "confidence": 0.7
                })
        
        return patterns
    
    async def _detect_content_patterns(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect content-based patterns"""
        patterns = []
        
        # Collect text content
        text_contents = []
        for item in data:
            for key, value in item.items():
                if isinstance(value, str) and len(value) > 10:
                    text_contents.append(value.lower())
        
        if text_contents:
            # Find common words/phrases
            word_counts = {}
            for content in text_contents:
                words = content.split()
                for word in words:
                    if len(word) > 3:  # Skip short words
                        word_counts[word] = word_counts.get(word, 0) + 1
            
            # Identify common words
            total_texts = len(text_contents)
            for word, count in word_counts.items():
                if count >= max(2, total_texts * 0.5):  # Appears in 50% of texts
                    patterns.append({
                        "type": "content",
                        "pattern": f"common_word:{word}",
                        "count": count,
                        "percentage": round((count / total_texts) * 100, 2),
                        "confidence": 0.6
                    })
        
        return patterns
    
    async def _background_learning_loop(self):
        """Background learning and maintenance loop"""
        while self.status == "active":
            try:
                # Update embeddings (simulated)
                await self._update_embeddings()
                
                # Clean up old patterns
                await self._cleanup_patterns()
                
                # Update memory structures
                await self._update_memory_structures()
                
                self.background_updates += 1
                await self._update_state()
                
                # Sleep for background update interval
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Error in background learning loop: {e}")
                await asyncio.sleep(600)  # Wait longer on error
    
    async def _pattern_detection_loop(self):
        """Continuous pattern detection loop"""
        while self.status == "active":
            try:
                # Get recent events for pattern analysis
                recent_events = await DatabaseManager.get_events(limit=50)
                
                if recent_events:
                    # Convert events to data format
                    event_data = []
                    for event in recent_events:
                        event_data.append({
                            "id": event.id,
                            "timestamp": event.timestamp.isoformat(),
                            "source": event.source,
                            "modality": event.modality,
                            "status": event.status,
                            "tags": event.tags
                        })
                    
                    # Detect patterns
                    patterns = await self.detect_patterns(event_data)
                    
                    # Store significant patterns
                    for pattern in patterns:
                        if pattern.get("confidence", 0) > 0.7:
                            pattern_type = pattern.get("type", "unknown")
                            if pattern_type not in self.pattern_cache:
                                self.pattern_cache[pattern_type] = []
                            self.pattern_cache[pattern_type].append(pattern)
                
                await asyncio.sleep(180)  # 3 minutes
                
            except Exception as e:
                logger.error(f"Error in pattern detection loop: {e}")
                await asyncio.sleep(300)
    
    async def _reflex_response_loop(self):
        """Monitor for events requiring reflex responses"""
        while self.status == "active":
            try:
                # Get pending events
                pending_events = await DatabaseManager.get_events(limit=20, status="pending")
                
                for event in pending_events:
                    # Try to generate reflex response
                    reflex = await self.process_reflex(event.id)
                    
                    if reflex:
                        # Save reflex response as metric
                        await DatabaseManager.save_metric("reflex_generated", 1.0, self.name)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in reflex response loop: {e}")
                await asyncio.sleep(60)
    
    async def _update_embeddings(self):
        """Update embeddings for recent content"""
        # Placeholder for embedding updates
        logger.debug("Updating embeddings (simulated)")
        await asyncio.sleep(0.1)
    
    async def _cleanup_patterns(self):
        """Clean up old pattern cache entries"""
        current_time = datetime.now()
        
        # Clean up patterns older than 1 hour
        for pattern_type in list(self.pattern_cache.keys()):
            old_patterns = []
            for pattern in self.pattern_cache[pattern_type]:
                if "timestamp" in pattern:
                    try:
                        pattern_time = datetime.fromisoformat(pattern["timestamp"])
                        if (current_time - pattern_time).total_seconds() > 3600:
                            old_patterns.append(pattern)
                    except:
                        pass
            
            # Remove old patterns
            for old_pattern in old_patterns:
                self.pattern_cache[pattern_type].remove(old_pattern)
            
            # Remove empty pattern types
            if not self.pattern_cache[pattern_type]:
                del self.pattern_cache[pattern_type]
    
    async def _update_memory_structures(self):
        """Update memory structures with new learnings"""
        # Placeholder for memory structure updates
        logger.debug("Updating memory structures (simulated)")
        await asyncio.sleep(0.1)
    
    async def _update_state(self):
        """Update component state in database"""
        state = AgentState(
            agent_name=self.name,
            status=self.status,
            last_activity=datetime.now(),
            metrics={
                "reflex_responses": float(self.reflex_responses),
                "background_updates": float(self.background_updates),
                "pattern_detections": float(self.pattern_detections),
                "pattern_types_cached": float(len(self.pattern_cache)),
                "reflex_threshold": round(self.reflex_threshold, 3)
            },
            config={
                "reflex_threshold": self.reflex_threshold,
                "pattern_cache_size": len(self.pattern_cache)
            }
        )
        await DatabaseManager.update_agent_state(state)
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current component status"""
        return {
            "name": self.name,
            "status": self.status,
            "reflex_responses": self.reflex_responses,
            "background_updates": self.background_updates,
            "pattern_detections": self.pattern_detections,
            "pattern_cache_size": len(self.pattern_cache),
            "reflex_threshold": round(self.reflex_threshold, 3)
        }