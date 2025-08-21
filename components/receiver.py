# components/receiver.py
import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging
import hashlib
import json
from core.database import DatabaseManager, Event, AgentState
from core.llm_client import llm_client

logger = logging.getLogger(__name__)

class ReceiverComponent:
    """
    Receiver (Input Gateway / Thalamus Analogy)
    - Universal entry point for all external signals
    - Normalizes inputs into standard schema
    - Routes signals to relevant agents
    """
    
    def __init__(self):
        self.name = "receiver"
        self.status = "idle"
        self.processed_count = 0
        self.error_count = 0
        self.duplicate_count = 0
        self.seen_fingerprints = set()
        self.routing_rules = {
            "text": ["self_agent"],
            "audio": ["self_agent", "conscious_agent"],
            "image": ["conscious_agent"],
            "event": ["unconscious_agent"],
            "webhook": ["self_agent"]
        }
    
    async def start(self):
        """Start the receiver component"""
        self.status = "active"
        await self._update_state()
        logger.info("Receiver component started")
    
    async def stop(self):
        """Stop the receiver component"""
        self.status = "stopped"
        await self._update_state()
        logger.info("Receiver component stopped")
    
    async def ingest(self, 
                    source: str, 
                    modality: str, 
                    payload: Dict[str, Any], 
                    tags: List[str] = None) -> str:
        """
        Ingest and process incoming data
        
        Args:
            source: Source of the data (user, webhook, system)
            modality: Type of data (text, audio, image, event)
            payload: Actual data content
            tags: Optional tags for categorization
        
        Returns:
            Event ID
        """
        try:
            # Generate event ID
            event_id = str(uuid.uuid4())
            
            # Create fingerprint for deduplication
            fingerprint = self._create_fingerprint(payload)
            
            # Check for duplicates
            if fingerprint in self.seen_fingerprints:
                self.duplicate_count += 1
                logger.warning(f"Duplicate event detected: {fingerprint}")
                raise ValueError("Duplicate event")
            
            self.seen_fingerprints.add(fingerprint)
            
            # Normalize input
            normalized_event = await self._normalize_input(
                event_id, source, modality, payload, tags or []
            )
            
            # Save to database
            await DatabaseManager.save_event(normalized_event)
            
            # Route to appropriate agents
            await self._route_event(normalized_event)
            
            self.processed_count += 1
            await self._update_state()
            
            logger.info(f"Event {event_id} processed successfully")
            return event_id
            
        except Exception as e:
            self.error_count += 1
            await self._update_state()
            logger.error(f"Error processing event: {e}")
            raise
    
    async def _normalize_input(self, 
                              event_id: str, 
                              source: str, 
                              modality: str, 
                              payload: Dict[str, Any], 
                              tags: List[str]) -> Event:
        """Normalize input into standard Event format"""
        
        # Add metadata
        normalized_payload = {
            **payload,
            "received_at": datetime.now().isoformat(),
            "fingerprint": self._create_fingerprint(payload)
        }
        
        # Enhance with AI analysis for text inputs
        if modality == "text" and "content" in payload:
            try:
                intent_analysis = await llm_client.analyze_intent(payload["content"])
                normalized_payload["intent_analysis"] = intent_analysis
                
                # Add intent-based tags
                if intent_analysis.get("intent"):
                    tags.append(f"intent:{intent_analysis['intent']}")
                if intent_analysis.get("urgency"):
                    tags.append(f"urgency:{intent_analysis['urgency']}")
                    
            except Exception as e:
                logger.warning(f"Intent analysis failed: {e}")
        
        return Event(
            id=event_id,
            timestamp=datetime.now(),
            source=source,
            modality=modality,
            payload=normalized_payload,
            tags=tags,
            status="normalized"
        )
    
    async def _route_event(self, event: Event):
        """Route event to appropriate agents based on modality and content"""
        routes = self.routing_rules.get(event.modality, ["self_agent"])
        
        # Enhanced routing based on content analysis
        if event.modality == "text" and "intent_analysis" in event.payload:
            intent = event.payload["intent_analysis"].get("intent", "")
            urgency = event.payload["intent_analysis"].get("urgency", "low")
            
            # High urgency items go to conscious agent
            if urgency == "high":
                routes.append("conscious_agent")
            
            # Specific intents route to specialized agents
            if "learn" in intent.lower() or "remember" in intent.lower():
                routes.append("memory_cache")
            
            if "search" in intent.lower() or "find" in intent.lower():
                routes.append("retriever")
        
        # Update event status
        event.status = "routed"
        await DatabaseManager.save_event(event)
        
        # TODO: Send to actual agent queues (placeholder for now)
        logger.info(f"Event {event.id} routed to: {routes}")
        
        # Save routing metric
        await DatabaseManager.save_metric("events_routed", 1.0, self.name)
    
    def _create_fingerprint(self, payload: Dict[str, Any]) -> str:
        """Create fingerprint for deduplication"""
        content = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    async def _update_state(self):
        """Update component state in database"""
        state = AgentState(
            agent_name=self.name,
            status=self.status,
            last_activity=datetime.now(),
            metrics={
                "processed_count": float(self.processed_count),
                "error_count": float(self.error_count),
                "duplicate_count": float(self.duplicate_count),
                "success_rate": round(
                    self.processed_count / max(self.processed_count + self.error_count, 1) * 100, 2
                )
            },
            config={
                "routing_rules": self.routing_rules
            }
        )
        await DatabaseManager.update_agent_state(state)
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current component status"""
        return {
            "name": self.name,
            "status": self.status,
            "processed_count": self.processed_count,
            "error_count": self.error_count,
            "duplicate_count": self.duplicate_count,
            "success_rate": round(
                self.processed_count / max(self.processed_count + self.error_count, 1) * 100, 2
            ),
            "routing_rules": self.routing_rules
        }
    
    async def update_routing_rules(self, new_rules: Dict[str, List[str]]):
        """Update routing rules dynamically"""
        self.routing_rules.update(new_rules)
        await self._update_state()
        logger.info(f"Routing rules updated: {new_rules}")

# Example usage functions for testing
async def test_receiver():
    """Test function for receiver component"""
    receiver = ReceiverComponent()
    await receiver.start()
    
    # Test text input
    event_id = await receiver.ingest(
        source="user",
        modality="text",
        payload={"content": "Hello, I need help with something urgent"},
        tags=["test"]
    )
    
    print(f"Created event: {event_id}")
    status = await receiver.get_status()
    print(f"Receiver status: {status}")
    
    await receiver.stop()

if __name__ == "__main__":
    asyncio.run(test_receiver())
