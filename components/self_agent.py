# components/self_agent.py
import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging
import json
import random
from core.database import DatabaseManager, Event, AgentState, MemoryItem
from core.llm_client import llm_client

logger = logging.getLogger(__name__)

class SelfAgentComponent:
    """
    Self-Agent (Core Learner / Internal Drive)
    - Persistent inquisitor and learning orchestrator
    - Actively probes inputs for new knowledge and meaning
    - Decides what to explore further, store, or discard
    - Feeds knowledge into the representation layer (Flywheel)
    """
    
    def __init__(self):
        self.name = "self_agent"
        self.status = "idle"
        self.observations_count = 0
        self.probes_generated = 0
        self.knowledge_items_created = 0
        self.exploration_policy = "epsilon_greedy"  # epsilon_greedy, thompson_sampling
        self.epsilon = 0.1  # for epsilon-greedy exploration
        self.active_hypotheses = {}
        self.learning_threshold = 0.7
        
    async def start(self):
        """Start the self-agent component"""
        self.status = "active"
        await self._update_state()
        logger.info("Self-Agent component started")
        
        # Start background learning loop
        asyncio.create_task(self._learning_loop())
    
    async def stop(self):
        """Stop the self-agent component"""
        self.status = "stopped"
        await self._update_state()
        logger.info("Self-Agent component stopped")
    
    async def observe(self, event_id: str) -> Dict[str, Any]:
        """
        Observe an event and start exploration cycle
        
        Args:
            event_id: ID of the event to observe
            
        Returns:
            Observation results and generated probes
        """
        try:
            # Fetch event from database
            events = await DatabaseManager.get_events(limit=1000)
            event = next((e for e in events if e.id == event_id), None)
            
            if not event:
                raise ValueError(f"Event {event_id} not found")
            
            self.observations_count += 1
            
            # Generate exploration probes
            probes = await self._generate_probes(event)
            
            # Create hypothesis
            hypothesis_id = await self._create_hypothesis(event, probes)
            
            # Execute probes and gather insights
            insights = await self._execute_probes(probes, event)
            
            # Update hypothesis with findings
            await self._update_hypothesis(hypothesis_id, insights)
            
            # Decide on commitment to long-term memory
            should_commit = await self._evaluate_for_commitment(hypothesis_id, insights)
            
            result = {
                "event_id": event_id,
                "hypothesis_id": hypothesis_id,
                "probes_generated": len(probes),
                "insights_gathered": len(insights),
                "should_commit": should_commit,
                "exploration_strategy": self.exploration_policy
            }
            
            await self._update_state()
            logger.info(f"Observation complete for event {event_id}: {result}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in observation: {e}")
            raise
    
    async def _generate_probes(self, event: Event) -> List[Dict[str, Any]]:
        """Generate clarifying questions and micro-queries"""
        probes = []
        
        # Content-based probes
        if event.modality == "text" and "content" in event.payload:
            content = event.payload["content"]
            
            # Generate clarifying questions
            clarifying_questions = await self._generate_clarifying_questions(content)
            
            for question in clarifying_questions:
                probes.append({
                    "type": "clarifying_question",
                    "question": question,
                    "target": "user"
                })
            
            # Generate knowledge retrieval queries
            retrieval_queries = await self._generate_retrieval_queries(content)
            
            for query in retrieval_queries:
                probes.append({
                    "type": "knowledge_retrieval",
                    "query": query,
                    "target": "retriever"
                })
        
        # Context-based probes
        if event.tags:
            for tag in event.tags:
                if tag.startswith("intent:"):
                    intent = tag.split(":")[1]
                    probes.append({
                        "type": "intent_exploration",
                        "intent": intent,
                        "target": "conscious_agent"
                    })
        
        self.probes_generated += len(probes)
        return probes
    
    async def _generate_clarifying_questions(self, content: str) -> List[str]:
        """Generate clarifying questions using LLM"""
        messages = [
            {
                "role": "system",
                "content": """You are a curious AI that generates clarifying questions to better understand user input. 
                Generate 2-3 specific, relevant questions that would help understand the user's intent, context, or needs better.
                Return only the questions, one per line."""
            },
            {
                "role": "user",
                "content": f"User said: '{content}'\n\nWhat clarifying questions should I ask?"
            }
        ]
        
        try:
            response = await llm_client.chat_completion(messages, temperature=0.7)
            questions = [q.strip() for q in response["content"].split("\n") if q.strip()]
            return questions[:3]  # Limit to 3 questions
        except Exception as e:
            logger.error(f"Failed to generate clarifying questions: {e}")
            return []
    
    async def _generate_retrieval_queries(self, content: str) -> List[str]:
        """Generate queries for knowledge retrieval"""
        messages = [
            {
                "role": "system",
                "content": """Generate 2-3 search queries that would help find relevant information to respond to the user's input.
                Focus on key concepts, entities, and topics mentioned.
                Return only the search queries, one per line."""
            },
            {
                "role": "user",
                "content": f"User input: '{content}'\n\nWhat should I search for?"
            }
        ]
        
        try:
            response = await llm_client.chat_completion(messages, temperature=0.5)
            queries = [q.strip() for q in response["content"].split("\n") if q.strip()]
            return queries[:3]  # Limit to 3 queries
        except Exception as e:
            logger.error(f"Failed to generate retrieval queries: {e}")
            return []
    
    async def _create_hypothesis(self, event: Event, probes: List[Dict[str, Any]]) -> str:
        """Create a hypothesis about the event and required responses"""
        hypothesis_id = str(uuid.uuid4())
        
        hypothesis = {
            "id": hypothesis_id,
            "event_id": event.id,
            "created_at": datetime.now().isoformat(),
            "initial_assessment": {
                "modality": event.modality,
                "source": event.source,
                "tags": event.tags,
                "probe_count": len(probes)
            },
            "probes": probes,
            "insights": [],
            "confidence": 0.0,
            "status": "active"
        }
        
        self.active_hypotheses[hypothesis_id] = hypothesis
        
        # Save to short-term memory
        memory_item = MemoryItem(
            id=f"hypothesis_{hypothesis_id}",
            scope="session",
            data=hypothesis,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        await DatabaseManager.save_memory_item(memory_item)
        return hypothesis_id
    
    async def _execute_probes(self, probes: List[Dict[str, Any]], event: Event) -> List[Dict[str, Any]]:
        """Execute probes and gather insights"""
        insights = []
        
        for probe in probes:
            try:
                if probe["type"] == "knowledge_retrieval":
                    # Simulate knowledge retrieval (placeholder)
                    insight = await self._simulate_knowledge_retrieval(probe["query"], event)
                    
                elif probe["type"] == "clarifying_question":
                    # For now, generate auto-response (in real system, would wait for user input)
                    insight = await self._simulate_clarifying_response(probe["question"], event)
                    
                elif probe["type"] == "intent_exploration":
                    insight = await self._explore_intent(probe["intent"], event)
                
                if insight:
                    insights.append(insight)
                    
            except Exception as e:
                logger.error(f"Failed to execute probe {probe}: {e}")
        
        return insights
    
    async def _simulate_knowledge_retrieval(self, query: str, event: Event) -> Dict[str, Any]:
        """Simulate knowledge retrieval (placeholder)"""
        # In real implementation, this would query the Retriever component
        return {
            "type": "knowledge_retrieval",
            "query": query,
            "results": f"Retrieved knowledge for: {query}",
            "relevance_score": random.uniform(0.3, 0.9),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _simulate_clarifying_response(self, question: str, event: Event) -> Dict[str, Any]:
        """Simulate response to clarifying question"""
        return {
            "type": "clarifying_response",
            "question": question,
            "insight": f"Clarification needed for: {question}",
            "confidence": random.uniform(0.4, 0.8),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _explore_intent(self, intent: str, event: Event) -> Dict[str, Any]:
        """Explore user intent more deeply"""
        return {
            "type": "intent_exploration",
            "intent": intent,
            "analysis": f"Intent '{intent}' requires further exploration",
            "suggested_actions": [f"action_for_{intent}"],
            "confidence": random.uniform(0.5, 0.9),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _update_hypothesis(self, hypothesis_id: str, insights: List[Dict[str, Any]]):
        """Update hypothesis with new insights"""
        if hypothesis_id in self.active_hypotheses:
            hypothesis = self.active_hypotheses[hypothesis_id]
            hypothesis["insights"].extend(insights)
            
            # Calculate confidence based on insights
            confidence_scores = [insight.get("confidence", 0.5) for insight in insights if "confidence" in insight]
            if confidence_scores:
                hypothesis["confidence"] = sum(confidence_scores) / len(confidence_scores)
            
            hypothesis["updated_at"] = datetime.now().isoformat()
            
            # Update in memory
            memory_item = MemoryItem(
                id=f"hypothesis_{hypothesis_id}",
                scope="session",
                data=hypothesis,
                created_at=datetime.fromisoformat(hypothesis["created_at"]),
                updated_at=datetime.now()
            )
            
            await DatabaseManager.save_memory_item(memory_item)
    
    async def _evaluate_for_commitment(self, hypothesis_id: str, insights: List[Dict[str, Any]]) -> bool:
        """Evaluate whether hypothesis should be committed to long-term memory"""
        if hypothesis_id not in self.active_hypotheses:
            return False
        
        hypothesis = self.active_hypotheses[hypothesis_id]
        confidence = hypothesis.get("confidence", 0.0)
        
        # Use exploration policy to decide
        if self.exploration_policy == "epsilon_greedy":
            if random.random() < self.epsilon:
                # Explore: commit randomly
                return random.choice([True, False])
            else:
                # Exploit: commit if confidence is high
                return confidence >= self.learning_threshold
        
        return confidence >= self.learning_threshold
    
    async def _learning_loop(self):
        """Background learning loop"""
        while self.status == "active":
            try:
                # Clean up old hypotheses
                await self._cleanup_hypotheses()
                
                # Update exploration parameters
                await self._update_exploration_parameters()
                
                # Sleep for a bit
                await asyncio.sleep(30)  # Run every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in learning loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _cleanup_hypotheses(self):
        """Clean up old hypotheses"""
        current_time = datetime.now()
        to_remove = []
        
        for hypothesis_id, hypothesis in self.active_hypotheses.items():
            created_at = datetime.fromisoformat(hypothesis["created_at"])
            age_hours = (current_time - created_at).total_seconds() / 3600
            
            # Remove hypotheses older than 1 hour
            if age_hours > 1:
                to_remove.append(hypothesis_id)
        
        for hypothesis_id in to_remove:
            del self.active_hypotheses[hypothesis_id]
    
    async def _update_exploration_parameters(self):
        """Update exploration parameters based on performance"""
        # Simple adaptive epsilon
        if self.observations_count > 0:
            success_rate = self.knowledge_items_created / self.observations_count
            if success_rate > 0.8:
                self.epsilon = max(0.05, self.epsilon * 0.95)  # Reduce exploration
            elif success_rate < 0.3:
                self.epsilon = min(0.3, self.epsilon * 1.05)  # Increase exploration
    
    async def _update_state(self):
        """Update component state in database"""
        state = AgentState(
            agent_name=self.name,
            status=self.status,
            last_activity=datetime.now(),
            metrics={
                "observations_count": float(self.observations_count),
                "probes_generated": float(self.probes_generated),
                "knowledge_items_created": float(self.knowledge_items_created),
                "active_hypotheses": float(len(self.active_hypotheses)),
                "exploration_epsilon": round(self.epsilon, 3),
                "learning_threshold": round(self.learning_threshold, 3)
            },
            config={
                "exploration_policy": self.exploration_policy,
                "epsilon": self.epsilon,
                "learning_threshold": self.learning_threshold
            }
        )
        await DatabaseManager.update_agent_state(state)
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current component status"""
        return {
            "name": self.name,
            "status": self.status,
            "observations_count": self.observations_count,
            "probes_generated": self.probes_generated,
            "knowledge_items_created": self.knowledge_items_created,
            "active_hypotheses": len(self.active_hypotheses),
            "exploration_policy": self.exploration_policy,
            "epsilon": round(self.epsilon, 3),
            "learning_threshold": round(self.learning_threshold, 3)
        }
    
    async def update_config(self, config: Dict[str, Any]):
        """Update component configuration"""
        if "exploration_policy" in config:
            self.exploration_policy = config["exploration_policy"]
        if "epsilon" in config:
            self.epsilon = config["epsilon"]
        if "learning_threshold" in config:
            self.learning_threshold = config["learning_threshold"]
        
        await self._update_state()
        logger.info(f"Self-Agent config updated: {config}")

# Example usage
async def test_self_agent():
    """Test function for self-agent component"""
    from components.receiver import ReceiverComponent
    
    # Create receiver and self-agent
    receiver = ReceiverComponent()
    self_agent = SelfAgentComponent()
    
    await receiver.start()
    await self_agent.start()
    
    # Create test event
    event_id = await receiver.ingest(
        source="user",
        modality="text",
        payload={"content": "I want to learn about machine learning"},
        tags=["learning"]
    )
    
    # Have self-agent observe the event
    result = await self_agent.observe(event_id)
    print(f"Observation result: {result}")
    
    # Check status
    status = await self_agent.get_status()
    print(f"Self-Agent status: {status}")
    
    await self_agent.stop()
    await receiver.stop()

if __name__ == "__main__":
    asyncio.run(test_self_agent())
