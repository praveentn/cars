# components/flywheel.py
import asyncio
import uuid
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import logging
import json

from core.database import DatabaseManager, AgentState
from core.llm_client import llm_client

logger = logging.getLogger(__name__)

class FlywheelComponent:
    """
    Flywheel (Meta-Learning & Evolution Layer)
    - Continuous self-improvement mechanism
    - Represents everything the system encounters in a conceptual graph
    - Identifies knowledge gaps, strengthens connections, adjusts heuristics
    - Enables meta-learning: refining prompts, updating weights, improving retrieval
    """
    
    def __init__(self):
        self.name = "flywheel"
        self.status = "idle"
        self.concept_graph = {}  # Node ID -> Node data
        self.connections = {}    # Connection ID -> Edge data
        self.experiments = {}    # Experiment ID -> Experiment data
        self.policies = {}       # Policy name -> Policy data
        
        self.nodes_created = 0
        self.connections_made = 0
        self.experiments_run = 0
        self.policy_updates = 0
        
        # Meta-learning parameters
        self.learning_rate = 0.1
        self.decay_rate = 0.01
        self.connection_threshold = 0.5
        self.experiment_duration = 3600  # 1 hour
        
        # Default policies
        self._initialize_default_policies()
    
    async def start(self):
        """Start the flywheel component"""
        self.status = "active"
        await self._update_state()
        logger.info("Flywheel component started")
        
        # Start background evolution loops
        asyncio.create_task(self._concept_evolution_loop())
        asyncio.create_task(self._experiment_runner_loop())
        asyncio.create_task(self._policy_optimization_loop())
    
    async def stop(self):
        """Stop the flywheel component"""
        self.status = "stopped"
        
        # Save all state to persistent storage
        await self._save_state()
        
        await self._update_state()
        logger.info("Flywheel component stopped")
    
    async def observe_representation(self, representation_data: Dict[str, Any]) -> str:
        """
        Observe and integrate new representation into the concept graph
        
        Args:
            representation_data: New representation to integrate
            
        Returns:
            Node ID of the created/updated concept
        """
        try:
            # Extract key concepts from the representation
            concepts = await self._extract_concepts(representation_data)
            
            node_ids = []
            for concept in concepts:
                node_id = await self._create_or_update_node(concept)
                node_ids.append(node_id)
            
            # Create connections between related concepts
            await self._create_concept_connections(node_ids, representation_data)
            
            # Update graph metrics
            await self._update_graph_metrics()
            
            logger.info(f"Integrated representation with {len(concepts)} concepts")
            return node_ids[0] if node_ids else ""
            
        except Exception as e:
            logger.error(f"Error observing representation: {e}")
            raise
    
    async def evolve_concept(self, concept_id: str, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evolve a concept based on new evidence
        
        Args:
            concept_id: ID of the concept to evolve
            evidence: Evidence for evolution
            
        Returns:
            Evolution results
        """
        try:
            if concept_id not in self.concept_graph:
                raise ValueError(f"Concept {concept_id} not found")
            
            concept = self.concept_graph[concept_id]
            
            # Calculate evidence strength
            evidence_strength = await self._calculate_evidence_strength(evidence)
            
            # Update concept weights and properties
            old_weight = concept.get("weight", 0.5)
            new_weight = old_weight + self.learning_rate * (evidence_strength - old_weight)
            
            concept["weight"] = max(0.0, min(1.0, new_weight))
            concept["last_updated"] = datetime.now()
            concept["update_count"] = concept.get("update_count", 0) + 1
            
            # Add evidence to concept history
            if "evidence_history" not in concept:
                concept["evidence_history"] = []
            
            concept["evidence_history"].append({
                "evidence": evidence,
                "strength": evidence_strength,
                "timestamp": datetime.now().isoformat()
            })
            
            # Keep history manageable
            if len(concept["evidence_history"]) > 20:
                concept["evidence_history"] = concept["evidence_history"][-15:]
            
            # Check for concept merging opportunities
            await self._check_concept_merging(concept_id)
            
            evolution_result = {
                "concept_id": concept_id,
                "old_weight": old_weight,
                "new_weight": concept["weight"],
                "evidence_strength": evidence_strength,
                "update_count": concept["update_count"]
            }
            
            logger.info(f"Evolved concept {concept_id}: weight {old_weight:.3f} -> {concept['weight']:.3f}")
            return evolution_result
            
        except Exception as e:
            logger.error(f"Error evolving concept {concept_id}: {e}")
            raise
    
    async def run_experiment(self, experiment_config: Dict[str, Any]) -> str:
        """
        Start a new A/B experiment
        
        Args:
            experiment_config: Configuration for the experiment
            
        Returns:
            Experiment ID
        """
        try:
            experiment_id = str(uuid.uuid4())
            
            experiment = {
                "id": experiment_id,
                "name": experiment_config.get("name", f"experiment_{experiment_id[:8]}"),
                "type": experiment_config.get("type", "ab_test"),
                "variants": experiment_config.get("variants", ["control", "treatment"]),
                "metrics": experiment_config.get("metrics", ["success_rate"]),
                "status": "running",
                "started_at": datetime.now(),
                "duration": experiment_config.get("duration", self.experiment_duration),
                "results": {},
                "participants": {},
                "config": experiment_config
            }
            
            self.experiments[experiment_id] = experiment
            self.experiments_run += 1
            
            await self._update_state()
            
            logger.info(f"Started experiment {experiment_id}: {experiment['name']}")
            return experiment_id
            
        except Exception as e:
            logger.error(f"Error running experiment: {e}")
            raise
    
    async def update_policy(self, policy_name: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a system policy based on learning
        
        Args:
            policy_name: Name of the policy to update
            updates: Updates to apply
            
        Returns:
            Updated policy
        """
        try:
            if policy_name not in self.policies:
                # Create new policy
                self.policies[policy_name] = {
                    "name": policy_name,
                    "created_at": datetime.now(),
                    "version": 1,
                    "parameters": {},
                    "performance_history": []
                }
            
            policy = self.policies[policy_name]
            
            # Apply updates
            old_parameters = policy["parameters"].copy()
            policy["parameters"].update(updates)
            policy["version"] += 1
            policy["updated_at"] = datetime.now()
            
            # Record performance if provided
            if "performance" in updates:
                policy["performance_history"].append({
                    "performance": updates["performance"],
                    "timestamp": datetime.now().isoformat(),
                    "version": policy["version"]
                })
                
                # Keep history manageable
                if len(policy["performance_history"]) > 50:
                    policy["performance_history"] = policy["performance_history"][-30:]
            
            self.policy_updates += 1
            await self._update_state()
            
            logger.info(f"Updated policy {policy_name} to version {policy['version']}")
            
            return {
                "policy_name": policy_name,
                "old_parameters": old_parameters,
                "new_parameters": policy["parameters"],
                "version": policy["version"]
            }
            
        except Exception as e:
            logger.error(f"Error updating policy {policy_name}: {e}")
            raise
    
    async def get_knowledge_gaps(self) -> List[Dict[str, Any]]:
        """
        Identify knowledge gaps in the concept graph
        
        Returns:
            List of identified knowledge gaps
        """
        try:
            gaps = []
            
            # Find isolated concepts (few connections)
            for node_id, node in self.concept_graph.items():
                connection_count = len([c for c in self.connections.values() 
                                     if c["source"] == node_id or c["target"] == node_id])
                
                if connection_count < 2 and node.get("weight", 0) > 0.3:
                    gaps.append({
                        "type": "isolated_concept",
                        "concept_id": node_id,
                        "concept_name": node.get("name", "unknown"),
                        "connection_count": connection_count,
                        "weight": node.get("weight", 0),
                        "priority": "medium"
                    })
            
            # Find weak connections that might need strengthening
            for conn_id, connection in self.connections.items():
                if connection.get("strength", 0) < 0.3:
                    gaps.append({
                        "type": "weak_connection",
                        "connection_id": conn_id,
                        "source": connection["source"],
                        "target": connection["target"],
                        "strength": connection.get("strength", 0),
                        "priority": "low"
                    })
            
            # Find concepts that haven't been updated recently
            cutoff_date = datetime.now() - timedelta(days=7)
            for node_id, node in self.concept_graph.items():
                last_updated = node.get("last_updated", node.get("created_at", datetime.now()))
                if isinstance(last_updated, str):
                    last_updated = datetime.fromisoformat(last_updated)
                
                if last_updated < cutoff_date and node.get("weight", 0) > 0.5:
                    gaps.append({
                        "type": "stale_concept",
                        "concept_id": node_id,
                        "concept_name": node.get("name", "unknown"),
                        "last_updated": last_updated.isoformat(),
                        "weight": node.get("weight", 0),
                        "priority": "high"
                    })
            
            # Sort by priority
            priority_order = {"high": 3, "medium": 2, "low": 1}
            gaps.sort(key=lambda x: priority_order.get(x["priority"], 0), reverse=True)
            
            return gaps[:20]  # Return top 20 gaps
            
        except Exception as e:
            logger.error(f"Error identifying knowledge gaps: {e}")
            return []
    
    async def _extract_concepts(self, representation_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract key concepts from representation data"""
        concepts = []
        
        # Extract from different data types
        if "text" in representation_data:
            text_concepts = await self._extract_text_concepts(representation_data["text"])
            concepts.extend(text_concepts)
        
        if "entities" in representation_data:
            for entity in representation_data["entities"]:
                concepts.append({
                    "name": entity.get("name", "unknown"),
                    "type": "entity",
                    "category": entity.get("category", "general"),
                    "confidence": entity.get("confidence", 0.5),
                    "source": "entity_extraction"
                })
        
        if "topics" in representation_data:
            for topic in representation_data["topics"]:
                concepts.append({
                    "name": topic.get("name", "unknown"),
                    "type": "topic",
                    "category": "subject",
                    "confidence": topic.get("confidence", 0.5),
                    "source": "topic_modeling"
                })
        
        return concepts
    
    async def _extract_text_concepts(self, text: str) -> List[Dict[str, Any]]:
        """Extract concepts from text using LLM"""
        try:
            messages = [
                {
                    "role": "system",
                    "content": """Extract key concepts from the given text. Return a JSON array of concepts with:
                    {
                        "name": "concept name",
                        "type": "entity|topic|skill|goal",
                        "category": "general category",
                        "confidence": 0.8
                    }
                    
                    Focus on important, specific concepts rather than common words."""
                },
                {
                    "role": "user",
                    "content": f"Text: {text}"
                }
            ]
            
            response = await llm_client.chat_completion(messages, temperature=0.3)
            
            try:
                concepts_data = json.loads(response["content"])
                
                # Validate and enhance concepts
                concepts = []
                for concept in concepts_data:
                    if isinstance(concept, dict) and "name" in concept:
                        enhanced_concept = {
                            "name": concept["name"],
                            "type": concept.get("type", "general"),
                            "category": concept.get("category", "general"),
                            "confidence": float(concept.get("confidence", 0.5)),
                            "source": "llm_extraction"
                        }
                        concepts.append(enhanced_concept)
                
                return concepts[:10]  # Limit to 10 concepts
                
            except json.JSONDecodeError:
                # Fallback: simple keyword extraction
                words = text.split()
                important_words = [w for w in words if len(w) > 4 and w.isalpha()]
                
                return [
                    {
                        "name": word,
                        "type": "keyword",
                        "category": "general",
                        "confidence": 0.3,
                        "source": "keyword_extraction"
                    }
                    for word in important_words[:5]
                ]
            
        except Exception as e:
            logger.error(f"Error extracting text concepts: {e}")
            return []
    
    async def _create_or_update_node(self, concept: Dict[str, Any]) -> str:
        """Create or update a concept node"""
        # Check if concept already exists
        existing_node = None
        for node_id, node in self.concept_graph.items():
            if node.get("name", "").lower() == concept["name"].lower():
                existing_node = node_id
                break
        
        if existing_node:
            # Update existing node
            node = self.concept_graph[existing_node]
            node["confidence"] = max(node.get("confidence", 0), concept["confidence"])
            node["last_updated"] = datetime.now()
            node["update_count"] = node.get("update_count", 0) + 1
            return existing_node
        else:
            # Create new node
            node_id = str(uuid.uuid4())
            
            node = {
                "id": node_id,
                "name": concept["name"],
                "type": concept["type"],
                "category": concept["category"],
                "confidence": concept["confidence"],
                "weight": concept["confidence"],  # Initial weight based on confidence
                "created_at": datetime.now(),
                "last_updated": datetime.now(),
                "update_count": 1,
                "source": concept["source"]
            }
            
            self.concept_graph[node_id] = node
            self.nodes_created += 1
            
            return node_id
    
    async def _create_concept_connections(self, node_ids: List[str], context: Dict[str, Any]):
        """Create connections between related concepts"""
        # Create connections between concepts that appeared together
        for i, node1_id in enumerate(node_ids):
            for node2_id in node_ids[i+1:]:
                await self._create_connection(node1_id, node2_id, "co_occurrence", 0.3)
    
    async def _create_connection(self, source_id: str, target_id: str, relation_type: str, strength: float):
        """Create a connection between two concepts"""
        # Check if connection already exists
        existing_conn = None
        for conn_id, conn in self.connections.items():
            if ((conn["source"] == source_id and conn["target"] == target_id) or
                (conn["source"] == target_id and conn["target"] == source_id)):
                existing_conn = conn_id
                break
        
        if existing_conn:
            # Strengthen existing connection
            connection = self.connections[existing_conn]
            old_strength = connection["strength"]
            new_strength = old_strength + self.learning_rate * (strength - old_strength)
            connection["strength"] = min(1.0, new_strength)
            connection["last_updated"] = datetime.now()
        else:
            # Create new connection
            conn_id = str(uuid.uuid4())
            
            connection = {
                "id": conn_id,
                "source": source_id,
                "target": target_id,
                "relation_type": relation_type,
                "strength": strength,
                "created_at": datetime.now(),
                "last_updated": datetime.now(),
                "evidence_count": 1
            }
            
            self.connections[conn_id] = connection
            self.connections_made += 1
    
    async def _calculate_evidence_strength(self, evidence: Dict[str, Any]) -> float:
        """Calculate the strength of evidence for concept evolution"""
        # Simple evidence strength calculation
        strength = 0.5  # Base strength
        
        # Adjust based on evidence type
        if evidence.get("type") == "positive_feedback":
            strength += 0.3
        elif evidence.get("type") == "negative_feedback":
            strength -= 0.3
        
        # Adjust based on confidence
        if "confidence" in evidence:
            confidence_factor = (evidence["confidence"] - 0.5) * 0.4
            strength += confidence_factor
        
        # Adjust based on source reliability
        if evidence.get("source") == "user_feedback":
            strength += 0.2
        elif evidence.get("source") == "system_metric":
            strength += 0.1
        
        return max(0.0, min(1.0, strength))
    
    async def _check_concept_merging(self, concept_id: str):
        """Check if concept should be merged with similar concepts"""
        concept = self.concept_graph[concept_id]
        
        # Find similar concepts
        similar_concepts = []
        for other_id, other_concept in self.concept_graph.items():
            if other_id == concept_id:
                continue
            
            # Simple similarity check based on name
            if (concept["name"].lower() in other_concept["name"].lower() or
                other_concept["name"].lower() in concept["name"].lower()):
                
                similarity = len(set(concept["name"].lower().split()) & 
                               set(other_concept["name"].lower().split()))
                
                if similarity > 0:
                    similar_concepts.append((other_id, similarity))
        
        # Merge with most similar concept if similarity is high
        if similar_concepts:
            similar_concepts.sort(key=lambda x: x[1], reverse=True)
            most_similar_id, similarity = similar_concepts[0]
            
            if similarity >= 2:  # At least 2 common words
                await self._merge_concepts(concept_id, most_similar_id)
    
    async def _merge_concepts(self, concept1_id: str, concept2_id: str):
        """Merge two similar concepts"""
        concept1 = self.concept_graph[concept1_id]
        concept2 = self.concept_graph[concept2_id]
        
        # Create merged concept
        merged_concept = {
            "id": concept1_id,  # Keep first concept's ID
            "name": concept1["name"] if len(concept1["name"]) > len(concept2["name"]) else concept2["name"],
            "type": concept1["type"],
            "category": concept1["category"],
            "confidence": max(concept1["confidence"], concept2["confidence"]),
            "weight": (concept1["weight"] + concept2["weight"]) / 2,
            "created_at": min(concept1["created_at"], concept2["created_at"]),
            "last_updated": datetime.now(),
            "update_count": concept1.get("update_count", 0) + concept2.get("update_count", 0),
            "merged_from": [concept1_id, concept2_id],
            "source": "concept_merge"
        }
        
        # Update concept graph
        self.concept_graph[concept1_id] = merged_concept
        del self.concept_graph[concept2_id]
        
        # Update connections to point to merged concept
        for conn in self.connections.values():
            if conn["source"] == concept2_id:
                conn["source"] = concept1_id
            if conn["target"] == concept2_id:
                conn["target"] = concept1_id
        
        logger.info(f"Merged concepts {concept1_id} and {concept2_id}")
    
    def _initialize_default_policies(self):
        """Initialize default system policies"""
        self.policies = {
            "routing": {
                "name": "routing",
                "created_at": datetime.now(),
                "version": 1,
                "parameters": {
                    "intent_threshold": 0.7,
                    "urgency_routing": True,
                    "load_balancing": True
                },
                "performance_history": []
            },
            "response_generation": {
                "name": "response_generation",
                "created_at": datetime.now(),
                "version": 1,
                "parameters": {
                    "temperature": 0.7,
                    "max_tokens": 1000,
                    "use_context": True
                },
                "performance_history": []
            },
            "learning": {
                "name": "learning",
                "created_at": datetime.now(),
                "version": 1,
                "parameters": {
                    "learning_rate": 0.1,
                    "decay_rate": 0.01,
                    "exploration_rate": 0.2
                },
                "performance_history": []
            }
        }
    
    async def _concept_evolution_loop(self):
        """Background loop for concept evolution"""
        while self.status == "active":
            try:
                # Apply decay to all concepts and connections
                await self._apply_decay()
                
                # Identify and merge similar concepts
                await self._batch_concept_merging()
                
                # Update graph metrics
                await self._update_graph_metrics()
                
                await asyncio.sleep(1800)  # 30 minutes
                
            except Exception as e:
                logger.error(f"Error in concept evolution loop: {e}")
                await asyncio.sleep(1800)
    
    async def _experiment_runner_loop(self):
        """Background loop for running experiments"""
        while self.status == "active":
            try:
                # Check and finalize completed experiments
                await self._check_experiment_completion()
                
                # Start new experiments if needed
                await self._start_automatic_experiments()
                
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Error in experiment runner loop: {e}")
                await asyncio.sleep(300)
    
    async def _policy_optimization_loop(self):
        """Background loop for policy optimization"""
        while self.status == "active":
            try:
                # Optimize policies based on performance
                await self._optimize_policies()
                
                await asyncio.sleep(3600)  # 1 hour
                
            except Exception as e:
                logger.error(f"Error in policy optimization loop: {e}")
                await asyncio.sleep(3600)
    
    async def _apply_decay(self):
        """Apply decay to concept weights and connection strengths"""
        for concept in self.concept_graph.values():
            old_weight = concept.get("weight", 0.5)
            new_weight = old_weight * (1 - self.decay_rate)
            concept["weight"] = max(0.1, new_weight)  # Minimum weight
        
        for connection in self.connections.values():
            old_strength = connection.get("strength", 0.5)
            new_strength = old_strength * (1 - self.decay_rate)
            connection["strength"] = max(0.1, new_strength)  # Minimum strength
    
    async def _batch_concept_merging(self):
        """Batch process concept merging"""
        # This would implement more sophisticated concept merging logic
        pass
    
    async def _update_graph_metrics(self):
        """Update graph-level metrics"""
        # Calculate various graph metrics like density, clustering coefficient, etc.
        pass
    
    async def _check_experiment_completion(self):
        """Check and finalize completed experiments"""
        current_time = datetime.now()
        
        for exp_id, experiment in list(self.experiments.items()):
            if experiment["status"] == "running":
                start_time = experiment["started_at"]
                duration = experiment["duration"]
                
                if (current_time - start_time).total_seconds() >= duration:
                    # Finalize experiment
                    experiment["status"] = "completed"
                    experiment["completed_at"] = current_time
                    
                    # Analyze results and update policies
                    await self._analyze_experiment_results(exp_id)
    
    async def _start_automatic_experiments(self):
        """Start new experiments automatically based on system needs"""
        # This would implement logic to automatically start experiments
        # when system performance degrades or new patterns are detected
        pass
    
    async def _optimize_policies(self):
        """Optimize policies based on performance history"""
        for policy_name, policy in self.policies.items():
            performance_history = policy.get("performance_history", [])
            
            if len(performance_history) >= 5:
                # Analyze performance trend
                recent_performance = [p["performance"] for p in performance_history[-5:]]
                avg_performance = sum(recent_performance) / len(recent_performance)
                
                # Simple optimization: adjust parameters if performance is low
                if avg_performance < 0.7:
                    # This would implement more sophisticated optimization
                    self.policy_updates += 1
                    logger.info(f"Optimizing policy {policy_name} due to low performance")
    
    async def _analyze_experiment_results(self, experiment_id: str):
        """Analyze experiment results and apply learnings"""
        experiment = self.experiments[experiment_id]
        
        # Simple result analysis (would be more sophisticated in practice)
        results = experiment.get("results", {})
        
        if results:
            # Determine winning variant
            best_variant = max(results.keys(), key=lambda k: results[k].get("success_rate", 0))
            
            # Update relevant policies
            if experiment.get("type") == "policy_test":
                policy_name = experiment["config"].get("policy_name")
                if policy_name and policy_name in self.policies:
                    winning_params = results[best_variant].get("parameters", {})
                    await self.update_policy(policy_name, winning_params)
        
        logger.info(f"Analyzed experiment {experiment_id}")
    
    async def _save_state(self):
        """Save flywheel state to persistent storage"""
        try:
            # In a real implementation, this would save to database
            state_data = {
                "concept_graph": self.concept_graph,
                "connections": self.connections,
                "policies": self.policies,
                "experiments": self.experiments,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.debug("Flywheel state saved")
            
        except Exception as e:
            logger.error(f"Error saving flywheel state: {e}")
    
    async def _update_state(self):
        """Update component state in database"""
        state = AgentState(
            agent_name=self.name,
            status=self.status,
            last_activity=datetime.now(),
            metrics={
                "nodes_created": float(self.nodes_created),
                "connections_made": float(self.connections_made),
                "experiments_run": float(self.experiments_run),
                "policy_updates": float(self.policy_updates),
                "total_nodes": float(len(self.concept_graph)),
                "total_connections": float(len(self.connections)),
                "active_experiments": float(len([e for e in self.experiments.values() if e["status"] == "running"]))
            },
            config={
                "learning_rate": self.learning_rate,
                "decay_rate": self.decay_rate,
                "connection_threshold": self.connection_threshold
            }
        )
        await DatabaseManager.update_agent_state(state)
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current component status"""
        return {
            "name": self.name,
            "status": self.status,
            "nodes_created": self.nodes_created,
            "connections_made": self.connections_made,
            "experiments_run": self.experiments_run,
            "policy_updates": self.policy_updates,
            "total_nodes": len(self.concept_graph),
            "total_connections": len(self.connections),
            "active_experiments": len([e for e in self.experiments.values() if e["status"] == "running"]),
            "policies": list(self.policies.keys()),
            "config": {
                "learning_rate": self.learning_rate,
                "decay_rate": self.decay_rate,
                "connection_threshold": self.connection_threshold
            }
        }