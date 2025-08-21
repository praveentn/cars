# api/concept_graph.py
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException
import logging
import json

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/graph")
async def get_concept_graph():
    """Get concept graph data for visualization"""
    try:
        # Mock data - in real implementation, this would come from Flywheel component
        nodes = [
            {"id": "ml", "name": "Machine Learning", "type": "concept", "weight": 0.9, "connections": 5},
            {"id": "ai", "name": "Artificial Intelligence", "type": "concept", "weight": 0.95, "connections": 8},
            {"id": "neural", "name": "Neural Networks", "type": "concept", "weight": 0.8, "connections": 4},
            {"id": "deep", "name": "Deep Learning", "type": "concept", "weight": 0.85, "connections": 6},
            {"id": "python", "name": "Python", "type": "skill", "weight": 0.7, "connections": 3},
            {"id": "fastapi", "name": "FastAPI", "type": "skill", "weight": 0.75, "connections": 2},
        ]
        
        edges = [
            {"source": "ai", "target": "ml", "strength": 0.9, "type": "contains"},
            {"source": "ml", "target": "neural", "strength": 0.8, "type": "uses"},
            {"source": "neural", "target": "deep", "strength": 0.95, "type": "subset"},
            {"source": "ml", "target": "python", "strength": 0.7, "type": "implemented_in"},
            {"source": "python", "target": "fastapi", "strength": 0.6, "type": "uses"},
            {"source": "ai", "target": "deep", "strength": 0.8, "type": "contains"},
        ]
        
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "total_concepts": len([n for n in nodes if n["type"] == "concept"]),
                "total_skills": len([n for n in nodes if n["type"] == "skill"]),
                "total_connections": len(edges),
                "last_updated": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting concept graph: {e}")
        raise HTTPException(status_code=500, detail="Failed to get concept graph")

@router.get("/learned-items")
async def get_learned_items(limit: int = 50):
    """Get recently learned items"""
    try:
        # Mock learned items - would come from Memory Cache and Retriever
        learned_items = [
            {
                "id": "learn_1",
                "type": "concept",
                "title": "Transformer Architecture",
                "content": "Learned about attention mechanisms and transformer models",
                "source": "user_input",
                "confidence": 0.85,
                "learned_at": datetime.now().isoformat(),
                "connections": ["neural", "deep", "ai"]
            },
            {
                "id": "learn_2", 
                "type": "skill",
                "title": "FastAPI Deployment",
                "content": "Learned deployment strategies for FastAPI applications",
                "source": "documentation",
                "confidence": 0.75,
                "learned_at": datetime.now().isoformat(),
                "connections": ["fastapi", "python"]
            }
        ]
        
        return {
            "items": learned_items[:limit],
            "total": len(learned_items),
            "categories": {
                "concepts": len([i for i in learned_items if i["type"] == "concept"]),
                "skills": len([i for i in learned_items if i["type"] == "skill"]),
                "facts": len([i for i in learned_items if i["type"] == "fact"])
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting learned items: {e}")
        raise HTTPException(status_code=500, detail="Failed to get learned items")

@router.get("/flywheel-state")
async def get_flywheel_state():
    """Get current flywheel/meta-learning state"""
    try:
        flywheel_state = {
            "experiments": {
                "active": 3,
                "completed": 15,
                "success_rate": 0.73
            },
            "policies": {
                "total": 8,
                "recently_updated": 2,
                "performance_improvement": 0.12
            },
            "concepts": {
                "total_nodes": 156,
                "total_connections": 234,
                "evolution_rate": 0.08
            },
            "learning_metrics": {
                "knowledge_gaps_identified": 12,
                "concepts_merged": 5,
                "new_patterns_detected": 8
            },
            "last_evolution": datetime.now().isoformat()
        }
        
        return flywheel_state
        
    except Exception as e:
        logger.error(f"Error getting flywheel state: {e}")
        raise HTTPException(status_code=500, detail="Failed to get flywheel state")