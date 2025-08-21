# api/agents.py
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging

from core.database import DatabaseManager, AgentState

logger = logging.getLogger(__name__)
router = APIRouter()

class AgentStatusResponse(BaseModel):
    name: str
    status: str
    last_activity: str
    processed_count: int
    error_count: int
    success_rate: float
    config: Dict[str, Any]

class AgentConfigUpdate(BaseModel):
    config: Dict[str, Any]

@router.get("/status", response_model=List[AgentStatusResponse])
async def get_all_agents_status():
    """Get status of all agents"""
    try:
        agent_states = await DatabaseManager.get_agent_states()
        
        agents_status = []
        for agent in agent_states:
            status_response = AgentStatusResponse(
                name=agent.agent_name,
                status=agent.status,
                last_activity=agent.last_activity.isoformat(),
                processed_count=int(agent.metrics.get("processed_count", 0)),
                error_count=int(agent.metrics.get("error_count", 0)),
                success_rate=float(agent.metrics.get("success_rate", 0)),
                config=agent.config
            )
            agents_status.append(status_response)
        
        return agents_status
        
    except Exception as e:
        logger.error(f"Error getting agents status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get agents status")

@router.get("/{agent_name}/status", response_model=AgentStatusResponse)
async def get_agent_status(agent_name: str):
    """Get status of a specific agent"""
    try:
        agent_states = await DatabaseManager.get_agent_states()
        agent = next((a for a in agent_states if a.agent_name == agent_name), None)
        
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found")
        
        return AgentStatusResponse(
            name=agent.agent_name,
            status=agent.status,
            last_activity=agent.last_activity.isoformat(),
            processed_count=int(agent.metrics.get("processed_count", 0)),
            error_count=int(agent.metrics.get("error_count", 0)),
            success_rate=float(agent.metrics.get("success_rate", 0)),
            config=agent.config
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent {agent_name} status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get agent {agent_name} status")

@router.put("/{agent_name}/config")
async def update_agent_config(agent_name: str, config_update: AgentConfigUpdate):
    """Update agent configuration"""
    try:
        # Get current agent state
        agent_states = await DatabaseManager.get_agent_states()
        agent = next((a for a in agent_states if a.agent_name == agent_name), None)
        
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found")
        
        # Update config
        updated_config = {**agent.config, **config_update.config}
        
        # Create updated agent state
        updated_agent = AgentState(
            agent_name=agent.agent_name,
            status=agent.status,
            last_activity=datetime.now(),
            metrics=agent.metrics,
            config=updated_config
        )
        
        await DatabaseManager.update_agent_state(updated_agent)
        
        return {
            "message": f"Agent {agent_name} configuration updated successfully",
            "updated_config": updated_config,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating agent {agent_name} config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update agent {agent_name} configuration")

@router.get("/{agent_name}/metrics")
async def get_agent_metrics(agent_name: str, limit: int = 100):
    """Get metrics for a specific agent"""
    try:
        metrics = await DatabaseManager.get_metrics(agent_name=agent_name, limit=limit)
        
        formatted_metrics = []
        for metric in metrics:
            formatted_metrics.append({
                "id": metric[0],
                "name": metric[1],
                "value": metric[2],
                "agent": metric[3],
                "timestamp": metric[4]
            })
        
        return formatted_metrics
        
    except Exception as e:
        logger.error(f"Error getting metrics for agent {agent_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics for agent {agent_name}")

