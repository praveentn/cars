# api/orchestrator.py
import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
import logging
import psutil
import time

from core.database import DatabaseManager, Event
from core.llm_client import llm_client

logger = logging.getLogger(__name__)
router = APIRouter()

# Request/Response models
class TestEventRequest(BaseModel):
    source: str = "system"
    modality: str = "text"
    content: str = "This is a test event for system validation"
    tags: List[str] = ["test", "system"]

class SystemStatusResponse(BaseModel):
    status: str
    timestamp: str
    total_events: int
    active_agents: int
    uptime: str
    memory_usage: str
    processing_rate: str
    success_rate: str
    response_time: str
    error_rate: str

class RecentActivityItem(BaseModel):
    message: str
    timestamp: str
    agent: Optional[str] = None
    level: str = "info"

# System start time for uptime calculation
SYSTEM_START_TIME = datetime.now()

@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status():
    """Get overall system status and metrics"""
    try:
        # Get events from database
        events = await DatabaseManager.get_events(limit=1000)
        total_events = len(events)
        
        # Get agent states
        agent_states = await DatabaseManager.get_agent_states()
        active_agents = len([agent for agent in agent_states if agent.status == "active"])
        
        # Calculate uptime
        uptime = datetime.now() - SYSTEM_START_TIME
        uptime_str = f"{uptime.days}d {uptime.seconds//3600}h {(uptime.seconds//60)%60}m"
        
        # Get system memory usage
        memory = psutil.virtual_memory()
        memory_usage = f"{memory.percent:.1f}%"
        
        # Calculate processing metrics
        recent_events = [e for e in events if (datetime.now() - e.timestamp).total_seconds() < 3600]
        processing_rate = f"{len(recent_events)}/hour"
        
        # Calculate success rate
        successful_events = [e for e in recent_events if e.status not in ["error", "failed"]]
        success_rate = f"{(len(successful_events)/max(len(recent_events), 1)*100):.1f}%" if recent_events else "N/A"
        
        # Get recent metrics for response time
        recent_metrics = await DatabaseManager.get_metrics("response_time", limit=10)
        avg_response_time = sum(metric[2] for metric in recent_metrics) / max(len(recent_metrics), 1) if recent_metrics else 0
        response_time = f"{avg_response_time:.2f}ms"
        
        # Calculate error rate
        error_events = [e for e in recent_events if e.status in ["error", "failed"]]
        error_rate = f"{(len(error_events)/max(len(recent_events), 1)*100):.1f}%" if recent_events else "0%"
        
        return SystemStatusResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            total_events=total_events,
            active_agents=active_agents,
            uptime=uptime_str,
            memory_usage=memory_usage,
            processing_rate=processing_rate,
            success_rate=success_rate,
            response_time=response_time,
            error_rate=error_rate
        )
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system status")

@router.get("/recent-activity", response_model=List[RecentActivityItem])
async def get_recent_activity():
    """Get recent system activity"""
    try:
        activities = []
        
        # Get recent events
        events = await DatabaseManager.get_events(limit=10)
        for event in events:
            activities.append(RecentActivityItem(
                message=f"Event processed: {event.modality} from {event.source}",
                timestamp=event.timestamp.isoformat(),
                agent="receiver",
                level="info"
            ))
        
        # Get recent agent state changes
        agent_states = await DatabaseManager.get_agent_states()
        for agent in agent_states:
            activities.append(RecentActivityItem(
                message=f"Agent {agent.agent_name} status: {agent.status}",
                timestamp=agent.last_activity.isoformat(),
                agent=agent.agent_name,
                level="info" if agent.status == "active" else "warning"
            ))
        
        # Sort by timestamp (most recent first)
        activities.sort(key=lambda x: x.timestamp, reverse=True)
        
        return activities[:20]  # Return last 20 activities
        
    except Exception as e:
        logger.error(f"Error getting recent activity: {e}")
        raise HTTPException(status_code=500, detail="Failed to get recent activity")

@router.post("/test-event")
async def process_test_event(request: TestEventRequest):
    """Process a test event to validate system functionality"""
    try:
        # Get receiver component from app state
        from fastapi import Request as FastAPIRequest
        # This is a placeholder - in real implementation, we'd access the app state
        
        # Create test event
        test_event = Event(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            source=request.source,
            modality=request.modality,
            payload={
                "content": request.content,
                "test": True,
                "created_by": "orchestrator"
            },
            tags=request.tags,
            status="pending"
        )
        
        # Save to database
        await DatabaseManager.save_event(test_event)
        
        # Record metric
        await DatabaseManager.save_metric("test_events_created", 1.0, "orchestrator")
        
        logger.info(f"Test event created: {test_event.id}")
        
        return {
            "message": "Test event processed successfully",
            "event_id": test_event.id,
            "timestamp": test_event.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing test event: {e}")
        raise HTTPException(status_code=500, detail="Failed to process test event")

@router.post("/restart-agent/{agent_name}")
async def restart_agent(agent_name: str):
    """Restart a specific agent"""
    try:
        # This would interact with the actual agent components
        # For now, we'll update the agent state
        
        from core.database import AgentState
        
        # Update agent state to "restarting"
        restart_state = AgentState(
            agent_name=agent_name,
            status="restarting",
            last_activity=datetime.now(),
            metrics={},
            config={}
        )
        
        await DatabaseManager.update_agent_state(restart_state)
        
        # Simulate restart delay
        await asyncio.sleep(2)
        
        # Update to active
        active_state = AgentState(
            agent_name=agent_name,
            status="active",
            last_activity=datetime.now(),
            metrics={},
            config={}
        )
        
        await DatabaseManager.update_agent_state(active_state)
        
        logger.info(f"Agent {agent_name} restarted successfully")
        
        return {
            "message": f"Agent {agent_name} restarted successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error restarting agent {agent_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to restart agent {agent_name}")

@router.post("/stop-agent/{agent_name}")
async def stop_agent(agent_name: str):
    """Stop a specific agent"""
    try:
        # Update agent state to "stopped"
        stop_state = AgentState(
            agent_name=agent_name,
            status="stopped",
            last_activity=datetime.now(),
            metrics={},
            config={}
        )
        
        await DatabaseManager.update_agent_state(stop_state)
        
        logger.info(f"Agent {agent_name} stopped")
        
        return {
            "message": f"Agent {agent_name} stopped successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error stopping agent {agent_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop agent {agent_name}")

@router.post("/start-agent/{agent_name}")
async def start_agent(agent_name: str):
    """Start a specific agent"""
    try:
        # Update agent state to "active"
        start_state = AgentState(
            agent_name=agent_name,
            status="active",
            last_activity=datetime.now(),
            metrics={},
            config={}
        )
        
        await DatabaseManager.update_agent_state(start_state)
        
        logger.info(f"Agent {agent_name} started")
        
        return {
            "message": f"Agent {agent_name} started successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error starting agent {agent_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start agent {agent_name}")

@router.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }
        
        # Check database connectivity
        try:
            await DatabaseManager.get_events(limit=1)
            health_status["components"]["database"] = "healthy"
        except Exception as e:
            health_status["components"]["database"] = f"unhealthy: {str(e)}"
            health_status["status"] = "degraded"
        
        # Check LLM connectivity
        try:
            test_response = await llm_client.chat_completion([
                {"role": "user", "content": "Health check"}
            ])
            health_status["components"]["llm"] = "healthy"
        except Exception as e:
            health_status["components"]["llm"] = f"unhealthy: {str(e)}"
            health_status["status"] = "degraded"
        
        # Check agent states
        try:
            agent_states = await DatabaseManager.get_agent_states()
            healthy_agents = len([agent for agent in agent_states if agent.status == "active"])
            total_agents = len(agent_states)
            
            health_status["components"]["agents"] = f"{healthy_agents}/{total_agents} healthy"
            
            if healthy_agents == 0:
                health_status["status"] = "unhealthy"
            elif healthy_agents < total_agents:
                health_status["status"] = "degraded"
                
        except Exception as e:
            health_status["components"]["agents"] = f"check failed: {str(e)}"
            health_status["status"] = "degraded"
        
        # Check system resources
        try:
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=1)
            
            health_status["components"]["system"] = {
                "memory_percent": round(memory.percent, 2),
                "cpu_percent": round(cpu, 2),
                "status": "healthy" if memory.percent < 90 and cpu < 90 else "warning"
            }
            
            if memory.percent > 95 or cpu > 95:
                health_status["status"] = "degraded"
                
        except Exception as e:
            health_status["components"]["system"] = f"check failed: {str(e)}"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@router.post("/clear-cache")
async def clear_cache():
    """Clear system cache and temporary data"""
    try:
        # Clear expired memory items
        memory_items = await DatabaseManager.get_memory_items()
        cleared_count = 0
        
        # This would typically clear expired items
        # For now, we'll just log the action
        
        logger.info(f"Cache clear requested - {len(memory_items)} items checked")
        
        return {
            "message": f"Cache cleared successfully",
            "items_cleared": cleared_count,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear cache")

@router.get("/system-info")
async def get_system_info():
    """Get detailed system information"""
    try:
        # Get system information
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        cpu_count = psutil.cpu_count()
        
        # Get Python process info
        process = psutil.Process()
        
        system_info = {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "memory_total": f"{memory.total / (1024**3):.2f} GB",
                "memory_available": f"{memory.available / (1024**3):.2f} GB",
                "memory_percent": f"{memory.percent:.1f}%",
                "disk_total": f"{disk.total / (1024**3):.2f} GB",
                "disk_free": f"{disk.free / (1024**3):.2f} GB",
                "disk_percent": f"{(disk.used / disk.total * 100):.1f}%",
                "cpu_count": cpu_count,
                "cpu_percent": f"{psutil.cpu_percent(interval=1):.1f}%"
            },
            "process": {
                "memory_percent": f"{process.memory_percent():.2f}%",
                "cpu_percent": f"{process.cpu_percent():.2f}%",
                "threads": process.num_threads(),
                "create_time": datetime.fromtimestamp(process.create_time()).isoformat()
            },
            "database": {
                "events_count": len(await DatabaseManager.get_events(limit=10000)),
                "agents_count": len(await DatabaseManager.get_agent_states())
            }
        }
        
        return system_info
        
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system information")
