# api/monitoring.py
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import JSONResponse
import logging
import json

from core.database import DatabaseManager

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/metrics")
async def get_system_metrics():
    """Get system performance metrics for charts"""
    try:
        # Get metrics from the last 24 hours
        all_metrics = await DatabaseManager.get_metrics(limit=1000)
        
        # Process metrics for chart display
        timestamps = []
        events_processed = []
        success_rates = []
        
        # Group metrics by time intervals (hourly)
        now = datetime.now()
        for i in range(24):
            hour_start = now - timedelta(hours=i+1)
            hour_end = now - timedelta(hours=i)
            
            hour_label = hour_start.strftime("%H:%M")
            timestamps.insert(0, hour_label)
            
            # Count events processed in this hour
            hour_events = len([m for m in all_metrics 
                             if m[1] == "events_processed" and 
                             hour_start <= datetime.fromisoformat(m[4]) < hour_end])
            events_processed.insert(0, hour_events)
            
            # Calculate success rate for this hour
            success_metrics = [m for m in all_metrics 
                             if m[1] == "success_rate" and 
                             hour_start <= datetime.fromisoformat(m[4]) < hour_end]
            avg_success = sum(m[2] for m in success_metrics) / max(len(success_metrics), 1) if success_metrics else 0
            success_rates.insert(0, round(avg_success, 2))
        
        return {
            "timestamps": timestamps,
            "events_processed": events_processed,
            "success_rates": success_rates
        }
        
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system metrics")

@router.get("/logs")
async def get_system_logs(limit: int = 100):
    """Get recent system logs"""
    try:
        # Get recent events as logs
        events = await DatabaseManager.get_events(limit=limit)
        
        logs = []
        for event in events:
            logs.append({
                "timestamp": event.timestamp.isoformat(),
                "level": "INFO" if event.status == "processed" else "WARNING",
                "message": f"Event {event.id}: {event.modality} from {event.source}",
                "details": {
                    "event_id": event.id,
                    "status": event.status,
                    "tags": event.tags
                }
            })
        
        return logs
        
    except Exception as e:
        logger.error(f"Error getting system logs: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system logs")

@router.get("/export-logs")
async def export_logs():
    """Export system logs as JSON file"""
    try:
        # Get comprehensive log data
        events = await DatabaseManager.get_events(limit=1000)
        agent_states = await DatabaseManager.get_agent_states()
        metrics = await DatabaseManager.get_metrics(limit=1000)
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "events": [
                {
                    "id": event.id,
                    "timestamp": event.timestamp.isoformat(),
                    "source": event.source,
                    "modality": event.modality,
                    "status": event.status,
                    "tags": event.tags
                }
                for event in events
            ],
            "agent_states": [
                {
                    "name": agent.agent_name,
                    "status": agent.status,
                    "last_activity": agent.last_activity.isoformat(),
                    "metrics": agent.metrics
                }
                for agent in agent_states
            ],
            "metrics": [
                {
                    "name": metric[1],
                    "value": metric[2],
                    "agent": metric[3],
                    "timestamp": metric[4]
                }
                for metric in metrics
            ]
        }
        
        # Create response with file download
        response = Response(
            content=json.dumps(export_data, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=system_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"}
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error exporting logs: {e}")
        raise HTTPException(status_code=500, detail="Failed to export logs")

@router.get("/health-summary")
async def get_health_summary():
    """Get summarized health information"""
    try:
        # Get agent health
        agent_states = await DatabaseManager.get_agent_states()
        agent_health = {
            "total": len(agent_states),
            "active": len([a for a in agent_states if a.status == "active"]),
            "idle": len([a for a in agent_states if a.status == "idle"]),
            "error": len([a for a in agent_states if a.status == "error"]),
            "stopped": len([a for a in agent_states if a.status == "stopped"])
        }
        
        # Get recent events health
        recent_events = await DatabaseManager.get_events(limit=100)
        event_health = {
            "total": len(recent_events),
            "processed": len([e for e in recent_events if e.status == "processed"]),
            "pending": len([e for e in recent_events if e.status == "pending"]),
            "error": len([e for e in recent_events if e.status == "error"])
        }
        
        # Calculate overall health score
        agent_score = (agent_health["active"] / max(agent_health["total"], 1)) * 100
        event_score = ((event_health["processed"] + event_health["pending"]) / max(event_health["total"], 1)) * 100
        overall_score = (agent_score + event_score) / 2
        
        # Determine health status
        if overall_score >= 80:
            health_status = "healthy"
        elif overall_score >= 60:
            health_status = "warning"
        else:
            health_status = "critical"
        
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_health": health_status,
            "health_score": round(overall_score, 2),
            "agent_health": agent_health,
            "event_health": event_health,
            "recommendations": get_health_recommendations(agent_health, event_health)
        }
        
    except Exception as e:
        logger.error(f"Error getting health summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to get health summary")

def get_health_recommendations(agent_health: Dict, event_health: Dict) -> List[str]:
    """Generate health recommendations based on current state"""
    recommendations = []
    
    if agent_health["error"] > 0:
        recommendations.append(f"⚠️ {agent_health['error']} agent(s) in error state - check logs")
    
    if agent_health["stopped"] > agent_health["active"]:
        recommendations.append("🔄 More agents stopped than active - consider restarting services")
    
    if event_health["error"] > event_health["processed"] * 0.1:
        recommendations.append("❌ High event error rate - review event processing logic")
    
    if event_health["pending"] > event_health["processed"]:
        recommendations.append("⏳ High number of pending events - may need to scale processing")
    
    if not recommendations:
        recommendations.append("✅ System operating normally")
    
    return recommendations

@router.post("/test-alert")
async def create_test_alert():
    """Create a test alert for monitoring validation"""
    try:
        # Save test metric
        await DatabaseManager.save_metric("test_alert", 1.0, "monitoring")
        
        return {
            "message": "Test alert created successfully",
            "timestamp": datetime.now().isoformat(),
            "alert_type": "test"
        }
        
    except Exception as e:
        logger.error(f"Error creating test alert: {e}")
        raise HTTPException(status_code=500, detail="Failed to create test alert")
