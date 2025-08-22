# crain_interface/api/chat.py
import asyncio
import httpx
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging
import json

from core.config import settings
from core.llm_client import llm_client

logger = logging.getLogger(__name__)
router = APIRouter()

class ChatMessage(BaseModel):
    message: str
    context: Optional[List[Dict[str, Any]]] = None  # Fixed: List instead of Dict

class ChatResponse(BaseModel):
    response: str
    data: Optional[Dict[str, Any]] = None
    timestamp: str

@router.post("/message", response_model=ChatResponse)
async def process_message(chat_message: ChatMessage):
    """Process natural language message and interact with Crain system"""
    try:
        user_message = chat_message.message.strip()
        logger.info(f"Processing message: {user_message[:50]}...")
        
        # Analyze user intent
        intent_analysis = await llm_client.analyze_intent(user_message)
        logger.info(f"Intent analysis: {intent_analysis}")
        
        # Route to appropriate handler based on intent
        if intent_analysis["intent"] == "query_status":
            response = await handle_status_query(intent_analysis, user_message)
        elif intent_analysis["intent"] == "get_metrics":
            response = await handle_metrics_query(intent_analysis, user_message)
        elif intent_analysis["intent"] == "search_knowledge":
            response = await handle_knowledge_search(intent_analysis, user_message)
        elif intent_analysis["intent"] == "teach_concept":
            response = await handle_teaching(intent_analysis, user_message)
        elif intent_analysis["intent"] == "explain_system":
            response = await handle_system_explanation(intent_analysis, user_message)
        else:
            response = await handle_general_query(user_message)
        
        return ChatResponse(
            response=response["text"],
            data=response.get("data"),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        return ChatResponse(
            response="I'm having trouble processing your request right now. Please try again.",
            data=None,
            timestamp=datetime.now().isoformat()
        )

async def handle_status_query(intent: Dict[str, Any], user_message: str) -> Dict[str, Any]:
    """Handle queries about agent status"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            logger.info(f"Connecting to: {settings.crain_api_url}")
            
            # Get agent status
            agents_response = await client.get(f"{settings.crain_api_url}/api/agents/status")
            agents_data = agents_response.json()
            
            # Get system status  
            system_response = await client.get(f"{settings.crain_api_url}/api/orchestrator/status")
            system_data = system_response.json()
            
            # Generate natural language response
            response_text = await generate_status_response(agents_data, system_data, user_message)
            
            return {
                "text": response_text,
                "data": {
                    "agents": agents_data,
                    "system": system_data
                }
            }
            
    except httpx.ConnectError as e:
        logger.error(f"Connection error to Crain system: {e}")
        return {
            "text": f"I can't connect to the Crain system at {settings.crain_api_url}. Please make sure it's running and accessible.",
            "data": None
        }
    except Exception as e:
        logger.error(f"Error handling status query: {e}")
        return {
            "text": "I'm having trouble accessing the system status right now. Please try again later.",
            "data": None
        }

async def handle_general_query(user_message: str) -> Dict[str, Any]:
    """Handle general queries with LLM"""
    messages = [
        {
            "role": "system",
            "content": """You are an AI assistant for the Crain Cognitive Architecture system. 
            You help users understand and interact with this advanced AI system.
            
            Be helpful, informative, and suggest specific things users can ask about:
            - Agent status and performance
            - System metrics and health
            - Knowledge and concepts learned
            - Teaching new information
            - How different components work
            
            Keep responses concise but informative."""
        },
        {
            "role": "user", 
            "content": user_message
        }
    ]
    
    try:
        response = await llm_client.chat_completion(messages)
        return {
            "text": response["content"],
            "data": None
        }
    except Exception as e:
        logger.error(f"Error in general query: {e}")
        return {
            "text": "Hello! I'm here to help you interact with the Crain system. You can ask me about agent status, system performance, knowledge base, or how different components work. What would you like to know?",
            "data": None
        }

# Add the remaining handler functions here...
async def handle_metrics_query(intent: Dict[str, Any], user_message: str) -> Dict[str, Any]:
    """Handle queries about metrics and performance"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            metrics_response = await client.get(f"{settings.crain_api_url}/api/monitoring/metrics")
            metrics_data = metrics_response.json()
            
            health_response = await client.get(f"{settings.crain_api_url}/api/monitoring/health-summary")
            health_data = health_response.json()
            
            response_text = await generate_metrics_response(metrics_data, health_data, user_message)
            
            return {
                "text": response_text,
                "data": {
                    "metrics": metrics_data,
                    "health": health_data
                }
            }
            
    except Exception as e:
        logger.error(f"Error handling metrics query: {e}")
        return {
            "text": "I couldn't retrieve the performance metrics. The monitoring system might be unavailable.",
            "data": None
        }

async def handle_knowledge_search(intent: Dict[str, Any], user_message: str) -> Dict[str, Any]:
    """Handle knowledge search queries"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Get concept graph
            graph_response = await client.get(f"{settings.crain_api_url}/api/concept-graph/graph")
            graph_data = graph_response.json()
            
            # Get learned items
            learned_response = await client.get(f"{settings.crain_api_url}/api/concept-graph/learned-items")
            learned_data = learned_response.json()
            
            response_text = await generate_knowledge_response(graph_data, learned_data, user_message)
            
            return {
                "text": response_text,
                "data": {
                    "graph": graph_data,
                    "learned_items": learned_data
                }
            }
            
    except Exception as e:
        logger.error(f"Error handling knowledge search: {e}")
        return {
            "text": "I'm having trouble accessing the knowledge base. Please try again later.",
            "data": None
        }

async def handle_teaching(intent: Dict[str, Any], user_message: str) -> Dict[str, Any]:
    """Handle teaching new concepts to the system"""
    try:
        response_text = f"Thank you for teaching me! I've noted this information: '{user_message}'. The system's Self-Agent will process this new information and update the concept graph accordingly."
        
        return {
            "text": response_text,
            "data": {
                "concept_added": True,
                "message": user_message
            }
        }
        
    except Exception as e:
        logger.error(f"Error handling teaching: {e}")
        return {
            "text": "I encountered an issue while trying to learn that concept. Please try rephrasing or try again later.",
            "data": None
        }

async def handle_system_explanation(intent: Dict[str, Any], user_message: str) -> Dict[str, Any]:
    """Handle requests for system explanations"""
    component = intent.get("target", "system")
    
    explanations = {
        "receiver": "The Receiver is like the system's sensory input center. It takes in all external information - text, events, webhooks - normalizes them into a standard format, and routes them to the appropriate agents for processing.",
        "self_agent": "The Self-Agent is the curious learner of the system. It actively explores new information, asks clarifying questions, and decides what knowledge should be stored long-term. Think of it as the system's internal drive for learning.",
        "conscious_agent": "The Conscious Agent handles deliberate, step-by-step reasoning. When complex problems need careful analysis or planning, this component breaks them down methodically and provides transparent reasoning.",
        "unconscious_agent": "The Unconscious Agent works in the background, handling routine tasks and quick responses. It processes patterns, updates memory, and can give fast reflex responses when appropriate.",
        "relationship_manager": "This component maintains context about users and conversations. It tracks communication style, preferences, and builds rapport over time to provide more personalized interactions.",
        "memory_cache": "The Memory Cache is the system's short-term working memory. It stores recent information, active thoughts, and bridges the gap between different agents during processing.",
        "retriever": "The Retriever manages the long-term knowledge base. It stores documents, finds relevant information using hybrid search, and provides knowledge context to other components.",
        "flywheel": "The Flywheel is the meta-learning system. It continuously improves the overall system by running experiments, updating policies, and evolving the concept graph based on what works best."
    }
    
    if component in explanations:
        response_text = explanations[component]
    else:
        response_text = "The Crain Cognitive Architecture is an AI system inspired by neuroscience, with 8 interconnected components that work together to process information, learn, and respond intelligently. Each component has a specialized role, from input processing to meta-learning."
    
    return {
        "text": response_text,
        "data": {"component": component}
    }

async def generate_status_response(agents_data: List[Dict], system_data: Dict, user_message: str) -> str:
    """Generate natural language response for status queries"""
    active_agents = len([a for a in agents_data if a.get("status") == "active"])
    total_agents = len(agents_data)
    
    status_summary = f"The system currently has {active_agents} out of {total_agents} agents running. "
    
    if active_agents == total_agents:
        status_summary += "All agents are active and functioning well! "
    elif active_agents == 0:
        status_summary += "All agents appear to be stopped. "
    else:
        status_summary += f"{total_agents - active_agents} agents are currently stopped. "
    
    status_summary += f"Overall system uptime is {system_data.get('uptime', 'unknown')} with {system_data.get('memory_usage', 'unknown')} memory usage."
    
    return status_summary

async def generate_metrics_response(metrics_data: Dict, health_data: Dict, user_message: str) -> str:
    """Generate natural language response for metrics queries"""
    health_status = health_data.get("overall_health", "unknown")
    health_score = health_data.get("health_score", 0)
    
    return f"The system is currently in {health_status} condition with a health score of {health_score}/100."

async def generate_knowledge_response(graph_data: Dict, learned_data: Dict, user_message: str) -> str:
    """Generate natural language response for knowledge queries"""
    metadata = graph_data.get("metadata", {})
    concepts = metadata.get("total_concepts", 0)
    skills = metadata.get("total_skills", 0)
    connections = metadata.get("total_connections", 0)
    
    return f"The knowledge base contains {concepts} concepts and {skills} skills, connected by {connections} relationships."

@router.get("/suggestions")
async def get_suggestions():
    """Get suggested questions for users"""
    suggestions = [
        "What's the current status of all agents?",
        "How is the system performing today?",
        "What concepts has the system learned recently?", 
        "Show me the knowledge graph",
        "How does the Self-Agent work?",
        "What's the system health score?",
        "Tell me about the Flywheel component",
        "How many events have been processed?",
        "What are the latest learned items?",
        "Explain the Conscious Agent"
    ]
    
    return {"suggestions": suggestions}