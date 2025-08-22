# crain_interface/core/llm_client.py
import asyncio
from openai import AsyncAzureOpenAI
from typing import List, Dict, Any, Optional
import logging
from .config import settings

logger = logging.getLogger(__name__)

class LLMClient:
    """Azure OpenAI client for natural language processing"""
    
    def __init__(self):
        self.client = AsyncAzureOpenAI(
            api_version=settings.azure_openai_api_version,
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
        )
        self.deployment = settings.azure_openai_deployment
    
    async def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate chat completion with Azure OpenAI"""
        try:
            response = await self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                temperature=temperature or settings.llm_temperature,
                max_tokens=max_tokens or settings.llm_max_tokens,
                **kwargs
            )
            
            return {
                "content": response.choices[0].message.content,
                "role": response.choices[0].message.role,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            raise
    
    async def analyze_intent(self, user_message: str) -> Dict[str, Any]:
        """Analyze user intent and extract parameters"""
        messages = [
            {
                "role": "system",
                "content": """You are an AI assistant that analyzes user queries about a cognitive architecture system.
                
                The system has these components:
                - Receiver: Processes incoming events
                - Self-Agent: Learning and exploration
                - Conscious Agent: Deliberative reasoning 
                - Unconscious Agent: Background processing
                - Relationship Manager: Context and social layer
                - Memory Cache: Short-term storage
                - Retriever: Knowledge retrieval
                - Flywheel: Meta-learning and evolution
                
                Available data types:
                - Agent status and metrics
                - System performance data
                - Concept graph and learned items
                - Event processing statistics
                - Knowledge base content
                
                Analyze the user's query and return JSON with:
                {
                    "intent": "query_status|teach_concept|search_knowledge|get_metrics|explain_system",
                    "target": "specific component or 'system'",
                    "parameters": {"key": "value"},
                    "confidence": 0.9
                }"""
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
        
        try:
            response = await self.chat_completion(messages, temperature=0.1)
            import json
            return json.loads(response["content"])
        except Exception as e:
            logger.error(f"Intent analysis failed: {e}")
            return {
                "intent": "general_query",
                "target": "system", 
                "parameters": {},
                "confidence": 0.5
            }

# Global client instance
llm_client = LLMClient()
