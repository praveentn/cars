# core/llm_client.py
import asyncio
from openai import AsyncAzureOpenAI
from typing import List, Dict, Any, Optional
import logging
from .config import settings

logger = logging.getLogger(__name__)

class LLMClient:
    """Azure OpenAI client wrapper"""
    
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
        """Generate chat completion"""
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
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate text embedding (placeholder for embedding model)"""
        # Note: Azure OpenAI embedding would require different endpoint/model
        # For now, return a placeholder embedding
        import hashlib
        import struct
        
        # Simple hash-based embedding for demonstration
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to float vector
        embedding = []
        for i in range(0, len(hash_bytes), 4):
            chunk = hash_bytes[i:i+4]
            if len(chunk) == 4:
                value = struct.unpack('f', chunk)[0]
                embedding.append(float(value))
        
        # Normalize to 512 dimensions
        while len(embedding) < 512:
            embedding.extend(embedding[:min(512-len(embedding), len(embedding))])
        
        return embedding[:512]
    
    async def analyze_intent(self, text: str) -> Dict[str, Any]:
        """Analyze user intent from text"""
        messages = [
            {
                "role": "system",
                "content": """You are an intent analysis expert. Analyze the user's input and return a JSON response with:
                {
                    "intent": "primary intent category",
                    "confidence": 0.95,
                    "entities": ["extracted", "entities"],
                    "sentiment": "positive/negative/neutral",
                    "urgency": "low/medium/high"
                }"""
            },
            {
                "role": "user",
                "content": text
            }
        ]
        
        try:
            response = await self.chat_completion(messages, temperature=0.1)
            import json
            return json.loads(response["content"])
        except Exception as e:
            logger.error(f"Intent analysis failed: {e}")
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "entities": [],
                "sentiment": "neutral",
                "urgency": "low"
            }

# Global LLM client instance
llm_client = LLMClient()