# crain_interface/core/config.py
import os
from pydantic import BaseModel
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseModel):
    # App Configuration
    app_name: str = "Crain AI Interface"
    app_env: str = os.getenv("APP_ENV", "dev")
    host: str = os.getenv("INTERFACE_HOST", "0.0.0.0")
    port: int = int(os.getenv("INTERFACE_PORT", "8081"))
    
    # Crain System Configuration
    crain_api_url: str = os.getenv("CRAIN_API_URL", "http://localhost:7788")
    
    # Azure OpenAI Configuration (same as main system)
    azure_openai_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_openai_api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    azure_openai_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    azure_openai_deployment: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
    
    # LLM Configuration
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "2000"))
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

settings = Settings()
