# agentic/core/config.py
import os
from pydantic import BaseModel
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseModel):
    # App Configuration
    app_name: str = "Agentic Framework"
    app_env: str = os.getenv("APP_ENV", "dev")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "7788"))
    
    # Azure OpenAI Configuration
    azure_openai_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_openai_api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    azure_openai_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    azure_openai_deployment: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
    azure_openai_model: str = os.getenv("AZURE_OPENAI_MODEL", "gpt-4")
    
    # LLM Configuration
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "4000"))
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    llm_context_limit: int = int(os.getenv("LLM_CONTEXT_LIMIT", "20000"))
    
    # Database
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///agentic.db")
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_format: str = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    def get_azure_openai_config(self) -> Dict[str, Any]:
        return {
            "endpoint": self.azure_openai_endpoint,
            "api_key": self.azure_openai_api_key,
            "api_version": self.azure_openai_api_version,
            "deployment": self.azure_openai_deployment,
            "model": self.azure_openai_model,
            "max_tokens": self.llm_max_tokens,
            "temperature": self.llm_temperature
        }

settings = Settings()