# crain_interface/app.py - Main FastAPI Application
import asyncio
import logging
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

from core.config import settings
from core.logger import setup_logging
from api.chat import router as chat_router

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Crain Natural Language Interface",
    description="Natural Language Interface for Cognitive Architecture System",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Include routers
app.include_router(chat_router, prefix="/api/chat", tags=["chat"])

@app.get("/", response_class=HTMLResponse)
async def main_interface(request: Request):
    """Main chat interface"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": "Crain AI Assistant"
    })

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "crain-interface"
    }

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        reload=False
    )
