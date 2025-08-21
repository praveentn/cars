# app.py
import asyncio
import logging
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
import uvicorn

from core.config import settings
from core.database import init_db, get_db
from core.logger import setup_logging
from components.receiver import ReceiverComponent
from components.self_agent import SelfAgentComponent
from components.conscious_agent import ConsciousAgentComponent
from components.unconscious_agent import UnconsciousAgentComponent
from components.relationship_manager import RelationshipManagerComponent
from components.memory_cache import MemoryCacheComponent
from components.retriever import RetrieverComponent
from components.flywheel import FlywheelComponent
from api.orchestrator import router as orchestrator_router
from api.agents import router as agents_router
from api.monitoring import router as monitoring_router
from api.concept_graph import router as concept_graph_router

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("Starting Cognitive Architecture Orchestrator...")
    
    # Initialize database
    await init_db()
    
    # Initialize all components
    app.state.receiver = ReceiverComponent()
    app.state.self_agent = SelfAgentComponent()
    app.state.conscious_agent = ConsciousAgentComponent()
    app.state.unconscious_agent = UnconsciousAgentComponent()
    app.state.relationship_manager = RelationshipManagerComponent()
    app.state.memory_cache = MemoryCacheComponent()
    app.state.retriever = RetrieverComponent()
    app.state.flywheel = FlywheelComponent()
    
    # Start background services
    for component_name in ['receiver', 'self_agent', 'conscious_agent', 'unconscious_agent', 
                          'relationship_manager', 'memory_cache', 'retriever', 'flywheel']:
        component = getattr(app.state, component_name)
        if hasattr(component, 'start'):
            await component.start()
    
    logger.info("All components initialized successfully")
    yield
    
    # Cleanup
    logger.info("Shutting down components...")
    for component_name in ['receiver', 'self_agent', 'conscious_agent', 'unconscious_agent', 
                          'relationship_manager', 'memory_cache', 'retriever', 'flywheel']:
        component = getattr(app.state, component_name)
        if hasattr(component, 'stop'):
            await component.stop()

# Create FastAPI app
app = FastAPI(
    title="Cognitive Architecture Orchestrator",
    description="Enterprise-grade AI Agent Management System",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Include routers
app.include_router(orchestrator_router, prefix="/api/orchestrator", tags=["orchestrator"])
app.include_router(agents_router, prefix="/api/agents", tags=["agents"])
app.include_router(monitoring_router, prefix="/api/monitoring", tags=["monitoring"])
app.include_router(concept_graph_router, prefix="/api/concept-graph", tags=["concept-graph"])

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse("pages/dashboard.html", {
        "request": request,
        "title": "Cognitive Architecture Dashboard",
        "active_page": "dashboard"
    })

@app.get("/agents", response_class=HTMLResponse)
async def agents_page(request: Request):
    """Agents management page"""
    return templates.TemplateResponse("pages/agents.html", {
        "request": request,
        "title": "Agent Management",
        "active_page": "agents"
    })

@app.get("/monitoring", response_class=HTMLResponse)
async def monitoring_page(request: Request):
    """System monitoring page"""
    return templates.TemplateResponse("pages/monitoring.html", {
        "request": request,
        "title": "System Monitoring",
        "active_page": "monitoring"
    })

@app.get("/knowledge", response_class=HTMLResponse)
async def knowledge_page(request: Request):
    """Knowledge management page"""
    return templates.TemplateResponse("pages/knowledge.html", {
        "request": request,
        "title": "Knowledge Management",
        "active_page": "knowledge"
    })

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        reload=True if settings.app_env == "dev" else False,
        log_level=settings.log_level.lower()
    )