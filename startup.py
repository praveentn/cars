# startup.py
"""
Startup script for Cognitive Architecture Orchestrator
"""
import asyncio
import sys
import os
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.config import settings
from core.database import init_db
from core.logger import setup_logging

async def initialize_system():
    """Initialize the cognitive architecture system"""
    print("🧠 Initializing Cognitive Architecture Orchestrator...")
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Check environment variables
    if not settings.azure_openai_api_key:
        logger.error("❌ Azure OpenAI API key not configured. Please check your .env file.")
        return False
    
    # Initialize database
    try:
        await init_db()
        logger.info("✅ Database initialized successfully")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        return False
    
    # Check static directories
    static_dir = project_root / "static"
    templates_dir = project_root / "templates"
    
    if not static_dir.exists():
        static_dir.mkdir(exist_ok=True)
        logger.info("📁 Created static directory")
    
    if not templates_dir.exists():
        logger.error("❌ Templates directory not found")
        return False
    
    logger.info("🚀 System initialization completed successfully")
    return True

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'fastapi', 'uvicorn', 'pydantic', 'aiosqlite', 
        'openai', 'jinja2', 'psutil', 'numpy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed")
    return True

def check_env_file():
    """Check if .env file exists and has required variables"""
    env_file = project_root / ".env"
    
    if not env_file.exists():
        print("❌ .env file not found")
        print("Please copy .env.example to .env and configure your settings")
        return False
    
    # Check for required environment variables
    required_vars = [
        'AZURE_OPENAI_ENDPOINT',
        'AZURE_OPENAI_API_KEY',
        'AZURE_OPENAI_DEPLOYMENT'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please check your .env file configuration")
        return False
    
    print("✅ Environment configuration is valid")
    return True

async def main():
    """Main startup function"""
    print("=" * 60)
    print("🧠 Cognitive Architecture Orchestrator")
    print("=" * 60)
    
    # Pre-flight checks
    if not check_dependencies():
        sys.exit(1)
    
    if not check_env_file():
        sys.exit(1)
    
    # Initialize system
    if not await initialize_system():
        sys.exit(1)
    
    print("\n🎉 System ready to start!")
    print(f"🌐 Application will be available at: http://{settings.host}:{settings.port}")
    print(f"📊 Dashboard: http://{settings.host}:{settings.port}/")
    print(f"🤖 Agent Management: http://{settings.host}:{settings.port}/agents")
    print(f"📈 Monitoring: http://{settings.host}:{settings.port}/monitoring")
    print(f"🧠 Knowledge: http://{settings.host}:{settings.port}/knowledge")
    print("\nStarting FastAPI server...")

if __name__ == "__main__":
    # Run startup checks
    asyncio.run(main())
    
    # Start the FastAPI application
    import uvicorn
    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level=settings.log_level.lower()
    )