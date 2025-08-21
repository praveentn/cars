# core/database.py
import asyncio
import aiosqlite
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
import json
import logging

logger = logging.getLogger(__name__)

class Event(BaseModel):
    id: str
    timestamp: datetime
    source: str
    modality: str
    payload: Dict[str, Any]
    tags: List[str] = []
    status: str = "pending"

class AgentState(BaseModel):
    agent_name: str
    status: str
    last_activity: datetime
    metrics: Dict[str, float] = {}
    config: Dict[str, Any] = {}

class MemoryItem(BaseModel):
    id: str
    scope: str  # session, user, global
    data: Dict[str, Any]
    expires_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

class KnowledgeChunk(BaseModel):
    id: str
    text: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = {}
    source: str
    created_at: datetime
    relevance_score: float = 0.0

class TaskPlan(BaseModel):
    id: str
    goal: str
    steps: List[Dict[str, Any]]
    status: str
    created_at: datetime
    updated_at: datetime
    agent_id: str

DATABASE_FILE = "cognitive_architecture.db"

async def init_db():
    """Initialize the database with required tables"""
    async with aiosqlite.connect(DATABASE_FILE) as db:
        # Events table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                source TEXT NOT NULL,
                modality TEXT NOT NULL,
                payload TEXT NOT NULL,
                tags TEXT,
                status TEXT DEFAULT 'pending',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Agent states table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS agent_states (
                agent_name TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                last_activity TEXT NOT NULL,
                metrics TEXT,
                config TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Memory items table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS memory_items (
                id TEXT PRIMARY KEY,
                scope TEXT NOT NULL,
                data TEXT NOT NULL,
                expires_at TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Knowledge chunks table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_chunks (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                embedding TEXT,
                metadata TEXT,
                source TEXT NOT NULL,
                relevance_score REAL DEFAULT 0.0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Task plans table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS task_plans (
                id TEXT PRIMARY KEY,
                goal TEXT NOT NULL,
                steps TEXT NOT NULL,
                status TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # System metrics table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                agent_name TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        await db.commit()
        logger.info("Database initialized successfully")

async def get_db():
    """Get database connection"""
    return aiosqlite.connect(DATABASE_FILE)

class DatabaseManager:
    """Database operations manager"""
    
    @staticmethod
    async def save_event(event: Event):
        """Save an event to the database"""
        async with aiosqlite.connect(DATABASE_FILE) as db:
            await db.execute("""
                INSERT INTO events (id, timestamp, source, modality, payload, tags, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                event.id,
                event.timestamp.isoformat(),
                event.source,
                event.modality,
                json.dumps(event.payload),
                json.dumps(event.tags),
                event.status
            ))
            await db.commit()
    
    @staticmethod
    async def get_events(limit: int = 100, status: Optional[str] = None) -> List[Event]:
        """Retrieve events from database"""
        async with aiosqlite.connect(DATABASE_FILE) as db:
            if status:
                cursor = await db.execute("""
                    SELECT * FROM events WHERE status = ? 
                    ORDER BY timestamp DESC LIMIT ?
                """, (status, limit))
            else:
                cursor = await db.execute("""
                    SELECT * FROM events ORDER BY timestamp DESC LIMIT ?
                """, (limit,))
            
            rows = await cursor.fetchall()
            return [
                Event(
                    id=row[0],
                    timestamp=datetime.fromisoformat(row[1]),
                    source=row[2],
                    modality=row[3],
                    payload=json.loads(row[4]),
                    tags=json.loads(row[5]) if row[5] else [],
                    status=row[6]
                )
                for row in rows
            ]
    
    @staticmethod
    async def update_agent_state(agent_state: AgentState):
        """Update agent state in database"""
        async with aiosqlite.connect(DATABASE_FILE) as db:
            await db.execute("""
                INSERT OR REPLACE INTO agent_states 
                (agent_name, status, last_activity, metrics, config, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                agent_state.agent_name,
                agent_state.status,
                agent_state.last_activity.isoformat(),
                json.dumps(agent_state.metrics),
                json.dumps(agent_state.config),
                datetime.now().isoformat()
            ))
            await db.commit()
    
    @staticmethod
    async def get_agent_states() -> List[AgentState]:
        """Get all agent states"""
        async with aiosqlite.connect(DATABASE_FILE) as db:
            cursor = await db.execute("SELECT * FROM agent_states")
            rows = await cursor.fetchall()
            return [
                AgentState(
                    agent_name=row[0],
                    status=row[1],
                    last_activity=datetime.fromisoformat(row[2]),
                    metrics=json.loads(row[3]) if row[3] else {},
                    config=json.loads(row[4]) if row[4] else {}
                )
                for row in rows
            ]
    
    @staticmethod
    async def save_memory_item(memory_item: MemoryItem):
        """Save memory item to database"""
        async with aiosqlite.connect(DATABASE_FILE) as db:
            await db.execute("""
                INSERT OR REPLACE INTO memory_items 
                (id, scope, data, expires_at, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                memory_item.id,
                memory_item.scope,
                json.dumps(memory_item.data),
                memory_item.expires_at.isoformat() if memory_item.expires_at else None,
                memory_item.created_at.isoformat(),
                memory_item.updated_at.isoformat()
            ))
            await db.commit()
    
    @staticmethod
    async def get_memory_items(scope: Optional[str] = None) -> List[MemoryItem]:
        """Get memory items by scope"""
        async with aiosqlite.connect(DATABASE_FILE) as db:
            if scope:
                cursor = await db.execute("""
                    SELECT * FROM memory_items WHERE scope = ? 
                    AND (expires_at IS NULL OR expires_at > ?)
                    ORDER BY updated_at DESC
                """, (scope, datetime.now().isoformat()))
            else:
                cursor = await db.execute("""
                    SELECT * FROM memory_items 
                    WHERE (expires_at IS NULL OR expires_at > ?)
                    ORDER BY updated_at DESC
                """, (datetime.now().isoformat(),))
            
            rows = await cursor.fetchall()
            return [
                MemoryItem(
                    id=row[0],
                    scope=row[1],
                    data=json.loads(row[2]),
                    expires_at=datetime.fromisoformat(row[3]) if row[3] else None,
                    created_at=datetime.fromisoformat(row[4]),
                    updated_at=datetime.fromisoformat(row[5])
                )
                for row in rows
            ]
    
    @staticmethod
    async def save_metric(metric_name: str, metric_value: float, agent_name: Optional[str] = None):
        """Save system metric"""
        async with aiosqlite.connect(DATABASE_FILE) as db:
            await db.execute("""
                INSERT INTO system_metrics (metric_name, metric_value, agent_name)
                VALUES (?, ?, ?)
            """, (metric_name, round(metric_value, 3), agent_name))
            await db.commit()
    
    @staticmethod
    async def get_metrics(metric_name: Optional[str] = None, agent_name: Optional[str] = None, limit: int = 100):
        """Get system metrics"""
        async with aiosqlite.connect(DATABASE_FILE) as db:
            query = "SELECT * FROM system_metrics WHERE 1=1"
            params = []
            
            if metric_name:
                query += " AND metric_name = ?"
                params.append(metric_name)
            
            if agent_name:
                query += " AND agent_name = ?"
                params.append(agent_name)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor = await db.execute(query, params)
            return await cursor.fetchall()
