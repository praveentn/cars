# components/memory_cache.py
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
import logging
import json
import hashlib

from core.database import DatabaseManager, AgentState, MemoryItem

logger = logging.getLogger(__name__)

class MemoryCacheComponent:
    """
    Memory Cache Service (Short-Term Store)
    - Fast-access working memory
    - Stores short-term facts relevant to current tasks
    - Keeps recent actions, mistakes, and quick learnings
    - Bridges conscious/unconscious agent activity
    """
    
    def __init__(self):
        self.name = "memory_cache"
        self.status = "idle"
        self.cache = {}  # In-memory cache
        self.access_count = 0
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        self.max_cache_size = 1000
        self.default_ttl = 3600  # 1 hour in seconds
        self.pinned_keys = set()
        
    async def start(self):
        """Start the memory cache component"""
        self.status = "active"
        await self._update_state()
        logger.info("Memory Cache component started")
        
        # Start background maintenance
        asyncio.create_task(self._cache_maintenance_loop())
        
        # Load existing memory items from database
        await self._load_from_database()
    
    async def stop(self):
        """Stop the memory cache component"""
        self.status = "stopped"
        
        # Save all cache items to database before stopping
        await self._save_to_database()
        
        await self._update_state()
        logger.info("Memory Cache component stopped")
    
    async def put(self, scope: str, key: str, data: Any, ttl: Optional[int] = None, pin: bool = False) -> bool:
        """
        Store data in cache
        
        Args:
            scope: Scope of the memory (session, user, task, global)
            key: Unique key for the data
            data: Data to store
            ttl: Time to live in seconds (None for default)
            pin: Whether to pin this item (prevent eviction)
            
        Returns:
            Success status
        """
        try:
            full_key = f"{scope}:{key}"
            
            # Check cache size and evict if necessary
            if len(self.cache) >= self.max_cache_size and full_key not in self.cache:
                await self._evict_lru()
            
            expires_at = None
            if ttl is not None:
                expires_at = datetime.now() + timedelta(seconds=ttl)
            elif ttl != 0:  # ttl=0 means no expiration
                expires_at = datetime.now() + timedelta(seconds=self.default_ttl)
            
            cache_item = {
                "scope": scope,
                "key": key,
                "data": data,
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
                "last_accessed": datetime.now(),
                "access_count": 1,
                "expires_at": expires_at,
                "pinned": pin,
                "size": len(json.dumps(data, default=str)) if isinstance(data, (dict, list)) else len(str(data))
            }
            
            self.cache[full_key] = cache_item
            
            if pin:
                self.pinned_keys.add(full_key)
            
            await self._update_state()
            logger.debug(f"Stored item in cache: {full_key}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing item in cache: {e}")
            return False
    
    async def get(self, scope: str, key: str) -> Optional[Any]:
        """
        Retrieve data from cache
        
        Args:
            scope: Scope of the memory
            key: Key of the data
            
        Returns:
            Cached data or None if not found/expired
        """
        try:
            full_key = f"{scope}:{key}"
            self.access_count += 1
            
            if full_key not in self.cache:
                self.miss_count += 1
                
                # Try loading from database
                db_item = await self._load_from_database_key(full_key)
                if db_item:
                    self.hit_count += 1
                    await self._update_state()
                    return db_item
                
                await self._update_state()
                return None
            
            item = self.cache[full_key]
            
            # Check if expired
            if item["expires_at"] and datetime.now() > item["expires_at"]:
                del self.cache[full_key]
                self.pinned_keys.discard(full_key)
                self.miss_count += 1
                await self._update_state()
                return None
            
            # Update access information
            item["last_accessed"] = datetime.now()
            item["access_count"] += 1
            
            self.hit_count += 1
            await self._update_state()
            
            logger.debug(f"Retrieved item from cache: {full_key}")
            return item["data"]
            
        except Exception as e:
            logger.error(f"Error retrieving item from cache: {e}")
            self.miss_count += 1
            await self._update_state()
            return None
    
    async def delete(self, scope: str, key: str) -> bool:
        """
        Delete item from cache
        
        Args:
            scope: Scope of the memory
            key: Key of the data
            
        Returns:
            Success status
        """
        try:
            full_key = f"{scope}:{key}"
            
            if full_key in self.cache:
                del self.cache[full_key]
                self.pinned_keys.discard(full_key)
                logger.debug(f"Deleted item from cache: {full_key}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting item from cache: {e}")
            return False
    
    async def merge(self, scope: str, key: str, data: Dict[str, Any]) -> bool:
        """
        Merge data with existing cached item
        
        Args:
            scope: Scope of the memory
            key: Key of the data
            data: Data to merge
            
        Returns:
            Success status
        """
        try:
            existing_data = await self.get(scope, key)
            
            if existing_data is None:
                # No existing data, store as new
                return await self.put(scope, key, data)
            
            if isinstance(existing_data, dict) and isinstance(data, dict):
                # Merge dictionaries
                merged_data = {**existing_data, **data}
                return await self.put(scope, key, merged_data)
            elif isinstance(existing_data, list) and isinstance(data, list):
                # Merge lists
                merged_data = existing_data + data
                return await self.put(scope, key, merged_data)
            else:
                # Can't merge, replace
                return await self.put(scope, key, data)
                
        except Exception as e:
            logger.error(f"Error merging item in cache: {e}")
            return False
    
    async def pin(self, scope: str, key: str) -> bool:
        """Pin an item to prevent eviction"""
        try:
            full_key = f"{scope}:{key}"
            
            if full_key in self.cache:
                self.cache[full_key]["pinned"] = True
                self.pinned_keys.add(full_key)
                logger.debug(f"Pinned cache item: {full_key}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error pinning cache item: {e}")
            return False
    
    async def unpin(self, scope: str, key: str) -> bool:
        """Unpin an item to allow eviction"""
        try:
            full_key = f"{scope}:{key}"
            
            if full_key in self.cache:
                self.cache[full_key]["pinned"] = False
                self.pinned_keys.discard(full_key)
                logger.debug(f"Unpinned cache item: {full_key}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error unpinning cache item: {e}")
            return False
    
    async def promote_to_longterm(self, scope: str, key: str) -> bool:
        """
        Promote a cache item to long-term storage (Flywheel)
        
        Args:
            scope: Scope of the memory
            key: Key of the data
            
        Returns:
            Success status
        """
        try:
            full_key = f"{scope}:{key}"
            
            if full_key not in self.cache:
                return False
            
            item = self.cache[full_key]
            
            # Create memory item for long-term storage
            memory_item = MemoryItem(
                id=f"promoted_{full_key}",
                scope="global",  # Promoted items go to global scope
                data={
                    "original_scope": scope,
                    "original_key": key,
                    "data": item["data"],
                    "promoted_at": datetime.now().isoformat(),
                    "access_count": item["access_count"],
                    "promotion_reason": "cache_promotion"
                },
                created_at=item["created_at"],
                updated_at=datetime.now()
            )
            
            # Save to database
            await DatabaseManager.save_memory_item(memory_item)
            
            # Keep in cache but mark as promoted
            item["promoted"] = True
            item["promoted_at"] = datetime.now()
            
            logger.info(f"Promoted cache item to long-term storage: {full_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error promoting cache item: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        hit_rate = (self.hit_count / max(self.access_count, 1)) * 100
        
        # Calculate memory usage
        total_size = sum(item["size"] for item in self.cache.values())
        
        # Count items by scope
        scope_counts = {}
        for full_key, item in self.cache.items():
            scope = item["scope"]
            scope_counts[scope] = scope_counts.get(scope, 0) + 1
        
        return {
            "total_items": len(self.cache),
            "pinned_items": len(self.pinned_keys),
            "access_count": self.access_count,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "eviction_count": self.eviction_count,
            "hit_rate": round(hit_rate, 2),
            "total_size_bytes": total_size,
            "avg_size_bytes": round(total_size / max(len(self.cache), 1), 2),
            "scope_distribution": scope_counts,
            "max_cache_size": self.max_cache_size,
            "cache_utilization": round((len(self.cache) / self.max_cache_size) * 100, 2)
        }
    
    async def clear_scope(self, scope: str) -> int:
        """Clear all items from a specific scope"""
        try:
            cleared_count = 0
            keys_to_remove = []
            
            for full_key, item in self.cache.items():
                if item["scope"] == scope and not item.get("pinned", False):
                    keys_to_remove.append(full_key)
            
            for key in keys_to_remove:
                del self.cache[key]
                self.pinned_keys.discard(key)
                cleared_count += 1
            
            logger.info(f"Cleared {cleared_count} items from scope: {scope}")
            return cleared_count
            
        except Exception as e:
            logger.error(f"Error clearing scope {scope}: {e}")
            return 0
    
    async def _evict_lru(self):
        """Evict least recently used item"""
        if not self.cache:
            return
        
        # Find LRU item that's not pinned
        lru_key = None
        lru_time = datetime.now()
        
        for full_key, item in self.cache.items():
            if not item.get("pinned", False) and item["last_accessed"] < lru_time:
                lru_time = item["last_accessed"]
                lru_key = full_key
        
        if lru_key:
            del self.cache[lru_key]
            self.eviction_count += 1
            logger.debug(f"Evicted LRU item: {lru_key}")
    
    async def _cache_maintenance_loop(self):
        """Background cache maintenance"""
        while self.status == "active":
            try:
                # Clean expired items
                await self._clean_expired()
                
                # Save cache state to database periodically
                await self._save_to_database()
                
                # Check for items to promote
                await self._check_promotion_candidates()
                
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Error in cache maintenance loop: {e}")
                await asyncio.sleep(300)
    
    async def _clean_expired(self):
        """Remove expired items from cache"""
        current_time = datetime.now()
        expired_keys = []
        
        for full_key, item in self.cache.items():
            if item["expires_at"] and current_time > item["expires_at"]:
                if not item.get("pinned", False):
                    expired_keys.append(full_key)
        
        for key in expired_keys:
            del self.cache[key]
            self.pinned_keys.discard(key)
        
        if expired_keys:
            logger.debug(f"Cleaned {len(expired_keys)} expired items")
    
    async def _check_promotion_candidates(self):
        """Check for items that should be promoted to long-term storage"""
        for full_key, item in self.cache.items():
            # Promote items with high access count or long duration
            if (item["access_count"] >= 10 or 
                (datetime.now() - item["created_at"]).total_seconds() > 7200):  # 2 hours
                
                if not item.get("promoted", False):
                    scope, key = full_key.split(":", 1)
                    await self.promote_to_longterm(scope, key)
    
    async def _save_to_database(self):
        """Save important cache items to database"""
        for full_key, item in self.cache.items():
            # Save pinned items and frequently accessed items
            if item.get("pinned", False) or item["access_count"] >= 5:
                try:
                    memory_item = MemoryItem(
                        id=f"cache_{full_key}",
                        scope=item["scope"],
                        data=item["data"],
                        created_at=item["created_at"],
                        updated_at=item["updated_at"]
                    )
                    
                    await DatabaseManager.save_memory_item(memory_item)
                    
                except Exception as e:
                    logger.error(f"Error saving cache item to database: {e}")
    
    async def _load_from_database(self):
        """Load memory items from database into cache"""
        try:
            memory_items = await DatabaseManager.get_memory_items()
            
            for item in memory_items:
                if item.id.startswith("cache_"):
                    full_key = item.id[6:]  # Remove "cache_" prefix
                    
                    cache_item = {
                        "scope": item.scope,
                        "key": full_key.split(":", 1)[1],
                        "data": item.data,
                        "created_at": item.created_at,
                        "updated_at": item.updated_at,
                        "last_accessed": item.updated_at,
                        "access_count": 1,
                        "expires_at": None,
                        "pinned": False,
                        "size": len(json.dumps(item.data, default=str))
                    }
                    
                    self.cache[full_key] = cache_item
            
            logger.info(f"Loaded {len(self.cache)} items from database")
            
        except Exception as e:
            logger.error(f"Error loading from database: {e}")
    
    async def _load_from_database_key(self, full_key: str) -> Optional[Any]:
        """Load a specific key from database"""
        try:
            memory_items = await DatabaseManager.get_memory_items()
            
            for item in memory_items:
                if item.id == f"cache_{full_key}":
                    # Add to cache
                    cache_item = {
                        "scope": item.scope,
                        "key": full_key.split(":", 1)[1],
                        "data": item.data,
                        "created_at": item.created_at,
                        "updated_at": item.updated_at,
                        "last_accessed": datetime.now(),
                        "access_count": 1,
                        "expires_at": None,
                        "pinned": False,
                        "size": len(json.dumps(item.data, default=str))
                    }
                    
                    self.cache[full_key] = cache_item
                    return item.data
            
            return None
            
        except Exception as e:
            logger.error(f"Error loading key from database: {e}")
            return None
    
    async def _update_state(self):
        """Update component state in database"""
        stats = await self.get_stats()
        
        state = AgentState(
            agent_name=self.name,
            status=self.status,
            last_activity=datetime.now(),
            metrics={
                "total_items": float(stats["total_items"]),
                "access_count": float(self.access_count),
                "hit_rate": round(stats["hit_rate"], 2),
                "eviction_count": float(self.eviction_count),
                "cache_utilization": round(stats["cache_utilization"], 2),
                "total_size_mb": round(stats["total_size_bytes"] / (1024*1024), 2)
            },
            config={
                "max_cache_size": self.max_cache_size,
                "default_ttl": self.default_ttl
            }
        )
        await DatabaseManager.update_agent_state(state)
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current component status"""
        stats = await self.get_stats()
        return {
            "name": self.name,
            "status": self.status,
            **stats
        }