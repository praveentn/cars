# components/retriever.py
import asyncio
import uuid
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import logging
import json
import hashlib
import math

from core.database import DatabaseManager, AgentState, KnowledgeChunk
from core.llm_client import llm_client

logger = logging.getLogger(__name__)

class RetrieverComponent:
    """
    Retriever (Relevancy & Recency; RAG)
    - Long-term knowledge with hybrid search and freshness weighting
    - Hybrid ranking: BM25 + vector similarity + metadata filters
    - Time-aware scoring with decay
    - Smart chunking with overlap and lineage tracking
    """
    
    def __init__(self):
        self.name = "retriever"
        self.status = "idle"
        self.knowledge_base = {}  # In-memory knowledge store
        self.embeddings_cache = {}  # Vector embeddings cache
        self.chunks_added = 0
        self.searches_performed = 0
        self.total_chunks = 0
        self.index_version = 1
        
        # Configuration
        self.chunk_size = 512
        self.chunk_overlap = 64
        self.max_chunks_per_doc = 50
        self.freshness_weight = 0.3
        self.similarity_weight = 0.4
        self.bm25_weight = 0.3
        self.freshness_half_life = 7 * 24 * 3600  # 7 days
        
    async def start(self):
        """Start the retriever component"""
        self.status = "active"
        await self._update_state()
        logger.info("Retriever component started")
        
        # Start background maintenance
        asyncio.create_task(self._index_maintenance_loop())
        
        # Load existing knowledge from database
        await self._load_knowledge_base()
    
    async def stop(self):
        """Stop the retriever component"""
        self.status = "stopped"
        
        # Save knowledge base to database
        await self._save_knowledge_base()
        
        await self._update_state()
        logger.info("Retriever component stopped")
    
    async def add_document(self, content: str, metadata: Dict[str, Any]) -> List[str]:
        """
        Add a document to the knowledge base
        
        Args:
            content: Text content of the document
            metadata: Document metadata (source, title, etc.)
            
        Returns:
            List of chunk IDs created
        """
        try:
            # Generate document ID
            doc_id = str(uuid.uuid4())
            
            # Create chunks from the document
            chunks = await self._create_chunks(content, doc_id, metadata)
            
            chunk_ids = []
            for chunk in chunks:
                chunk_id = await self._store_chunk(chunk)
                chunk_ids.append(chunk_id)
            
            self.chunks_added += len(chunks)
            self.total_chunks += len(chunks)
            await self._update_state()
            
            logger.info(f"Added document {doc_id} with {len(chunks)} chunks")
            return chunk_ids
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            raise
    
    async def search(self, 
                    query: str, 
                    k: int = 8, 
                    filters: Dict[str, Any] = None,
                    include_scores: bool = True) -> List[Dict[str, Any]]:
        """
        Search the knowledge base using hybrid retrieval
        
        Args:
            query: Search query
            k: Number of results to return
            filters: Metadata filters to apply
            include_scores: Whether to include relevance scores
            
        Returns:
            List of search results with chunks and scores
        """
        try:
            self.searches_performed += 1
            
            # Generate query embedding
            query_embedding = await llm_client.generate_embedding(query)
            
            # Get all relevant chunks
            candidates = await self._get_candidate_chunks(filters or {})
            
            if not candidates:
                return []
            
            # Score each candidate
            scored_results = []
            for chunk_id, chunk in candidates.items():
                score = await self._calculate_hybrid_score(
                    query, query_embedding, chunk
                )
                
                result = {
                    "chunk_id": chunk_id,
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                    "source": chunk.get("source", "unknown"),
                    "created_at": chunk.get("created_at", datetime.now()).isoformat(),
                }
                
                if include_scores:
                    result["relevance_score"] = round(score, 4)
                    result["score_breakdown"] = chunk.get("score_breakdown", {})
                
                scored_results.append((score, result))
            
            # Sort by score and return top k
            scored_results.sort(key=lambda x: x[0], reverse=True)
            results = [result for _, result in scored_results[:k]]
            
            await self._update_state()
            
            logger.info(f"Search performed: '{query}' returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error performing search: {e}")
            return []
    
    async def get_similar_chunks(self, chunk_id: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Find chunks similar to a given chunk
        
        Args:
            chunk_id: ID of the reference chunk
            k: Number of similar chunks to return
            
        Returns:
            List of similar chunks
        """
        try:
            if chunk_id not in self.knowledge_base:
                return []
            
            reference_chunk = self.knowledge_base[chunk_id]
            reference_embedding = reference_chunk.get("embedding")
            
            if not reference_embedding:
                return []
            
            # Calculate similarity with all other chunks
            similarities = []
            for other_id, other_chunk in self.knowledge_base.items():
                if other_id == chunk_id:
                    continue
                
                other_embedding = other_chunk.get("embedding")
                if other_embedding:
                    similarity = self._cosine_similarity(reference_embedding, other_embedding)
                    similarities.append((similarity, other_id, other_chunk))
            
            # Sort by similarity and return top k
            similarities.sort(key=lambda x: x[0], reverse=True)
            
            results = []
            for similarity, chunk_id, chunk in similarities[:k]:
                results.append({
                    "chunk_id": chunk_id,
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                    "similarity_score": round(similarity, 4)
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error finding similar chunks: {e}")
            return []
    
    async def update_chunk(self, chunk_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an existing chunk
        
        Args:
            chunk_id: ID of the chunk to update
            updates: Updates to apply
            
        Returns:
            Success status
        """
        try:
            if chunk_id not in self.knowledge_base:
                return False
            
            chunk = self.knowledge_base[chunk_id]
            
            # Apply updates
            for key, value in updates.items():
                if key in ["text", "metadata", "relevance_score"]:
                    chunk[key] = value
            
            # Update timestamp
            chunk["updated_at"] = datetime.now()
            
            # Regenerate embedding if text was updated
            if "text" in updates:
                chunk["embedding"] = await llm_client.generate_embedding(updates["text"])
            
            logger.info(f"Updated chunk {chunk_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating chunk {chunk_id}: {e}")
            return False
    
    async def delete_chunk(self, chunk_id: str) -> bool:
        """
        Delete a chunk from the knowledge base
        
        Args:
            chunk_id: ID of the chunk to delete
            
        Returns:
            Success status
        """
        try:
            if chunk_id in self.knowledge_base:
                del self.knowledge_base[chunk_id]
                self.total_chunks -= 1
                logger.info(f"Deleted chunk {chunk_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting chunk {chunk_id}: {e}")
            return False
    
    async def reindex(self) -> Dict[str, Any]:
        """
        Rebuild the search index
        
        Returns:
            Reindexing statistics
        """
        try:
            start_time = datetime.now()
            
            # Regenerate embeddings for all chunks
            regenerated_count = 0
            for chunk_id, chunk in self.knowledge_base.items():
                if "text" in chunk:
                    chunk["embedding"] = await llm_client.generate_embedding(chunk["text"])
                    regenerated_count += 1
            
            # Update index version
            self.index_version += 1
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            stats = {
                "chunks_reindexed": regenerated_count,
                "duration_seconds": round(duration, 2),
                "new_index_version": self.index_version,
                "timestamp": end_time.isoformat()
            }
            
            logger.info(f"Reindexing completed: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error during reindexing: {e}")
            raise
    
    async def _create_chunks(self, content: str, doc_id: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create chunks from document content"""
        chunks = []
        
        # Simple text chunking (word-based)
        words = content.split()
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            if len(chunk_text.strip()) < 50:  # Skip very short chunks
                continue
            
            chunk = {
                "text": chunk_text,
                "doc_id": doc_id,
                "chunk_index": len(chunks),
                "metadata": {**metadata, "chunk_index": len(chunks)},
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
                "source": metadata.get("source", "unknown")
            }
            
            chunks.append(chunk)
            
            if len(chunks) >= self.max_chunks_per_doc:
                break
        
        return chunks
    
    async def _store_chunk(self, chunk: Dict[str, Any]) -> str:
        """Store a chunk in the knowledge base"""
        chunk_id = str(uuid.uuid4())
        
        # Generate embedding
        embedding = await llm_client.generate_embedding(chunk["text"])
        chunk["embedding"] = embedding
        
        # Store in memory
        self.knowledge_base[chunk_id] = chunk
        
        # Create database record
        knowledge_chunk = KnowledgeChunk(
            id=chunk_id,
            text=chunk["text"],
            embedding=embedding,
            metadata=chunk["metadata"],
            source=chunk["source"],
            created_at=chunk["created_at"],
            relevance_score=0.0
        )
        
        # Save to database (placeholder - would use actual database)
        logger.debug(f"Stored chunk {chunk_id} in knowledge base")
        
        return chunk_id
    
    async def _get_candidate_chunks(self, filters: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Get candidate chunks based on filters"""
        candidates = {}
        
        for chunk_id, chunk in self.knowledge_base.items():
            # Apply filters
            include_chunk = True
            
            for filter_key, filter_value in filters.items():
                if filter_key in chunk["metadata"]:
                    chunk_value = chunk["metadata"][filter_key]
                    if isinstance(filter_value, list):
                        if chunk_value not in filter_value:
                            include_chunk = False
                            break
                    elif chunk_value != filter_value:
                        include_chunk = False
                        break
            
            if include_chunk:
                candidates[chunk_id] = chunk
        
        return candidates
    
    async def _calculate_hybrid_score(self, 
                                    query: str, 
                                    query_embedding: List[float], 
                                    chunk: Dict[str, Any]) -> float:
        """Calculate hybrid relevance score"""
        
        # 1. Semantic similarity (vector similarity)
        chunk_embedding = chunk.get("embedding", [])
        if chunk_embedding:
            semantic_score = self._cosine_similarity(query_embedding, chunk_embedding)
        else:
            semantic_score = 0.0
        
        # 2. BM25 score (term frequency)
        bm25_score = self._calculate_bm25_score(query, chunk["text"])
        
        # 3. Freshness score (time decay)
        freshness_score = self._calculate_freshness_score(chunk.get("created_at", datetime.now()))
        
        # Combine scores
        hybrid_score = (
            self.similarity_weight * semantic_score +
            self.bm25_weight * bm25_score +
            self.freshness_weight * freshness_score
        )
        
        # Store score breakdown for debugging
        chunk["score_breakdown"] = {
            "semantic": round(semantic_score, 4),
            "bm25": round(bm25_score, 4),
            "freshness": round(freshness_score, 4),
            "hybrid": round(hybrid_score, 4)
        }
        
        return hybrid_score
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            if not vec1 or not vec2 or len(vec1) != len(vec2):
                return 0.0
            
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = math.sqrt(sum(a * a for a in vec1))
            magnitude2 = math.sqrt(sum(a * a for a in vec2))
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            return dot_product / (magnitude1 * magnitude2)
            
        except Exception:
            return 0.0
    
    def _calculate_bm25_score(self, query: str, text: str) -> float:
        """Calculate BM25 score (simplified implementation)"""
        try:
            # Simple BM25 approximation
            query_terms = query.lower().split()
            text_terms = text.lower().split()
            
            if not query_terms or not text_terms:
                return 0.0
            
            # Term frequency
            tf_score = 0.0
            for term in query_terms:
                tf = text_terms.count(term)
                if tf > 0:
                    # BM25 TF component (simplified)
                    k1 = 1.2
                    tf_component = (tf * (k1 + 1)) / (tf + k1)
                    tf_score += tf_component
            
            # Normalize by query length
            return tf_score / len(query_terms)
            
        except Exception:
            return 0.0
    
    def _calculate_freshness_score(self, created_at: datetime) -> float:
        """Calculate freshness score with exponential decay"""
        try:
            age_seconds = (datetime.now() - created_at).total_seconds()
            
            # Exponential decay: score = 0.5^(age / half_life)
            freshness_score = 0.5 ** (age_seconds / self.freshness_half_life)
            
            return freshness_score
            
        except Exception:
            return 0.0
    
    async def _index_maintenance_loop(self):
        """Background index maintenance"""
        while self.status == "active":
            try:
                # Clean up old, low-relevance chunks
                await self._cleanup_irrelevant_chunks()
                
                # Update relevance scores based on usage
                await self._update_relevance_scores()
                
                # Save knowledge base periodically
                await self._save_knowledge_base()
                
                await asyncio.sleep(1800)  # 30 minutes
                
            except Exception as e:
                logger.error(f"Error in index maintenance loop: {e}")
                await asyncio.sleep(1800)
    
    async def _cleanup_irrelevant_chunks(self):
        """Remove chunks with consistently low relevance"""
        to_remove = []
        
        for chunk_id, chunk in self.knowledge_base.items():
            relevance_score = chunk.get("relevance_score", 0.0)
            age_days = (datetime.now() - chunk.get("created_at", datetime.now())).days
            
            # Remove old chunks with very low relevance
            if age_days > 30 and relevance_score < 0.1:
                to_remove.append(chunk_id)
        
        for chunk_id in to_remove:
            await self.delete_chunk(chunk_id)
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} irrelevant chunks")
    
    async def _update_relevance_scores(self):
        """Update relevance scores based on usage patterns"""
        # This would typically track which chunks are frequently retrieved
        # and update their relevance scores accordingly
        pass
    
    async def _save_knowledge_base(self):
        """Save knowledge base to persistent storage"""
        try:
            # Save important chunks to database
            saved_count = 0
            for chunk_id, chunk in self.knowledge_base.items():
                knowledge_chunk = KnowledgeChunk(
                    id=chunk_id,
                    text=chunk["text"],
                    embedding=chunk.get("embedding", []),
                    metadata=chunk["metadata"],
                    source=chunk["source"],
                    created_at=chunk["created_at"],
                    relevance_score=chunk.get("relevance_score", 0.0)
                )
                
                # In a real implementation, this would save to the database
                saved_count += 1
            
            logger.debug(f"Saved {saved_count} chunks to database")
            
        except Exception as e:
            logger.error(f"Error saving knowledge base: {e}")
    
    async def _load_knowledge_base(self):
        """Load knowledge base from persistent storage"""
        try:
            # In a real implementation, this would load from the database
            logger.info("Knowledge base loaded from database")
            
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
    
    async def _update_state(self):
        """Update component state in database"""
        state = AgentState(
            agent_name=self.name,
            status=self.status,
            last_activity=datetime.now(),
            metrics={
                "total_chunks": float(self.total_chunks),
                "chunks_added": float(self.chunks_added),
                "searches_performed": float(self.searches_performed),
                "index_version": float(self.index_version),
                "knowledge_base_size_mb": round(len(json.dumps(self.knowledge_base, default=str)) / (1024*1024), 2)
            },
            config={
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "freshness_weight": self.freshness_weight,
                "similarity_weight": self.similarity_weight,
                "bm25_weight": self.bm25_weight
            }
        )
        await DatabaseManager.update_agent_state(state)
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current component status"""
        return {
            "name": self.name,
            "status": self.status,
            "total_chunks": self.total_chunks,
            "chunks_added": self.chunks_added,
            "searches_performed": self.searches_performed,
            "index_version": self.index_version,
            "knowledge_base_size": len(self.knowledge_base),
            "config": {
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "weights": {
                    "freshness": self.freshness_weight,
                    "similarity": self.similarity_weight,
                    "bm25": self.bm25_weight
                }
            }
        }

