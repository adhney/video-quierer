"""
Multi-level caching system for high-performance video search
"""

import time
import hashlib
import pickle
import logging
from typing import Any, Dict, Optional, List
from collections import OrderedDict
from abc import ABC, abstractmethod
import threading
import asyncio
import aioredis
import redis

logger = logging.getLogger(__name__)


class CacheBackend(ABC):
    """Abstract base class for cache backends"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        pass
    
    @abstractmethod
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        pass
    
    @abstractmethod
    def clear(self) -> None:
        pass


class LRUCache(CacheBackend):
    """
    Thread-safe Least Recently Used cache implementation
    Optimized for high-frequency access
    """
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.lock = threading.RLock()
        
        # Stats
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            # Move to end (most recent)
            value = self.cache.pop(key)
            self.cache[key] = value
            self.hits += 1
            return value
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        with self.lock:
            if key in self.cache:
                # Update existing
                self.cache.pop(key)
            elif len(self.cache) >= self.capacity:
                # Evict least recently used
                self.cache.popitem(last=False)
                self.evictions += 1
            
            # Add new item
            self.cache[key] = {
                'value': value,
                'timestamp': time.time(),
                'ttl': ttl
            }
            return True
    
    def delete(self, key: str) -> bool:
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def clear(self) -> None:
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
            self.evictions = 0
    
    def _cleanup_expired(self) -> None:
        """Remove expired items"""
        current_time = time.time()
        with self.lock:
            expired_keys = []
            for key, item in self.cache.items():
                if item.get('ttl') and current_time - item['timestamp'] > item['ttl']:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache[key]
    
    def get_stats(self) -> Dict[str, Any]:
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'hit_rate': hit_rate,
            'size': len(self.cache),
            'capacity': self.capacity
        }


class RedisCache(CacheBackend):
    """
    Redis-based cache for distributed caching
    """
    
    def __init__(
        self, 
        host: str = 'localhost',
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        prefix: str = 'video_search:'
    ):
        self.prefix = prefix
        try:
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=False,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {host}:{port}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    def _make_key(self, key: str) -> str:
        return f"{self.prefix}{key}"
    
    def get(self, key: str) -> Optional[Any]:
        if not self.redis_client:
            return None
        
        try:
            data = self.redis_client.get(self._make_key(key))
            if data:
                return pickle.loads(data)
            return None
            
        except Exception as e:
            logger.warning(f"Redis GET error: {e}")
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        if not self.redis_client:
            return False
        
        try:
            data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            return self.redis_client.set(
                self._make_key(key), 
                data, 
                ex=ttl  # TTL in seconds
            )
            
        except Exception as e:
            logger.warning(f"Redis PUT error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        if not self.redis_client:
            return False
        
        try:
            return bool(self.redis_client.delete(self._make_key(key)))
        except Exception as e:
            logger.warning(f"Redis DELETE error: {e}")
            return False
    
    def clear(self) -> None:
        if not self.redis_client:
            return
        
        try:
            # Delete all keys with our prefix
            keys = self.redis_client.keys(f"{self.prefix}*")
            if keys:
                self.redis_client.delete(*keys)
        except Exception as e:
            logger.warning(f"Redis CLEAR error: {e}")


class MultiLevelCache:
    """
    Hierarchical caching system with multiple tiers
    L1: In-memory (fastest)
    L2: Redis (fast)
    L3: Optional persistent storage (fallback)
    """
    
    def __init__(
        self,
        l1_capacity: int = 1000,
        l2_config: Optional[Dict] = None,
        ttl_seconds: int = 300,
        enable_l2: bool = True
    ):
        self.ttl_seconds = ttl_seconds
        
        # L1 Cache - In-memory
        self.l1_cache = LRUCache(capacity=l1_capacity)
        
        # L2 Cache - Redis
        self.l2_cache = None
        if enable_l2 and l2_config:
            try:
                self.l2_cache = RedisCache(**l2_config)
            except Exception as e:
                logger.warning(f"L2 cache disabled: {e}")
        
        # Statistics
        self.stats = {
            'l1_hits': 0,
            'l2_hits': 0,
            'misses': 0,
            'total_requests': 0
        }
    
    def _compute_cache_key(self, key: str) -> str:
        """
        Generate consistent cache key with hash for long keys
        """
        if len(key) > 100:
            # Hash long keys
            return hashlib.md5(key.encode()).hexdigest()
        return key
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache, checking tiers in order
        """
        cache_key = self._compute_cache_key(key)
        self.stats['total_requests'] += 1
        
        # L1 - Memory
        value = self.l1_cache.get(cache_key)
        if value is not None:
            self.stats['l1_hits'] += 1
            # Check TTL
            if self._is_expired(value):
                self.l1_cache.delete(cache_key)
                if self.l2_cache:
                    self.l2_cache.delete(cache_key)
                return None
            return value['value']
        
        # L2 - Redis
        if self.l2_cache:
            value = self.l2_cache.get(cache_key)
            if value is not None:
                self.stats['l2_hits'] += 1
                # Promote to L1
                self.l1_cache.put(cache_key, value, self.ttl_seconds)
                return value
        
        # Cache miss
        self.stats['misses'] += 1
        return None
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Put value in all cache tiers
        """
        cache_key = self._compute_cache_key(key)
        effective_ttl = ttl or self.ttl_seconds
        
        # Store in L1
        success_l1 = self.l1_cache.put(cache_key, value, effective_ttl)
        
        # Store in L2
        success_l2 = True
        if self.l2_cache:
            success_l2 = self.l2_cache.put(cache_key, value, effective_ttl)
        
        return success_l1 and success_l2
    
    def delete(self, key: str) -> bool:
        """
        Delete from all cache tiers
        """
        cache_key = self._compute_cache_key(key)
        
        success_l1 = self.l1_cache.delete(cache_key)
        success_l2 = True
        
        if self.l2_cache:
            success_l2 = self.l2_cache.delete(cache_key)
        
        return success_l1 or success_l2
    
    def clear(self) -> None:
        """
        Clear all cache tiers
        """
        self.l1_cache.clear()
        if self.l2_cache:
            self.l2_cache.clear()
        
        # Reset stats
        self.stats = {
            'l1_hits': 0,
            'l2_hits': 0,
            'misses': 0,
            'total_requests': 0
        }
    
    def _is_expired(self, cache_item: Dict) -> bool:
        """
        Check if cache item has expired
        """
        if not cache_item.get('ttl'):
            return False
        
        return time.time() - cache_item['timestamp'] > cache_item['ttl']
    
    def get_hit_rate(self) -> float:
        """
        Calculate overall cache hit rate
        """
        total = self.stats['total_requests']
        if total == 0:
            return 0.0
        
        hits = self.stats['l1_hits'] + self.stats['l2_hits']
        return hits / total
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics
        """
        total = self.stats['total_requests']
        
        stats = {
            'total_requests': total,
            'l1_hits': self.stats['l1_hits'],
            'l2_hits': self.stats['l2_hits'],
            'misses': self.stats['misses'],
            'overall_hit_rate': self.get_hit_rate(),
            'l1_hit_rate': self.stats['l1_hits'] / total if total > 0 else 0,
            'l2_hit_rate': self.stats['l2_hits'] / total if total > 0 else 0
        }
        
        # Add L1 stats
        l1_stats = self.l1_cache.get_stats()
        for key, value in l1_stats.items():
            stats[f'l1_{key}'] = value
        
        return stats


class QueryResultCache:
    """
    Specialized cache for video search query results
    Handles result expiration and similarity-based caching
    """
    
    def __init__(self, cache: MultiLevelCache, similarity_threshold: float = 0.95):
        self.cache = cache
        self.similarity_threshold = similarity_threshold
        self.query_vectors = {}  # Store query vectors for similarity matching
    
    def get_cached_results(
        self, 
        query_vector: Any, 
        k: int, 
        query_text: Optional[str] = None
    ) -> Optional[List[Dict]]:
        """
        Get cached results for similar queries
        """
        # Create cache key
        if query_text:
            cache_key = f"text_query:{hashlib.md5(query_text.encode()).hexdigest()}:{k}"
        else:
            # Use vector hash for image queries
            vector_hash = hashlib.md5(query_vector.tobytes()).hexdigest()
            cache_key = f"vector_query:{vector_hash}:{k}"
        
        # Try exact match first
        results = self.cache.get(cache_key)
        if results is not None:
            return results
        
        # For vector queries, try similarity matching
        if query_text is None:
            return self._find_similar_cached_query(query_vector, k)
        
        return None
    
    def cache_results(
        self,
        query_vector: Any,
        k: int,
        results: List[Dict],
        query_text: Optional[str] = None,
        ttl: Optional[int] = None
    ) -> None:
        """
        Cache query results
        """
        # Create cache key
        if query_text:
            cache_key = f"text_query:{hashlib.md5(query_text.encode()).hexdigest()}:{k}"
        else:
            vector_hash = hashlib.md5(query_vector.tobytes()).hexdigest()
            cache_key = f"vector_query:{vector_hash}:{k}"
            
            # Store vector for similarity matching
            self.query_vectors[cache_key] = query_vector
        
        # Cache results
        self.cache.put(cache_key, results, ttl)
    
    def _find_similar_cached_query(self, query_vector: Any, k: int) -> Optional[List[Dict]]:
        """
        Find cached results for similar query vectors
        """
        try:
            import numpy as np
            
            best_similarity = 0
            best_results = None
            
            for cached_key, cached_vector in self.query_vectors.items():
                # Check if this is for the same k value
                if not cached_key.endswith(f":{k}"):
                    continue
                
                # Calculate cosine similarity
                similarity = np.dot(query_vector, cached_vector) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(cached_vector)
                )
                
                if similarity > self.similarity_threshold and similarity > best_similarity:
                    # Try to get cached results
                    results = self.cache.get(cached_key)
                    if results is not None:
                        best_similarity = similarity
                        best_results = results
            
            return best_results
            
        except Exception as e:
            logger.warning(f"Similarity matching failed: {e}")
            return None
    
    def invalidate_results(self, video_id: str) -> None:
        """
        Invalidate cached results that might contain this video
        """
        # This is a simplified implementation
        # In production, you'd maintain an index of which queries contain which videos
        logger.info(f"Invalidating cache for video: {video_id}")
        # For now, we just clear everything (could be optimized)
        self.cache.clear()


class CacheWarmer:
    """
    Pre-warm cache with popular queries
    """
    
    def __init__(self, cache: MultiLevelCache, video_search_system):
        self.cache = cache
        self.video_search_system = video_search_system
        self.popular_queries = []
    
    def add_popular_query(self, query: str, weight: float = 1.0) -> None:
        """
        Add query to pre-warming list
        """
        self.popular_queries.append({
            'query': query,
            'weight': weight
        })
    
    async def warm_cache(self) -> None:
        """
        Pre-warm cache with popular queries
        """
        logger.info(f"Warming cache with {len(self.popular_queries)} queries")
        
        for query_info in self.popular_queries:
            try:
                query = query_info['query']
                
                # Check if already cached
                cache_key = f"text_query:{hashlib.md5(query.encode()).hexdigest()}:5"
                if self.cache.get(cache_key) is not None:
                    continue
                
                # Perform search and cache results
                results = await self.video_search_system.search(query, k=5)
                
                # Cache with longer TTL for popular queries
                extended_ttl = self.cache.ttl_seconds * 2
                # Note: This would require modifying the search system to use our cache
                
                logger.info(f"Warmed cache for query: {query}")
                
            except Exception as e:
                logger.warning(f"Failed to warm cache for query '{query_info['query']}': {e}")
        
        logger.info("Cache warming completed")


class CacheMetrics:
    """
    Detailed metrics and monitoring for cache performance
    """
    
    def __init__(self, cache: MultiLevelCache):
        self.cache = cache
        self.start_time = time.time()
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report
        """
        stats = self.cache.get_stats()
        uptime = time.time() - self.start_time
        
        return {
            'uptime_seconds': uptime,
            'requests_per_second': stats['total_requests'] / uptime if uptime > 0 else 0,
            'cache_efficiency': {
                'overall_hit_rate': stats['overall_hit_rate'],
                'l1_hit_rate': stats['l1_hit_rate'],
                'l2_hit_rate': stats['l2_hit_rate']
            },
            'memory_usage': {
                'l1_size': stats['l1_size'],
                'l1_capacity': stats['l1_capacity'],
                'l1_utilization': stats['l1_size'] / stats['l1_capacity'] if stats['l1_capacity'] > 0 else 0
            },
            'operations': {
                'total_requests': stats['total_requests'],
                'l1_hits': stats['l1_hits'],
                'l2_hits': stats['l2_hits'],
                'misses': stats['misses'],
                'evictions': stats.get('l1_evictions', 0)
            }
        }
    
    def export_prometheus_metrics(self) -> str:
        """
        Export metrics in Prometheus format
        """
        stats = self.cache.get_stats()
        uptime = time.time() - self.start_time
        
        metrics = [
            f"video_search_cache_requests_total {stats['total_requests']}",
            f"video_search_cache_hits_total{{tier=\"l1\"}} {stats['l1_hits']}",
            f"video_search_cache_hits_total{{tier=\"l2\"}} {stats['l2_hits']}",
            f"video_search_cache_misses_total {stats['misses']}",
            f"video_search_cache_hit_rate {{tier=\"overall\"}} {stats['overall_hit_rate']}",
            f"video_search_cache_hit_rate{{tier=\"l1\"}} {stats['l1_hit_rate']}",
            f"video_search_cache_hit_rate{{tier=\"l2\"}} {stats['l2_hit_rate']}",
            f"video_search_cache_size{{tier=\"l1\"}} {stats['l1_size']}",
            f"video_search_cache_capacity{{tier=\"l1\"}} {stats['l1_capacity']}",
            f"video_search_cache_uptime_seconds {uptime}",
            f"video_search_cache_requests_per_second {stats['total_requests'] / uptime if uptime > 0 else 0}"
        ]
        
        return '\n'.join(metrics)
