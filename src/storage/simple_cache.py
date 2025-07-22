"""
Simplified caching system for basic demo
No async dependencies
"""

import time
import hashlib
import pickle
import logging
from typing import Any, Dict, Optional, List
from collections import OrderedDict
from abc import ABC, abstractmethod
import threading

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


class SimpleCache:
    """
    Simple cache implementation for basic demo
    No Redis dependencies
    """
    
    def __init__(self, l1_capacity: int = 1000, ttl_seconds: int = 300):
        self.ttl_seconds = ttl_seconds
        
        # L1 Cache - In-memory only
        self.l1_cache = LRUCache(capacity=l1_capacity)
        
        # Statistics
        self.stats = {
            'l1_hits': 0,
            'l2_hits': 0,
            'misses': 0,
            'total_requests': 0
        }
    
    def _compute_cache_key(self, key: str) -> str:
        """Generate consistent cache key"""
        if len(key) > 100:
            return hashlib.md5(key.encode()).hexdigest()
        return key
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        cache_key = self._compute_cache_key(key)
        self.stats['total_requests'] += 1
        
        # L1 - Memory
        value = self.l1_cache.get(cache_key)
        if value is not None:
            self.stats['l1_hits'] += 1
            # Check TTL
            if self._is_expired(value):
                self.l1_cache.delete(cache_key)
                return None
            return value['value']
        
        # Cache miss
        self.stats['misses'] += 1
        return None
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Put value in cache"""
        cache_key = self._compute_cache_key(key)
        effective_ttl = ttl or self.ttl_seconds
        
        return self.l1_cache.put(cache_key, value, effective_ttl)
    
    def delete(self, key: str) -> bool:
        """Delete from cache"""
        cache_key = self._compute_cache_key(key)
        return self.l1_cache.delete(cache_key)
    
    def clear(self) -> None:
        """Clear cache"""
        self.l1_cache.clear()
        self.stats = {
            'l1_hits': 0,
            'l2_hits': 0,
            'misses': 0,
            'total_requests': 0
        }
    
    def _is_expired(self, cache_item: Dict) -> bool:
        """Check if cache item has expired"""
        if not cache_item.get('ttl'):
            return False
        
        return time.time() - cache_item['timestamp'] > cache_item['ttl']
    
    def get_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.stats['total_requests']
        if total == 0:
            return 0.0
        
        hits = self.stats['l1_hits'] + self.stats['l2_hits']
        return hits / total
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.stats['total_requests']
        
        l1_stats = self.l1_cache.get_stats()
        
        return {
            'total_requests': total,
            'l1_hits': self.stats['l1_hits'],
            'l2_hits': self.stats['l2_hits'],
            'misses': self.stats['misses'],
            'overall_hit_rate': self.get_hit_rate(),
            'l1_hit_rate': self.stats['l1_hits'] / total if total > 0 else 0,
            'l2_hit_rate': self.stats['l2_hits'] / total if total > 0 else 0,
            'l1_size': l1_stats['size'],
            'l1_capacity': l1_stats['capacity'],
            'l1_evictions': l1_stats['evictions']
        }


# Aliases for compatibility
MultiLevelCache = SimpleCache
QueryResultCache = SimpleCache
