"""
Main Video Search System - Integrates all components
Optimized for sub-second latency and high throughput
"""

import os
import time
import uuid
import logging
import asyncio
from typing import List, Dict, Any, Optional, Union
import numpy as np
import yaml
from pathlib import Path

# Core components
from core.frame_extractor import OptimizedFrameExtractor, choose_optimal_strategy
from core.feature_extractor import FeatureExtractor, BatchProcessor
from indexes.hnsw import OptimizedHNSWIndex
from storage.cache import MultiLevelCache, QueryResultCache
from utils.metrics import SystemMetrics
from utils.config import load_config

logger = logging.getLogger(__name__)


class VideoSearchSystem:
    """
    High-performance video search system with HNSW index
    Achieves sub-second query latency for millions of videos
    """
    
    def __init__(self, config_path: Optional[str] = None):
        # Load configuration
        if config_path:
            self.config = load_config(config_path)
        else:
            # Use default config
            default_config_path = Path(__file__).parent / 'config' / 'default.yaml'
            self.config = load_config(str(default_config_path))
        
        # Initialize components
        self._initialize_components()
        
        # System state
        self.is_ready = False
        self.startup_time = time.time()
        
        logger.info("VideoSearchSystem initialized")
    
    def _initialize_components(self):
        """Initialize all system components"""
        
        # Frame Extractor
        self.frame_extractor = OptimizedFrameExtractor(
            sample_rate=self.config['frame_extraction']['sample_rate'],
            strategy='hybrid',  # Use hybrid for best results
            max_frames_per_video=self.config['frame_extraction']['max_frames_per_video'],
            frame_size=tuple(self.config['frame_extraction']['frame_size']),
            quality_filter=True
        )
        
        # Feature Extractor
        self.feature_extractor = FeatureExtractor(
            model_name=self.config['feature_extraction']['model_name'],
            device=self.config['feature_extraction']['device'],
            batch_size=self.config['feature_extraction']['batch_size']
        )
        
        # Batch Processor for async operations
        self.batch_processor = BatchProcessor(
            self.feature_extractor,
            timeout_ms=self.config['batch_processing']['timeout_ms']
        )
        
        # HNSW Index (fastest approach from the document)
        index_config = self.config['index']['hnsw']
        self.index = OptimizedHNSWIndex(
            dimension=self.config['index']['dimension'],
            M=index_config['M'],
            ef_construction=index_config['ef_construction'],
            ef_search=index_config['ef_search'],
            max_M=index_config['max_M'],
            level_generation_factor=index_config.get('level_generation_factor', 1.0 / np.log(2.0)),
            num_threads=self.config['performance']['num_threads'],
            use_numpy_optimization=True
        )
        
        # Multi-level Cache System
        cache_config = self.config['cache']
        l2_config = None
        if cache_config.get('enable_cache') and 'l2_host' in cache_config:
            # Parse Redis URL
            redis_url = cache_config['l2_host']
            if redis_url.startswith('redis://'):
                host_port = redis_url.replace('redis://', '').split(':')
                l2_config = {
                    'host': host_port[0],
                    'port': int(host_port[1]) if len(host_port) > 1 else 6379
                }
        
        self.cache = MultiLevelCache(
            l1_capacity=cache_config['l1_capacity'],
            l2_config=l2_config,
            ttl_seconds=cache_config['ttl_seconds'],
            enable_l2=cache_config.get('enable_cache', True)
        )
        
        # Query Result Cache
        self.query_cache = QueryResultCache(self.cache)
        
        # System Metrics
        self.metrics = SystemMetrics()
        
        # Video Metadata Storage
        self.video_metadata = {}  # In production, use proper database
        
        logger.info("All components initialized successfully")
    
    async def add_video(self, video_path: str, video_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Add video to the search index
        Complete ingestion pipeline with optimizations
        """
        if video_id is None:
            video_id = str(uuid.uuid4())
        
        start_time = time.time()
        
        try:
            # Validate video file
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            logger.info(f"Adding video {video_id}: {video_path}")
            
            # Step 1: Extract frames using optimal strategy
            extraction_start = time.time()
            frames_data = self.frame_extractor.extract_frames(video_path)
            
            if not frames_data:
                raise ValueError("No frames extracted from video")
            
            self.metrics.record_counter('frames_extracted', len(frames_data))
            self.metrics.record_histogram('frame_extraction_time', 
                                        (time.time() - extraction_start) * 1000)
            
            # Step 2: Extract features in batches
            feature_start = time.time()
            features_data = self.feature_extractor.extract_from_video_frames(frames_data)
            
            self.metrics.record_histogram('feature_extraction_time',
                                        (time.time() - feature_start) * 1000)
            
            # Step 3: Add to HNSW index
            index_start = time.time()
            vectors = []
            node_ids = []
            timestamps = []
            
            for i, frame_data in enumerate(features_data):
                vector = frame_data['features']
                node_id = f"{video_id}_{i}"
                
                vectors.append(vector)
                node_ids.append(node_id)
                timestamps.append(frame_data['timestamp'])
                
                # Store metadata mapping
                self.video_metadata[node_id] = {
                    'video_id': video_id,
                    'timestamp': frame_data['timestamp'],
                    'frame_number': frame_data.get('frame_number', i),
                    'video_path': video_path
                }
            
            # Batch add to index for better performance
            self.index.add_batch(vectors, node_ids)
            
            index_time = time.time() - index_start
            self.metrics.record_histogram('index_insertion_time', index_time * 1000)
            
            # Step 4: Store video metadata
            video_duration = max(timestamps) if timestamps else 0
            self.video_metadata[f"video_{video_id}"] = {
                'video_id': video_id,
                'path': video_path,
                'duration': video_duration,
                'frame_count': len(features_data),
                'indexed_at': time.time(),
                'file_size': os.path.getsize(video_path)
            }
            
            # Step 5: Invalidate related cache entries
            self.query_cache.invalidate_results(video_id)
            
            # Record final metrics
            total_time = time.time() - start_time
            self.metrics.record_counter('videos_indexed', 1)
            self.metrics.record_histogram('total_ingestion_time', total_time * 1000)
            
            result = {
                'status': 'success',
                'video_id': video_id,
                'frames_indexed': len(features_data),
                'processing_time': total_time,
                'performance': {
                    'frame_extraction_time': extraction_start,
                    'feature_extraction_time': feature_start,
                    'index_insertion_time': index_time,
                    'total_time': total_time
                }
            }
            
            logger.info(f"Video indexed successfully: {video_id} "
                       f"({len(features_data)} frames in {total_time:.2f}s)")
            
            return result
            
        except Exception as e:
            self.metrics.record_counter('ingestion_errors', 1)
            logger.error(f"Failed to add video {video_id}: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    async def search(
        self, 
        query: Union[str, np.ndarray], 
        k: int = 5,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Search for similar content with sub-second latency
        Supports both text and image queries
        """
        start_time = time.time()
        query_id = str(uuid.uuid4())
        
        try:
            self.metrics.record_counter('searches', 1)
            
            # Step 1: Check cache
            cached_results = None
            if use_cache:
                if isinstance(query, str):
                    cached_results = self.query_cache.get_cached_results(
                        None, k, query_text=query
                    )
                else:
                    cached_results = self.query_cache.get_cached_results(
                        query, k
                    )
                
                if cached_results is not None:
                    self.metrics.record_counter('cache_hits', 1)
                    search_time = (time.time() - start_time) * 1000
                    self.metrics.record_histogram('search_latency', search_time)
                    
                    return {
                        'results': cached_results,
                        'search_time_ms': search_time,
                        'from_cache': True,
                        'query_id': query_id
                    }
            
            self.metrics.record_counter('cache_misses', 1)
            
            # Step 2: Extract query features
            feature_start = time.time()
            if isinstance(query, str):
                # Text query
                query_vector = self.feature_extractor.extract_text_features(query)
            else:
                # Image query - assume it's already a numpy array or process it
                if not isinstance(query, np.ndarray):
                    query_vector = self.feature_extractor.extract_features(query)
                else:
                    query_vector = query
            
            feature_time = time.time() - feature_start
            self.metrics.record_histogram('query_feature_extraction', feature_time * 1000)
            
            # Step 3: Search HNSW index
            search_start = time.time()
            search_results = self.index.search(query_vector, k * 2)  # Over-fetch for deduplication
            
            index_search_time = time.time() - search_start
            self.metrics.record_histogram('index_search_time', index_search_time * 1000)
            
            # Step 4: Enrich results with metadata and deduplicate
            enrichment_start = time.time()
            final_results = []
            seen_videos = set()
            
            for result in search_results:
                node_id = result['id']
                metadata = self.video_metadata.get(node_id, {})
                
                video_id = metadata.get('video_id')
                if not video_id or video_id in seen_videos:
                    continue
                
                seen_videos.add(video_id)
                
                # Get video metadata
                video_meta = self.video_metadata.get(f"video_{video_id}", {})
                
                enriched_result = {
                    'video_id': video_id,
                    'timestamp': metadata.get('timestamp', 0),
                    'frame_number': metadata.get('frame_number', 0),
                    'score': result['score'],
                    'distance': result['distance'],
                    'video_duration': video_meta.get('duration', 0),
                    'video_path': video_meta.get('path', ''),
                    'file_size': video_meta.get('file_size', 0)
                }
                
                # Add thumbnail URL if configured
                if 'thumbnail_base_url' in self.config.get('api', {}):
                    base_url = self.config['api']['thumbnail_base_url']
                    timestamp = enriched_result['timestamp']
                    enriched_result['thumbnail_url'] = f"{base_url}/{video_id}/thumbnail_{timestamp:.2f}.jpg"
                
                final_results.append(enriched_result)
                
                if len(final_results) >= k:
                    break
            
            enrichment_time = time.time() - enrichment_start
            self.metrics.record_histogram('result_enrichment_time', enrichment_time * 1000)
            
            # Step 5: Cache results
            if use_cache and final_results:
                if isinstance(query, str):
                    self.query_cache.cache_results(None, k, final_results, query_text=query)
                else:
                    self.query_cache.cache_results(query_vector, k, final_results)
            
            # Calculate total search time
            total_search_time = (time.time() - start_time) * 1000
            self.metrics.record_histogram('search_latency', total_search_time)
            
            result_data = {
                'results': final_results,
                'search_time_ms': total_search_time,
                'from_cache': False,
                'query_id': query_id,
                'performance': {
                    'feature_extraction_ms': feature_time * 1000,
                    'index_search_ms': index_search_time * 1000,
                    'result_enrichment_ms': enrichment_time * 1000,
                    'total_results_found': len(search_results),
                    'results_after_dedup': len(final_results)
                }
            }
            
            logger.info(f"Search completed: {query_id} "
                       f"({len(final_results)} results in {total_search_time:.1f}ms)")
            
            return result_data
            
        except Exception as e:
            self.metrics.record_counter('search_errors', 1)
            logger.error(f"Search failed for query {query_id}: {e}")
            
            return {
                'results': [],
                'search_time_ms': (time.time() - start_time) * 1000,
                'error': str(e),
                'query_id': query_id
            }
    
    async def search_batch(
        self, 
        queries: List[Union[str, np.ndarray]], 
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Process multiple search queries in parallel
        """
        tasks = []
        for query in queries:
            task = asyncio.create_task(self.search(query, k))
            tasks.append(task)
        
        return await asyncio.gather(*tasks)
    
    def get_video_info(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about indexed video
        """
        return self.video_metadata.get(f"video_{video_id}")
    
    def list_videos(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        List indexed videos with pagination
        """
        video_keys = [k for k in self.video_metadata.keys() if k.startswith('video_')]
        video_keys.sort()
        
        paginated_keys = video_keys[offset:offset + limit]
        
        return [
            self.video_metadata[key] 
            for key in paginated_keys
        ]
    
    def delete_video(self, video_id: str) -> bool:
        """
        Remove video from index
        """
        try:
            # Find all nodes for this video
            nodes_to_remove = []
            for node_id, metadata in self.video_metadata.items():
                if metadata.get('video_id') == video_id:
                    nodes_to_remove.append(node_id)
            
            # Remove from index (note: HNSW doesn't support deletion in this implementation)
            # In production, you'd need to rebuild the index or use a different approach
            logger.warning("Video deletion requires index rebuild (not implemented)")
            
            # Remove metadata
            for node_id in nodes_to_remove:
                if node_id in self.video_metadata:
                    del self.video_metadata[node_id]
            
            # Remove video metadata
            video_key = f"video_{video_id}"
            if video_key in self.video_metadata:
                del self.video_metadata[video_key]
            
            # Invalidate cache
            self.query_cache.invalidate_results(video_id)
            
            self.metrics.record_counter('videos_deleted', 1)
            logger.info(f"Video deleted: {video_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete video {video_id}: {e}")
            return False
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics
        """
        uptime = time.time() - self.startup_time
        
        # Index stats
        index_stats = self.index.get_stats()
        
        # Feature extractor stats
        feature_stats = self.feature_extractor.get_stats()
        
        # Cache stats
        cache_stats = self.cache.get_stats()
        
        # System metrics
        metrics_summary = self.metrics.get_summary()
        
        return {
            'uptime_seconds': uptime,
            'system_ready': self.is_ready,
            'video_count': len([k for k in self.video_metadata.keys() if k.startswith('video_')]),
            'total_frames_indexed': index_stats['element_count'],
            'index_performance': {
                'avg_search_time_ms': index_stats['avg_search_time_ms'],
                'p95_search_time_ms': index_stats['p95_search_time_ms'],
                'total_searches': index_stats['total_searches']
            },
            'feature_extraction': feature_stats,
            'cache_performance': cache_stats,
            'metrics': metrics_summary
        }
    
    def save_index(self, filepath: str) -> bool:
        """
        Save index to disk for persistence
        """
        try:
            self.index.save(filepath)
            logger.info(f"Index saved to: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            return False
    
    def load_index(self, filepath: str) -> bool:
        """
        Load index from disk
        """
        try:
            self.index.load(filepath)
            logger.info(f"Index loaded from: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """
        System health check
        """
        health_status = {
            'status': 'healthy',
            'timestamp': time.time(),
            'components': {}
        }
        
        # Check index
        try:
            if self.index.size() > 0:
                # Test search
                test_vector = np.random.randn(self.config['index']['dimension'])
                test_results = self.index.search(test_vector, 1)
                health_status['components']['index'] = {
                    'status': 'healthy',
                    'size': self.index.size()
                }
            else:
                health_status['components']['index'] = {
                    'status': 'empty',
                    'size': 0
                }
        except Exception as e:
            health_status['components']['index'] = {
                'status': 'error',
                'error': str(e)
            }
        
        # Check feature extractor
        try:
            test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            features = self.feature_extractor.extract_features(test_image)
            health_status['components']['feature_extractor'] = {
                'status': 'healthy',
                'output_dim': len(features)
            }
        except Exception as e:
            health_status['components']['feature_extractor'] = {
                'status': 'error',
                'error': str(e)
            }
        
        # Check cache
        try:
            test_key = 'health_check'
            self.cache.put(test_key, 'test_value')
            retrieved = self.cache.get(test_key)
            
            if retrieved == 'test_value':
                health_status['components']['cache'] = {'status': 'healthy'}
            else:
                health_status['components']['cache'] = {'status': 'degraded'}
                
            self.cache.delete(test_key)
            
        except Exception as e:
            health_status['components']['cache'] = {
                'status': 'error',
                'error': str(e)
            }
        
        # Overall status
        component_statuses = [comp['status'] for comp in health_status['components'].values()]
        if 'error' in component_statuses:
            health_status['status'] = 'unhealthy'
        elif 'degraded' in component_statuses:
            health_status['status'] = 'degraded'
        
        return health_status
    
    async def startup(self):
        """
        Complete system startup and readiness check
        """
        logger.info("Starting up VideoSearchSystem...")
        
        try:
            # Warm up components
            logger.info("Warming up feature extractor...")
            dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            self.feature_extractor.extract_features(dummy_image)
            
            # Set ready flag
            self.is_ready = True
            
            logger.info("VideoSearchSystem startup complete")
            
        except Exception as e:
            logger.error(f"Startup failed: {e}")
            raise
    
    async def shutdown(self):
        """
        Graceful shutdown
        """
        logger.info("Shutting down VideoSearchSystem...")
        
        try:
            # Save index if configured
            if hasattr(self.config, 'persistence') and self.config['persistence'].get('auto_save'):
                save_path = self.config['persistence']['index_path']
                self.save_index(save_path)
            
            # Clean up thread pools
            if hasattr(self.feature_extractor, 'thread_pool'):
                self.feature_extractor.thread_pool.shutdown()
            
            if hasattr(self.index, 'thread_pool'):
                self.index.thread_pool.shutdown()
            
            logger.info("VideoSearchSystem shutdown complete")
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")
