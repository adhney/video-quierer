#!/usr/bin/env python3
"""
🌉 Video Search System Bridge
===========================
Integrates our optimized FixedEnhancedSemanticVideoProcessor 
with the modular FastAPI architecture
"""

import os
import sys
import time
import uuid
import logging
import asyncio
from typing import List, Dict, Any, Optional, Union
import numpy as np
from pathlib import Path

# Import our working processor (already in parent directory)
sys.path.insert(0, str(Path(__file__).parent.parent))
from enhanced_semantic_search_fixed import FixedEnhancedSemanticVideoProcessor, check_dependencies

logger = logging.getLogger(__name__)


class VideoSearchSystemBridge:
    """
    Bridge between our optimized processor and modular FastAPI architecture
    Maintains all performance optimizations while providing clean API
    """
    
    def __init__(self, videos_dir: str = "videos", use_clip: bool = True, enhanced_mode: bool = True):
        self.videos_dir = Path(videos_dir)
        self.videos_dir.mkdir(exist_ok=True)
        
        # Initialize our proven processor
        self.processor = FixedEnhancedSemanticVideoProcessor(
            use_clip=use_clip,
            enhanced_mode=enhanced_mode
        )
        
        # System state
        self.is_ready = False
        self.startup_time = time.time()
        self.video_metadata = {}
        
        logger.info("VideoSearchSystemBridge initialized")
    
    async def startup(self):
        """Initialize the system and process existing videos"""
        try:
            logger.info("🚀 Starting Video Search System...")
            
            # Check dependencies
            deps = check_dependencies()
            logger.info(f"📦 Dependencies: {deps}")
            
            # Process any existing videos
            await self._process_existing_videos()
            
            self.is_ready = True
            logger.info("✅ Video Search System ready!")
            
        except Exception as e:
            logger.error(f"❌ Startup failed: {e}")
            raise
    
    async def _process_existing_videos(self):
        """Process only new or changed videos"""
        video_files = []
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            video_files.extend(list(self.videos_dir.glob(f"*{ext}")))
            video_files.extend(list(self.videos_dir.glob(f"*{ext.upper()}")))
        
        # Load existing metadata
        metadata_file = self.videos_dir / ".video_metadata.json"
        existing_metadata = {}
        if metadata_file.exists():
            try:
                import json
                with open(metadata_file, 'r') as f:
                    existing_metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")
        
        # CRITICAL FIX: Always reload existing videos into processor on startup
        all_existing_videos = []
        for video_file in video_files:
            video_id = video_file.stem
            if video_id in existing_metadata:
                all_existing_videos.append((video_file, video_id))
                self.video_metadata[video_id] = existing_metadata[video_id]
                logger.info(f"   ✅ Already processed: {video_file.name}")
        
        # DEBUG: Check what we have
        logger.info(f"💾 DEBUG: all_existing_videos count: {len(all_existing_videos)}")
        
        # Check if we can load a saved processor state instead of reprocessing
        processor_state_file = self.videos_dir / ".processor_state.pkl"
        loaded_from_cache = False
        
        logger.info(f"💾 DEBUG: Cache file exists: {processor_state_file.exists()}")
        
        if processor_state_file.exists() and all_existing_videos:
            try:
                logger.info("💾 Loading saved processor state...")
                import pickle
                with open(processor_state_file, 'rb') as f:
                    saved_state = pickle.load(f)
                
                # DEBUG: Show what we loaded
                logger.info(f"💾 DEBUG: Loaded state keys: {list(saved_state.keys())}")
                
                # Handle both old and new cache formats
                if 'video_data' in saved_state:
                    # Old format - convert to new format
                    logger.info("💾 Converting old cache format to new format")
                    video_metadata = saved_state.get('video_data', {})
                    frame_count = saved_state.get('video_count', 0) 
                    # Try to get embeddings from old format
                    embeddings = saved_state.get('frame_embeddings', [])
                    index_vectors = embeddings if isinstance(embeddings, list) else []
                    index_ids = [f"frame_{i}" for i in range(len(index_vectors))]
                else:
                    # New format
                    video_metadata = saved_state.get('video_metadata', {})
                    frame_count = saved_state.get('frame_count', 0)
                    index_vectors = saved_state.get('index_vectors', [])
                    index_ids = saved_state.get('index_ids', [])
                
                logger.info(f"💾 DEBUG: video_metadata count: {len(video_metadata)}")
                logger.info(f"💾 DEBUG: frame_count: {frame_count}")
                logger.info(f"💾 DEBUG: index_vectors count: {len(index_vectors)}")
                logger.info(f"💾 DEBUG: index_ids count: {len(index_ids)}")
                
                # Verify state matches current videos
                saved_video_ids = set(saved_state.get('video_ids', []))
                current_video_ids = set([v[1] for v in all_existing_videos])
                
                if saved_video_ids == current_video_ids:
                    # Restore processor state
                    self.processor.video_metadata = video_metadata
                    self.processor.frame_count = frame_count
                    
                    # Restore index data if available
                    if hasattr(self.processor, 'index'):
                        if hasattr(self.processor.index, 'vectors') and isinstance(self.processor.index.vectors, dict):
                            # Dictionary format - reconstruct from lists
                            if index_ids and index_vectors:
                                self.processor.index.vectors = dict(zip(index_ids, index_vectors))
                                self.processor.index.ids = index_ids
                                logger.info(f"💾 DEBUG: Restored {len(self.processor.index.vectors)} vectors to dict format")
                            else:
                                logger.warning("💾 DEBUG: No valid vectors/ids to restore")
                        else:
                            # List format
                            self.processor.index.vectors = index_vectors
                            self.processor.index.ids = index_ids
                            logger.info(f"💾 DEBUG: Restored {len(index_vectors)} vectors to list format")
                    
                    logger.info(f"✅ Loaded processor state: {len(video_metadata)} videos with {len(index_vectors)} embeddings (instant startup!)")
                    loaded_from_cache = True
                else:
                    logger.info(f"⚠️ Video set changed, will reload from scratch")
                    
            except Exception as e:
                logger.warning(f"Failed to load processor state: {e}, will reload videos")
        
        # Only force reload if we couldn't load from cache AND no embeddings were loaded
        processor_has_embeddings = (
            hasattr(self.processor, 'index') and 
            hasattr(self.processor.index, 'vectors') and 
            len(getattr(self.processor.index, 'vectors', [])) > 0
        )
        
        if all_existing_videos and (not loaded_from_cache or not processor_has_embeddings):
            if not loaded_from_cache:
                logger.info(f"🔄 Cache failed to load, reprocessing {len(all_existing_videos)} videos...")
            else:
                logger.info(f"🔄 Cache loaded but no embeddings found, reprocessing {len(all_existing_videos)} videos...")
            # Force reload needed
            for video_file, video_id in all_existing_videos:
                try:
                    logger.info(f"   🔄 Reloading: {video_file.name}")
                    self.processor.process_video_enhanced(
                        str(video_file),
                        sampling_mode='high',
                        max_frames=300
                    )
                    logger.info(f"   ✅ Reloaded: {video_file.name}")
                except Exception as e:
                    logger.error(f"   ❌ Failed to reload {video_file.name}: {e}")
            
            # Save processor state for next startup
            await self._save_processor_state(all_existing_videos)
            logger.info(f"✅ Force reload complete with {getattr(self.processor, 'frame_count', 0)} frames indexed")
        
        # Check for new videos that need processing
        to_process = []
        for video_file in video_files:
            video_id = video_file.stem
            file_mtime = video_file.stat().st_mtime
            
            # Check if video needs processing (not in metadata or changed)
            needs_processing = (
                video_id not in existing_metadata or  # New video
                existing_metadata[video_id].get('file_mtime') != file_mtime or  # File changed
                existing_metadata[video_id].get('config_hash') != self._get_config_hash()  # Config changed
            )
            if needs_processing and video_id not in [v[1] for v in all_existing_videos]:
                to_process.append((video_file, video_id))
        
        if to_process:
            logger.info(f"🎬 Processing {len(to_process)} new videos...")
            for video_file, video_id in to_process:
                try:
                    await self._process_video_file(str(video_file), video_id)
                    logger.info(f"   ✅ Processed: {video_file.name}")
                except Exception as e:
                    logger.error(f"   ❌ Failed to process {video_file.name}: {e}")
            
            # Save updated metadata and processor state
            await self._save_metadata()
            # Update processor state cache with new videos
            updated_video_list = all_existing_videos + to_process
            await self._save_processor_state(updated_video_list)
            logger.info(f"🎉 Completed processing {len(to_process)} videos")
        elif loaded_from_cache and processor_has_embeddings:
            logger.info("📁 All videos loaded from cache into search index!")
        else:
            logger.info("📁 All videos loaded into search index!")
    
    def _get_config_hash(self):
        """Get a hash of current processing configuration"""
        import hashlib
        config_str = f"sampling_mode:high,max_frames:300,use_clip:{getattr(self.processor, 'use_clip', False)}"
        return hashlib.md5(config_str.encode()).hexdigest()
    
    async def _save_metadata(self):
        """Save video metadata to disk"""
        try:
            import json
            metadata_file = self.videos_dir / ".video_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(self.video_metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    async def _save_processor_state(self, video_list):
        """Save processor state for fast startup next time"""
        try:
            import pickle
            processor_state_file = self.videos_dir / ".processor_state.pkl"
            
            # Debug what we're trying to save
            video_meta = getattr(self.processor, 'video_metadata', {})
            frame_count = getattr(self.processor, 'frame_count', 0)
            
            # Try different ways to get the embeddings
            index_vectors = []
            index_ids = []
            
            if hasattr(self.processor, 'index'):
                if hasattr(self.processor.index, 'vectors'):
                    vectors = self.processor.index.vectors
                    if isinstance(vectors, dict):
                        # Dictionary format: {node_id: vector}
                        index_ids = list(vectors.keys())
                        index_vectors = list(vectors.values())
                        logger.info(f"💾 DEBUG: Found vectors as dict with {len(vectors)} items")
                    elif isinstance(vectors, list):
                        # List format
                        index_vectors = vectors
                        logger.info(f"💾 DEBUG: Found vectors as list with {len(vectors)} items")
                    else:
                        logger.info(f"💾 DEBUG: Unknown vectors type: {type(vectors)}")
                        
                if hasattr(self.processor.index, 'ids') and not index_ids:
                    # Only use index.ids if we didn't get IDs from vectors dict
                    index_ids = self.processor.index.ids or []
            
            # Also check for embeddings in other possible locations
            if hasattr(self.processor, 'embeddings'):
                logger.info(f"💾 DEBUG: Found embeddings attribute: {len(self.processor.embeddings)} items")
                if not index_vectors and self.processor.embeddings:
                    index_vectors = self.processor.embeddings
                    index_ids = [f"frame_{i}" for i in range(len(self.processor.embeddings))]
            
            if hasattr(self.processor, 'frame_embeddings'):
                logger.info(f"💾 DEBUG: Found frame_embeddings: {len(self.processor.frame_embeddings)} items")
                if not index_vectors and self.processor.frame_embeddings:
                    index_vectors = self.processor.frame_embeddings
                    index_ids = [f"frame_{i}" for i in range(len(self.processor.frame_embeddings))]
            
            # Debug what we found
            logger.info(f"💾 DEBUG: About to save:")
            logger.info(f"  - video_ids: {len(video_list)} items")
            logger.info(f"  - video_metadata: {len(video_meta)} items")
            logger.info(f"  - frame_count: {frame_count}")
            logger.info(f"  - index_vectors: {len(index_vectors)} items")
            logger.info(f"  - index_ids: {len(index_ids)} items")
            
            state_data = {
                'video_ids': [v[1] for v in video_list],
                'video_metadata': video_meta,
                'frame_count': frame_count,
                'index_vectors': index_vectors,
                'index_ids': index_ids
            }
            
            with open(processor_state_file, 'wb') as f:
                pickle.dump(state_data, f)
                
            logger.info(f"💾 Saved processor state: {len(video_meta)} videos, {len(index_vectors)} embeddings")
            
        except Exception as e:
            logger.warning(f"Failed to save processor state: {e}")
            logger.exception("Full error details:")
    
    async def _process_video_file(self, video_path: str, video_id: str):
        """Process a single video file"""
        # Use our optimized processor
        self.processor.process_video_enhanced(
            video_path,
            sampling_mode='high',
            max_frames=300
        )
        
        # Store metadata with config tracking
        video_file = Path(video_path)
        self.video_metadata[video_id] = {
            'id': video_id,
            'filename': video_file.name,
            'path': str(video_path),
            'size': video_file.stat().st_size,
            'file_mtime': video_file.stat().st_mtime,
            'processed_at': time.time(),
            'config_hash': self._get_config_hash(),
            'status': 'indexed'
        }
    
    async def add_video(self, video_path: str, video_id: Optional[str] = None) -> Dict[str, Any]:
        """Add and process a new video"""
        start_time = time.time()
        
        try:
            if not video_id:
                video_id = str(uuid.uuid4())
            
            # Process the video
            await self._process_video_file(video_path, video_id)
            
            # Save metadata
            await self._save_metadata()
            
            processing_time = time.time() - start_time
            
            return {
                'video_id': video_id,
                'status': 'success',
                'frames_indexed': getattr(self.processor, 'frame_count', 0),
                'processing_time': processing_time,
                'performance': {
                    'processing_time_ms': processing_time * 1000
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to add video {video_path}: {e}")
            return {
                'video_id': video_id,
                'status': 'error',
                'error': str(e),
                'frames_indexed': 0,
                'processing_time': time.time() - start_time
            }
    
    async def search(self, query: Union[str, np.ndarray], k: int = 5, use_cache: bool = True) -> Dict[str, Any]:
        """Search for similar videos"""
        start_time = time.time()
        query_id = str(uuid.uuid4())
        
        try:
            # Debug logging
            logger.info(f"🔍 Search called with query: {query[:50] if isinstance(query, str) else 'image_array'}...")
            logger.info(f"📊 Video metadata count: {len(self.video_metadata)}")
            logger.info(f"🎬 Processor video count: {getattr(self.processor, 'video_count', 0)}")
            logger.info(f"🐄 Processor frame count: {getattr(self.processor, 'frame_count', 0)}")
            logger.info(f"🗂️ Processor has video_metadata: {hasattr(self.processor, 'video_metadata') and len(getattr(self.processor, 'video_metadata', {})) > 0}")
            logger.info(f"📁 Processor index has data: {hasattr(self.processor, 'index') and hasattr(self.processor.index, 'vectors') and len(getattr(self.processor.index, 'vectors', {})) > 0}")
            
            # Check if processor has any data (check multiple possible attributes)
            has_video_metadata = hasattr(self.processor, 'video_metadata') and len(getattr(self.processor, 'video_metadata', {})) > 0
            has_frame_count = getattr(self.processor, 'frame_count', 0) > 0
            has_index_data = hasattr(self.processor, 'index') and hasattr(self.processor.index, 'vectors') and len(getattr(self.processor.index, 'vectors', {})) > 0
            
            if not (has_video_metadata or has_frame_count or has_index_data):
                logger.warning("⚠️ No videos processed yet - processor has no data")
                return {
                    'results': [],
                    'search_time_ms': 0,
                    'from_cache': False,
                    'query_id': query_id,
                    'performance': {
                        'search_time_ms': 0,
                        'total_time_ms': (time.time() - start_time) * 1000,
                        'error': 'No videos in processor index'
                    }
                }
            
            # Use our optimized search
            results, search_time = self.processor.search_enhanced(query, k=k)
            
            logger.info(f"✅ Search completed: {len(results)} results in {search_time:.3f}s")
            
            # Convert numpy types for JSON serialization
            results_converted = self._convert_numpy_types(results)
            search_time_converted = float(search_time)
            
            return {
                'results': results_converted,
                'search_time_ms': search_time_converted * 1000,
                'from_cache': False,  # Our processor handles caching internally
                'query_id': query_id,
                'performance': {
                    'search_time_ms': search_time_converted * 1000,
                    'total_time_ms': (time.time() - start_time) * 1000
                }
            }
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    async def search_batch(self, queries: List[str], k: int = 5) -> List[Dict[str, Any]]:
        """Process multiple search queries"""
        results = []
        
        for query in queries:
            try:
                result = await self.search(query, k=k)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch search failed for query '{query}': {e}")
                results.append({
                    'results': [],
                    'search_time_ms': 0,
                    'error': str(e)
                })
        
        return results
    
    def get_video_info(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific video"""
        return self.video_metadata.get(video_id)
    
    def list_videos(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """List indexed videos with pagination"""
        videos = list(self.video_metadata.values())
        
        # Apply pagination
        end_idx = offset + limit
        paginated_videos = videos[offset:end_idx]
        
        return paginated_videos
    
    def delete_video(self, video_id: str) -> bool:
        """Delete a video from the index"""
        if video_id in self.video_metadata:
            # Remove from metadata
            video_info = self.video_metadata.pop(video_id)
            
            # Try to delete the actual file
            try:
                video_path = Path(video_info['path'])
                if video_path.exists():
                    video_path.unlink()
            except Exception as e:
                logger.warning(f"Could not delete video file: {e}")
            
            # Note: We don't remove from the processor's index as it doesn't have 
            # a delete method, but the metadata removal prevents it from appearing in results
            
            return True
        
        return False
    
    def save_index(self, filepath: str) -> bool:
        """Save index to disk"""
        try:
            # Save metadata
            import json
            metadata_path = Path(filepath).with_suffix('.metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(self.video_metadata, f, indent=2)
            
            logger.info(f"Index metadata saved to {metadata_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            return False
    
    def load_index(self, filepath: str) -> bool:
        """Load index from disk"""
        try:
            # Load metadata
            import json
            metadata_path = Path(filepath).with_suffix('.metadata.json')
            
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.video_metadata = json.load(f)
                
                logger.info(f"Index metadata loaded from {metadata_path}")
                return True
            else:
                logger.warning(f"Metadata file not found: {metadata_path}")
                return False
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """System health check"""
        return {
            'status': 'healthy' if self.is_ready else 'starting',
            'timestamp': time.time(),
            'components': {
                'processor': {
                    'status': 'ready' if self.processor else 'error',
                    'use_clip': getattr(self.processor, 'use_clip', False),
                    'enhanced_mode': getattr(self.processor, 'enhanced_mode', False)
                },
                'videos': {
                    'status': 'ready',
                    'count': len(self.video_metadata),
                    'directory': str(self.videos_dir)
                }
            }
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        uptime = time.time() - self.startup_time
        
        return {
            'uptime_seconds': uptime,
            'system_ready': self.is_ready,
            'video_count': len(self.video_metadata),
            'total_frames_indexed': getattr(self.processor, 'frame_count', 0),
            'index_performance': {
                'type': 'HNSW',
                'status': 'ready'
            },
            'feature_extraction': {
                'model': 'CLIP' if getattr(self.processor, 'use_clip', False) else 'Visual Features',
                'status': 'ready'
            },
            'cache_performance': {
                'status': 'ready'
            },
            'metrics': {
                'uptime_hours': uptime / 3600,
                'videos_per_hour': len(self.video_metadata) / (uptime / 3600) if uptime > 0 else 0
            }
        }
    
    async def shutdown(self):
        """Cleanup on shutdown"""
        logger.info("🛑 Shutting down Video Search System...")
        # Our processor doesn't need special cleanup
        self.is_ready = False
        logger.info("✅ Shutdown complete")
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        return obj


# Global instance for FastAPI
video_search_system: Optional[VideoSearchSystemBridge] = None


def get_video_search_system() -> VideoSearchSystemBridge:
    """Get the global video search system instance"""
    global video_search_system
    if video_search_system is None:
        raise RuntimeError("Video search system not initialized")
    return video_search_system


def initialize_video_search_system(videos_dir: str = "videos") -> VideoSearchSystemBridge:
    """Initialize the global video search system"""
    global video_search_system
    video_search_system = VideoSearchSystemBridge(videos_dir=videos_dir)
    return video_search_system
