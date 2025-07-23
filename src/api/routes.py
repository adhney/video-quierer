"""
🚀 FastAPI Routes - Modular Video Search API
==========================================
Clean API endpoints using our optimized video search system
"""

from video_search_overhaul import VideoSearchSystem
import asyncio
import time
import uuid
import os
import shutil
import json
from typing import List, Dict, Any, Optional, Union
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import numpy as np
from PIL import Image
import io
import base64

# Import our bridge system
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Global video system instance for routes
_video_system = None
_server_start_time = time.time()  # Track when the server started


def get_video_search_system():
    """Get the video search system instance"""
    global _video_system, current_config
    if _video_system is None:
        # Load configuration if not already loaded
        if current_config is None:
            current_config = load_config_from_file()

        # Create system with current configuration
        _video_system = VideoSearchSystem("videos", current_config)
        _video_system.startup()
    return _video_system


# Configure logging
logger = logging.getLogger(__name__)

# Pydantic models


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query (text or base64 image)")
    k: int = Field(5, ge=1, le=50, description="Number of results to return")
    use_cache: bool = Field(True, description="Whether to use caching")


class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    search_time_ms: float
    from_cache: bool = False
    query_id: str
    performance: Optional[Dict[str, Any]] = None


class VideoUploadResponse(BaseModel):
    video_id: str
    status: str
    frames_indexed: int
    processing_time: float
    performance: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    timestamp: float
    components: Dict[str, Dict[str, Any]]


class SystemStatsResponse(BaseModel):
    uptime_seconds: float
    system_ready: bool
    video_count: int
    total_frames_indexed: int
    index_performance: Dict[str, Any]
    feature_extraction: Dict[str, Any]
    cache_performance: Dict[str, Any]
    metrics: Dict[str, Any]


class BatchSearchRequest(BaseModel):
    queries: List[str] = Field(..., description="List of search queries")
    k: int = Field(5, ge=1, le=50, description="Number of results per query")


class ConfigurationModel(BaseModel):
    sampling_mode: str = "high"
    max_frames: int = 300
    use_clip: bool = True
    enhanced_mode: bool = True
    default_results: int = 10
    cache_search: bool = True
    search_timeout: int = 30
    auto_save: bool = True
    log_level: str = "INFO"


class ConfigurationResponse(BaseModel):
    success: bool
    config: Optional[ConfigurationModel] = None
    message: Optional[str] = None


class CacheStats(BaseModel):
    embeddings_count: int
    videos_count: int
    cache_size_mb: float
    last_updated: str
    cache_file_exists: bool
    video_hashes_count: int


class CacheResponse(BaseModel):
    success: bool
    stats: Optional[CacheStats] = None
    message: Optional[str] = None


class CacheHealthResult(BaseModel):
    success: bool
    issues: List[str]
    recommendations: List[str]
    total_checks: int
    passed_checks: int


# Configuration
UPLOAD_FOLDER = Path("videos")
MAX_FILE_SIZE = 1024 * 1024 * 1024  # 1GB

# Global configuration for routes
current_config = None
config_file_path = Path("config.json")

# Configuration management functions


def load_config_from_file() -> ConfigurationModel:
    """Load configuration from file"""
    global current_config
    try:
        if config_file_path.exists():
            with open(config_file_path, 'r') as f:
                data = json.load(f)
                current_config = ConfigurationModel(**data)
                logger.info("Configuration loaded from file")
        else:
            logger.info("No config file found, using defaults")
            current_config = ConfigurationModel()
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        current_config = ConfigurationModel()
    return current_config


def save_config_to_file(config: ConfigurationModel) -> bool:
    """Save configuration to file"""
    try:
        with open(config_file_path, 'w') as f:
            json.dump(config.dict(), f, indent=2)
        logger.info("Configuration saved to file")
        return True
    except Exception as e:
        logger.error(f"Failed to save config: {e}")
        return False


def get_cache_stats() -> CacheStats:
    """Get cache statistics"""
    try:
        cache_path = Path("videos/video_search_cache.pkl")
        cache_size_mb = 0.0
        cache_exists = False
        last_updated = "Never"

        if cache_path.exists():
            cache_exists = True
            cache_size_mb = cache_path.stat().st_size / (1024 * 1024)
            last_updated = time.strftime("%Y-%m-%d %H:%M:%S",
                                         time.localtime(cache_path.stat().st_mtime))

        system = get_video_search_system()
        embeddings_count = len(system.index.embeddings) if system else 0
        videos_count = len(set(meta['video_name']
                           for meta in system.index.metadata)) if system else 0
        video_hashes_count = len(system.index.video_hashes) if system else 0

        return CacheStats(
            embeddings_count=embeddings_count,
            videos_count=videos_count,
            cache_size_mb=round(cache_size_mb, 2),
            last_updated=last_updated,
            cache_file_exists=cache_exists,
            video_hashes_count=video_hashes_count
        )
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        return CacheStats(
            embeddings_count=0,
            videos_count=0,
            cache_size_mb=0.0,
            last_updated="Error",
            cache_file_exists=False,
            video_hashes_count=0
        )


def perform_cache_health_check() -> CacheHealthResult:
    """Perform health check on cache"""
    issues = []
    recommendations = []
    checks = [
        "Cache file exists",
        "Video system initialized",
        "Embeddings present",
        "Video metadata consistent",
        "Video files accessible"
    ]

    passed = 0

    try:
        # Check cache file exists
        cache_path = Path("videos/video_search_cache.pkl")
        if cache_path.exists():
            passed += 1
        else:
            issues.append("Cache file does not exist")
            recommendations.append("Run rebuild cache to create cache file")

        # Check video system
        system = get_video_search_system()
        if system:
            passed += 1
        else:
            issues.append("Video system not initialized")
            recommendations.append("Restart the server")

        # Check embeddings
        if system and len(system.index.embeddings) > 0:
            passed += 1
        else:
            issues.append("No embeddings found")
            recommendations.append(
                "Process some videos to generate embeddings")

        # Check metadata consistency
        if system:
            if len(system.index.embeddings) == len(system.index.metadata):
                passed += 1
            else:
                issues.append("Embeddings and metadata count mismatch")
                recommendations.append("Rebuild cache to fix inconsistencies")

        # Check video files
        if system:
            videos_dir = Path("videos")
            existing_videos = set()
            if videos_dir.exists():
                for meta in system.index.metadata:
                    video_path = videos_dir / meta['video_name']
                    if video_path.exists():
                        existing_videos.add(meta['video_name'])

                if len(existing_videos) == len(set(meta['video_name'] for meta in system.index.metadata)):
                    passed += 1
                else:
                    issues.append("Some indexed videos are missing from disk")
                    recommendations.append(
                        "Remove missing videos from index or restore video files")

        return CacheHealthResult(
            success=len(issues) == 0,
            issues=issues,
            recommendations=recommendations,
            total_checks=len(checks),
            passed_checks=passed
        )

    except Exception as e:
        return CacheHealthResult(
            success=False,
            issues=[f"Health check failed: {str(e)}"],
            recommendations=["Check system logs and restart if necessary"],
            total_checks=len(checks),
            passed_checks=0
        )


def _get_youtube_format(quality: str) -> str:
    """Get yt-dlp format string based on quality preference"""
    format_map = {
        'best': 'best[ext=mp4]/best',
        '720p': 'best[height<=720][ext=mp4]/best[height<=720]',
        '480p': 'best[height<=480][ext=mp4]/best[height<=480]',
        '360p': 'best[height<=360][ext=mp4]/best[height<=360]',
        'worst': 'worst[ext=mp4]/worst'
    }
    return format_map.get(quality, 'best[ext=mp4]/best')


def create_api_routes(app: FastAPI):
    """Create all API routes"""

    @app.get("/api", tags=["system"])
    async def api_root():
        """API information and available endpoints"""
        return {
            "name": "🎬 Video Search API",
            "version": "2.1.0",
            "description": "High-performance semantic video search system",
            "features": [
                "CLIP-powered semantic search",
                "Multiple video format support",
                "YouTube download integration",
                "Frame-level search results",
                "Configuration management",
                "Cache optimization"
            ],
            "endpoints": {
                "documentation": "/api/docs",
                "health": "/api/health",
                "search": "/api/search",
                "upload": "/api/videos/upload",
                "videos": "/api/videos",
                "configuration": "/api/config",
                "cache": "/api/cache/stats"
            }
        }

    @app.get("/api/health", tags=["system"])
    async def health_check():
        """System health check endpoint"""
        try:
            system = get_video_search_system()
            return {
                "status": "healthy" if system else "starting",
                "timestamp": time.time(),
                "components": {
                    "video_system": {"status": "healthy" if system else "not_ready"},
                    "index": {"status": "healthy" if system and len(system.index.embeddings) > 0 else "empty"}
                }
            }
        except Exception as e:
            return {
                "status": "error",
                "timestamp": time.time(),
                "components": {"error": str(e)}
            }

    @app.get("/api/stats", tags=["system"])
    async def get_system_stats():
        """Get comprehensive system statistics"""
        try:
            system = get_video_search_system()
            if not system:
                raise HTTPException(status_code=503, detail="System not ready")

            # Calculate actual uptime since server started
            uptime_seconds = time.time() - _server_start_time

            return {
                "uptime_seconds": uptime_seconds,
                "system_ready": True,
                "video_count": len(set(meta['video_name'] for meta in system.index.metadata)),
                "total_frames_indexed": len(system.index.embeddings),
                "index_performance": {"embeddings_count": len(system.index.embeddings)},
                "feature_extraction": {"processor_type": "CLIP" if system.processor.use_clip else "Visual"},
                "cache_performance": {"cache_exists": system.cache_path.exists()},
                "metrics": {"total_videos": len(system.index.video_hashes)}
            }
        except Exception as e:
            raise HTTPException(
                status_code=503, detail=f"System not ready: {e}")

    @app.post("/api/videos/upload", tags=["videos"])
    async def upload_video(
        file: UploadFile = File(...),
        video_id: Optional[str] = Form(None)
    ):
        """Upload and index a video file"""
        try:
            system = get_video_search_system()

            # Validate file
            if not file.filename:
                raise HTTPException(status_code=400, detail="No file provided")

            # Generate video ID if not provided
            if not video_id:
                video_id = str(uuid.uuid4())

            # Check file type
            allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in allowed_extensions:
                raise HTTPException(
                    status_code=400, detail=f"Unsupported file type: {file_ext}")

            # Save uploaded file
            filename = f"{video_id}_{file.filename}"
            file_path = UPLOAD_FOLDER / filename

            start_time = time.time()

            try:
                with open(file_path, "wb") as buffer:
                    content = await file.read()
                    if len(content) > MAX_FILE_SIZE:
                        raise HTTPException(
                            status_code=413, detail="File too large (max 1GB)")
                    buffer.write(content)

                logger.info(f"📁 File uploaded: {file_path}")

                # Process video using the simple system
                frames_before = len(system.index.embeddings)
                system._process_single_video(file_path)
                frames_after = len(system.index.embeddings)

                # Update hash and save cache
                system.index.video_hashes[filename] = system.processor.get_video_hash(
                    file_path)
                system.index.save_to_disk(system.cache_path)

                processing_time = time.time() - start_time
                frames_indexed = frames_after - frames_before

                return {
                    "video_id": video_id,
                    "status": "success",
                    "frames_indexed": frames_indexed,
                    "processing_time": processing_time,
                    "performance": {"frames_per_second": frames_indexed / processing_time if processing_time > 0 else 0}
                }

            except HTTPException:
                raise
            except Exception as e:
                # Clean up file on error
                try:
                    file_path.unlink()
                except:
                    pass
                raise HTTPException(
                    status_code=500, detail=f"Upload failed: {e}")

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Upload endpoint failed: {e}")
            raise HTTPException(status_code=503, detail="System not ready")

    @app.post("/api/videos/download-youtube", tags=["videos"])
    async def download_youtube_video(request: dict):
        """Download and index a video from YouTube"""
        try:
            system = get_video_search_system()

            url = request.get('url', '').strip()
            quality = request.get('quality', 'best')
            config = request.get('config', {})

            if not url:
                raise HTTPException(status_code=400, detail="No URL provided")

            # Basic YouTube URL validation
            if 'youtube.com/watch' not in url and 'youtu.be/' not in url:
                raise HTTPException(
                    status_code=400, detail="Invalid YouTube URL")

            start_time = time.time()

            try:
                # Import yt-dlp for YouTube downloads
                import yt_dlp
                import yt_dlp

                # Generate unique filename
                video_id = str(uuid.uuid4())
                temp_filename = f"{video_id}_%(title)s.%(ext)s"
                output_path = UPLOAD_FOLDER / temp_filename

                # Configure yt-dlp options
                ydl_opts = {
                    'format': _get_youtube_format(quality),
                    'outtmpl': str(output_path),
                    'restrictfilenames': True,
                    'no_warnings': True,
                    'extractaudio': False,
                    'audioformat': 'mp3',
                    'embed_subs': False,
                    'writesubtitles': False,
                    'writeautomaticsub': False,
                }

                # Download the video
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    # Get video info first
                    info = ydl.extract_info(url, download=False)
                    video_title = info.get('title', 'Unknown')

                    logger.info(f"📺 Downloading: {video_title}")

                    # Download the video
                    ydl.download([url])

                # Find the downloaded file (yt-dlp changes the filename)
                downloaded_files = list(UPLOAD_FOLDER.glob(f"{video_id}_*"))
                if not downloaded_files:
                    raise HTTPException(
                        status_code=500, detail="Download completed but file not found")

                video_path = downloaded_files[0]
                logger.info(f"📁 Video downloaded: {video_path}")

                # Process video using the system with config
                frames_before = len(system.index.embeddings)

                # Apply configuration if provided
                if config:
                    # Temporarily update system config for this processing
                    temp_config = system.config or type('Config', (), {})()
                    for key, value in config.items():
                        setattr(temp_config, key, value)
                    system._process_single_video(video_path, temp_config)
                else:
                    system._process_single_video(video_path, system.config)

                frames_after = len(system.index.embeddings)

                # Update hash and save cache
                system.index.video_hashes[video_path.name] = system.processor.get_video_hash(
                    video_path)
                system.index.save_to_disk(system.cache_path)

                processing_time = time.time() - start_time
                frames_indexed = frames_after - frames_before

                return {
                    "video_id": video_id,
                    "status": "success",
                    "title": video_title,
                    "filename": video_path.name,
                    "frames_indexed": frames_indexed,
                    "processing_time": processing_time,
                    "quality": quality,
                    "url": url,
                    "performance": {
                        "frames_per_second": frames_indexed / processing_time if processing_time > 0 else 0
                    }
                }

            except ImportError:
                raise HTTPException(
                    status_code=500,
                    detail="yt-dlp not installed. Install with: pip install yt-dlp"
                )
            except Exception as e:
                # Clean up any partial downloads
                try:
                    for file in UPLOAD_FOLDER.glob(f"{video_id}_*"):
                        file.unlink()
                except:
                    pass
                raise HTTPException(
                    status_code=500, detail=f"YouTube download failed: {str(e)}")

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"YouTube download endpoint failed: {e}")
            raise HTTPException(status_code=503, detail="System not ready")

    @app.post("/api/search", tags=["search"])
    async def search_videos(request: SearchRequest):
        """Search for similar videos using semantic text search"""
        try:
            system = get_video_search_system()

            query = request.query.strip()
            if not query:
                raise HTTPException(
                    status_code=400, detail="No query provided")

            start_time = time.time()

            # Use the simple search method
            results = system.search(query, k=request.k)

            search_time_ms = (time.time() - start_time) * 1000

            return {
                "results": results,
                "search_time_ms": search_time_ms,
                "from_cache": request.use_cache,
                "query_id": str(uuid.uuid4()),
                "performance": {"results_count": len(results)}
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise HTTPException(status_code=500, detail=f"Search failed: {e}")

    @app.post("/api/search/batch")
    async def batch_search(request: BatchSearchRequest):
        """Process multiple search queries in parallel"""
        try:
            system = get_video_search_system()

            results = []
            for query in request.queries:
                search_results = system.search(query, k=request.k)
                results.append({
                    "query": query,
                    "results": search_results,
                    "count": len(search_results)
                })

            return {
                "results": results,
                "query_count": len(request.queries),
                "total_results": sum(len(r["results"]) for r in results)
            }

        except Exception as e:
            logger.error(f"Batch search failed: {e}")
            raise HTTPException(
                status_code=500, detail=f"Batch search failed: {e}")

    @app.get("/api/videos/{video_id}")
    async def get_video_info(video_id: str):
        """Get information about a specific video"""
        try:
            system = get_video_search_system()

            # Find video in metadata
            for meta in system.index.metadata:
                if video_id in meta['video_name']:
                    video_path = Path("videos") / meta['video_name']
                    return {
                        "video_id": video_id,
                        "filename": meta['video_name'],
                        "exists": video_path.exists(),
                        "frame_count": len([m for m in system.index.metadata if m['video_name'] == meta['video_name']])
                    }

            raise HTTPException(status_code=404, detail="Video not found")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=503, detail="System not ready")

    @app.get("/api/videos")
    async def list_videos(limit: int = 100, offset: int = 0):
        """List indexed videos with pagination"""
        try:
            system = get_video_search_system()

            if limit > 1000:
                raise HTTPException(
                    status_code=400, detail="Limit too large (max 1000)")

            # Get unique videos from metadata
            unique_videos = {}
            for meta in system.index.metadata:
                video_name = meta['video_name']
                if video_name not in unique_videos:
                    video_path = Path("videos") / video_name
                    unique_videos[video_name] = {
                        "filename": video_name,
                        "video_id": video_name.replace('.mp4', '').replace('.', '_'),
                        "frame_count": 0,
                        "size": video_path.stat().st_size if video_path.exists() else 0,
                        "processed_at": time.time()
                    }
                unique_videos[video_name]["frame_count"] += 1

            videos = list(unique_videos.values())[offset:offset + limit]

            return {
                "videos": videos,
                "count": len(videos),
                "limit": limit,
                "offset": offset
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=503, detail="System not ready")

    @app.delete("/api/videos/{video_id}")
    async def delete_video(video_id: str):
        """Delete a video from the index"""
        try:
            system = get_video_search_system()

            # Find video by video_id (which is derived from filename)
            target_video_name = None

            # Look through metadata to find the actual video filename
            for meta in system.index.metadata:
                video_name = meta['video_name']
                # The video_id is generated as video_name.replace('.mp4', '').replace('.', '_')
                # So we need to reverse this to find the matching video
                if video_name.replace('.mp4', '').replace('.mov', '').replace('.avi', '').replace('.mkv', '').replace('.', '_') == video_id:
                    target_video_name = video_name
                    break

            if not target_video_name:
                # Fallback: try to find by partial filename match
                video_files = list(Path("videos").glob(f"*{video_id}*"))
                if not video_files:
                    raise HTTPException(
                        status_code=404, detail="Video not found")
                target_video_name = video_files[0].name

            # Remove the video file
            video_path = Path("videos") / target_video_name
            success = False

            if video_path.exists():
                video_path.unlink()
                success = True
                logger.info(f"🗑️ Deleted video file: {video_path}")

                # Remove from cache hash
                if target_video_name in system.index.video_hashes:
                    del system.index.video_hashes[target_video_name]

                # Find indices of embeddings to remove BEFORE modifying metadata
                indices_to_remove = []
                for i, meta in enumerate(system.index.metadata):
                    if meta['video_name'] == target_video_name:
                        indices_to_remove.append(i)

                # Remove embeddings in reverse order to maintain indices
                for idx in reversed(indices_to_remove):
                    if idx < len(system.index.embeddings):
                        system.index.embeddings.pop(idx)

                # Remove all metadata entries for this video
                system.index.metadata = [
                    meta for meta in system.index.metadata
                    if meta['video_name'] != target_video_name
                ]

            if not success:
                raise HTTPException(
                    status_code=404, detail="Video not found or deletion failed")

            # Save updated cache
            system.index.save_to_disk(system.cache_path)

            return {"status": "deleted", "video_id": video_id, "filename": target_video_name}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Delete video failed: {e}")
            raise HTTPException(status_code=503, detail="System not ready")

    @app.post("/api/index/save")
    async def save_index(filepath: str):
        """Save index to disk"""
        try:
            system = get_video_search_system()

            success = system.index.save_to_disk(Path(filepath))
            if not success:
                raise HTTPException(
                    status_code=500, detail="Failed to save index")

            return {"status": "saved", "filepath": filepath}

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Save failed: {e}")

    @app.post("/api/index/load")
    async def load_index(filepath: str):
        """Load index from disk"""
        try:
            system = get_video_search_system()

            success = system.index.load_from_disk(Path(filepath))
            if not success:
                raise HTTPException(
                    status_code=500, detail="Failed to load index")

            return {"status": "loaded", "filepath": filepath}

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Load failed: {e}")

    @app.get("/videos/{filename}")
    async def serve_video(filename: str):
        """Serve video files for playback"""
        try:
            logger.info(f"Serving video: {filename}")
            video_path = UPLOAD_FOLDER / filename

            if not video_path.exists():
                logger.error(f"Video not found: {video_path}")
                raise HTTPException(
                    status_code=404, detail=f"Video not found: {filename}")

            logger.info(f"Video found at: {video_path}")

            from fastapi.responses import FileResponse
            return FileResponse(
                path=str(video_path),
                media_type="video/mp4",
                headers={"Accept-Ranges": "bytes"}
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error serving video {filename}: {e}")
            raise HTTPException(
                status_code=500, detail=f"Error serving video: {e}")

    # Legacy endpoints for compatibility with existing Flask frontend
    @app.post("/search")
    async def search_legacy(request: dict):
        """Legacy search endpoint for compatibility"""
        search_request = SearchRequest(
            query=request.get('query', ''),
            k=request.get('k', 5),
            use_cache=request.get('use_cache', True)
        )
        result = await search_videos(search_request)

        # Convert to legacy format
        return {
            'success': True,
            'results': result.results,
            'search_time': result.search_time_ms / 1000,
            'query': search_request.query
        }

    @app.get("/videos")
    async def list_videos_legacy():
        """Legacy videos endpoint"""
        try:
            result = await list_videos()
            return {"videos": [
                {
                    "name": video.get("filename", "unknown"),
                    "size": video.get("size", 0),
                    "modified": video.get("processed_at", 0)
                }
                for video in result["videos"]
            ]}
        except Exception as e:
            return {"error": str(e)}

    # Configuration Endpoints
    @app.get("/api/config", response_model=ConfigurationResponse, tags=["configuration"])
    async def get_configuration():
        """Get current system configuration"""
        try:
            global current_config
            if current_config is None:
                current_config = load_config_from_file()
            return ConfigurationResponse(
                success=True,
                config=current_config,
                message="Configuration retrieved successfully"
            )
        except Exception as e:
            logger.error(f"Failed to get configuration: {e}")
            return ConfigurationResponse(
                success=False,
                message=f"Failed to get configuration: {str(e)}"
            )

    @app.post("/api/config", response_model=ConfigurationResponse)
    async def update_configuration(config: ConfigurationModel):
        """Update configuration"""
        global current_config
        try:
            current_config = config
            success = save_config_to_file(config)

            # Update logging level if changed
            if config.log_level in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
                logging.getLogger().setLevel(getattr(logging, config.log_level))

            return ConfigurationResponse(
                success=success,
                config=current_config,
                message="Configuration updated successfully" if success else "Failed to save configuration"
            )
        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            return ConfigurationResponse(
                success=False,
                message=f"Failed to update configuration: {str(e)}"
            )

    @app.post("/api/config/reset")
    async def reset_configuration():
        """Reset configuration to defaults"""
        global current_config
        try:
            current_config = ConfigurationModel()
            success = save_config_to_file(current_config)

            return ConfigurationResponse(
                success=success,
                config=current_config,
                message="Configuration reset to defaults" if success else "Failed to save default configuration"
            )
        except Exception as e:
            logger.error(f"Failed to reset configuration: {e}")
            return ConfigurationResponse(
                success=False,
                message=f"Failed to reset configuration: {str(e)}"
            )

    # Cache Management Endpoints
    @app.get("/api/cache/stats", tags=["cache"])
    async def get_cache_stats_endpoint():
        """Get detailed cache statistics"""
        try:
            stats = get_cache_stats()
            # Return flat format that frontend expects
            return {
                "success": True,
                "embeddings": stats.embeddings_count,
                "videos": stats.videos_count,
                "size": stats.cache_size_mb * 1024 * 1024,  # Convert back to bytes
                "last_updated": int(time.mktime(time.strptime(stats.last_updated, "%Y-%m-%d %H:%M:%S"))) if stats.last_updated != "Never" and stats.last_updated != "Error" else None,
                "cache_file_exists": stats.cache_file_exists,
                "video_hashes_count": stats.video_hashes_count
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {
                "success": False,
                "embeddings": 0,
                "videos": 0,
                "size": 0,
                "last_updated": None,
                "cache_file_exists": False,
                "video_hashes_count": 0
            }

    @app.post("/api/cache/rebuild")
    async def rebuild_cache():
        """Rebuild the entire cache"""
        try:
            global current_config, _video_system

            # Load current configuration
            if current_config is None:
                current_config = load_config_from_file()

            # Clear existing cache
            old_system = get_video_search_system()
            if old_system:
                old_system.index.embeddings = []
                old_system.index.metadata = []
                old_system.index.video_hashes = {}

            # Create a new system with current configuration
            _video_system = VideoSearchSystem("videos", current_config)

            # Rebuild cache by reprocessing all videos
            _video_system.startup()

            # Save the new cache
            _video_system.index.save_to_disk(_video_system.cache_path)

            stats = get_cache_stats()
            return CacheResponse(
                success=True,
                stats=stats,
                message=f"Cache rebuilt successfully with config: max_frames={current_config.max_frames}, use_clip={current_config.use_clip}"
            )
        except Exception as e:
            logger.error(f"Failed to rebuild cache: {e}")
            return CacheResponse(
                success=False,
                message=f"Failed to rebuild cache: {str(e)}"
            )

    @app.post("/api/cache/clear")
    async def clear_cache():
        """Clear the cache completely"""
        try:
            system = get_video_search_system()
            if not system:
                raise HTTPException(status_code=503, detail="System not ready")

            # Clear in-memory cache
            system.index.embeddings = []
            system.index.metadata = []
            system.index.video_hashes = {}

            # Remove cache file
            cache_path = Path("videos/video_search_cache.pkl")
            if cache_path.exists():
                cache_path.unlink()

            stats = get_cache_stats()
            return CacheResponse(
                success=True,
                stats=stats,
                message="Cache cleared successfully"
            )
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return CacheResponse(
                success=False,
                message=f"Failed to clear cache: {str(e)}"
            )

    @app.get("/api/cache/health")
    async def cache_health_check():
        """Perform cache health check"""
        try:
            result = perform_cache_health_check()
            return result
        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
            return CacheHealthResult(
                success=False,
                issues=[f"Health check failed: {str(e)}"],
                recommendations=["Check system logs and restart if necessary"],
                total_checks=0,
                passed_checks=0
            )

    @app.post("/api/cache/export")
    async def export_cache():
        """Export cache to a downloadable file"""
        try:
            system = get_video_search_system()
            if not system:
                raise HTTPException(status_code=503, detail="System not ready")

            cache_path = Path("videos/video_search_cache.pkl")
            if not cache_path.exists():
                raise HTTPException(
                    status_code=404, detail="Cache file not found")

            return FileResponse(
                path=str(cache_path),
                filename="video_search_cache_export.pkl",
                media_type="application/octet-stream"
            )
        except Exception as e:
            logger.error(f"Failed to export cache: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to export cache: {str(e)}")

    @app.post("/api/cache/import")
    async def import_cache(file: UploadFile = File(...)):
        """Import cache from uploaded file"""
        try:
            system = get_video_search_system()
            if not system:
                raise HTTPException(status_code=503, detail="System not ready")

            # Validate file type
            if not file.filename.endswith('.pkl'):
                raise HTTPException(
                    status_code=400, detail="Invalid file type. Must be a .pkl file")

            # Save uploaded cache file
            cache_path = Path("videos/video_search_cache.pkl")

            with open(cache_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

            # Reload the cache
            system.index.load_from_disk(cache_path)

            stats = get_cache_stats()
            return CacheResponse(
                success=True,
                stats=stats,
                message="Cache imported successfully"
            )
        except Exception as e:
            logger.error(f"Failed to import cache: {e}")
            return CacheResponse(
                success=False,
                message=f"Failed to import cache: {str(e)}"
            )

    return app
