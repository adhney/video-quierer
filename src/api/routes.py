"""
🚀 FastAPI Routes - Modular Video Search API
==========================================
Clean API endpoints using our optimized video search system
"""

import asyncio
import time
import uuid
import os
import shutil
from typing import List, Dict, Any, Optional, Union
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import numpy as np
from PIL import Image
import io
import base64

# Import our bridge system
from video_search_system_bridge import get_video_search_system

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


# Configuration
UPLOAD_FOLDER = Path("videos")
MAX_FILE_SIZE = 1024 * 1024 * 1024  # 1GB


def create_api_routes(app: FastAPI):
    """Create all API routes"""

    @app.get("/api/health", response_model=HealthResponse)
    async def health_check():
        """System health check endpoint"""
        try:
            system = get_video_search_system()
            health_data = await system.health_check()
            return HealthResponse(**health_data)
        except Exception as e:
            raise HTTPException(
                status_code=503, detail=f"System not ready: {e}")

    @app.get("/api/stats", response_model=SystemStatsResponse)
    async def get_system_stats():
        """Get comprehensive system statistics"""
        try:
            system = get_video_search_system()
            stats = system.get_system_stats()
            return SystemStatsResponse(**stats)
        except Exception as e:
            raise HTTPException(
                status_code=503, detail=f"System not ready: {e}")

    @app.post("/api/videos/upload", response_model=VideoUploadResponse)
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

            try:
                with open(file_path, "wb") as buffer:
                    content = await file.read()
                    if len(content) > MAX_FILE_SIZE:
                        raise HTTPException(
                            status_code=413, detail="File too large (max 1GB)")
                    buffer.write(content)

                logger.info(f"📁 File uploaded: {file_path}")

                # Process video
                result = await system.add_video(str(file_path), video_id)

                if result['status'] == 'error':
                    # Clean up file on error
                    try:
                        file_path.unlink()
                    except:
                        pass
                    raise HTTPException(
                        status_code=500, detail=result['error'])

                return VideoUploadResponse(**result)

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

    @app.post("/api/search", response_model=SearchResponse)
    async def search_videos(request: SearchRequest):
        """Search for similar videos"""
        try:
            system = get_video_search_system()

            query = request.query.strip()
            if not query:
                raise HTTPException(
                    status_code=400, detail="No query provided")

            # Check if query is a base64 encoded image
            if query.startswith('data:image/'):
                try:
                    header, data = query.split(',', 1)
                    image_data = base64.b64decode(data)
                    image = Image.open(io.BytesIO(image_data))

                    # Convert to numpy array
                    query_array = np.array(image)

                    # Search with image
                    results = await system.search(
                        query_array,
                        k=request.k,
                        use_cache=request.use_cache
                    )

                except Exception as e:
                    raise HTTPException(
                        status_code=400, detail=f"Invalid image data: {e}")
            else:
                # Text search
                results = await system.search(
                    query,
                    k=request.k,
                    use_cache=request.use_cache
                )

            return SearchResponse(**results)

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

            results = await system.search_batch(request.queries, request.k)

            return {
                "results": results,
                "query_count": len(request.queries),
                "total_results": sum(len(r.get('results', [])) for r in results)
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
            video_info = system.get_video_info(video_id)

            if not video_info:
                raise HTTPException(status_code=404, detail="Video not found")

            return video_info
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

            videos = system.list_videos(limit, offset)

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

            success = system.delete_video(video_id)

            if not success:
                raise HTTPException(
                    status_code=404, detail="Video not found or deletion failed")

            return {"status": "deleted", "video_id": video_id}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=503, detail="System not ready")

    @app.post("/api/index/save")
    async def save_index(filepath: str):
        """Save index to disk"""
        try:
            system = get_video_search_system()

            success = system.save_index(filepath)
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

            success = system.load_index(filepath)
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
                raise HTTPException(status_code=404, detail=f"Video not found: {filename}")
            
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
            raise HTTPException(status_code=500, detail=f"Error serving video: {e}")

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

    return app
