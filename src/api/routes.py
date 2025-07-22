"""
FastAPI REST API for Video Search System
High-performance async API with sub-second response times
"""

import asyncio
import time
import uuid
from typing import List, Dict, Any, Optional, Union
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import numpy as np
from PIL import Image
import io
import base64

from video_search_system import VideoSearchSystem
from utils.config import load_config, get_default_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global system instance
video_search_system: Optional[VideoSearchSystem] = None

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

# Create FastAPI app
app = FastAPI(
    title="Video Search API",
    description="High-performance video search with HNSW index",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize the video search system"""
    global video_search_system
    
    try:
        logger.info("Starting Video Search API...")
        
        # Load configuration
        config_path = Path(__file__).parent.parent.parent / 'config' / 'default.yaml'
        
        if config_path.exists():
            video_search_system = VideoSearchSystem(str(config_path))
        else:
            logger.warning("Config file not found, using defaults")
            video_search_system = VideoSearchSystem()
        
        # Complete system startup
        await video_search_system.startup()
        
        logger.info("Video Search API started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start Video Search API: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global video_search_system
    
    if video_search_system:
        await video_search_system.shutdown()
        logger.info("Video Search API shutdown complete")

# API Routes

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    System health check endpoint
    """
    if not video_search_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    health_data = await video_search_system.health_check()
    return HealthResponse(**health_data)

@app.get("/stats", response_model=SystemStatsResponse)
async def get_system_stats():
    """
    Get comprehensive system statistics
    """
    if not video_search_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    stats = video_search_system.get_system_stats()
    return SystemStatsResponse(**stats)

@app.get("/metrics")
async def get_metrics():
    """
    Get Prometheus-compatible metrics
    """
    if not video_search_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    metrics_text = video_search_system.metrics.export_prometheus()
    return JSONResponse(
        content=metrics_text,
        media_type="text/plain"
    )

@app.post("/videos/upload", response_model=VideoUploadResponse)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    video_id: Optional[str] = Form(None)
):
    """
    Upload and index a video file
    Processing happens in background for large files
    """
    if not video_search_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file size (5GB limit)
    MAX_FILE_SIZE = 5 * 1024 * 1024 * 1024
    if hasattr(file, 'size') and file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
    
    # Generate video ID if not provided
    if not video_id:
        video_id = str(uuid.uuid4())
    
    try:
        # Save uploaded file temporarily
        temp_dir = Path("/tmp/video_search")
        temp_dir.mkdir(exist_ok=True)
        
        temp_file_path = temp_dir / f"{video_id}_{file.filename}"
        
        # Write file
        with open(temp_file_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)
        
        logger.info(f"File uploaded: {temp_file_path}")
        
        # Process video (this may take time)
        start_time = time.time()
        
        result = await video_search_system.add_video(str(temp_file_path), video_id)
        
        # Clean up temp file
        try:
            temp_file_path.unlink()
        except Exception:
            pass
        
        if result['status'] == 'error':
            raise HTTPException(status_code=500, detail=result['error'])
        
        return VideoUploadResponse(**result)
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=SearchResponse)
async def search_videos(request: SearchRequest):
    """
    Search for similar videos
    Supports both text and image queries
    """
    if not video_search_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        query = request.query
        
        # Check if query is a base64 encoded image
        if query.startswith('data:image/'):
            # Decode base64 image
            try:
                header, data = query.split(',', 1)
                image_data = base64.b64decode(data)
                image = Image.open(io.BytesIO(image_data))
                
                # Convert to numpy array
                query_array = np.array(image)
                
                # Search with image
                results = await video_search_system.search(
                    query_array, 
                    k=request.k,
                    use_cache=request.use_cache
                )
                
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")
        else:
            # Text search
            results = await video_search_system.search(
                query,
                k=request.k,
                use_cache=request.use_cache
            )
        
        return SearchResponse(**results)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/batch")
async def batch_search(request: BatchSearchRequest):
    """
    Process multiple search queries in parallel
    """
    if not video_search_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        results = await video_search_system.search_batch(request.queries, request.k)
        
        return {
            "results": results,
            "query_count": len(request.queries),
            "total_results": sum(len(r['results']) for r in results)
        }
        
    except Exception as e:
        logger.error(f"Batch search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/videos/{video_id}")
async def get_video_info(video_id: str):
    """
    Get information about a specific video
    """
    if not video_search_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    video_info = video_search_system.get_video_info(video_id)
    
    if not video_info:
        raise HTTPException(status_code=404, detail="Video not found")
    
    return video_info

@app.get("/videos")
async def list_videos(limit: int = 100, offset: int = 0):
    """
    List indexed videos with pagination
    """
    if not video_search_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if limit > 1000:
        raise HTTPException(status_code=400, detail="Limit too large (max 1000)")
    
    videos = video_search_system.list_videos(limit, offset)
    
    return {
        "videos": videos,
        "count": len(videos),
        "limit": limit,
        "offset": offset
    }

@app.delete("/videos/{video_id}")
async def delete_video(video_id: str):
    """
    Delete a video from the index
    """
    if not video_search_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    success = video_search_system.delete_video(video_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Video not found or deletion failed")
    
    return {"status": "deleted", "video_id": video_id}

@app.post("/index/save")
async def save_index(filepath: str):
    """
    Save index to disk
    """
    if not video_search_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        success = video_search_system.save_index(filepath)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save index")
        
        return {"status": "saved", "filepath": filepath}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index/load")
async def load_index(filepath: str):
    """
    Load index from disk
    """
    if not video_search_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        success = video_search_system.load_index(filepath)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to load index")
        
        return {"status": "loaded", "filepath": filepath}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cache/clear")
async def clear_cache():
    """
    Clear all cached data
    """
    if not video_search_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    video_search_system.cache.clear()
    
    return {"status": "cleared"}

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler
    """
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Main entry point
def create_app(config_path: Optional[str] = None) -> FastAPI:
    """
    Application factory
    """
    return app

def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 1,
    config_path: Optional[str] = None
):
    """
    Run the API server
    """
    uvicorn.run(
        "src.api.routes:app",
        host=host,
        port=port,
        workers=workers,
        reload=False,
        access_log=True,
        log_level="info"
    )

if __name__ == "__main__":
    run_server()
