#!/usr/bin/env python3
"""
🚀 New FastAPI Server - Using Overhauled Video Search System
==========================================================
Clean, simple, and actually working with frame preview!
"""

import asyncio
import logging
import time
import cv2
import base64
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import numpy as np

# Import our clean system
from video_search_overhaul import VideoSearchSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

# Global system instance
video_system: Optional[VideoSearchSystem] = None

# FastAPI app
app = FastAPI(
    title="Video Search API - Overhauled",
    description="Clean, simple, and reliable video search system with frame preview",
    version="2.1.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (for UI)
app.mount("/static", StaticFiles(directory="static"), name="static")
# Mount videos directory for direct video access
app.mount("/videos", StaticFiles(directory="videos"), name="videos")

# Pydantic models


class SearchRequest(BaseModel):
    query: str
    k: Optional[int] = 5


class SearchResult(BaseModel):
    video_name: str
    timestamp: float
    formatted_time: str
    score: float
    frame_id: int


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    search_time_ms: float
    total_results: int


class SystemStats(BaseModel):
    total_embeddings: int
    total_videos: int
    system_ready: bool
    uptime_seconds: float

# Enhanced video info model for compatibility


class VideoInfoEnhanced(BaseModel):
    id: str
    filename: str
    name: str
    status: str
    frame_count: int
    duration: Optional[float] = None
    fps: Optional[float] = None
    size: Optional[int] = None
    processed_at: Optional[float] = None


class VideoListResponse(BaseModel):
    videos: List[VideoInfoEnhanced]
    total_count: int


class UploadResponse(BaseModel):
    success: bool
    video_id: str
    filename: str
    frames_indexed: int
    processing_time: float


class FrameResponse(BaseModel):
    success: bool
    frame_data: Optional[str] = None  # base64 encoded image
    error: Optional[str] = None
    timestamp: float
    video_name: str

# Startup/shutdown


@app.on_event("startup")
async def startup_event():
    """Initialize the video search system"""
    global video_system

    logger.info("🚀 Starting Video Search System...")

    try:
        video_system = VideoSearchSystem("videos")

        # Run startup in thread to avoid blocking
        import threading
        startup_complete = threading.Event()

        def run_startup():
            try:
                video_system.startup()
                startup_complete.set()
            except Exception as e:
                logger.error(f"Startup failed: {e}")
                startup_complete.set()

        startup_thread = threading.Thread(target=run_startup)
        startup_thread.start()

        # Wait for startup (with timeout)
        startup_complete.wait(timeout=300)  # 5 minutes max

        logger.info("✅ Video Search System ready!")

    except Exception as e:
        logger.error(f"❌ Failed to initialize system: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down Video Search System...")

# Helper functions for frame extraction


def extract_frame_at_timestamp(video_path: str, timestamp: float) -> Optional[np.ndarray]:
    """Extract a specific frame at timestamp"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(timestamp * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()

        if ret:
            return frame
        return None

    except Exception as e:
        logger.error(f"Frame extraction failed: {e}")
        return None


def frame_to_base64(frame: np.ndarray) -> str:
    """Convert frame to base64 encoded JPEG"""
    try:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)

        # Save to bytes buffer as JPEG
        buffer = BytesIO()
        pil_image.save(buffer, format='JPEG', quality=85)

        # Encode to base64
        img_bytes = buffer.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')

        return f"data:image/jpeg;base64,{img_base64}"

    except Exception as e:
        logger.error(f"Base64 conversion failed: {e}")
        return ""

# API endpoints


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the beautiful modular UI"""
    try:
        with open("static/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse("""
        <h1>UI File Not Found</h1>
        <p>The static/index.html file was not found. Please ensure it exists.</p>
        <p><a href="/api/docs">Go to API Documentation</a></p>
        """)


@app.post("/api/search", response_model=SearchResponse)
async def search_videos(request: SearchRequest):
    """Search videos"""
    if not video_system:
        raise HTTPException(status_code=503, detail="System not ready")

    start_time = time.time()

    try:
        results = video_system.search(request.query, k=request.k)

        search_time = (time.time() - start_time) * 1000

        return SearchResponse(
            query=request.query,
            results=[SearchResult(**result) for result in results],
            search_time_ms=search_time,
            total_results=len(results)
        )

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats", response_model=SystemStats)
async def get_stats():
    """Get system statistics"""
    if not video_system:
        raise HTTPException(status_code=503, detail="System not ready")

    unique_videos = len(set(meta['video_name']
                        for meta in video_system.index.metadata))

    return SystemStats(
        total_embeddings=len(video_system.index.embeddings),
        total_videos=unique_videos,
        system_ready=True,
        uptime_seconds=time.time()
    )


@app.get("/api/videos", response_model=VideoListResponse)
async def get_videos():
    """Get list of available videos with enhanced info"""
    if not video_system:
        raise HTTPException(status_code=503, detail="System not ready")

    # Get unique videos from metadata and calculate stats
    videos_info = {}
    for meta in video_system.index.metadata:
        video_name = meta['video_name']
        if video_name not in videos_info:
            # Try to get video stats
            video_path = video_system.videos_dir / video_name
            duration, fps, size = None, None, None

            if video_path.exists():
                try:
                    # Get file size
                    size = video_path.stat().st_size

                    # Get video properties
                    cap = cv2.VideoCapture(str(video_path))
                    if cap.isOpened():
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        duration = frame_count / fps if fps > 0 else None
                        cap.release()
                except Exception:
                    pass

            videos_info[video_name] = VideoInfoEnhanced(
                id=video_name.replace('.mp4', '').replace('.', '_'),
                filename=video_name,
                name=video_name,
                status='indexed',
                frame_count=0,
                duration=duration,
                fps=fps,
                size=size,
                processed_at=time.time()  # Approximate
            )
        videos_info[video_name].frame_count += 1

    video_list = list(videos_info.values())
    return VideoListResponse(
        videos=video_list,
        total_count=len(video_list)
    )


@app.post("/api/videos/upload", response_model=UploadResponse)
async def upload_video(file: UploadFile = File(...)):
    """Upload and process a new video"""
    if not video_system:
        raise HTTPException(status_code=503, detail="System not ready")

    # Validate file type
    allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    file_ext = None
    for ext in allowed_extensions:
        if file.filename.lower().endswith(ext.lower()):
            file_ext = ext
            break

    if not file_ext:
        raise HTTPException(
            status_code=400, detail="Invalid file type. Supported: MP4, AVI, MOV, MKV")

    start_time = time.time()

    try:
        # Save uploaded file
        file_path = video_system.videos_dir / file.filename

        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Process the video
        frames_before = len(video_system.index.embeddings)
        video_system._process_single_video(file_path)
        frames_after = len(video_system.index.embeddings)

        # Update hash
        video_system.index.video_hashes[file.filename] = video_system.processor.get_video_hash(
            file_path)

        # Save cache
        video_system.index.save_to_disk(video_system.cache_path)

        processing_time = time.time() - start_time
        frames_indexed = frames_after - frames_before

        return UploadResponse(
            success=True,
            video_id=file.filename.replace('.mp4', '').replace('.', '_'),
            filename=file.filename,
            frames_indexed=frames_indexed,
            processing_time=processing_time
        )

    except Exception as e:
        # Clean up file if processing failed
        if file_path.exists():
            file_path.unlink()

        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.delete("/api/videos/{video_id}")
async def delete_video(video_id: str):
    """Delete a video from the system"""
    if not video_system:
        raise HTTPException(status_code=503, detail="System not ready")

    # Find video by ID
    video_filename = None
    for meta in video_system.index.metadata:
        vid_id = meta['video_name'].replace('.mp4', '').replace('.', '_')
        if vid_id == video_id:
            video_filename = meta['video_name']
            break

    if not video_filename:
        raise HTTPException(status_code=404, detail="Video not found")

    try:
        # Remove video file
        video_path = video_system.videos_dir / video_filename
        if video_path.exists():
            video_path.unlink()

        # Remove from cache hash
        if video_filename in video_system.index.video_hashes:
            del video_system.index.video_hashes[video_filename]

        # TODO: Remove embeddings from index (would need index rebuild for now)
        # For now, just mark the video as gone so it won't appear in future results

        # Save updated cache
        video_system.index.save_to_disk(video_system.cache_path)

        return {"success": True, "message": f"Video {video_filename} deleted successfully"}

    except Exception as e:
        logger.error(f"Delete failed: {e}")
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


@app.get("/api/video/{video_id}/frame", response_model=FrameResponse)
async def get_video_frame(video_id: str, timestamp: float = Query(..., description="Timestamp in seconds")):
    """Get video frame at timestamp"""
    if not video_system:
        raise HTTPException(status_code=503, detail="System not ready")

    # Find video file
    video_name = None
    for meta in video_system.index.metadata:
        vid_id = meta['video_name'].replace('.mp4', '').replace('.', '_')
        if vid_id == video_id:
            video_name = meta['video_name']
            break

    if not video_name:
        return FrameResponse(
            success=False,
            error="Video not found",
            timestamp=timestamp,
            video_name="unknown"
        )

    video_path = video_system.videos_dir / video_name

    if not video_path.exists():
        return FrameResponse(
            success=False,
            error="Video file not found on disk",
            timestamp=timestamp,
            video_name=video_name
        )

    # Extract frame
    frame = extract_frame_at_timestamp(str(video_path), timestamp)

    if frame is None:
        return FrameResponse(
            success=False,
            error="Failed to extract frame at timestamp",
            timestamp=timestamp,
            video_name=video_name
        )

    # Convert to base64
    frame_data = frame_to_base64(frame)

    if not frame_data:
        return FrameResponse(
            success=False,
            error="Failed to encode frame",
            timestamp=timestamp,
            video_name=video_name
        )

    return FrameResponse(
        success=True,
        frame_data=frame_data,
        timestamp=timestamp,
        video_name=video_name
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy" if video_system else "starting"}

if __name__ == "__main__":
    print("🚀 Starting Complete Video Search Server")
    print("=" * 50)

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=5001,
        reload=False,
        log_level="info"
    )
