# 🎬 Video Search System - Clean & Working

A semantic video search system with CLIP embeddings and beautiful UI.

## ✅ What Works
- **Fast startup** - Loads 1200+ embeddings instantly from cache
- **Semantic search** - CLIP-powered understanding of video content  
- **Beautiful UI** - Modern web interface with frame previews
- **File management** - Upload, delete, and organize videos
- **Frame extraction** - View exact video frames at search timestamps

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the system
python clean_server.py

# Open browser
open http://localhost:5001
```

## 📁 File Structure

```
video-search/
├── clean_server.py              # Main FastAPI server
├── video_search_overhaul.py     # Core video processing
├── static/index.html             # Web interface
├── videos/                       # Video files
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## 🔍 Usage

1. **Search Videos**: Enter semantic queries like "phone apps", "car driving", "goal celebration"
2. **Upload Videos**: Drag & drop MP4/AVI/MOV/MKV files for automatic processing
3. **View Results**: Click search results to see video frames at exact timestamps
4. **Manage Library**: View, organize, and delete indexed videos

## ⚡ Performance

- **Search Speed**: Sub-second semantic search across thousands of frames
- **Cache System**: Reliable persistence - restarts are instant
- **Memory Efficient**: Smart video sampling and embedding storage
- **CLIP Integration**: State-of-the-art semantic understanding

## 🛠️ API Endpoints

- `GET /` - Web interface
- `POST /api/search` - Semantic video search
- `GET /api/videos` - List indexed videos
- `POST /api/videos/upload` - Upload new videos
- `DELETE /api/videos/{id}` - Remove videos
- `GET /api/video/{id}/frame?timestamp=X` - Extract frame
- `GET /api/stats` - System statistics

## 🎯 Features

- **CLIP Embeddings** - Semantic understanding of video content
- **Smart Sampling** - Efficient frame extraction (300 frames/video)
- **Cache System** - Reliable state persistence
- **Web UI** - Beautiful, responsive interface
- **File Upload** - Drag & drop with progress tracking
- **Frame Preview** - Base64 image extraction
- **Video Streaming** - Direct video playback with timestamp jumping

Built with FastAPI, CLIP, OpenCV, and modern web technologies.
