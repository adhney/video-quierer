# 🎬 Video Search System - Advanced Semantic Search Platform

A high-performance semantic video search system with CLIP embeddings, YouTube integration, and enterprise-grade features.

## ✨ Key Features

### 🔍 **Advanced Search**

- **Semantic Understanding** - CLIP-powered content recognition
- **Sub-second Latency** - Optimized HNSW vector indexing
- **Multi-modal Queries** - Text and image search support
- **Intelligent Ranking** - Relevance-based result scoring

### 📺 **Video Management**

- **Multiple Upload Methods** - File upload, YouTube downloads
- **Format Support** - MP4, AVI, MOV, MKV, and more
- **Smart Processing** - Configurable frame sampling strategies
- **Auto-indexing** - Automatic feature extraction and indexing

### ⚙️ **Configuration System**

- **Sampling Modes** - Ultra-high, high, medium, low quality
- **Frame Limits** - Configurable max frames per video
- **CLIP Integration** - Toggle advanced AI processing
- **Performance Tuning** - Adjustable timeout and cache settings

### 🗄️ **Cache Management**

- **Multi-level Caching** - L1 memory + L2 persistent storage
- **Cache Operations** - Rebuild, clear, export, import
- **Health Monitoring** - Real-time cache statistics
- **Auto-persistence** - Reliable state management

### 🌐 **Modern Web Interface**

- **Responsive Design** - Works on desktop and mobile
- **Real-time Stats** - System health and performance metrics
- **Interactive UI** - Tabbed interface with drag & drop
- **Video Preview** - In-browser playback with timestamp navigation

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the system
python server.py

# Open browser
open http://localhost:8000
```

## 📁 Architecture

```
video-search/
├── server.py                    # Main FastAPI application
├── video_search_overhaul.py     # Core video processing engine
├── static/
│   └── index.html               # Modern web interface
├── src/
│   ├── api/
│   │   └── routes.py            # Modular API endpoints
│   ├── core/
│   │   ├── frame_extractor.py   # Smart frame sampling
│   │   └── feature_extractor.py # CLIP embedding extraction
│   ├── indexes/
│   │   └── hnsw.py             # High-performance vector index
│   ├── storage/
│   │   ├── cache.py            # Multi-level caching
│   │   └── simple_cache.py     # Lightweight cache implementation
│   ├── utils/
│   │   ├── config.py           # Configuration management
│   │   └── metrics.py          # Performance monitoring
│   └── video_search_system.py  # Main system orchestration
├── videos/                      # Video storage directory
├── config.json                  # System configuration
└── requirements.txt             # Python dependencies
```

## 🎯 Advanced Usage

### Configuration Management

```bash
# Access configuration via web UI
http://localhost:8000 → Configuration Tab

# Available settings:
- sampling_mode: ultra_high, high, medium, low
- max_frames: 50-1000 frames per video
- use_clip: Enable/disable CLIP processing
- enhanced_mode: Advanced processing features
- cache_search: Enable search result caching
```

### YouTube Integration

```bash
# Download and index YouTube videos
http://localhost:8000 → Upload Video Tab → YouTube URL

# Supported formats:
- YouTube videos and playlists
- Quality selection (best, worst, specific)
- Automatic processing and indexing
```

### Cache Operations

```bash
# Via web interface:
http://localhost:8000 → Cache Manager Tab

# Available operations:
- Rebuild: Reprocess all videos with current config
- Clear: Remove all cached data
- Export: Backup cache to file
- Import: Restore cache from backup
- Health: Check cache integrity
```

## ⚡ Performance

### Search Performance

- **Latency**: <100ms for most queries
- **Throughput**: 100+ searches/second
- **Scalability**: Handles 10,000+ videos efficiently
- **Memory Usage**: Optimized embedding storage

### Processing Performance

- **Frame Extraction**: 30-60 FPS processing
- **Feature Extraction**: GPU-accelerated CLIP
- **Indexing**: Batch processing for efficiency
- **Storage**: Compressed cache format

### Configuration Impact

| Sampling Mode | Frames/Video | Quality   | Speed     |
| ------------- | ------------ | --------- | --------- |
| Ultra High    | 500-1000     | Excellent | Slow      |
| High          | 200-500      | Very Good | Medium    |
| Medium        | 100-200      | Good      | Fast      |
| Low           | 50-100       | Basic     | Very Fast |

## 🛠️ API Documentation

### Core Endpoints

```bash
# System Information
GET  /api                        # API overview
GET  /api/health                 # Health check
GET  /api/stats                  # System statistics

# Video Management
POST /api/videos/upload          # Upload video file
POST /api/videos/download-youtube # Download from YouTube
GET  /api/videos                 # List all videos
GET  /api/videos/{id}            # Get video info
DELETE /api/videos/{id}          # Delete video

# Search
POST /api/search                 # Semantic search
POST /api/search/batch           # Batch search

# Configuration
GET  /api/config                 # Get configuration
POST /api/config                 # Update configuration
POST /api/config/reset           # Reset to defaults

# Cache Management
GET  /api/cache/stats            # Cache statistics
POST /api/cache/rebuild          # Rebuild cache
POST /api/cache/clear            # Clear cache
GET  /api/cache/health           # Cache health check
POST /api/cache/export           # Export cache
POST /api/cache/import           # Import cache
```

### Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔧 Configuration

### Environment Variables

```bash
# Optional environment configuration
export VIDEO_SEARCH_PORT=8000
export VIDEO_SEARCH_HOST=0.0.0.0
export VIDEO_SEARCH_LOG_LEVEL=INFO
```

### Configuration File (config.json)

```json
{
  "sampling_mode": "ultra_high",
  "max_frames": 500,
  "use_clip": true,
  "enhanced_mode": true,
  "default_results": 10,
  "cache_search": true,
  "search_timeout": 30,
  "auto_save": true,
  "log_level": "INFO"
}
```

## 🐛 Troubleshooting

### Common Issues

1. **Slow processing**: Lower sampling_mode or max_frames
2. **Memory issues**: Reduce max_frames or disable CLIP
3. **Cache corruption**: Use cache rebuild function
4. **YouTube download fails**: Check yt-dlp installation

### Performance Optimization

- Use GPU for CLIP processing when available
- Adjust sampling mode based on use case
- Enable caching for repeated searches
- Regular cache maintenance

## 📈 System Requirements

### Minimum Requirements

- **RAM**: 8GB (16GB recommended)
- **Storage**: 10GB free space
- **CPU**: 4+ cores recommended
- **Python**: 3.8+ required

### Recommended for Production

- **RAM**: 32GB+
- **Storage**: SSD with 100GB+
- **GPU**: NVIDIA GPU for CLIP acceleration
- **CPU**: 8+ cores Intel/AMD

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **OpenAI CLIP** - Semantic understanding capabilities
- **HNSW** - High-performance vector indexing
- **FastAPI** - Modern web framework
- **yt-dlp** - YouTube download functionality

---

Built with ❤️ using FastAPI, CLIP, OpenCV, and modern web technologies.
