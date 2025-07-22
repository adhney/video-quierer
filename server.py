#!/usr/bin/env python3
"""
🌐 Local Video Search Server - With Progress Bar
================================================
Web interface for uploading videos and YouTube downloads with real-time progress
"""

from flask import Flask, request, render_template_string, jsonify
from flask_socketio import SocketIO, emit
import os
import sys
from pathlib import Path
import subprocess
import json
import time
import numpy as np
import threading
import re

# Import our semantic search system
from semantic_video_search import SemanticVideoProcessor, check_dependencies

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuration
UPLOAD_FOLDER = Path("videos")
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

# Global search processor
search_processor = None


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


def format_timestamp(seconds):
    """Convert seconds to readable timestamp format like 1m50s"""
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if hours > 0:
        return f"{hours}h{minutes}m{secs}s"
    elif minutes > 0:
        return f"{minutes}m{secs}s"
    else:
        return f"{secs}s"


def init_search_system():
    """Initialize the search system"""
    global search_processor
    if search_processor is None:
        deps = check_dependencies()
        use_clip = deps['torch'] and deps['clip']
        search_processor = SemanticVideoProcessor(use_clip=use_clip)

        # Process existing videos
        video_files = []
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            video_files.extend(list(UPLOAD_FOLDER.glob(f"*{ext}")))

        for video_file in video_files:
            search_processor.process_video(str(video_file))

    return search_processor


def download_youtube_video_with_progress(url, output_path, session_id):
    """Download video from YouTube using yt-dlp with real-time progress"""
    try:
        # Check if yt-dlp is available
        result = subprocess.run(['yt-dlp', '--version'],
                                capture_output=True, text=True)
        if result.returncode != 0:
            socketio.emit('download_error', {
                          'error': 'yt-dlp not installed'}, room=session_id)
            return False, "yt-dlp not installed. Install with: pip install yt-dlp"

        socketio.emit('download_progress', {
                      'progress': 0, 'status': 'Starting download...'}, room=session_id)

        # Download video with progress
        cmd = [
            'yt-dlp',
            '-f', 'best[height<=720]',  # Limit to 720p for faster processing
            '-o', str(output_path / '%(title)s.%(ext)s'),
            '--restrict-filenames',  # Safe filenames
            '--newline',  # Each progress update on new line
            url
        ]

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   universal_newlines=True, bufsize=1)

        # Track progress in real-time
        progress = 0
        for line in process.stdout:
            if '[download]' in line:
                # Extract progress percentage
                progress_match = re.search(r'(\d+\.?\d*)%', line)
                if progress_match:
                    progress = float(progress_match.group(1))
                    socketio.emit('download_progress', {
                        'progress': progress,
                        'status': f'Downloading... {progress:.1f}%'
                    }, room=session_id)

                # Extract file size and speed info
                if 'of' in line:
                    parts = line.split()
                    try:
                        # Look for file size and speed patterns
                        for i, part in enumerate(parts):
                            if part == 'of' and i+1 < len(parts):
                                size = parts[i+1]
                                if i+3 < len(parts) and parts[i+2] == 'at':
                                    speed = parts[i+3]
                                    socketio.emit('download_progress', {
                                        'progress': progress,
                                        'status': f'Downloading {size} at {speed} ({progress:.1f}%)'
                                    }, room=session_id)
                                break
                    except:
                        pass

        process.wait()

        if process.returncode == 0:
            socketio.emit('download_progress', {
                'progress': 100,
                'status': 'Download completed! Processing video...'
            }, room=session_id)

            # Find the downloaded file
            downloaded_files = list(output_path.glob(
                "*.mp4")) + list(output_path.glob("*.mkv"))
            if downloaded_files:
                return True, str(downloaded_files[-1])  # Return latest file
            else:
                return False, "Download completed but file not found"
        else:
            stderr = process.stderr.read()
            socketio.emit('download_error', {
                          'error': f'Download failed: {stderr}'}, room=session_id)
            return False, f"Download failed: {stderr}"

    except Exception as e:
        socketio.emit('download_error', {'error': str(e)}, room=session_id)
        return False, f"Error: {str(e)}"


# Enhanced HTML Template with Progress Bar
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎬 Semantic Video Search</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.0/socket.io.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        h1 { color: #333; text-align: center; }
        .upload-section, .search-section {
            margin: 20px 0;
            padding: 20px;
            border: 2px dashed #ddd;
            border-radius: 8px;
        }
        .youtube-section {
            background: #ffe6e6;
            border-color: #ff9999;
        }
        .file-section {
            background: #e6f3ff;
            border-color: #99ccff;
        }
        .search-section {
            background: #f0f8e6;
            border-color: #99cc99;
        }
        input[type="text"], input[type="file"] {
            width: 100%;
            padding: 12px;
            margin: 10px 0;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 16px;
        }
        button {
            background: #007bff;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            margin: 5px;
        }
        button:hover { background: #0056b3; }
        button:disabled { 
            background: #6c757d; 
            cursor: not-allowed; 
        }
        .youtube-btn { background: #ff0000; }
        .youtube-btn:hover { background: #cc0000; }
        
        /* Progress Bar Styles */
        .progress-container {
            width: 100%;
            background-color: #f0f0f0;
            border-radius: 4px;
            margin: 15px 0;
            display: none;
        }
        .progress-bar {
            height: 20px;
            background: linear-gradient(45deg, #007bff, #0056b3);
            border-radius: 4px;
            transition: width 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        .progress-bar::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, transparent 25%, rgba(255,255,255,0.2) 25%, rgba(255,255,255,0.2) 50%, transparent 50%, transparent 75%, rgba(255,255,255,0.2) 75%);
            background-size: 20px 20px;
            animation: progress-animation 1s linear infinite;
        }
        @keyframes progress-animation {
            0% { transform: translateX(-20px); }
            100% { transform: translateX(20px); }
        }
        .progress-text {
            text-align: center;
            margin: 5px 0;
            font-weight: bold;
            color: #333;
        }
        
        .results {
            margin-top: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 5px;
        }
        .result-item {
            padding: 10px;
            margin: 10px 0;
            background: white;
            border-left: 4px solid #007bff;
            border-radius: 4px;
        }
        .status {
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
        }
        .status.success { background: #d4edda; color: #155724; }
        .status.error { background: #f8d7da; color: #721c24; }
        .status.info { background: #d1ecf1; color: #0c5460; }
        .video-list {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .video-card {
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎬 Semantic Video Search Server</h1>
        
        <!-- YouTube Download Section -->
        <div class="upload-section youtube-section">
            <h3>📺 Download from YouTube</h3>
            <input type="text" id="youtube-url" placeholder="Enter YouTube URL">
            <button class="youtube-btn" id="download-btn" onclick="downloadYoutube()">📥 Download & Process</button>
            
            <!-- Progress Bar -->
            <div class="progress-container" id="progress-container">
                <div class="progress-text" id="progress-text">Preparing download...</div>
                <div class="progress-bar" id="progress-bar" style="width: 0%"></div>
            </div>
        </div>
        
        <!-- File Upload Section -->
        <div class="upload-section file-section">
            <h3>📁 Upload Video File</h3>
            <input type="file" id="video-file" accept=".mp4,.avi,.mov,.mkv">
            <button onclick="uploadFile()">📤 Upload & Process</button>
        </div>
        
        <!-- Search Section -->
        <div class="search-section">
            <h3>🔍 Search Videos</h3>
            <input type="text" id="search-query" placeholder="Enter search query">
            <button onclick="searchVideos()">🔍 Search</button>
            <button onclick="searchVideos(true)">⚡ Fast Search</button>
        </div>
        
        <!-- Status Messages -->
        <div id="status"></div>
        
        <!-- Search Results -->
        <div id="results"></div>
        
        <!-- Video List -->
        <div class="container">
            <h3>📼 Available Videos</h3>
            <div id="video-list" class="video-list"></div>
        </div>
    </div>

    <script>
        // Initialize Socket.IO
        const socket = io();
        
        function showStatus(message, type = 'info') {
            const status = document.getElementById('status');
            status.innerHTML = `<div class="status ${type}">${message}</div>`;
            setTimeout(() => status.innerHTML = '', 5000);
        }
        
        function showProgress(show = true) {
            const container = document.getElementById('progress-container');
            const btn = document.getElementById('download-btn');
            container.style.display = show ? 'block' : 'none';
            btn.disabled = show;
            
            if (!show) {
                // Reset progress
                document.getElementById('progress-bar').style.width = '0%';
                document.getElementById('progress-text').textContent = 'Preparing download...';
            }
        }
        
        function updateProgress(progress, status) {
            document.getElementById('progress-bar').style.width = progress + '%';
            document.getElementById('progress-text').textContent = status;
        }
        
        // Socket.IO event listeners
        socket.on('download_progress', function(data) {
            updateProgress(data.progress, data.status);
        });
        
        socket.on('download_error', function(data) {
            showStatus(`❌ Download failed: ${data.error}`, 'error');
            showProgress(false);
        });
        
        async function downloadYoutube() {
            const url = document.getElementById('youtube-url').value.trim();
            if (!url) {
                showStatus('Please enter a YouTube URL', 'error');
                return;
            }
            
            showProgress(true);
            showStatus('Starting YouTube download...', 'info');
            
            try {
                const response = await fetch('/download-youtube', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        url: url,
                        session_id: socket.id 
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    updateProgress(100, 'Video processed successfully!');
                    showStatus(`✅ Video downloaded: ${result.filename}`, 'success');
                    loadVideoList();
                    setTimeout(() => showProgress(false), 2000);
                } else {
                    showStatus(`❌ Download failed: ${result.error}`, 'error');
                    showProgress(false);
                }
            } catch (error) {
                showStatus(`❌ Error: ${error.message}`, 'error');
                showProgress(false);
            }
        }
        
        async function uploadFile() {
            const fileInput = document.getElementById('video-file');
            const file = fileInput.files[0];
            
            if (!file) {
                showStatus('Please select a video file', 'error');
                return;
            }
            
            showStatus('Uploading and processing video...', 'info');
            
            const formData = new FormData();
            formData.append('video', file);
            
            try {
                const response = await fetch('/upload-video', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.success) {
                    showStatus(`✅ Video uploaded: ${result.filename}`, 'success');
                    loadVideoList();
                } else {
                    showStatus(`❌ Upload failed: ${result.error}`, 'error');
                }
            } catch (error) {
                showStatus(`❌ Error: ${error.message}`, 'error');
            }
        }
        
        async function searchVideos(fastMode = false) {
            const query = document.getElementById('search-query').value.trim();
            if (!query) {
                showStatus('Please enter a search query', 'error');
                return;
            }
            
            showStatus(`🔍 Searching for: "${query}"...`, 'info');
            
            try {
                const response = await fetch('/search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        query: query,
                        fast_mode: fastMode,
                        k: 5 
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    displayResults(result.results, result.search_time, fastMode);
                } else {
                    showStatus(`❌ Search failed: ${result.error}`, 'error');
                }
            } catch (error) {
                showStatus(`❌ Error: ${error.message}`, 'error');
            }
        }
        
        function displayResults(results, searchTime, fastMode) {
            const resultsDiv = document.getElementById('results');
            
            if (results.length === 0) {
                resultsDiv.innerHTML = '<div class="results"><p>No results found</p></div>';
                return;
            }
            
            const mode = fastMode ? '⚡ Fast' : '🧠 Semantic';
            let html = `<div class="results">
                <h4>${mode} Search Results (${searchTime.toFixed(1)}ms)</h4>`;
            
            results.forEach((result, index) => {
                const formattedTime = formatTimestamp(result.timestamp);
                html += `<div class="result-item">
                    <strong>${index + 1}. ${result.video_name}</strong><br>
                    📍 Timestamp: ${formattedTime}<br>
                    📊 Score: ${result.score.toFixed(3)}
                </div>`;
            });
            
            html += '</div>';
            resultsDiv.innerHTML = html;
        }
        
        function formatTimestamp(seconds) {
            const totalSeconds = Math.floor(seconds);
            const hours = Math.floor(totalSeconds / 3600);
            const minutes = Math.floor((totalSeconds % 3600) / 60);
            const secs = totalSeconds % 60;
            
            if (hours > 0) {
                return `${hours}h${minutes}m${secs}s`;
            } else if (minutes > 0) {
                return `${minutes}m${secs}s`;
            } else {
                return `${secs}s`;
            }
        }
        
        async function loadVideoList() {
            try {
                const response = await fetch('/videos');
                const result = await response.json();
                
                const videoList = document.getElementById('video-list');
                
                if (result.videos.length === 0) {
                    videoList.innerHTML = '<p>No videos uploaded yet</p>';
                    return;
                }
                
                let html = '';
                result.videos.forEach(video => {
                    html += `<div class="video-card">
                        <h4>📹 ${video.name}</h4>
                        <p>📊 Size: ${(video.size / 1024 / 1024).toFixed(1)} MB</p>
                        <p>📅 Added: ${new Date(video.modified * 1000).toLocaleString()}</p>
                    </div>`;
                });
                
                videoList.innerHTML = html;
            } catch (error) {
                console.error('Error loading video list:', error);
            }
        }
        
        // Load video list on page load
        document.addEventListener('DOMContentLoaded', loadVideoList);
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """Main page"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/download-youtube', methods=['POST'])
def download_youtube():
    """Download video from YouTube with progress tracking"""
    try:
        data = request.json
        url = data.get('url', '').strip()
        session_id = data.get('session_id')

        if not url:
            return jsonify({'success': False, 'error': 'No URL provided'})

        # Run download in background thread to enable real-time progress
        def download_task():
            success, result = download_youtube_video_with_progress(
                url, UPLOAD_FOLDER, session_id)

            if success:
                # Process the video with our search system
                processor = init_search_system()
                socketio.emit('download_progress', {
                    'progress': 100,
                    'status': 'Processing video for search...'
                }, room=session_id)

                processor.process_video(result)

                socketio.emit('download_complete', {
                    'success': True,
                    'filename': Path(result).name
                }, room=session_id)
            else:
                socketio.emit('download_complete', {
                    'success': False,
                    'error': result
                }, room=session_id)

        # Start download in background
        thread = threading.Thread(target=download_task)
        thread.daemon = True
        thread.start()

        return jsonify({'success': True, 'message': 'Download started'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/upload-video', methods=['POST'])
def upload_video():
    """Upload video file"""
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'No video file provided'})

        file = request.files['video']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})

        # Save the file
        filename = file.filename
        filepath = UPLOAD_FOLDER / filename
        file.save(str(filepath))

        # Process the video with our search system
        processor = init_search_system()
        processor.process_video(str(filepath))

        return jsonify({
            'success': True,
            'filename': filename,
            'message': 'Video uploaded and processed successfully'
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/search', methods=['POST'])
def search():
    """Search videos - FIXED VERSION"""
    try:
        data = request.json
        query = data.get('query', '').strip()
        fast_mode = data.get('fast_mode', False)
        k = data.get('k', 5)

        if not query:
            return jsonify({'success': False, 'error': 'No query provided'})

        processor = init_search_system()

        # Override processor mode if fast_mode is specified
        if fast_mode:
            processor.use_clip = False

        results, search_time = processor.search(query, k=k)

        # Convert numpy types to native Python types for JSON serialization
        results_converted = convert_numpy_types(results)
        search_time_converted = convert_numpy_types(search_time)

        return jsonify({
            'success': True,
            'results': results_converted,
            'search_time': search_time_converted,
            'query': query
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/videos')
def list_videos():
    """List available videos"""
    try:
        videos = []
        for video_file in UPLOAD_FOLDER.iterdir():
            if video_file.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                stat = video_file.stat()
                videos.append({
                    'name': video_file.name,
                    'size': stat.st_size,
                    'modified': stat.st_mtime
                })

        return jsonify({'videos': videos})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    print("🌐 Starting Semantic Video Search Server with Progress Bar")
    print("=" * 60)

    # Check dependencies
    deps = check_dependencies()
    print("📦 Available components:")
    for name, version in deps.items():
        if version:
            print(f"   ✅ {name}: {version}")
        else:
            print(f"   ❌ {name}: Not available")

    # Check for yt-dlp
    try:
        result = subprocess.run(['yt-dlp', '--version'],
                                capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ yt-dlp: Available")
        else:
            print(f"   ⚠️  yt-dlp: Install with 'pip install yt-dlp'")
    except:
        print(f"   ⚠️  yt-dlp: Install with 'pip install yt-dlp'")

    print(f"\n🚀 Server starting with real-time progress...")
    print(f"📱 Open your browser to: http://localhost:5000")
    print(f"📁 Videos will be saved to: {UPLOAD_FOLDER.absolute()}")
    print(f"🔍 Both semantic and fast search available!")
    print(f"📊 Real-time download progress enabled!")

    socketio.run(app, debug=True, host='0.0.0.0', port=5001)
