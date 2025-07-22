#!/usr/bin/env python3
"""
🧠 Semantic Mode Only - AI-Powered Search  
Forces the system to use full semantic understanding
"""

from semantic_video_search import SemanticVideoProcessor, check_dependencies
from pathlib import Path

def main():
    print("🧠 Semantic Mode Video Search")
    print("=" * 40)
    print("🎯 Forcing AI semantic understanding")
    
    # Check if AI is available
    deps = check_dependencies()
    if not (deps['torch'] and deps['clip']):
        print("❌ AI dependencies not available!")
        print("Install: pip install torch transformers")
        return
    
    # Force semantic mode by setting use_clip=True
    processor = SemanticVideoProcessor(use_clip=True)
    
    # Check for videos
    videos_dir = Path("videos")
    video_files = []
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        video_files.extend(list(videos_dir.glob(f"*{ext}")))
    
    if not video_files:
        print("📁 No video files found in videos/ directory")
        return
    
    # Process videos
    for video_file in video_files:
        processor.process_video(str(video_file))
    
    print(f"\n🧠 Semantic Understanding Search (1-2s)")
    print("Try: 'system architecture', 'database design', 'microservices'")
    
    while True:
        try:
            query = input("\n🔍 Semantic search: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            if query:
                processor.search(query, k=5)
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
