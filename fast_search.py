#!/usr/bin/env python3
"""
🚀 Fast Mode Only - Visual Features Search
Forces the system to use only fast visual analysis
"""

from semantic_video_search import SemanticVideoProcessor, check_dependencies
from pathlib import Path

def main():
    print("⚡ Fast Mode Video Search")
    print("=" * 40)
    print("🔧 Forcing visual features only (no AI)")
    
    # Force fast mode by setting use_clip=False
    processor = SemanticVideoProcessor(use_clip=False)
    
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
    
    print(f"\n⚡ Fast Visual Search (< 10ms)")
    print("Try: 'bright slide', 'complex diagram', 'simple layout'")
    
    while True:
        try:
            query = input("\n🔍 Visual search: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            if query:
                processor.search(query, k=5)
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
