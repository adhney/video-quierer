#!/usr/bin/env python3
"""
🚀 Video Search System - Quick Start
===================================
Simple setup verification for the semantic video search system.
"""

import os
import sys
from pathlib import Path

def check_dependencies():
    """Check if required packages are available"""
    print("🔍 Checking dependencies...")
    
    required = ['cv2', 'numpy']
    optional = ['torch', 'transformers']
    
    missing_required = []
    missing_optional = []
    
    for pkg in required:
        try:
            __import__(pkg)
            print(f"  ✅ {pkg}")
        except ImportError:
            missing_required.append(pkg)
            print(f"  ❌ {pkg}")
    
    for pkg in optional:
        try:
            __import__(pkg)
            print(f"  ✅ {pkg} (semantic features enabled)")
        except ImportError:
            missing_optional.append(pkg)
            print(f"  ⚠️  {pkg} (semantic features disabled)")
    
    if missing_required:
        print(f"\n❌ Missing required packages: {', '.join(missing_required)}")
        print("Install with: pip install opencv-python numpy")
        return False
    
    if missing_optional:
        print(f"\n💡 Optional packages missing: {', '.join(missing_optional)}")
        print("For full semantic search: pip install torch transformers")
    
    return True

def check_videos_folder():
    """Check if videos folder exists and has content"""
    print("\n📁 Checking videos folder...")
    
    videos_path = Path("videos")
    if not videos_path.exists():
        print("  📁 Creating videos folder...")
        videos_path.mkdir()
        print("  ✅ Videos folder created")
        return False
    
    video_files = list(videos_path.glob("*.mp4")) + list(videos_path.glob("*.avi")) + \
                 list(videos_path.glob("*.mov")) + list(videos_path.glob("*.mkv"))
    
    if not video_files:
        print("  ⚠️  No video files found in videos/ folder")
        print("  💡 Add some .mp4, .avi, .mov, or .mkv files to test")
        return False
    
    print(f"  ✅ Found {len(video_files)} video file(s):")
    for video in video_files[:3]:  # Show first 3
        print(f"    📹 {video.name}")
    if len(video_files) > 3:
        print(f"    ... and {len(video_files) - 3} more")
    
    return True

def test_semantic_search():
    """Test the semantic video search system"""
    print("\n🧠 Testing semantic video search...")
    
    try:
        # Import our working system - correct class name
        from semantic_video_search import SemanticVideoProcessor
        
        # Initialize
        search_system = SemanticVideoProcessor()
        print("  ✅ Search system initialized")
        
        # Check if we have videos to test
        videos_path = Path("videos")
        video_files = list(videos_path.glob("*.mp4")) + list(videos_path.glob("*.avi"))
        
        if video_files:
            print(f"  🎬 Ready to search {len(video_files)} video(s)")
            print("\n🎯 Example search commands:")
            print("    python semantic_video_search.py")
            print("    🔍 Semantic search: your search query here")
        else:
            print("  ⚠️  Add videos to test search functionality")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def main():
    """Main quickstart function"""
    print("🚀 Video Search System - Quick Start")
    print("=" * 50)
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Check videos folder
    videos_ok = check_videos_folder()
    
    # Test system
    system_ok = test_semantic_search()
    
    print("\n" + "=" * 50)
    
    if deps_ok and system_ok:
        print("🎉 System is ready!")
        if videos_ok:
            print("✨ You can start searching your videos")
            print("\nRun: python semantic_video_search.py")
        else:
            print("📹 Add videos to the videos/ folder to start searching")
    else:
        print("⚠️  Setup incomplete - check errors above")
    
    print("\n📚 Next Steps:")
    print("  1. Add video files to videos/ folder")
    print("  2. Run: python semantic_video_search.py")
    print("  3. Enter search queries like 'system design' or 'architecture'")

if __name__ == "__main__":
    main()
