#!/usr/bin/env python3
"""
🗑️ Cache Management Utility
===========================
Clear cached video data and manage your video search index
"""

import os
import shutil
from pathlib import Path
import json

def clear_video_cache():
    """Clear all cached video data"""
    print("🗑️ Clearing Video Cache")
    print("=" * 30)
    
    # Paths to clear
    cache_paths = [
        Path("videos"),           # Video files
        Path("__pycache__"),      # Python cache
        Path("src/__pycache__"),  # Source cache
        Path(".cache"),           # General cache
        Path("cache"),            # Cache directory
    ]
    
    # Look for any cache/index files
    cache_files = [
        "video_index.pkl",
        "search_cache.json", 
        "processed_videos.json",
        "hnsw_index.bin",
        "video_metadata.json"
    ]
    
    removed_count = 0
    
    # Remove cache directories
    for cache_path in cache_paths:
        if cache_path.exists():
            if cache_path.name == "videos":
                # Special handling for videos folder
                videos_removed = clear_videos_folder(cache_path)
                removed_count += videos_removed
            else:
                try:
                    shutil.rmtree(cache_path)
                    print(f"  ✅ Removed directory: {cache_path}")
                    removed_count += 1
                except Exception as e:
                    print(f"  ⚠️  Could not remove {cache_path}: {e}")
    
    # Remove cache files
    for cache_file in cache_files:
        cache_file_path = Path(cache_file)
        if cache_file_path.exists():
            try:
                cache_file_path.unlink()
                print(f"  ✅ Removed file: {cache_file}")
                removed_count += 1
            except Exception as e:
                print(f"  ⚠️  Could not remove {cache_file}: {e}")
    
    # Look in src directory for cache files
    src_dir = Path("src")
    if src_dir.exists():
        for cache_file in cache_files:
            src_cache_file = src_dir / cache_file
            if src_cache_file.exists():
                try:
                    src_cache_file.unlink()
                    print(f"  ✅ Removed src file: {src_cache_file}")
                    removed_count += 1
                except Exception as e:
                    print(f"  ⚠️  Could not remove {src_cache_file}: {e}")
    
    print(f"\n📊 Removed {removed_count} cached items")
    return removed_count

def clear_videos_folder(videos_path):
    """Clear videos folder and show what's being removed"""
    if not videos_path.exists():
        print("  📁 Videos folder doesn't exist")
        return 0
    
    video_files = []
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        video_files.extend(list(videos_path.glob(f"*{ext}")))
        video_files.extend(list(videos_path.glob(f"*{ext.upper()}")))
    
    if not video_files:
        print("  📁 No video files found to remove")
        return 0
    
    print(f"  📹 Found {len(video_files)} video files to remove:")
    for video_file in video_files:
        print(f"    🎬 {video_file.name}")
    
    # Ask for confirmation
    response = input(f"\n❓ Remove all {len(video_files)} video files? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        removed = 0
        for video_file in video_files:
            try:
                video_file.unlink()
                print(f"    ✅ Removed: {video_file.name}")
                removed += 1
            except Exception as e:
                print(f"    ❌ Failed to remove {video_file.name}: {e}")
        return removed
    else:
        print("  🚫 Video removal cancelled")
        return 0

def selective_video_removal():
    """Remove specific videos by choice"""
    videos_path = Path("videos")
    if not videos_path.exists():
        print("📁 No videos folder found")
        return
    
    video_files = []
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        video_files.extend(list(videos_path.glob(f"*{ext}")))
        video_files.extend(list(videos_path.glob(f"*{ext.upper()}")))
    
    if not video_files:
        print("📁 No video files found")
        return
    
    print("📹 Available video files:")
    for i, video_file in enumerate(video_files, 1):
        size_mb = video_file.stat().st_size / (1024 * 1024)
        print(f"  {i}. {video_file.name} ({size_mb:.1f} MB)")
    
    print(f"  {len(video_files) + 1}. Remove ALL videos")
    print(f"  0. Cancel")
    
    try:
        choice = input(f"\n❓ Enter number(s) to remove (e.g., 1,3,4 or 'all'): ").strip()
        
        if choice == '0':
            print("🚫 Cancelled")
            return
        
        if choice.lower() == 'all' or choice == str(len(video_files) + 1):
            # Remove all videos
            removed = 0
            for video_file in video_files:
                try:
                    video_file.unlink()
                    print(f"✅ Removed: {video_file.name}")
                    removed += 1
                except Exception as e:
                    print(f"❌ Failed to remove {video_file.name}: {e}")
            print(f"\n📊 Removed {removed} video files")
            return
        
        # Parse individual choices
        choices = [int(x.strip()) for x in choice.split(',')]
        removed = 0
        
        for choice_num in choices:
            if 1 <= choice_num <= len(video_files):
                video_file = video_files[choice_num - 1]
                try:
                    video_file.unlink()
                    print(f"✅ Removed: {video_file.name}")
                    removed += 1
                except Exception as e:
                    print(f"❌ Failed to remove {video_file.name}: {e}")
            else:
                print(f"⚠️  Invalid choice: {choice_num}")
        
        print(f"\n📊 Removed {removed} video files")
        
    except ValueError:
        print("❌ Invalid input. Please enter numbers separated by commas.")
    except KeyboardInterrupt:
        print("\n🚫 Cancelled")

def show_cache_status():
    """Show current cache status"""
    print("📊 Cache Status")
    print("=" * 20)
    
    # Check videos
    videos_path = Path("videos")
    if videos_path.exists():
        video_files = []
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            video_files.extend(list(videos_path.glob(f"*{ext}")))
        
        print(f"📹 Videos: {len(video_files)} files")
        for video_file in video_files:
            size_mb = video_file.stat().st_size / (1024 * 1024)
            print(f"  🎬 {video_file.name} ({size_mb:.1f} MB)")
    else:
        print("📹 Videos: No videos folder")
    
    # Check cache files
    cache_files = [
        "video_index.pkl",
        "search_cache.json", 
        "processed_videos.json",
        "hnsw_index.bin",
        "video_metadata.json"
    ]
    
    cache_found = []
    for cache_file in cache_files:
        if Path(cache_file).exists():
            cache_found.append(cache_file)
        if Path("src") / cache_file:
            cache_found.append(f"src/{cache_file}")
    
    if cache_found:
        print(f"💾 Cache files: {len(cache_found)} found")
        for cache_file in cache_found:
            print(f"  📄 {cache_file}")
    else:
        print("💾 Cache files: None found")
    
    # Check directories
    cache_dirs = ["__pycache__", "src/__pycache__", ".cache", "cache"]
    dirs_found = [d for d in cache_dirs if Path(d).exists()]
    
    if dirs_found:
        print(f"📁 Cache directories: {len(dirs_found)} found")
        for cache_dir in dirs_found:
            print(f"  📂 {cache_dir}")
    else:
        print("📁 Cache directories: None found")

def main():
    """Main cache management function"""
    print("🗑️ Video Search Cache Manager")
    print("=" * 40)
    
    while True:
        print("\n🎛️ Options:")
        print("  1. Show cache status")
        print("  2. Clear ALL cache (videos + index)")
        print("  3. Remove specific videos")
        print("  4. Clear cache but keep videos")
        print("  0. Exit")
        
        try:
            choice = input("\n❓ Choose option: ").strip()
            
            if choice == '0':
                print("👋 Goodbye!")
                break
            elif choice == '1':
                show_cache_status()
            elif choice == '2':
                removed = clear_video_cache()
                if removed > 0:
                    print("✅ Cache cleared successfully!")
                    print("💡 Run your server again to rebuild the cache")
                else:
                    print("ℹ️  No cache found to clear")
            elif choice == '3':
                selective_video_removal()
            elif choice == '4':
                # Clear cache but keep videos
                print("🗑️ Clearing cache (keeping videos)")
                cache_paths = [
                    Path("__pycache__"),
                    Path("src/__pycache__"),
                    Path(".cache"),
                    Path("cache"),
                ]
                
                cache_files = [
                    "video_index.pkl",
                    "search_cache.json", 
                    "processed_videos.json",
                    "hnsw_index.bin",
                    "video_metadata.json"
                ]
                
                removed = 0
                for cache_path in cache_paths:
                    if cache_path.exists():
                        try:
                            shutil.rmtree(cache_path)
                            print(f"  ✅ Removed: {cache_path}")
                            removed += 1
                        except Exception as e:
                            print(f"  ❌ Failed: {cache_path} - {e}")
                
                for cache_file in cache_files:
                    for location in [Path("."), Path("src")]:
                        cache_file_path = location / cache_file
                        if cache_file_path.exists():
                            try:
                                cache_file_path.unlink()
                                print(f"  ✅ Removed: {cache_file_path}")
                                removed += 1
                            except Exception as e:
                                print(f"  ❌ Failed: {cache_file_path} - {e}")
                
                print(f"📊 Removed {removed} cache items (videos preserved)")
            else:
                print("❌ Invalid option")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
