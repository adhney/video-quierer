#!/usr/bin/env python3
"""
🚀 Complete Video Search System Overhaul
========================================
Simple, reliable, and actually working caching system
"""

import os
import sys
import time
import cv2
import json
import pickle
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class SimpleVideoIndex:
    """Dead simple but reliable video index"""

    def __init__(self):
        self.embeddings = []  # List of numpy arrays
        self.metadata = []    # List of metadata dicts
        self.video_hashes = {}  # Track processed videos

    def add_frame(self, embedding: np.ndarray, video_name: str, timestamp: float):
        """Add a frame embedding with metadata"""
        self.embeddings.append(embedding.astype(np.float32))
        self.metadata.append({
            'video_name': video_name,
            'timestamp': timestamp,
            'frame_id': len(self.embeddings) - 1
        })

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Simple cosine similarity search"""
        if not self.embeddings:
            return []

        # Convert embeddings to numpy array for fast computation
        embeddings_array = np.vstack(self.embeddings)

        # Normalize query
        query_norm = query_embedding / \
            (np.linalg.norm(query_embedding) + 1e-10)

        # Compute similarities
        similarities = np.dot(embeddings_array, query_norm)

        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in top_indices:
            metadata = self.metadata[idx].copy()
            metadata['score'] = float(similarities[idx])
            results.append(metadata)

        return results

    def save_to_disk(self, cache_path: Path):
        """Save index to disk"""
        try:
            cache_data = {
                'embeddings': self.embeddings,
                'metadata': self.metadata,
                'video_hashes': self.video_hashes,
                'version': '1.0'
            }

            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)

            logger.info(
                f"💾 Saved {len(self.embeddings)} embeddings to {cache_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
            return False

    def load_from_disk(self, cache_path: Path) -> bool:
        """Load index from disk"""
        try:
            if not cache_path.exists():
                return False

            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)

            self.embeddings = cache_data.get('embeddings', [])
            self.metadata = cache_data.get('metadata', [])
            self.video_hashes = cache_data.get('video_hashes', {})

            logger.info(
                f"💾 Loaded {len(self.embeddings)} embeddings from {cache_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return False


class VideoProcessor:
    """Simple, reliable video processor"""

    def __init__(self, use_clip: bool = True):
        self.use_clip = use_clip
        self.clip_model = None
        self.clip_processor = None

        if use_clip:
            self._init_clip()

    def _init_clip(self):
        """Initialize CLIP model"""
        try:
            from transformers import CLIPProcessor, CLIPModel
            import torch

            logger.info("🧠 Loading CLIP...")
            self.clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32")

            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
            self.clip_model.to(self.device)
            self.clip_model.eval()

            logger.info(f"✅ CLIP loaded on {self.device}")

        except Exception as e:
            logger.error(f"CLIP failed to load: {e}")
            self.use_clip = False

    def get_video_hash(self, video_path: Path) -> str:
        """Get unique hash for video (based on size + mtime)"""
        stat = video_path.stat()
        hash_input = f"{video_path.name}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.md5(hash_input.encode()).hexdigest()

    def extract_frames(self, video_path: Path, max_frames: int = 300, sampling_mode: str = "high") -> List[Dict]:
        """Extract frames from video with different sampling strategies"""
        logger.info(
            f"📼 Extracting frames from {video_path.name} (mode: {sampling_mode})")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Cannot open {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate sampling strategy based on mode
        if sampling_mode == "ultra_high":
            # For ultra_high, sample every frame if possible, or use minimal interval
            # More aggressive sampling
            interval = max(1, total_frames // (max_frames * 2))
            logger.info(f"🔥 Ultra-high sampling mode: interval={interval}")
        elif sampling_mode == "high":
            # Standard high-quality sampling
            interval = max(1, total_frames // max_frames)
            logger.info(f"📈 High sampling mode: interval={interval}")
        elif sampling_mode == "medium":
            # Medium quality - larger intervals
            interval = max(1, total_frames // (max_frames // 2))
            logger.info(f"📊 Medium sampling mode: interval={interval}")
        else:  # low
            # Low quality - even larger intervals
            interval = max(1, total_frames // (max_frames // 4))
            logger.info(f"📉 Low sampling mode: interval={interval}")

        frames = []
        frame_number = 0

        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number % interval == 0:
                timestamp = frame_number / fps if fps > 0 else frame_number / 30
                frames.append({
                    'frame': frame,
                    'timestamp': timestamp
                })

            frame_number += 1

        cap.release()
        logger.info(
            f"✅ Extracted {len(frames)} frames with {sampling_mode} sampling")
        return frames

    def get_frame_embedding(self, frame: np.ndarray) -> np.ndarray:
        """Get embedding for a single frame"""
        if self.use_clip:
            return self._get_clip_embedding(frame)
        else:
            return self._get_visual_features(frame)

    def _get_clip_embedding(self, frame: np.ndarray) -> np.ndarray:
        """Get CLIP embedding"""
        try:
            import torch
            from PIL import Image

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # Process with CLIP
            inputs = self.clip_processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                features = self.clip_model.get_image_features(**inputs)
                features = features / features.norm(p=2, dim=-1, keepdim=True)

            return features.cpu().numpy().flatten().astype(np.float32)

        except Exception as e:
            logger.error(f"CLIP embedding failed: {e}")
            return self._get_visual_features(frame)

    def _get_visual_features(self, frame: np.ndarray) -> np.ndarray:
        """Get basic visual features as fallback"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Basic features
        features = []

        # Color statistics
        features.extend([
            np.mean(gray),
            np.std(gray),
            np.median(gray)
        ])

        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        features.append(np.sum(edges > 0) / edges.size)

        # Histogram features
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
        features.extend(hist)

        # Pad to 512 dimensions
        features = np.array(features, dtype=np.float32)
        if len(features) < 512:
            features = np.pad(features, (0, 512 - len(features)), 'constant')
        else:
            features = features[:512]

        # Normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm

        return features

    def encode_text_query(self, query: str) -> np.ndarray:
        """Encode text query to embedding"""
        if self.use_clip:
            return self._encode_clip_text(query)
        else:
            return self._encode_visual_query(query)

    def _encode_clip_text(self, query: str) -> np.ndarray:
        """Encode text with CLIP"""
        try:
            import torch

            inputs = self.clip_processor(
                text=[query], return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                features = self.clip_model.get_text_features(**inputs)
                features = features / features.norm(p=2, dim=-1, keepdim=True)

            return features.cpu().numpy().flatten().astype(np.float32)

        except Exception as e:
            logger.error(f"CLIP text encoding failed: {e}")
            return self._encode_visual_query(query)

    def _encode_visual_query(self, query: str) -> np.ndarray:
        """Encode query based on visual keywords"""
        features = np.zeros(512, dtype=np.float32)
        query_lower = query.lower()

        # Map keywords to feature positions
        if 'bright' in query_lower:
            features[0] = 0.8
        if 'dark' in query_lower:
            features[0] = 0.2
        if 'phone' in query_lower or 'app' in query_lower:
            features[10] = 0.9
        if 'car' in query_lower or 'vehicle' in query_lower:
            features[20] = 0.9
        if 'goal' in query_lower or 'football' in query_lower:
            features[30] = 0.9

        # Normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        else:
            features = np.random.normal(0, 0.1, 512).astype(np.float32)
            features = features / np.linalg.norm(features)

        return features


class VideoSearchSystem:
    """Complete overhaul - simple and reliable"""

    def __init__(self, videos_dir: str = "videos", config=None):
        self.videos_dir = Path(videos_dir)
        self.videos_dir.mkdir(exist_ok=True)
        self.config = config  # Store configuration

        self.cache_path = self.videos_dir / "video_search_cache.pkl"
        self.index = SimpleVideoIndex()

        # Use configuration for processor settings
        use_clip = True
        if config and hasattr(config, 'use_clip'):
            use_clip = config.use_clip

        self.processor = VideoProcessor(use_clip=use_clip)

        logger.info("🚀 Video Search System initialized")

    def startup(self):
        """Initialize system and load/process videos"""
        logger.info("🔄 Starting up...")

        # Try to load cache first
        if self.index.load_from_disk(self.cache_path):
            logger.info(
                f"✅ Loaded cache with {len(self.index.embeddings)} embeddings")

            # Check if any videos are new or changed
            current_videos = self._get_current_videos()
            needs_update = self._check_videos_changed(current_videos)

            if needs_update:
                logger.info("📼 Some videos changed, updating...")
                self._process_changed_videos(current_videos)
            else:
                logger.info("📁 All videos up to date!")
                return
        else:
            logger.info("💾 No cache found, processing all videos...")
            current_videos = self._get_current_videos()
            self._process_all_videos(current_videos)

        # Save cache
        self.index.save_to_disk(self.cache_path)
        logger.info("✅ Startup complete!")

    def _get_current_videos(self) -> List[Path]:
        """Get all video files in directory"""
        videos = []
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            videos.extend(self.videos_dir.glob(f"*{ext}"))
            videos.extend(self.videos_dir.glob(f"*{ext.upper()}"))
        return videos

    def _check_videos_changed(self, current_videos: List[Path]) -> bool:
        """Check if any videos are new or changed"""
        for video in current_videos:
            current_hash = self.processor.get_video_hash(video)
            if video.name not in self.index.video_hashes:
                return True  # New video
            if self.index.video_hashes[video.name] != current_hash:
                return True  # Changed video
        return False

    def _process_changed_videos(self, current_videos: List[Path]):
        """Process only new/changed videos"""
        for video in current_videos:
            current_hash = self.processor.get_video_hash(video)

            # Check if video needs processing
            if (video.name not in self.index.video_hashes or
                    self.index.video_hashes[video.name] != current_hash):

                logger.info(f"🎬 Processing {video.name}")
                self._process_single_video(video, self.config)
                self.index.video_hashes[video.name] = current_hash

    def _process_all_videos(self, videos: List[Path]):
        """Process all videos from scratch"""
        for video in videos:
            logger.info(f"🎬 Processing {video.name}")
            self._process_single_video(video, self.config)
            self.index.video_hashes[video.name] = self.processor.get_video_hash(
                video)

    def _process_single_video(self, video_path: Path, config=None):
        """Process a single video"""
        max_frames = 300  # Default
        sampling_mode = "high"  # Default

        if config and hasattr(config, 'max_frames'):
            max_frames = config.max_frames

        if config and hasattr(config, 'sampling_mode'):
            sampling_mode = config.sampling_mode

        frames = self.processor.extract_frames(
            video_path, max_frames=max_frames, sampling_mode=sampling_mode)

        logger.info(f"🔍 Generating embeddings for {len(frames)} frames...")

        for i, frame_data in enumerate(frames):
            embedding = self.processor.get_frame_embedding(frame_data['frame'])
            self.index.add_frame(
                embedding=embedding,
                video_name=video_path.name,
                timestamp=frame_data['timestamp']
            )

            if (i + 1) % 50 == 0:
                logger.info(f"   📊 Processed {i + 1}/{len(frames)} frames")

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for videos"""
        logger.info(f"🔍 Searching: {query}")

        # Encode query
        query_embedding = self.processor.encode_text_query(query)

        # Search index
        results = self.index.search(query_embedding, k)

        # Format results
        for result in results:
            minutes = int(result['timestamp'] // 60)
            seconds = int(result['timestamp'] % 60)
            result['formatted_time'] = f"{minutes}m{seconds}s"

        logger.info(f"✅ Found {len(results)} results")
        return results


def main():
    """Test the overhauled system"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(name)s:%(message)s'
    )

    print("🚀 Video Search System - Complete Overhaul")
    print("=" * 50)

    # Initialize system
    system = VideoSearchSystem("videos")

    # Startup (load cache or process videos)
    system.startup()

    # Interactive search
    print("\n🔍 Search ready! Try queries like:")
    print("  • 'phone apps'")
    print("  • 'car interior'")
    print("  • 'goal celebration'")
    print("  • 'bright presentation'")
    print("Type 'quit' to exit")

    while True:
        try:
            query = input("\n🔍 Search: ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                break

            if query:
                results = system.search(query, k=5)

                if results:
                    print(f"\n📊 Results:")
                    for i, result in enumerate(results, 1):
                        print(
                            f"   {i}. {result['video_name']} at {result['formatted_time']}")
                        print(f"      Score: {result['score']:.3f}")
                else:
                    print("❌ No results found")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break


if __name__ == "__main__":
    main()
