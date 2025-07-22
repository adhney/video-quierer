"""
Optimized Frame Extraction with Multiple Sampling Strategies
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Generator
import time
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class FrameSampler(ABC):
    """Abstract base class for frame sampling strategies"""
    
    @abstractmethod
    def extract_frames(self, video_path: str) -> List[Dict[str, Any]]:
        pass


class UniformFrameSampler(FrameSampler):
    """
    Sample frames at fixed time intervals
    Optimized for predictable memory usage
    """
    
    def __init__(self, sample_rate: float = 1.0, max_frames: int = 3600):
        self.sample_rate = sample_rate  # frames per second
        self.max_frames = max_frames
    
    def extract_frames(self, video_path: str) -> List[Dict[str, Any]]:
        """
        Extract frames at uniform intervals
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of frame data with timestamps
        """
        video = cv2.VideoCapture(video_path)
        
        if not video.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        try:
            video_fps = video.get(cv2.CAP_PROP_FPS)
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / video_fps if video_fps > 0 else 0
            
            # Calculate frame interval
            frame_interval = max(1, int(video_fps / self.sample_rate))
            
            frames_data = []
            frame_number = 0
            extracted_count = 0
            
            logger.info(f"Extracting frames from {video_path}: "
                       f"FPS={video_fps:.2f}, Duration={duration:.2f}s, "
                       f"Interval={frame_interval}")
            
            while frame_number < total_frames and extracted_count < self.max_frames:
                # Seek to specific frame for efficiency
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = video.read()
                
                if not ret:
                    break
                
                timestamp = frame_number / video_fps
                frames_data.append({
                    'frame': frame,
                    'timestamp': timestamp,
                    'frame_number': frame_number,
                    'video_path': video_path
                })
                
                frame_number += frame_interval
                extracted_count += 1
            
            logger.info(f"Extracted {len(frames_data)} frames from {video_path}")
            return frames_data
            
        finally:
            video.release()


class AdaptiveFrameSampler(FrameSampler):
    """
    Sample based on scene changes
    Captures all significant visual transitions
    """
    
    def __init__(
        self, 
        threshold: float = 30.0, 
        min_interval: float = 0.5,
        max_frames: int = 3600
    ):
        self.threshold = threshold
        self.min_interval = min_interval  # Minimum seconds between frames
        self.max_frames = max_frames
    
    def extract_frames(self, video_path: str) -> List[Dict[str, Any]]:
        """
        Extract frames based on scene changes
        """
        video = cv2.VideoCapture(video_path)
        
        if not video.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        try:
            video_fps = video.get(cv2.CAP_PROP_FPS)
            min_frame_interval = int(self.min_interval * video_fps)
            
            frames_data = []
            prev_frame = None
            frame_number = 0
            last_extracted_frame = -min_frame_interval
            
            logger.info(f"Adaptive extraction from {video_path}: "
                       f"threshold={self.threshold}, min_interval={self.min_interval}s")
            
            while len(frames_data) < self.max_frames:
                ret, frame = video.read()
                if not ret:
                    break
                
                # Always include first frame
                if prev_frame is None:
                    timestamp = frame_number / video_fps
                    frames_data.append({
                        'frame': frame,
                        'timestamp': timestamp,
                        'frame_number': frame_number,
                        'video_path': video_path,
                        'scene_change_score': 0.0
                    })
                    last_extracted_frame = frame_number
                
                elif frame_number - last_extracted_frame >= min_frame_interval:
                    # Calculate scene change
                    change_score = self._calculate_frame_difference(prev_frame, frame)
                    
                    if change_score > self.threshold:
                        timestamp = frame_number / video_fps
                        frames_data.append({
                            'frame': frame,
                            'timestamp': timestamp,
                            'frame_number': frame_number,
                            'video_path': video_path,
                            'scene_change_score': change_score
                        })
                        last_extracted_frame = frame_number
                
                prev_frame = frame
                frame_number += 1
            
            logger.info(f"Extracted {len(frames_data)} frames with scene changes")
            return frames_data
            
        finally:
            video.release()
    
    def _calculate_frame_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Calculate structural difference between frames
        Uses multiple methods for robust detection
        """
        # Convert to grayscale for faster computation
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Mean Squared Error
        mse = np.mean((gray1.astype(float) - gray2.astype(float)) ** 2)
        
        # Method 2: Histogram difference (for lighting changes)
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
        hist_diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
        
        # Combine metrics
        return mse + hist_diff * 0.01


class HybridFrameSampler(FrameSampler):
    """
    Combines uniform and adaptive sampling
    Ensures minimum coverage while capturing important changes
    """
    
    def __init__(
        self,
        base_sample_rate: float = 0.5,  # Base uniform sampling
        scene_threshold: float = 25.0,
        max_frames: int = 3600
    ):
        self.uniform_sampler = UniformFrameSampler(base_sample_rate, max_frames // 2)
        self.adaptive_sampler = AdaptiveFrameSampler(scene_threshold, 0.5, max_frames // 2)
    
    def extract_frames(self, video_path: str) -> List[Dict[str, Any]]:
        """
        Extract frames using both strategies and merge
        """
        # Get frames from both samplers
        uniform_frames = self.uniform_sampler.extract_frames(video_path)
        adaptive_frames = self.adaptive_sampler.extract_frames(video_path)
        
        # Merge and deduplicate based on timestamps
        all_frames = {}
        
        # Add uniform frames
        for frame_data in uniform_frames:
            timestamp = frame_data['timestamp']
            frame_data['sampling_method'] = 'uniform'
            all_frames[timestamp] = frame_data
        
        # Add adaptive frames (may overwrite uniform ones)
        for frame_data in adaptive_frames:
            timestamp = frame_data['timestamp']
            # Keep adaptive if it's close to uniform frame (prefer scene change detection)
            existing = all_frames.get(timestamp)
            if existing is None or abs(timestamp - min(all_frames.keys(), 
                                                      key=lambda x: abs(x - timestamp))) > 0.5:
                frame_data['sampling_method'] = 'adaptive'
                all_frames[timestamp] = frame_data
        
        # Sort by timestamp and return
        sorted_frames = sorted(all_frames.values(), key=lambda x: x['timestamp'])
        
        logger.info(f"Hybrid sampling extracted {len(sorted_frames)} frames "
                   f"({len(uniform_frames)} uniform + {len(adaptive_frames)} adaptive)")
        
        return sorted_frames


class OptimizedFrameExtractor:
    """
    Main frame extraction class with optimizations
    """
    
    def __init__(
        self,
        sample_rate: float = 1.0,
        strategy: str = 'uniform',
        max_frames_per_video: int = 3600,
        frame_size: tuple = (224, 224),
        quality_filter: bool = True
    ):
        self.sample_rate = sample_rate
        self.max_frames_per_video = max_frames_per_video
        self.frame_size = frame_size
        self.quality_filter = quality_filter
        
        # Initialize sampler
        if strategy == 'uniform':
            self.sampler = UniformFrameSampler(sample_rate, max_frames_per_video)
        elif strategy == 'adaptive':
            self.sampler = AdaptiveFrameSampler(max_frames=max_frames_per_video)
        elif strategy == 'hybrid':
            self.sampler = HybridFrameSampler(sample_rate * 0.7, max_frames=max_frames_per_video)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def extract_frames(self, video_path: str) -> List[Dict[str, Any]]:
        """
        Extract and preprocess frames from video
        """
        start_time = time.time()
        
        # Extract frames using selected strategy
        frames_data = self.sampler.extract_frames(video_path)
        
        # Post-process frames
        processed_frames = []
        for frame_data in frames_data:
            frame = frame_data['frame']
            
            # Resize frame
            if self.frame_size and frame.shape[:2] != self.frame_size:
                frame = cv2.resize(frame, self.frame_size)
            
            # Quality filtering
            if self.quality_filter and self._is_low_quality(frame):
                continue
            
            # Update frame data
            frame_data['frame'] = frame
            frame_data['processing_time'] = time.time() - start_time
            processed_frames.append(frame_data)
        
        total_time = time.time() - start_time
        logger.info(f"Frame extraction completed: {len(processed_frames)} frames "
                   f"in {total_time:.2f}s ({len(processed_frames)/total_time:.1f} fps)")
        
        return processed_frames
    
    def _is_low_quality(self, frame: np.ndarray) -> bool:
        """
        Filter out low quality frames
        """
        # Check for very dark or very bright frames
        mean_brightness = np.mean(frame)
        if mean_brightness < 20 or mean_brightness > 235:
            return True
        
        # Check for blurry frames using Laplacian variance
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 100:  # Threshold for blur detection
            return True
        
        return False
    
    def extract_frames_generator(self, video_path: str) -> Generator[Dict[str, Any], None, None]:
        """
        Memory-efficient frame extraction using generator
        """
        video = cv2.VideoCapture(video_path)
        
        if not video.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        try:
            video_fps = video.get(cv2.CAP_PROP_FPS)
            frame_interval = max(1, int(video_fps / self.sample_rate))
            
            frame_number = 0
            extracted_count = 0
            
            while extracted_count < self.max_frames_per_video:
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = video.read()
                
                if not ret:
                    break
                
                # Resize frame
                if self.frame_size:
                    frame = cv2.resize(frame, self.frame_size)
                
                # Quality check
                if self.quality_filter and self._is_low_quality(frame):
                    frame_number += frame_interval
                    continue
                
                timestamp = frame_number / video_fps
                yield {
                    'frame': frame,
                    'timestamp': timestamp,
                    'frame_number': frame_number,
                    'video_path': video_path
                }
                
                frame_number += frame_interval
                extracted_count += 1
                
        finally:
            video.release()


def choose_optimal_strategy(video_path: str) -> str:
    """
    Automatically choose the best extraction strategy based on video properties
    """
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        return 'uniform'  # Fallback
    
    try:
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Simple heuristics
        if duration < 300:  # Less than 5 minutes
            return 'uniform'  # Fast videos benefit from uniform sampling
        elif duration > 3600:  # More than 1 hour
            return 'adaptive'  # Long videos need scene detection
        else:
            return 'hybrid'  # Medium videos benefit from both
            
    finally:
        video.release()
