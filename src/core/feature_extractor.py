"""
Optimized Feature Extraction using Pre-trained Models
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Union, Optional
import cv2
from PIL import Image
import time
import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio
from transformers import CLIPProcessor, CLIPModel
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    High-performance feature extraction for images and video frames
    Uses CLIP model for semantic understanding
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "auto",
        batch_size: int = 32,
        num_threads: int = 4,
        cache_model: bool = True
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.cache_model = cache_model
        
        # Determine device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = None
        self.processor = None
        self._load_model()
        
        # Preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
        
        # Thread pool for CPU operations
        self.thread_pool = ThreadPoolExecutor(max_workers=num_threads)
        
        # Performance tracking
        self.extraction_times = []
        self.total_processed = 0
    
    def _load_model(self) -> None:
        """Load and initialize the CLIP model"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            start_time = time.time()
            
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model = CLIPModel.from_pretrained(self.model_name)
            
            # Move to device and set evaluation mode
            self.model.to(self.device)
            self.model.eval()
            
            # Enable optimizations
            if hasattr(torch, 'compile') and self.device.type == 'cuda':
                try:
                    self.model = torch.compile(self.model)
                    logger.info("Model compilation enabled")
                except Exception as e:
                    logger.warning(f"Model compilation failed: {e}")
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f}s")
            
            # Get output dimension
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
                dummy_output = self.model.get_image_features(dummy_input)
                self.output_dim = dummy_output.shape[1]
                logger.info(f"Feature dimension: {self.output_dim}")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _preprocess_image(self, image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Preprocess single image for model input
        """
        if isinstance(image, np.ndarray):
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # Apply transforms
        return self.transform(image)
    
    def _preprocess_batch(self, images: List[Union[np.ndarray, Image.Image]]) -> torch.Tensor:
        """
        Preprocess batch of images efficiently
        """
        # Use thread pool for parallel preprocessing
        if self.num_threads > 1 and len(images) > 4:
            processed_images = list(self.thread_pool.map(self._preprocess_image, images))
        else:
            processed_images = [self._preprocess_image(img) for img in images]
        
        # Stack into batch
        return torch.stack(processed_images)
    
    def extract_features(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Extract features from a single image
        """
        return self.extract_batch([image])[0]
    
    def extract_batch(self, images: List[Union[np.ndarray, Image.Image]]) -> np.ndarray:
        """
        Extract features from batch of images
        Optimized for throughput
        """
        if not images:
            return np.array([])
        
        start_time = time.time()
        
        try:
            # Preprocess batch
            batch_tensor = self._preprocess_batch(images)
            batch_tensor = batch_tensor.to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model.get_image_features(batch_tensor)
                
                # L2 normalize features
                features = nn.functional.normalize(features, p=2, dim=1)
            
            # Convert to numpy
            features_np = features.cpu().numpy().astype(np.float32)
            
            # Update metrics
            extraction_time = time.time() - start_time
            self.extraction_times.append(extraction_time)
            self.total_processed += len(images)
            
            if len(self.extraction_times) % 100 == 0:
                avg_time = np.mean(self.extraction_times[-100:])
                throughput = len(images) / extraction_time
                logger.info(f"Feature extraction: {throughput:.1f} images/sec, "
                          f"avg batch time: {avg_time:.3f}s")
            
            return features_np
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise
    
    def extract_from_video_frames(self, frames_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract features from video frames with metadata preservation
        """
        if not frames_data:
            return []
        
        logger.info(f"Extracting features from {len(frames_data)} frames")
        start_time = time.time()
        
        # Process in batches
        results = []
        for i in range(0, len(frames_data), self.batch_size):
            batch_frames = frames_data[i:i + self.batch_size]
            batch_images = [frame_data['frame'] for frame_data in batch_frames]
            
            # Extract features
            batch_features = self.extract_batch(batch_images)
            
            # Combine with metadata
            for frame_data, features in zip(batch_frames, batch_features):
                result = frame_data.copy()
                result['features'] = features
                result['feature_extraction_time'] = time.time() - start_time
                results.append(result)
        
        total_time = time.time() - start_time
        logger.info(f"Feature extraction completed: {len(results)} frames "
                   f"in {total_time:.2f}s ({len(results)/total_time:.1f} fps)")
        
        return results
    
    async def extract_batch_async(self, images: List[Union[np.ndarray, Image.Image]]) -> np.ndarray:
        """
        Asynchronous batch feature extraction
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.extract_batch, images)
    
    def extract_text_features(self, text: str) -> np.ndarray:
        """
        Extract features from text using CLIP text encoder
        """
        try:
            inputs = self.processor(text=[text], return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                text_features = nn.functional.normalize(text_features, p=2, dim=1)
            
            return text_features.cpu().numpy().astype(np.float32)[0]
            
        except Exception as e:
            logger.error(f"Text feature extraction failed: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics
        """
        if not self.extraction_times:
            return {
                'total_processed': 0,
                'avg_extraction_time': 0,
                'throughput': 0
            }
        
        avg_time = np.mean(self.extraction_times)
        total_images = sum(self.batch_size for _ in self.extraction_times)
        throughput = total_images / sum(self.extraction_times) if self.extraction_times else 0
        
        return {
            'total_processed': self.total_processed,
            'avg_extraction_time': avg_time,
            'throughput_images_per_sec': throughput,
            'device': str(self.device),
            'model_name': self.model_name,
            'output_dimension': self.output_dim
        }


class BatchProcessor:
    """
    Optimized batch processor for handling multiple extraction requests
    """
    
    def __init__(self, feature_extractor: FeatureExtractor, timeout_ms: int = 10):
        self.feature_extractor = feature_extractor
        self.timeout_ms = timeout_ms
        self.pending_requests = []
        self.results = {}
        self._processing = False
    
    async def process_request(self, request_id: str, images: List[Union[np.ndarray, Image.Image]]) -> np.ndarray:
        """
        Add request to batch and wait for result
        """
        # Create future for this request
        future = asyncio.Future()
        
        # Add to pending batch
        self.pending_requests.append({
            'id': request_id,
            'images': images,
            'future': future
        })
        
        # Process batch if full or start timeout
        if len(self.pending_requests) >= self.feature_extractor.batch_size:
            await self._process_batch()
        else:
            # Schedule timeout processing
            asyncio.create_task(self._timeout_processor())
        
        # Wait for result
        return await future
    
    async def _timeout_processor(self):
        """
        Process batch after timeout
        """
        await asyncio.sleep(self.timeout_ms / 1000.0)
        if self.pending_requests and not self._processing:
            await self._process_batch()
    
    async def _process_batch(self):
        """
        Process all pending requests in a batch
        """
        if not self.pending_requests or self._processing:
            return
        
        self._processing = True
        
        try:
            # Get batch to process
            batch = self.pending_requests[:self.feature_extractor.batch_size]
            self.pending_requests = self.pending_requests[self.feature_extractor.batch_size:]
            
            # Collect all images
            all_images = []
            request_indices = []
            
            for i, request in enumerate(batch):
                for image in request['images']:
                    all_images.append(image)
                    request_indices.append(i)
            
            # Process batch
            if all_images:
                features = await self.feature_extractor.extract_batch_async(all_images)
                
                # Distribute results back to requests
                feature_idx = 0
                for i, request in enumerate(batch):
                    request_features = []
                    for _ in request['images']:
                        if feature_idx < len(features):
                            request_features.append(features[feature_idx])
                            feature_idx += 1
                    
                    # Set result
                    if len(request_features) == 1:
                        request['future'].set_result(request_features[0])
                    else:
                        request['future'].set_result(np.array(request_features))
            
        except Exception as e:
            # Set exception for all pending requests
            for request in batch:
                if not request['future'].done():
                    request['future'].set_exception(e)
        
        finally:
            self._processing = False


class CachedFeatureExtractor(FeatureExtractor):
    """
    Feature extractor with caching to avoid recomputation
    """
    
    def __init__(self, *args, cache_size: int = 10000, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _get_image_hash(self, image: Union[np.ndarray, Image.Image]) -> str:
        """
        Generate hash for image caching
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Use a subset of pixels for faster hashing
        if image.size > 10000:
            # Sample pixels for large images
            step = int(np.sqrt(image.size / 1000))
            sample = image[::step, ::step]
        else:
            sample = image
        
        return str(hash(sample.tobytes()))
    
    def extract_features(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Extract features with caching
        """
        # Check cache
        image_hash = self._get_image_hash(image)
        
        if image_hash in self.cache:
            self.cache_hits += 1
            return self.cache[image_hash].copy()
        
        # Extract features
        self.cache_misses += 1
        features = super().extract_features(image)
        
        # Store in cache
        if len(self.cache) < self.cache_size:
            self.cache[image_hash] = features.copy()
        elif self.cache_size > 0:
            # Remove random item to make space
            remove_key = next(iter(self.cache))
            del self.cache[remove_key]
            self.cache[image_hash] = features.copy()
        
        return features
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics
        """
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache),
            'max_cache_size': self.cache_size
        }
