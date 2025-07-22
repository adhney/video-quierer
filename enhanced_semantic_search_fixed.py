#!/usr/bin/env python3
"""
🎬 Enhanced Semantic Video Search - Fixed Version
=================================================
Improved version with better frame sampling, more features, and higher accuracy
Compatible with existing HNSW index
"""

import sys
import os
import time
import cv2
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def check_dependencies():
    """Check what we have available"""
    available = {}
    
    try:
        import cv2
        available['opencv'] = cv2.__version__
    except ImportError:
        available['opencv'] = None
    
    try:
        import torch
        available['torch'] = torch.__version__
    except ImportError:
        available['torch'] = None
    
    try:
        from transformers import CLIPProcessor, CLIPModel
        available['clip'] = True
    except ImportError:
        available['clip'] = False
    
    return available

class FixedEnhancedSemanticVideoProcessor:
    """Enhanced video processor with high-quality frame extraction - Fixed for existing HNSW"""
    
    def __init__(self, use_clip=False, enhanced_mode=True):
        # Import the correct HNSW class
        try:
            from indexes.hnsw import HNSWIndex
            self.index = HNSWIndex(dimension=512, M=16, ef_construction=200)
        except ImportError:
            print("⚠️ Using fallback simple index")
            self.index = self._create_fallback_index()
        
        self.video_metadata = {}
        self.frame_count = 0
        self.use_clip = use_clip
        self.enhanced_mode = enhanced_mode
        
        # Enhanced sampling parameters
        self.frame_sampling_rates = {
            'ultra_high': 5.0,    # 5 frames per second
            'high': 2.0,          # 2 frames per second  
            'medium': 1.0,        # 1 frame per second
            'low': 0.5,           # 1 frame every 2 seconds
            'adaptive': 'auto'    # Smart sampling based on content
        }
        
        # Initialize CLIP if available
        if use_clip:
            try:
                from transformers import CLIPProcessor, CLIPModel
                import torch
                
                print("🧠 Loading CLIP model for semantic understanding...")
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.clip_model.to(self.device)
                self.clip_model.eval()
                print(f"✅ CLIP loaded on {self.device}")
            except Exception as e:
                print(f"⚠️ CLIP loading failed: {e}")
                self.use_clip = False
        
        if not self.use_clip:
            print("🔧 Using enhanced visual features (no CLIP)")
        
        print("🎬 Enhanced Semantic Video Processor initialized")
        if enhanced_mode:
            print("⚡ Enhanced mode: More frames, better quality, smarter sampling")
    
    def _create_fallback_index(self):
        """Create a simple fallback index if HNSW import fails"""
        class SimpleIndex:
            def __init__(self):
                self.vectors = {}
                self.ids = []
            
            def add(self, vector, node_id):
                self.vectors[node_id] = vector
                self.ids.append(node_id)
            
            def build(self, ef=None):
                pass
            
            def search(self, query_vector, k=5):
                if not self.vectors:
                    return [], []
                
                distances = []
                for node_id, vector in self.vectors.items():
                    dist = 1.0 - np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
                    distances.append((dist, node_id))
                
                distances.sort()
                k = min(k, len(distances))
                
                return [node_id for _, node_id in distances[:k]], [dist for dist, _ in distances[:k]]
        
        return SimpleIndex()
    
    def detect_scene_changes(self, frames, threshold=0.3):
        """Detect scene changes to sample more intelligently"""
        if len(frames) < 2:
            return [0]
        
        scene_changes = [0]  # Always include first frame
        
        for i in range(1, len(frames)):
            # Convert to grayscale for comparison
            gray1 = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            
            # Calculate histogram difference
            hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
            
            # Compare histograms
            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            
            if correlation < (1 - threshold):  # Significant change
                scene_changes.append(i)
        
        return scene_changes
    
    def extract_frames_enhanced(self, video_path, sampling_rate='high', max_frames=300):
        """Enhanced frame extraction with multiple sampling strategies"""
        print(f"📼 Enhanced frame extraction from: {os.path.basename(video_path)}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else total_frames / 30
        
        print(f"   📊 Video: {duration:.1f}s, {fps:.1f}fps, {total_frames} frames")
        
        # Determine sampling interval
        if isinstance(sampling_rate, str):
            target_fps = self.frame_sampling_rates.get(sampling_rate, 1.0)
        else:
            target_fps = sampling_rate
        
        frame_interval = max(1, int(fps / target_fps)) if fps > 0 else 30
        
        # Limit total frames extracted
        estimated_frames = total_frames // frame_interval
        if estimated_frames > max_frames:
            frame_interval = total_frames // max_frames
            print(f"   🎯 Adjusting interval to limit to {max_frames} frames")
        
        print(f"   ⚙️ Sampling every {frame_interval} frames ({target_fps} fps target)")
        
        frames_data = []
        frame_number = 0
        extracted = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_number % frame_interval == 0:
                timestamp = frame_number / fps if fps > 0 else frame_number / 30
                
                frames_data.append({
                    'frame': frame,
                    'timestamp': timestamp,
                    'frame_number': frame_number
                })
                extracted += 1
                
                if extracted >= max_frames:
                    print(f"   ⚠️ Reached max frames limit ({max_frames})")
                    break
            
            frame_number += 1
        
        cap.release()
        print(f"   ✅ Extracted {extracted} frames")
        return frames_data
    
    def extract_enhanced_visual_features(self, frame):
        """Extract enhanced visual features from frame"""
        # Convert to different color spaces for richer features
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        features = {}
        
        # Basic statistics (expanded)
        features['brightness_mean'] = float(np.mean(gray))
        features['brightness_std'] = float(np.std(gray))
        features['brightness_median'] = float(np.median(gray))
        
        # Color analysis
        features['hue_mean'] = float(np.mean(hsv[:,:,0]))
        features['saturation_mean'] = float(np.mean(hsv[:,:,1]))
        features['value_mean'] = float(np.mean(hsv[:,:,2]))
        
        # Edge analysis (enhanced)
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = float(np.sum(edges > 0) / edges.size)
        features['edge_strength'] = float(np.mean(edges[edges > 0])) if np.any(edges > 0) else 0.0
        
        # Contrast analysis
        features['contrast'] = float(np.std(gray))
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-10)  # Normalize
        features['histogram_entropy'] = float(-np.sum(hist * np.log2(hist + 1e-10)))
        
        # Line detection (for presentations/diagrams)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
        features['line_count'] = float(len(lines)) if lines is not None else 0.0
        features['has_lines'] = float(features['line_count'] > 10)
        
        # Shape analysis
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        features['shape_count'] = float(len(contours))
        
        # Analyze contour complexity
        if contours:
            areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 100]
            features['large_shapes'] = float(len(areas))
            features['avg_shape_area'] = float(np.mean(areas)) if areas else 0.0
        else:
            features['large_shapes'] = 0.0
            features['avg_shape_area'] = 0.0
        
        # Presentation slide detection
        h, w = gray.shape
        # Check for white/light background (common in presentations)
        background_brightness = np.mean(gray[h//4:3*h//4, w//4:3*w//4])
        features['is_bright_slide'] = float(background_brightness > 200)
        features['is_dark_slide'] = float(background_brightness < 50)
        
        # Text region detection (approximation)
        # Look for horizontal text-like patterns
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        text_regions = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, horizontal_kernel)
        features['text_region_density'] = float(np.sum(text_regions > 0) / text_regions.size)
        
        return features
    
    def get_clip_features(self, frame):
        """Get CLIP embeddings for semantic understanding"""
        if not self.use_clip:
            return np.zeros(512, dtype=np.float32)
        
        try:
            import torch
            from PIL import Image
            
            # Convert frame to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Process with CLIP
            inputs = self.clip_processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            
            return image_features.cpu().numpy().flatten().astype(np.float32)
        
        except Exception as e:
            print(f"⚠️ CLIP feature extraction failed: {e}")
            return np.zeros(512, dtype=np.float32)
    
    def process_video_enhanced(self, video_path, sampling_mode='high', max_frames=300):
        """Process video with enhanced frame extraction and features"""
        print(f"\n🎬 Processing video: {os.path.basename(video_path)}")
        print(f"⚙️ Sampling mode: {sampling_mode}, Max frames: {max_frames}")
        
        start_time = time.time()
        
        # Extract frames with enhanced sampling
        frames_data = self.extract_frames_enhanced(video_path, sampling_mode, max_frames)
        
        if not frames_data:
            print("❌ No frames extracted")
            return
        
        video_name = os.path.basename(video_path)
        frame_features = []
        
        print(f"🔍 Extracting features from {len(frames_data)} frames...")
        
        for i, frame_data in enumerate(frames_data):
            frame = frame_data['frame']
            timestamp = frame_data['timestamp']
            
            # Extract enhanced visual features
            visual_features = self.extract_enhanced_visual_features(frame)
            
            # Get CLIP semantic features if available
            if self.use_clip:
                clip_features = self.get_clip_features(frame)
                # Combine visual and semantic features
                visual_array = np.array(list(visual_features.values()), dtype=np.float32)
                
                # Ensure consistent dimensions - truncate visual features to fit with CLIP
                max_visual_features = 512 - len(clip_features)  # Reserve space for CLIP
                if len(visual_array) > max_visual_features:
                    visual_array = visual_array[:max_visual_features]
                
                combined_features = np.concatenate([visual_array, clip_features])
                
                # Ensure exactly 512 dimensions
                if len(combined_features) > 512:
                    combined_features = combined_features[:512]
                elif len(combined_features) < 512:
                    combined_features = np.pad(combined_features, 
                                             (0, 512 - len(combined_features)), 
                                             'constant')
                combined_features = combined_features.astype(np.float32)
            else:
                # Use enhanced visual features only
                visual_array = np.array(list(visual_features.values()), dtype=np.float32)
                # Ensure exactly 512 dimensions
                if len(visual_array) > 512:
                    combined_features = visual_array[:512]
                elif len(visual_array) < 512:
                    combined_features = np.pad(visual_array, 
                                             (0, 512 - len(visual_array)), 
                                             'constant')
                else:
                    combined_features = visual_array
                combined_features = combined_features.astype(np.float32)
            
            # Normalize features
            norm = np.linalg.norm(combined_features)
            if norm > 0:
                combined_features = combined_features / norm
            else:
                print(f"⚠️ Warning: Zero norm feature vector at frame {i}, using small random values")
                combined_features = np.random.normal(0, 0.01, 512).astype(np.float32)
                combined_features = combined_features / np.linalg.norm(combined_features)
            
            # Add to index - using correct method with dimension validation
            if len(combined_features) != 512:
                print(f"⚠️ Warning: Feature vector has {len(combined_features)} dimensions, expected 512")
                if len(combined_features) > 512:
                    combined_features = combined_features[:512]
                else:
                    combined_features = np.pad(combined_features, (0, 512 - len(combined_features)), 'constant')
                combined_features = combined_features.astype(np.float32)
            
            self.index.add(combined_features, self.frame_count)
            
            # Store metadata
            self.video_metadata[self.frame_count] = {
                'video_name': video_name,
                'timestamp': timestamp,
                'visual_features': visual_features,
                'has_clip': self.use_clip
            }
            
            frame_features.append(combined_features)
            self.frame_count += 1
            
            # Progress indicator
            if (i + 1) % 50 == 0:
                print(f"   📊 Processed {i + 1}/{len(frames_data)} frames")
        
        # Build the index
        print("🔨 Building search index...")
        if hasattr(self.index, 'build'):
            self.index.build(ef=200)
        
        processing_time = time.time() - start_time
        print(f"✅ Video processed in {processing_time:.1f}s")
        print(f"📊 Total frames in index: {self.frame_count}")
        
        # Show statistics
        if frames_data:
            recent_metadata = [self.video_metadata[i] for i in range(self.frame_count - len(frames_data), self.frame_count)]
            
            bright_frames = sum(1 for meta in recent_metadata if meta['visual_features']['is_bright_slide'] > 0.5)
            text_frames = sum(1 for meta in recent_metadata if meta['visual_features']['text_region_density'] > 0.1)
            
            print(f"📈 Video characteristics:")
            print(f"   💡 Bright slides: {bright_frames}/{len(frames_data)} ({100*bright_frames/len(frames_data):.1f}%)")
            print(f"   📝 Text-heavy frames: {text_frames}/{len(frames_data)} ({100*text_frames/len(frames_data):.1f}%)")
    
    def search_enhanced(self, query, k=5):
        """Enhanced search with feature boosting"""
        if self.frame_count == 0:
            print("⚠️ No videos processed yet")
            return [], 0
        
        start_time = time.time()
        
        if self.use_clip:
            # Semantic search using CLIP
            try:
                import torch
                
                # Encode the query
                inputs = self.clip_processor(text=[query], return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    text_features = self.clip_model.get_text_features(**inputs)
                    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
                
                query_vector = text_features.cpu().numpy().flatten().astype(np.float32)
                
                # Ensure exactly 512 dimensions
                if len(query_vector) > 512:
                    query_vector = query_vector[:512]
                elif len(query_vector) < 512:
                    query_vector = np.pad(query_vector, (0, 512 - len(query_vector)), 'constant')
                
                # Normalize
                norm = np.linalg.norm(query_vector)
                if norm > 0:
                    query_vector = query_vector / norm
                else:
                    print("⚠️ Warning: Zero norm query vector")
                    query_vector = np.ones(512, dtype=np.float32) / np.sqrt(512)
                
                # Search the index
                if hasattr(self.index, 'search'):
                    search_results = self.index.search(query_vector, k=min(k*3, self.frame_count))
                    indices = [result['id'] for result in search_results]
                    distances = [result['distance'] for result in search_results]
                else:
                    # Fallback search
                    indices, distances = self._fallback_search(query_vector, k*3)
                
            except Exception as e:
                print(f"⚠️ CLIP search failed: {e}")
                # Fallback to visual feature search
                indices, distances = self._enhanced_visual_search(query, k)
        else:
            # Enhanced visual feature search
            indices, distances = self._enhanced_visual_search(query, k)
        
        # Process results
        results = []
        for idx, distance in zip(indices, distances):
            if idx in self.video_metadata:
                metadata = self.video_metadata[idx]
                
                results.append({
                    'video_name': metadata['video_name'],
                    'timestamp': metadata['timestamp'],
                    'score': 1.0 - distance,  # Convert distance to similarity score
                    'distance': distance
                })
        
        # Sort by score (higher is better)
        results.sort(key=lambda x: x['score'], reverse=True)
        
        search_time = (time.time() - start_time) * 1000
        
        # Return top k results
        final_results = results[:k]
        
        return final_results, search_time
    
    def _enhanced_visual_search(self, query, k):
        """Enhanced visual search based on query keywords"""
        # Create query vector based on visual characteristics
        query_features = np.zeros(512, dtype=np.float32)
        
        query_lower = query.lower()
        
        # Map query terms to visual features (first 20 positions for visual features)
        if 'bright' in query_lower:
            query_features[0] = 200.0  # High brightness
        if 'dark' in query_lower:
            query_features[0] = 50.0   # Low brightness
        if 'complex' in query_lower or 'diagram' in query_lower:
            query_features[6] = 0.2  # High edge density
        if 'simple' in query_lower:
            query_features[6] = 0.05 # Low edge density
        if 'colorful' in query_lower:
            query_features[4] = 150.0  # High saturation
        if 'text' in query_lower or 'presentation' in query_lower:
            query_features[19] = 0.15 # High text region density
        
        # Normalize and ensure exactly 512 dimensions
        norm = np.linalg.norm(query_features)
        if norm > 0:
            query_features = query_features / norm
        else:
            print("⚠️ Warning: Zero norm visual query features")
            query_features = np.ones(512, dtype=np.float32) / np.sqrt(512)
        
        # Double-check dimensions
        if len(query_features) != 512:
            if len(query_features) > 512:
                query_features = query_features[:512]
            else:
                query_features = np.pad(query_features, (0, 512 - len(query_features)), 'constant')
            query_features = query_features.astype(np.float32)
        
        if hasattr(self.index, 'search'):
            search_results = self.index.search(query_features, k=min(k*2, self.frame_count))
            indices = [result['id'] for result in search_results]
            distances = [result['distance'] for result in search_results]
            return indices, distances
        else:
            return self._fallback_search(query_features, k*2)
    
    def _fallback_search(self, query_vector, k):
        """Fallback search when main search fails"""
        if not hasattr(self.index, 'vectors'):
            return [], []
        
        distances = []
        for node_id, vector in self.index.vectors.items():
            dist = 1.0 - np.dot(query_vector, vector)
            distances.append((dist, node_id))
        
        distances.sort()
        k = min(k, len(distances))
        
        return [node_id for _, node_id in distances[:k]], [dist for dist, _ in distances[:k]]

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

def main():
    """Enhanced main function with configuration options"""
    print("🎬 Enhanced Semantic Video Search (Fixed)")
    print("=" * 50)
    
    # Check dependencies
    deps = check_dependencies()
    print("📦 Available components:")
    for name, version in deps.items():
        if version:
            print(f"   ✅ {name}: {version}")
        else:
            print(f"   ❌ {name}: Not available")
    
    use_clip = deps['torch'] and deps['clip']
    
    # Configuration menu
    print("\n⚙️ Enhancement Configuration:")
    print("1. Sampling modes:")
    print("   • ultra_high: 5 fps (most frames, best quality)")
    print("   • high: 2 fps (good balance)")  
    print("   • medium: 1 fps (standard)")
    
    sampling_mode = input("Choose sampling mode (ultra_high/high/medium) [high]: ").strip() or 'high'
    max_frames = int(input("Maximum frames per video [300]: ").strip() or '300')
    
    print(f"\n🎯 Using {sampling_mode} sampling with max {max_frames} frames")
    
    # Initialize enhanced processor
    processor = FixedEnhancedSemanticVideoProcessor(use_clip=use_clip, enhanced_mode=True)
    
    # Check for videos
    videos_dir = Path("videos")
    if not videos_dir.exists():
        videos_dir.mkdir()
        print(f"\n📁 Created videos directory: {videos_dir.absolute()}")
        print("📋 Add your video files and run again!")
        return 0
    
    # Find videos
    video_files = []
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        video_files.extend(list(videos_dir.glob(f"*{ext}")))
        video_files.extend(list(videos_dir.glob(f"*{ext.upper()}")))
    
    if not video_files:
        print("📁 No video files found in videos/ directory")
        return 0
    
    print(f"\n📼 Found videos: {[f.name for f in video_files]}")
    
    # Process videos with enhanced extraction
    for video_file in video_files:
        processor.process_video_enhanced(str(video_file), sampling_mode, max_frames)
    
    # Interactive search with enhanced features
    print(f"\n🎯 Enhanced Semantic Search")
    print("-" * 40)
    print("🔍 Enhanced features:")
    print("  • More frames for better accuracy")
    print("  • Enhanced visual analysis")
    print("  • Smart content detection")
    print("\nTry queries like:")
    print("  • 'bright presentation slide'")
    print("  • 'complex system diagram'") 
    print("  • 'text heavy frame'")
    print("  • 'colorful chart'")
    print("  • 'goal celebration'")
    print("  • 'car interior'")
    print("Type 'quit' to exit")
    
    while True:
        try:
            query = input("\n🔍 Enhanced search: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if query:
                results, search_time = processor.search_enhanced(query, k=5)
                
                print(f"⏱️  Enhanced search: {search_time:.1f}ms")
                print(f"📊 Results:")
                
                for i, result in enumerate(results, 1):
                    formatted_time = format_timestamp(result['timestamp'])
                    print(f"   {i}. {result['video_name']} at {formatted_time}")
                    print(f"      Score: {result['score']:.3f}")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
    
    print(f"\n🎉 Enhanced semantic video search completed!")
    return 0

if __name__ == "__main__":
    exit(main())
