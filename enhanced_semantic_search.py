#!/usr/bin/env python3
"""
🎬 Enhanced Semantic Video Search - High Quality Frame Extraction
================================================================
Improved version with better frame sampling, more features, and higher accuracy
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

class EnhancedSemanticVideoProcessor:
    """Enhanced video processor with high-quality frame extraction and better features"""
    
    def __init__(self, use_clip=False, enhanced_mode=True):
        # Import only what we need
        from indexes.hnsw import OptimizedHNSWIndex
        
        self.index = OptimizedHNSWIndex(dimension=512, M=16, ef_construction=200)
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
    
    def adaptive_frame_sampling(self, video_path, target_frames=200):
        """Smart frame sampling based on content analysis"""
        print(f"🎯 Adaptive sampling targeting {target_frames} frames")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else total_frames / 30
        
        print(f"   📊 Video: {duration:.1f}s, {fps:.1f}fps, {total_frames} total frames")
        
        # First pass: Sample uniformly for scene detection
        preview_frames = []
        preview_interval = max(1, total_frames // 100)  # Sample ~100 frames for analysis
        
        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_number % preview_interval == 0:
                preview_frames.append(frame.copy())
            
            frame_number += 1
        
        cap.release()
        
        # Detect scene changes
        scene_changes = self.detect_scene_changes(preview_frames)
        print(f"   🎬 Detected {len(scene_changes)} scene changes")
        
        # Second pass: Extract frames based on scene analysis
        cap = cv2.VideoCapture(video_path)
        frames_data = []
        
        # Calculate smart sampling intervals
        if len(scene_changes) > 1:
            # More frames around scene changes
            frames_per_scene = target_frames // len(scene_changes)
            frames_per_scene = max(5, min(frames_per_scene, 20))  # 5-20 frames per scene
        else:
            # Uniform sampling
            frames_per_scene = target_frames // 10
        
        frame_number = 0
        extracted = 0
        
        for scene_idx in range(len(scene_changes)):
            scene_start = scene_changes[scene_idx] * preview_interval
            scene_end = scene_changes[scene_idx + 1] * preview_interval if scene_idx + 1 < len(scene_changes) else total_frames
            
            # Extract frames from this scene
            scene_interval = max(1, (scene_end - scene_start) // frames_per_scene)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, scene_start)
            scene_frame_num = scene_start
            
            while scene_frame_num < scene_end and extracted < target_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if (scene_frame_num - scene_start) % scene_interval == 0:
                    timestamp = scene_frame_num / fps if fps > 0 else scene_frame_num / 30
                    
                    frames_data.append({
                        'frame': frame,
                        'timestamp': timestamp,
                        'frame_number': scene_frame_num,
                        'scene_id': scene_idx
                    })
                    extracted += 1
                
                scene_frame_num += 1
        
        cap.release()
        print(f"   ✅ Extracted {extracted} frames using adaptive sampling")
        return frames_data
    
    def extract_frames_enhanced(self, video_path, sampling_rate='high', max_frames=300):
        """Enhanced frame extraction with multiple sampling strategies"""
        print(f"📼 Enhanced frame extraction from: {os.path.basename(video_path)}")
        
        if sampling_rate == 'adaptive':
            return self.adaptive_frame_sampling(video_path, target_frames=max_frames)
        
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
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        features = {}
        
        # Basic statistics (expanded)
        features['brightness_mean'] = np.mean(gray)
        features['brightness_std'] = np.std(gray)
        features['brightness_median'] = np.median(gray)
        
        # Color analysis
        features['hue_mean'] = np.mean(hsv[:,:,0])
        features['saturation_mean'] = np.mean(hsv[:,:,1])
        features['value_mean'] = np.mean(hsv[:,:,2])
        
        # LAB color space
        features['l_mean'] = np.mean(lab[:,:,0])  # Lightness
        features['a_mean'] = np.mean(lab[:,:,1])  # Green-Red
        features['b_mean'] = np.mean(lab[:,:,2])  # Blue-Yellow
        
        # Edge analysis (enhanced)
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / edges.size
        features['edge_strength'] = np.mean(edges[edges > 0]) if np.any(edges > 0) else 0
        
        # Texture analysis
        # Local Binary Pattern approximation
        kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
        texture = cv2.filter2D(gray, -1, kernel)
        features['texture_variance'] = np.var(texture)
        features['texture_mean'] = np.mean(np.abs(texture))
        
        # Contrast analysis
        features['contrast'] = np.std(gray)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        features['histogram_entropy'] = -np.sum(hist * np.log2(hist + 1e-10))
        
        # Line detection (for presentations/diagrams)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
        features['line_count'] = len(lines) if lines is not None else 0
        features['has_lines'] = features['line_count'] > 10
        
        # Shape analysis
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        features['shape_count'] = len(contours)
        
        # Analyze contour complexity
        if contours:
            areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 100]
            features['large_shapes'] = len(areas)
            features['avg_shape_area'] = np.mean(areas) if areas else 0
        else:
            features['large_shapes'] = 0
            features['avg_shape_area'] = 0
        
        # Grid/structure detection (for presentations)
        # Check for regular patterns
        gray_small = cv2.resize(gray, (64, 64))
        fft = np.fft.fft2(gray_small)
        fft_magnitude = np.abs(fft)
        features['structure_score'] = np.std(fft_magnitude)
        
        # Presentation slide detection
        h, w = gray.shape
        # Check for white/light background (common in presentations)
        background_brightness = np.mean(gray[h//4:3*h//4, w//4:3*w//4])
        features['is_bright_slide'] = background_brightness > 200
        features['is_dark_slide'] = background_brightness < 50
        
        # Text region detection (approximation)
        # Look for horizontal text-like patterns
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        text_regions = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, horizontal_kernel)
        features['text_region_density'] = np.sum(text_regions > 0) / text_regions.size
        
        return features
    
    def get_clip_features(self, frame):
        """Get CLIP embeddings for semantic understanding"""
        if not self.use_clip:
            return np.zeros(512)
        
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
            
            return image_features.cpu().numpy().flatten()
        
        except Exception as e:
            print(f"⚠️ CLIP feature extraction failed: {e}")
            return np.zeros(512)
    
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
                combined_features = np.concatenate([
                    np.array(list(visual_features.values())),
                    clip_features
                ])
            else:
                # Use enhanced visual features only
                combined_features = np.array(list(visual_features.values()))
                # Pad to 512 dimensions for consistency
                if len(combined_features) < 512:
                    combined_features = np.pad(combined_features, 
                                             (0, 512 - len(combined_features)), 
                                             'constant')
                combined_features = combined_features[:512]  # Ensure exactly 512 dims
            
            # Add to index
            self.index.add_item(self.frame_count, combined_features)
            
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
        self.index.build(ef=200)
        
        processing_time = time.time() - start_time
        print(f"✅ Video processed in {processing_time:.1f}s")
        print(f"📊 Total frames in index: {self.frame_count}")
        
        # Show statistics
        if frames_data:
            avg_features = np.mean([list(fd['visual_features'].values()) 
                                  for fd in [self.video_metadata[i] 
                                           for i in range(self.frame_count - len(frames_data), self.frame_count)]], axis=0)
            
            bright_frames = sum(1 for i in range(self.frame_count - len(frames_data), self.frame_count) 
                              if self.video_metadata[i]['visual_features']['is_bright_slide'])
            text_frames = sum(1 for i in range(self.frame_count - len(frames_data), self.frame_count) 
                            if self.video_metadata[i]['visual_features']['text_region_density'] > 0.1)
            
            print(f"📈 Video characteristics:")
            print(f"   💡 Bright slides: {bright_frames}/{len(frames_data)} ({100*bright_frames/len(frames_data):.1f}%)")
            print(f"   📝 Text-heavy frames: {text_frames}/{len(frames_data)} ({100*text_frames/len(frames_data):.1f}%)")
            print(f"   🎨 Average brightness: {avg_features[0]:.1f}")
            print(f"   🔍 Average edge density: {avg_features[3]:.3f}")
    
    def search_enhanced(self, query, k=5, boost_factors=None):
        """Enhanced search with feature boosting"""
        if self.frame_count == 0:
            print("⚠️ No videos processed yet")
            return [], 0
        
        start_time = time.time()
        
        # Default boost factors for different types of content
        if boost_factors is None:
            boost_factors = {
                'bright_slide': 1.2,      # Boost bright presentation slides
                'text_heavy': 1.1,        # Boost frames with text
                'complex_diagram': 1.3,   # Boost complex visual content
                'high_contrast': 1.1      # Boost high contrast content
            }
        
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
                
                query_vector = text_features.cpu().numpy().flatten()
                
                # Search the index
                indices, distances = self.index.query(query_vector, k=min(k*3, self.frame_count))
                
            except Exception as e:
                print(f"⚠️ CLIP search failed: {e}")
                # Fallback to visual feature search
                indices, distances = self._fallback_visual_search(query, k)
        else:
            # Enhanced visual feature search
            indices, distances = self._enhanced_visual_search(query, k)
        
        # Apply boost factors and re-rank
        enhanced_results = []
        for idx, distance in zip(indices, distances):
            if idx in self.video_metadata:
                metadata = self.video_metadata[idx]
                visual_features = metadata['visual_features']
                
                # Calculate boost score
                boost_score = 1.0
                
                if 'bright' in query.lower() and visual_features['is_bright_slide']:
                    boost_score *= boost_factors['bright_slide']
                
                if ('text' in query.lower() or 'presentation' in query.lower()) and visual_features['text_region_density'] > 0.1:
                    boost_score *= boost_factors['text_heavy']
                
                if ('complex' in query.lower() or 'diagram' in query.lower()) and visual_features['edge_density'] > 0.1:
                    boost_score *= boost_factors['complex_diagram']
                
                if visual_features['contrast'] > 50:
                    boost_score *= boost_factors['high_contrast']
                
                # Calculate final score (lower distance is better, higher boost is better)
                final_score = (1.0 - distance) * boost_score
                
                enhanced_results.append({
                    'video_name': metadata['video_name'],
                    'timestamp': metadata['timestamp'],
                    'score': final_score,
                    'original_distance': distance,
                    'boost_applied': boost_score
                })
        
        # Sort by enhanced score
        enhanced_results.sort(key=lambda x: x['score'], reverse=True)
        
        search_time = (time.time() - start_time) * 1000
        
        # Return top k results
        final_results = enhanced_results[:k]
        
        return final_results, search_time
    
    def _enhanced_visual_search(self, query, k):
        """Enhanced visual search based on query keywords"""
        # Create query vector based on visual characteristics
        query_features = np.zeros(512)
        
        query_lower = query.lower()
        
        # Map query terms to visual features
        if 'bright' in query_lower:
            query_features[0] = 200  # High brightness
        if 'dark' in query_lower:
            query_features[0] = 50   # Low brightness
        if 'complex' in query_lower or 'diagram' in query_lower:
            query_features[3] = 0.2  # High edge density
        if 'simple' in query_lower:
            query_features[3] = 0.05 # Low edge density
        if 'colorful' in query_lower:
            query_features[4] = 150  # High saturation
        if 'text' in query_lower or 'presentation' in query_lower:
            query_features[20] = 0.15 # High text region density
        
        # Normalize
        if np.linalg.norm(query_features) > 0:
            query_features = query_features / np.linalg.norm(query_features)
        
        return self.index.query(query_features, k=min(k*2, self.frame_count))
    
    def _fallback_visual_search(self, query, k):
        """Fallback visual search when CLIP fails"""
        return self._enhanced_visual_search(query, k)

def main():
    """Enhanced main function with configuration options"""
    print("🎬 Enhanced Semantic Video Search")
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
    print("   • adaptive: Smart scene-based sampling")
    
    sampling_mode = input("Choose sampling mode (ultra_high/high/medium/adaptive) [high]: ").strip() or 'high'
    max_frames = int(input("Maximum frames per video [300]: ").strip() or '300')
    
    print(f"\n🎯 Using {sampling_mode} sampling with max {max_frames} frames")
    
    # Initialize enhanced processor
    processor = EnhancedSemanticVideoProcessor(use_clip=use_clip, enhanced_mode=True)
    
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
    print("  • Scene change detection")
    print("  • Smart content boosting")
    print("  • Enhanced visual analysis")
    print("\nTry queries like:")
    print("  • 'bright presentation slide'")
    print("  • 'complex system diagram'") 
    print("  • 'text heavy frame'")
    print("  • 'colorful chart'")
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
                    boost_info = f" (boost: {result['boost_applied']:.2f}x)" if result['boost_applied'] != 1.0 else ""
                    print(f"   {i}. {result['video_name']} at {result['timestamp']:.1f}s")
                    print(f"      Score: {result['score']:.3f}{boost_info}")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
    
    print(f"\n🎉 Enhanced semantic video search completed!")
    return 0

if __name__ == "__main__":
    exit(main())
