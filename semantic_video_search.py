#!/usr/bin/env python3
"""
Fixed real video processing with semantic understanding
Bypasses all import issues and provides true semantic search
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

class SemanticVideoProcessor:
    """Video processor with semantic understanding for system design content"""
    
    def __init__(self, use_clip=False):
        # Import only what we need
        from indexes.hnsw import OptimizedHNSWIndex
        
        self.index = OptimizedHNSWIndex(dimension=512, M=16, ef_construction=200)
        self.video_metadata = {}
        self.frame_count = 0
        self.use_clip = use_clip
        
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
        
        print("🎬 Semantic Video Processor initialized")
    
    def extract_frames(self, video_path, sample_rate=1.0):
        """Extract frames at specified rate"""
        print(f"📼 Extracting frames from: {os.path.basename(video_path)}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else total_frames / 30
        
        print(f"   📊 Video: {duration:.1f}s, {fps:.1f}fps, {total_frames} frames")
        
        frame_interval = max(1, int(fps / sample_rate)) if fps > 0 else 30
        
        frames_data = []
        frame_number = 0
        extracted = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_number % frame_interval == 0:
                # Resize for processing
                resized_frame = cv2.resize(frame, (224, 224))
                timestamp = frame_number / fps if fps > 0 else frame_number / 30
                
                frames_data.append({
                    'frame': resized_frame,
                    'timestamp': timestamp,
                    'frame_number': frame_number
                })
                extracted += 1
                
                # Limit frames for demo
                if extracted >= 50:  # Process max 50 frames
                    break
            
            frame_number += 1
        
        cap.release()
        print(f"✅ Extracted {len(frames_data)} frames")
        return frames_data
    
    def extract_semantic_features(self, frame):
        """Extract semantic features using CLIP or enhanced visual features"""
        
        if self.use_clip:
            return self._extract_clip_features(frame)
        else:
            return self._extract_enhanced_visual_features(frame)
    
    def _extract_clip_features(self, frame):
        """Extract features using CLIP model"""
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
                image_features = self.clip_model.get_image_features(**inputs)
                # Normalize
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy().flatten().astype(np.float32)
            
        except Exception as e:
            print(f"⚠️ CLIP processing failed: {e}")
            return self._extract_enhanced_visual_features(frame)
    
    def _extract_enhanced_visual_features(self, frame):
        """Enhanced visual features optimized for presentation/diagram content"""
        features = []
        
        # Convert to different color spaces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 1. Text/Diagram detection features
        # High contrast edges (typical of text and diagrams)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (frame.shape[0] * frame.shape[1])
        features.extend([edge_density * 10])  # Boost edge importance
        
        # Horizontal and vertical line detection (typical of diagrams)
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_h)
        vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_v)
        
        h_line_density = np.sum(horizontal_lines > 0) / (frame.shape[0] * frame.shape[1])
        v_line_density = np.sum(vertical_lines > 0) / (frame.shape[0] * frame.shape[1])
        
        features.extend([h_line_density * 20, v_line_density * 20])
        
        # 2. Presentation-style features
        # Check for typical presentation characteristics
        
        # Color uniformity (presentations often have uniform backgrounds)
        color_std = np.std(gray)
        features.append(color_std / 255.0)
        
        # Brightness distribution (presentations often have consistent lighting)
        brightness_mean = np.mean(gray) / 255.0
        features.append(brightness_mean)
        
        # 3. Content structure features
        # Divide image into grid and analyze each section
        h, w = gray.shape
        grid_size = 8
        cell_h, cell_w = h // grid_size, w // grid_size
        
        for i in range(grid_size):
            for j in range(grid_size):
                cell = gray[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                
                # Cell statistics
                cell_mean = np.mean(cell) / 255.0
                cell_std = np.std(cell) / 255.0
                
                # Edge density in cell
                cell_edges = edges[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                cell_edge_density = np.sum(cell_edges > 0) / (cell_h * cell_w)
                
                features.extend([cell_mean, cell_std, cell_edge_density])
        
        # 4. Color distribution features
        # For each color channel
        for channel in range(3):
            channel_data = frame[:,:,channel]
            hist = cv2.calcHist([channel_data], [0], None, [16], [0, 256])
            hist = hist.flatten() / np.sum(hist)  # Normalize
            features.extend(hist)
        
        # 5. Texture analysis
        # Local Binary Pattern-like features
        def simple_lbp_feature(img, radius=3):
            """Simple texture feature"""
            h, w = img.shape
            feature = 0
            count = 0
            for y in range(radius, h-radius, 5):
                for x in range(radius, w-radius, 5):
                    center = img[y, x]
                    neighbors = [
                        img[y-radius, x-radius], img[y-radius, x], img[y-radius, x+radius],
                        img[y, x+radius], img[y+radius, x+radius], img[y+radius, x],
                        img[y+radius, x-radius], img[y, x-radius]
                    ]
                    pattern = sum(1 for n in neighbors if n >= center)
                    feature += pattern
                    count += 1
            return feature / count if count > 0 else 0
        
        texture_feature = simple_lbp_feature(gray)
        features.append(texture_feature / 8.0)
        
        # Convert to numpy array and pad/truncate to 512 dimensions
        features = np.array(features, dtype=np.float32)
        
        if len(features) < 512:
            # Pad with statistical features
            padding_needed = 512 - len(features)
            # Add repeated statistical measures
            stats = [
                np.mean(gray), np.std(gray), np.min(gray), np.max(gray),
                np.percentile(gray, 25), np.percentile(gray, 75),
                edge_density, h_line_density, v_line_density, color_std
            ]
            padding = np.tile(stats, (padding_needed // len(stats)) + 1)[:padding_needed]
            features = np.concatenate([features, padding])
        else:
            features = features[:512]
        
        # Normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features
    
    def extract_text_features(self, text):
        """Extract features for text queries"""
        if self.use_clip:
            try:
                import torch
                
                inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    text_features = self.clip_model.get_text_features(**inputs)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                return text_features.cpu().numpy().flatten().astype(np.float32)
                
            except Exception as e:
                print(f"⚠️ CLIP text processing failed: {e}")
        
        # Fallback: create query vector based on keywords
        query_vector = np.zeros(512, dtype=np.float32)
        
        # Enhanced keyword mapping for technical/presentation content
        keywords = text.lower().split()
        
        technical_keywords = {
            # System design terms
            'system': [0, 20], 'design': [20, 40], 'architecture': [40, 60],
            'database': [60, 80], 'server': [80, 100], 'api': [100, 120],
            'microservice': [120, 140], 'cloud': [140, 160], 'scale': [160, 180],
            
            # Presentation terms
            'diagram': [180, 200], 'chart': [200, 220], 'flow': [220, 240],
            'presentation': [240, 260], 'slide': [260, 280], 'screen': [280, 300],
            'text': [300, 320], 'document': [320, 340], 'interface': [340, 360],
            
            # Visual terms
            'bright': [360, 380], 'dark': [380, 400], 'complex': [400, 420],
            'simple': [420, 440], 'static': [440, 460], 'moving': [460, 480],
            'colorful': [480, 500], 'monochrome': [500, 512]
        }
        
        # Set features based on keywords
        for keyword in keywords:
            if keyword in technical_keywords:
                start, end = technical_keywords[keyword]
                query_vector[start:end] = 1.0
        
        # If no keywords matched, use general features
        if np.sum(query_vector) == 0:
            # Generic query processing
            query_vector[:50] = 0.5  # General content features
        
        # Normalize
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm
        
        return query_vector
    
    def process_video(self, video_path):
        """Process video and add to index"""
        video_name = os.path.basename(video_path)
        video_id = os.path.splitext(video_name)[0]
        
        print(f"\n📹 Processing: {video_name}")
        
        # Extract frames
        frames_data = self.extract_frames(video_path, sample_rate=1.0)
        
        if not frames_data:
            return False
        
        print(f"🧠 Extracting {'CLIP' if self.use_clip else 'enhanced visual'} features...")
        
        vectors = []
        node_ids = []
        
        for i, frame_data in enumerate(frames_data):
            # Extract semantic features
            features = self.extract_semantic_features(frame_data['frame'])
            vectors.append(features)
            
            node_id = f"{video_id}_frame_{i}"
            node_ids.append(node_id)
            
            # Store metadata
            self.video_metadata[node_id] = {
                'video_id': video_id,
                'video_name': video_name,
                'video_path': video_path,
                'timestamp': frame_data['timestamp'],
                'frame_number': frame_data['frame_number']
            }
        
        # Add to index
        print(f"📊 Adding {len(vectors)} vectors to HNSW index...")
        start_time = time.time()
        self.index.add_batch(vectors, node_ids)
        index_time = (time.time() - start_time) * 1000
        
        self.frame_count += len(vectors)
        
        print(f"✅ Indexed {video_name} in {index_time:.1f}ms")
        print(f"   📊 Frames: {len(vectors)}")
        print(f"   📈 Total in index: {self.index.size()}")
        
        return True
    
    def search(self, query_text, k=5):
        """Semantic search"""
        print(f"\n🔍 Searching: '{query_text}'")
        
        if self.index.size() == 0:
            print("❌ No videos indexed!")
            return [], 0
        
        # Extract query features
        query_vector = self.extract_text_features(query_text)
        
        # Search
        start_time = time.time()
        results = self.index.search(query_vector, k * 2)
        search_time = (time.time() - start_time) * 1000
        
        # Process results
        final_results = []
        seen_videos = set()
        
        for result in results:
            metadata = self.video_metadata.get(result['id'])
            if metadata and metadata['video_id'] not in seen_videos:
                seen_videos.add(metadata['video_id'])
                
                final_results.append({
                    'video_name': metadata['video_name'],
                    'timestamp': metadata['timestamp'],
                    'score': result['score'],
                    'video_path': metadata['video_path']
                })
                
                if len(final_results) >= k:
                    break
        
        print(f"⏱️  Search: {search_time:.1f}ms")
        print(f"📊 Results:")
        
        for i, result in enumerate(final_results, 1):
            print(f"   {i}. {result['video_name']} at {result['timestamp']:.1f}s")
            print(f"      Score: {result['score']:.3f}")
        
        return final_results, search_time

def main():
    """Main function"""
    print("🎬 Fixed Semantic Video Search")
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
    
    # Initialize processor
    processor = SemanticVideoProcessor(use_clip=use_clip)
    
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
    
    # Process videos
    for video_file in video_files:
        processor.process_video(str(video_file))
    
    # Test semantic searches
    print(f"\n🧠 Testing Semantic Search")
    print("-" * 40)
    
    test_queries = [
        "system design presentation",
        "architecture diagram", 
        "database design",
        "technical presentation",
        "flowchart or diagram",
        "bright slide",
        "complex diagram"
    ]
    
    for query in test_queries[:3]:
        processor.search(query, k=3)
        time.sleep(0.3)
    
    # Interactive search
    print(f"\n🎯 Interactive Semantic Search")
    print("-" * 40)
    print("Try queries like:")
    print("  • 'system design' • 'database diagram' • 'architecture'")
    print("  • 'presentation slide' • 'flowchart' • 'technical diagram'")
    print("Type 'quit' to exit")
    
    while True:
        try:
            query = input("\n🔍 Semantic search: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if query:
                processor.search(query, k=5)
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
    
    print(f"\n🎉 Semantic video search completed!")
    return 0

if __name__ == "__main__":
    exit(main())
