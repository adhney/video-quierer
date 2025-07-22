"""
System metrics and monitoring
"""

import time
import threading
from typing import Dict, Any, List
from collections import defaultdict, deque
import numpy as np


class SystemMetrics:
    """
    Thread-safe metrics collection and reporting
    """
    
    def __init__(self, max_histogram_size: int = 10000):
        self.max_histogram_size = max_histogram_size
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Metrics storage
        self.counters = defaultdict(int)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(lambda: deque(maxlen=max_histogram_size))
        
        # System info
        self.start_time = time.time()
    
    def record_counter(self, name: str, value: int = 1):
        """Increment counter metric"""
        with self.lock:
            self.counters[name] += value
    
    def record_gauge(self, name: str, value: float):
        """Set gauge metric to specific value"""
        with self.lock:
            self.gauges[name] = value
    
    def record_histogram(self, name: str, value: float):
        """Add value to histogram"""
        with self.lock:
            self.histograms[name].append(value)
    
    def get_counter(self, name: str) -> int:
        """Get counter value"""
        with self.lock:
            return self.counters[name]
    
    def get_gauge(self, name: str) -> float:
        """Get gauge value"""
        with self.lock:
            return self.gauges[name]
    
    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        """Get histogram statistics"""
        with self.lock:
            values = list(self.histograms[name])
            
            if not values:
                return {
                    'count': 0,
                    'min': 0,
                    'max': 0,
                    'mean': 0,
                    'p50': 0,
                    'p95': 0,
                    'p99': 0
                }
            
            values = np.array(values)
            return {
                'count': len(values),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'mean': float(np.mean(values)),
                'p50': float(np.percentile(values, 50)),
                'p95': float(np.percentile(values, 95)),
                'p99': float(np.percentile(values, 99))
            }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get complete metrics summary"""
        with self.lock:
            summary = {
                'counters': dict(self.counters),
                'gauges': dict(self.gauges),
                'histograms': {}
            }
            
            for name in self.histograms.keys():
                summary['histograms'][name] = self.get_histogram_stats(name)
            
            # Add system info
            summary['uptime_seconds'] = time.time() - self.start_time
            
            return summary
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        with self.lock:
            lines = []
            
            # Counters
            for name, value in self.counters.items():
                lines.append(f"video_search_{name}_total {value}")
            
            # Gauges
            for name, value in self.gauges.items():
                lines.append(f"video_search_{name} {value}")
            
            # Histograms
            for name in self.histograms.keys():
                stats = self.get_histogram_stats(name)
                if stats['count'] > 0:
                    lines.extend([
                        f"video_search_{name}_count {stats['count']}",
                        f"video_search_{name}_min {stats['min']}",
                        f"video_search_{name}_max {stats['max']}",
                        f"video_search_{name}_mean {stats['mean']}",
                        f"video_search_{name}_p50 {stats['p50']}",
                        f"video_search_{name}_p95 {stats['p95']}",
                        f"video_search_{name}_p99 {stats['p99']}"
                    ])
            
            # System metrics
            lines.append(f"video_search_uptime_seconds {time.time() - self.start_time}")
            
            return '\n'.join(lines)
    
    def reset(self):
        """Reset all metrics"""
        with self.lock:
            self.counters.clear()
            self.gauges.clear()
            self.histograms.clear()
            self.start_time = time.time()
