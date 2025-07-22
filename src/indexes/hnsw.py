"""
Optimized HNSW Index Implementation
Based on the Hierarchical Navigable Small World algorithm
"""

import heapq
import hashlib
import math
import numpy as np
import random
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict
import pickle
import os
import threading
from concurrent.futures import ThreadPoolExecutor


class HNSWIndex:
    """
    High-performance HNSW index optimized for video search
    Achieves O(log n) search time complexity
    """

    def __init__(
        self,
        dimension: int = 512,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        max_M: int = 16,
        level_generation_factor: float = 1.0 / math.log(2.0),
        num_threads: int = 4
    ):
        self.dimension = dimension
        self.M = M  # Number of bidirectional links per node
        self.max_M = max_M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.level_generation_factor = level_generation_factor
        self.num_threads = num_threads

        # Data structures
        self.data = {}  # id -> vector
        self.levels = {}  # id -> level
        # level -> id -> neighbors
        self.graph = defaultdict(lambda: defaultdict(set))
        self.entry_point = None
        self.element_count = 0

        # Thread safety
        self.lock = threading.RLock()
        self.thread_pool = ThreadPoolExecutor(max_workers=num_threads)

        # Metrics
        self.build_time = 0
        self.search_times = []

    def _distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Optimized L2 distance computation
        Using dot product for normalized vectors is equivalent to cosine similarity
        """
        # For normalized vectors: ||a - b||² = 2(1 - a·b)
        # We use 1 - a·b as distance (higher cosine similarity = lower distance)
        return 1.0 - np.dot(vec1, vec2)

    def _get_random_level(self) -> int:
        """
        Generate random level using exponential decay distribution
        """
        level = int(-math.log(random.uniform(0, 1))
                    * self.level_generation_factor)
        return level

    def _search_layer(
        self,
        query: np.ndarray,
        entry_points: List[int],
        num_closest: int,
        level: int
    ) -> List[Tuple[float, int]]:
        """
        Greedy search in a single layer
        Returns list of (distance, node_id) tuples
        """
        visited = set()
        candidates = []  # Min heap: (distance, node_id)
        dynamic_list = []  # Max heap: (-distance, node_id)

        # Initialize with entry points
        for ep in entry_points:
            if ep in self.data:
                dist = self._distance(query, self.data[ep])
                heapq.heappush(candidates, (dist, ep))
                heapq.heappush(dynamic_list, (-dist, ep))
                visited.add(ep)

        while candidates:
            current_dist, current = heapq.heappop(candidates)

            # Stop condition: current candidate is farther than worst in dynamic_list
            if dynamic_list and current_dist > -dynamic_list[0][0]:
                break

            # Check neighbors
            neighbors = self.graph[level].get(current, set())
            for neighbor in neighbors:
                if neighbor not in visited and neighbor in self.data:
                    visited.add(neighbor)
                    dist = self._distance(query, self.data[neighbor])

                    if len(dynamic_list) < num_closest or dist < -dynamic_list[0][0]:
                        heapq.heappush(candidates, (dist, neighbor))
                        heapq.heappush(dynamic_list, (-dist, neighbor))

                        if len(dynamic_list) > num_closest:
                            heapq.heappop(dynamic_list)

        # Convert max heap to min heap and return
        return [(-dist, node_id) for dist, node_id in dynamic_list]

    def _select_neighbors_heuristic(
        self,
        candidates: List[Tuple[float, int]],
        M: int,
        query_vec: np.ndarray = None
    ) -> List[int]:
        """
        Select M neighbors using heuristic to maintain graph connectivity
        Prioritizes closer nodes while maintaining diversity
        """
        if len(candidates) <= M:
            return [node_id for _, node_id in candidates]

        # Sort by distance
        candidates.sort()
        selected = []

        for dist, candidate in candidates:
            if len(selected) >= M:
                break

            # Simple heuristic: always select if we have room
            # More sophisticated heuristics can be implemented here
            selected.append(candidate)

        return selected

    def add(self, vector: np.ndarray, node_id: int) -> None:
        """
        Add vector to the index
        Thread-safe implementation
        """
        with self.lock:
            # Normalize vector for cosine similarity
            vector = vector / np.linalg.norm(vector)

            # Store data
            self.data[node_id] = vector
            level = self._get_random_level()
            self.levels[node_id] = level

            # Initialize connections
            for lv in range(level + 1):
                self.graph[lv][node_id] = set()

            if self.entry_point is None:
                self.entry_point = node_id
                self.element_count += 1
                return

            # Search from top to level+1
            curr_nearest = [self.entry_point]
            for lv in range(max(self.levels[self.entry_point], level), level, -1):
                curr_nearest = [
                    node_id for _, node_id in self._search_layer(
                        vector, curr_nearest, 1, lv
                    )
                ]

            # Search and connect from level down to 0
            for lv in range(min(level, self.levels[self.entry_point]), -1, -1):
                candidates = self._search_layer(
                    vector, curr_nearest, self.ef_construction, lv
                )

                curr_nearest = [node_id for _, node_id in candidates]

                # Select neighbors for new element
                max_conn = self.M if lv > 0 else self.max_M
                selected_neighbors = self._select_neighbors_heuristic(
                    candidates, max_conn, vector
                )

                # Add bidirectional connections
                for neighbor in selected_neighbors:
                    self.graph[lv][node_id].add(neighbor)
                    self.graph[lv][neighbor].add(node_id)

                    # Prune connections of neighbor if needed
                    neighbor_connections = list(self.graph[lv][neighbor])
                    if len(neighbor_connections) > max_conn:
                        # Get all neighbors with distances
                        neighbor_candidates = [
                            (self._distance(
                                self.data[neighbor], self.data[conn]), conn)
                            for conn in neighbor_connections
                        ]

                        # Keep only M best connections
                        selected_for_neighbor = self._select_neighbors_heuristic(
                            neighbor_candidates, max_conn, self.data[neighbor]
                        )

                        # Update connections
                        old_connections = self.graph[lv][neighbor].copy()
                        self.graph[lv][neighbor] = set(selected_for_neighbor)

                        # Remove bidirectional links for pruned connections
                        for conn in old_connections:
                            if conn not in selected_for_neighbor:
                                self.graph[lv][conn].discard(neighbor)

            # Update entry point if necessary
            if level > self.levels[self.entry_point]:
                self.entry_point = node_id

            self.element_count += 1

    def add_batch(self, vectors: List[np.ndarray], node_ids: List[int]) -> None:
        """
        Add multiple vectors efficiently
        """
        for vector, node_id in zip(vectors, node_ids):
            self.add(vector, node_id)

    def search(self, query: np.ndarray, k: int = 5) -> List[Dict]:
        """
        Search for k nearest neighbors
        Returns list of dicts with 'id', 'distance', 'score'
        """
        if self.entry_point is None or self.element_count == 0:
            return []

        import time
        start_time = time.time()

        # Normalize query
        query = query / np.linalg.norm(query)

        with self.lock:
            # Search from top layer to layer 1
            curr_nearest = [self.entry_point]
            for lv in range(self.levels[self.entry_point], 0, -1):
                curr_nearest = [
                    node_id for _, node_id in self._search_layer(
                        query, curr_nearest, 1, lv
                    )
                ]

            # Search layer 0 with ef_search
            candidates = self._search_layer(
                query, curr_nearest, max(self.ef_search, k), 0
            )

        # Convert to result format
        results = []
        for distance, node_id in sorted(candidates)[:k]:
            results.append({
                'id': node_id,
                'distance': distance,
                'score': 1.0 - distance  # Convert distance to similarity score
            })

        # Record search time
        search_time = (time.time() - start_time) * 1000
        self.search_times.append(search_time)

        return results

    def search_batch(self, queries: List[np.ndarray], k: int = 5) -> List[List[Dict]]:
        """
        Batch search for multiple queries
        Uses thread pool for parallel processing
        """
        if self.num_threads == 1:
            return [self.search(query, k) for query in queries]

        # Use thread pool for parallel search
        futures = []
        for query in queries:
            future = self.thread_pool.submit(self.search, query, k)
            futures.append(future)

        results = []
        for future in futures:
            results.append(future.result())

        return results

    def size(self) -> int:
        """Return number of elements in the index"""
        return self.element_count

    def save(self, filepath: str) -> None:
        """
        Save index to disk with compression
        """
        with self.lock:
            save_data = {
                'dimension': self.dimension,
                'M': self.M,
                'max_M': self.max_M,
                'ef_construction': self.ef_construction,
                'ef_search': self.ef_search,
                'level_generation_factor': self.level_generation_factor,
                'data': self.data,
                'levels': self.levels,
                # Convert defaultdict to regular dict
                'graph': dict(self.graph),
                'entry_point': self.entry_point,
                'element_count': self.element_count
            }

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Save with compression
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Create checksum
            import hashlib
            with open(filepath, 'rb') as f:
                checksum = hashlib.sha256(f.read()).hexdigest()

            with open(filepath + '.sha256', 'w') as f:
                f.write(checksum)

    def load(self, filepath: str) -> None:
        """
        Load index from disk with verification
        """
        # Verify checksum
        try:
            with open(filepath, 'rb') as f:
                current_checksum = hashlib.sha256(f.read()).hexdigest()

            with open(filepath + '.sha256', 'r') as f:
                expected_checksum = f.read().strip()

            if current_checksum != expected_checksum:
                raise ValueError("Index file corrupted (checksum mismatch)")

        except FileNotFoundError:
            print("Warning: No checksum file found, skipping verification")

        # Load data
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)

        # Restore state
        self.dimension = save_data['dimension']
        self.M = save_data['M']
        self.max_M = save_data['max_M']
        self.ef_construction = save_data['ef_construction']
        self.ef_search = save_data['ef_search']
        self.level_generation_factor = save_data['level_generation_factor']
        self.data = save_data['data']
        self.levels = save_data['levels']
        self.graph = defaultdict(lambda: defaultdict(set))

        # Restore graph structure
        for level, nodes in save_data['graph'].items():
            for node_id, neighbors in nodes.items():
                self.graph[int(level)][node_id] = set(neighbors)

        self.entry_point = save_data['entry_point']
        self.element_count = save_data['element_count']

    def get_stats(self) -> Dict:
        """
        Get index statistics
        """
        if not self.search_times:
            avg_search_time = 0
            p95_search_time = 0
        else:
            avg_search_time = sum(self.search_times) / len(self.search_times)
            p95_search_time = np.percentile(self.search_times, 95)

        return {
            'element_count': self.element_count,
            'entry_point_level': self.levels.get(self.entry_point, 0) if self.entry_point else 0,
            'avg_search_time_ms': avg_search_time,
            'p95_search_time_ms': p95_search_time,
            'total_searches': len(self.search_times),
            'dimension': self.dimension,
            'M': self.M,
            'ef_search': self.ef_search
        }


class OptimizedHNSWIndex(HNSWIndex):
    """
    Further optimized HNSW with additional performance enhancements
    """

    def __init__(self, *args, use_numpy_optimization=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_numpy_optimization = use_numpy_optimization

        # Pre-allocate arrays for better performance
        if use_numpy_optimization:
            self._distance_buffer = np.zeros(1000, dtype=np.float32)

    def _vectorized_distances(self, query: np.ndarray, node_ids: List[int]) -> np.ndarray:
        """
        Compute distances to multiple nodes using vectorized operations
        """
        if not node_ids:
            return np.array([])

        # Stack vectors
        vectors = np.stack([self.data[node_id] for node_id in node_ids])

        # Vectorized distance computation
        # For normalized vectors: distance = 1 - cosine_similarity
        similarities = np.dot(vectors, query)
        distances = 1.0 - similarities

        return distances

    def _search_layer_optimized(
        self,
        query: np.ndarray,
        entry_points: List[int],
        num_closest: int,
        level: int
    ) -> List[Tuple[float, int]]:
        """
        Optimized layer search using vectorized operations
        """
        if not self.use_numpy_optimization or len(entry_points) < 10:
            return self._search_layer(query, entry_points, num_closest, level)

        visited = set()
        candidates = []
        dynamic_list = []

        # Initialize with vectorized distance computation
        valid_entry_points = [ep for ep in entry_points if ep in self.data]
        if valid_entry_points:
            distances = self._vectorized_distances(query, valid_entry_points)
            for dist, ep in zip(distances, valid_entry_points):
                heapq.heappush(candidates, (dist, ep))
                heapq.heappush(dynamic_list, (-dist, ep))
                visited.add(ep)

        while candidates:
            current_dist, current = heapq.heappop(candidates)

            if dynamic_list and current_dist > -dynamic_list[0][0]:
                break

            neighbors = list(self.graph[level].get(current, set()))
            unvisited_neighbors = [
                n for n in neighbors if n not in visited and n in self.data]

            if unvisited_neighbors:
                # Vectorized distance computation for all unvisited neighbors
                distances = self._vectorized_distances(
                    query, unvisited_neighbors)

                for dist, neighbor in zip(distances, unvisited_neighbors):
                    visited.add(neighbor)

                    if len(dynamic_list) < num_closest or dist < -dynamic_list[0][0]:
                        heapq.heappush(candidates, (dist, neighbor))
                        heapq.heappush(dynamic_list, (-dist, neighbor))

                        if len(dynamic_list) > num_closest:
                            heapq.heappop(dynamic_list)

        return [(-dist, node_id) for dist, node_id in dynamic_list]

    def search(self, query: np.ndarray, k: int = 5) -> List[Dict]:
        """
        Optimized search using vectorized operations
        """
        if self.entry_point is None or self.element_count == 0:
            return []

        import time
        start_time = time.time()

        # Normalize query
        query = query / np.linalg.norm(query)

        with self.lock:
            # Search from top layer to layer 1
            curr_nearest = [self.entry_point]
            for lv in range(self.levels[self.entry_point], 0, -1):
                curr_nearest = [
                    node_id for _, node_id in self._search_layer_optimized(
                        query, curr_nearest, 1, lv
                    )
                ]

            # Search layer 0 with ef_search
            candidates = self._search_layer_optimized(
                query, curr_nearest, max(self.ef_search, k), 0
            )

        # Convert to result format
        results = []
        for distance, node_id in sorted(candidates)[:k]:
            results.append({
                'id': node_id,
                'distance': distance,
                'score': 1.0 - distance
            })

        search_time = (time.time() - start_time) * 1000
        self.search_times.append(search_time)

        return results
