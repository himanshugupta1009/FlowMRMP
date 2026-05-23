import numpy as np
from numba import njit, int32, float64

@njit
def euclidean_distance(p1, p2):
    d = 0.0
    for i in range(p1.shape[0]):
        diff = p1[i] - p2[i]
        d += diff * diff
    return np.sqrt(d)

@njit
def nearest_neighbor(points, active_mask, query_point):
    """
    Brute-force nearest neighbor search with mask filtering.
    This can later be replaced by a KD-tree search.
    """
    best_idx = -1
    best_dist = 1e18
    for i in range(points.shape[0]):
        if active_mask[i] == 1:  # only consider active nodes
            dist = euclidean_distance(points[i], query_point)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
    return best_idx, best_dist

class KDTreeNumba:
    def __init__(self, dim, capacity):
        self.points = np.zeros((capacity, dim), dtype=np.float64)
        self.parent = np.full(capacity, -1, dtype=np.int32)
        self.active_mask = np.zeros(capacity, dtype=np.int32)
        self.size = 0
        self.dim = dim
        self.capacity = capacity

    def insert(self, point, parent_idx):
        if self.size >= self.capacity:
            raise RuntimeError("Tree capacity exceeded.")
        self.points[self.size] = point
        self.parent[self.size] = parent_idx
        self.active_mask[self.size] = 1
        self.size += 1
        return self.size - 1

    def deactivate(self, idx):
        self.active_mask[idx] = 0

    def nearest(self, query_point):
        return nearest_neighbor(self.points[:self.size], self.active_mask[:self.size], query_point)
