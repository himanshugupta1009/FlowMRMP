import numpy as np
from scipy.spatial import cKDTree


# ============================================================
# KD-tree for [v, phi] state of second-order car
# ============================================================

class VPhiTree:
    """
    KD-tree index for hybrid state [v, phi], ignoring x,y,0.

    Feature embedding:
        x0 = v   / v_scale
        x1 = phi / phi_scale

    Euclidean distance in this 2D feature space corresponds exactly to

        d^2 = (Δv / v_scale)^2 + (Δphi / phi_scale)^2
    """

    def __init__(self,
                 v,
                 phi,
                 ids=None,
                 v_scale=1.0,
                 phi_scale=1.0):
        v = np.asarray(v, dtype=np.float64)
        phi = np.asarray(phi, dtype=np.float64)
        assert v.shape == phi.shape
        n = v.shape[0]

        if ids is None:
            ids = np.arange(n, dtype=np.int64)
        else:
            ids = np.asarray(ids, dtype=np.int64)
            assert ids.shape == v.shape

        self.ids = ids
        self.v = v
        self.phi = phi

        self.v_scale = float(v_scale)
        self.phi_scale = float(phi_scale)

        # Embed into 2D feature space
        X = np.empty((n, 2), dtype=np.float64)
        X[:, 0] = v   / self.v_scale
        X[:, 1] = phi / self.phi_scale

        self._X = X
        self.tree = cKDTree(X)

        self._query_buf = np.empty(2, dtype=np.float64)
        self._empty_idx = np.empty(0, dtype=np.int64)
        
        # Preallocate a big buffer to hold radius query results
        # self._candidate_indices_buffer = np.empty(n, dtype=np.int64)

    def _embed_query(self, query):
        x = self._query_buf
        x[0] = float(query[0])   / self.v_scale
        x[1] = float(query[1]) / self.phi_scale
        return x

    def radius_query(self, query, delta):
        """
        query = (v_q, phi_q)
        Return IDs whose hybrid distance to (v_q, phi_q) is <= delta:

            (Δv / v_scale)^2 + (Δphi / phi_scale)^2 <= delta^2
        """
        q = self._embed_query(query)
        cand_idx = self.tree.query_ball_point(q, r=delta)

        # m = len(cand_idx)
        # if m == 0:
        #     return self._empty_idx

        # buffer = self._candidate_indices_buffer
        # for i in range(m):
        #     buffer[i] = cand_idx[i]

        # return self.ids[cand_idx]
        # return cand_idx
        # return buffer[:m].copy() 
        # Used copy above to avoid overwriting output from past calls 
        # in future calls

        return np.asarray(cand_idx, dtype=np.int64)
        #Note - this only returns the indices in the kd-tree, not the original IDs
        #However, in our usage, the indices in the kd-tree correspond to the original IDs.
        #So it works for now. But be careful if you use this class elsewhere.



"""

import sys
sys.path.append('./src')
import numpy as np
from edge_bundle import EdgeBundle
from kd_tree_second_order_car import VPhiTree

N = 100000
v = np.random.uniform(-1.0, 1.0, size=N)
phi = np.random.uniform(-np.pi/3, np.pi/3, size=N)
edge_ids = np.arange(N, dtype=np.int64)

kd_tree_SOC = VPhiTree(v, phi, ids=edge_ids, v_scale=1.0, phi_scale=1.0)

vq, phiq, δ = 0.5, 0.1, 0.1
ids = kd_tree_SOC.radius_query([vq, phiq], δ)  # np.ndarray of IDs

for i in range(len(ids)):
    idx = ids[i]
    print("Query : v =", vq, ", phi =", phiq, ", delta =", δ)
    print(f"ID: {idx}, phi: {phi[idx]:.4f}, v: {v[idx]:.4f}")

import time
t0 = time.perf_counter()
near_ids = kd_tree_SOC.radius_query([vq, phiq], δ)
t1 = time.perf_counter()
print(f"Single radius_query: {(t1 - t0)*1e6:.2f} µs, found {near_ids.size} neighbors")


δ = 0.1

num_tries = 10000
vq_array = np.random.uniform(-1.0, 1.0, size=num_tries)
phiq_array = np.random.uniform(-np.pi/3, np.pi/3, size=num_tries)
start = time.perf_counter()
for i in range(num_tries):
    kd_tree_SOC.radius_query([vq_array[i], phiq_array[i]], δ)
end = time.perf_counter()

print(f"Average time: {(end - start)/num_tries*1e6:.2f} µs per call")


"""