import numpy as np
from numba import njit

# ---------------------------
# Numba helpers (no Python in hot loop)
# ---------------------------

@njit(cache=True)
def _clamp_int(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x

@njit(cache=True)
def _grid_radius_query3_numba(vx, vy, vz,
                             order, offsets,
                             vmin, h, nx, ny, nz,
                             sx, sy, sz,
                             qx, qy, qz,
                             delta,
                             out_idx):
    """
    Writes matching point indices into out_idx.
    Returns count (may exceed out_idx.size -> indicates truncation).
    """
    inv_sx = 1.0 / sx
    inv_sy = 1.0 / sy
    inv_sz = 1.0 / sz
    d2 = delta * delta

    # query cell
    ixq = int((qx - vmin) / h)
    iyq = int((qy - vmin) / h)
    izq = int((qz - vmin) / h)

    ixq = _clamp_int(ixq, 0, nx - 1)
    iyq = _clamp_int(iyq, 0, ny - 1)
    izq = _clamp_int(izq, 0, nz - 1)

    rcell = int(np.ceil(delta / h))
    if rcell < 0:
        rcell = 0

    count = 0

    for dz in range(-rcell, rcell + 1):
        izn = izq + dz
        if izn < 0 or izn >= nz:
            continue
        for dy in range(-rcell, rcell + 1):
            iyn = iyq + dy
            if iyn < 0 or iyn >= ny:
                continue
            base = nx * (iyn + ny * izn)
            for dx in range(-rcell, rcell + 1):
                ixn = ixq + dx
                if ixn < 0 or ixn >= nx:
                    continue

                cell = ixn + base
                a = offsets[cell]
                b = offsets[cell + 1]

                for t in range(a, b):
                    idx = order[t]

                    dvx = (vx[idx] - qx) * inv_sx
                    dvy = (vy[idx] - qy) * inv_sy
                    dvz = (vz[idx] - qz) * inv_sz

                    if dvx * dvx + dvy * dvy + dvz * dvz <= d2:
                        if count < out_idx.size:
                            out_idx[count] = idx
                        count += 1

    return count


# ---------------------------
# VxyzTree: grid-backed implementation
# ---------------------------

class VxyzGridTree:
    """
    Fast uniform-grid index over velocities [vx, vy, vz].

    Embedding/scaling matches your old KD-tree class:
        distance^2 = (Δvx/sx)^2 + (Δvy/sy)^2 + (Δvz/sz)^2
    and radius query returns points with distance <= delta.

    API preserved:
        - radius_query(query, delta) -> np.int64 array of indices (or IDs if you pass ids)
        - knn_query: provided as a slow fallback (optional)
    """

    def __init__(self, vx, vy, vz, ids=None, scales=(1.0, 1.0, 1.0),
                 vmin=None, vmax=None, cell_size=None,
                 initial_out_capacity=4096,
                 return_ids=False):
        """
        Parameters
        ----------
        vx, vy, vz : arrays
        ids : optional original IDs (same length). If return_ids=True, radius_query returns ids.
        scales : (sx, sy, sz)
        vmin, vmax : bounds for the grid (float). If None, inferred from data min/max.
        cell_size : h, grid cell size. If None, uses vmax-vmin / 64 as a heuristic;
                    best is usually ~ delta or delta/2 (delta varies, so pick your typical).
        initial_out_capacity : starting buffer size for radius query outputs.
        return_ids : if True, radius_query returns ids[...] instead of raw indices.
                     (Default False to match your current “indices correspond to IDs anyway”.)
        """
        vx = np.asarray(vx, dtype=np.float64)
        vy = np.asarray(vy, dtype=np.float64)
        vz = np.asarray(vz, dtype=np.float64)
        assert vx.shape == vy.shape == vz.shape
        n = vx.size

        if ids is None:
            ids = np.arange(n, dtype=np.int64)
        else:
            ids = np.asarray(ids, dtype=np.int64)
            assert ids.shape == vx.shape

        sx, sy, sz = map(float, scales)
        if sx <= 0 or sy <= 0 or sz <= 0:
            raise ValueError("All scales must be > 0.")

        # bounds
        if vmin is None:
            vmin = float(min(vx.min(), vy.min(), vz.min()))
        else:
            vmin = float(vmin)
        if vmax is None:
            vmax = float(max(vx.max(), vy.max(), vz.max()))
        else:
            vmax = float(vmax)

        if vmax <= vmin:
            raise ValueError("vmax must be > vmin.")

        # heuristic default cell size if not provided
        if cell_size is None:
            # heuristic: 64 cells across the range
            cell_size = (vmax - vmin) / 64.0
        cell_size = float(cell_size)
        if cell_size <= 0:
            raise ValueError("cell_size must be > 0.")

        # grid resolution (same along all axes)
        nx = int(np.ceil((vmax - vmin) / cell_size))
        if nx < 1:
            nx = 1
        ny = nx
        nz = nx
        num_cells = nx * ny * nz

        # cell ids
        ix = np.floor((vx - vmin) / cell_size).astype(np.int64)
        iy = np.floor((vy - vmin) / cell_size).astype(np.int64)
        iz = np.floor((vz - vmin) / cell_size).astype(np.int64)
        ix = np.clip(ix, 0, nx - 1)
        iy = np.clip(iy, 0, ny - 1)
        iz = np.clip(iz, 0, nz - 1)
        key = ix + nx * (iy + ny * iz)  # linear cell key

        # sort by cell key, keep original indices
        order = np.argsort(key, kind="mergesort").astype(np.int64)
        key_sorted = key[order]

        # counts per cell and prefix sum offsets
        counts = np.bincount(key_sorted, minlength=num_cells).astype(np.int64)
        offsets = np.empty(num_cells + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])

        # store
        self._vx = vx
        self._vy = vy
        self._vz = vz
        self.ids = ids

        self.vx_scale = sx
        self.vy_scale = sy
        self.vz_scale = sz

        self._vmin = vmin
        self._vmax = vmax
        self._h = cell_size
        self._nx = nx
        self._ny = ny
        self._nz = nz

        self._order = order
        self._offsets = offsets

        self._query_buf = np.empty(3, dtype=np.float64)

        self._out = np.empty(int(initial_out_capacity), dtype=np.int64)
        self._return_ids = bool(return_ids)

        # one-time warmup so first call isn't slow (optional but nice)
        _ = _grid_radius_query3_numba(self._vx, self._vy, self._vz,
                                     self._order, self._offsets,
                                     self._vmin, self._h, self._nx, self._ny, self._nz,
                                     self.vx_scale, self.vy_scale, self.vz_scale,
                                     0.0, 0.0, 0.0, 0.0,
                                     self._out)

    def _embed_query(self, query):
        q = self._query_buf
        q[0] = float(query[0])
        q[1] = float(query[1])
        q[2] = float(query[2])
        return q

    def _ensure_out_capacity(self, needed):
        # grow to next power-of-two-ish to avoid frequent reallocs
        cap = self._out.size
        if needed <= cap:
            return
        new_cap = cap
        while new_cap < needed:
            new_cap *= 2
        self._out = np.empty(new_cap, dtype=np.int64)

    def radius_query(self, query, delta):
        """
        Return indices (or ids if return_ids=True) within radius delta
        in scaled velocity space.
        """
        q = self._embed_query(query)
        cnt = _grid_radius_query3_numba(
            self._vx, self._vy, self._vz,
            self._order, self._offsets,
            self._vmin, self._h, self._nx, self._ny, self._nz,
            self.vx_scale, self.vy_scale, self.vz_scale,
            q[0], q[1], q[2],
            float(delta),
            self._out
        )

        if cnt <= self._out.size:
            out = self._out[:cnt].copy()  # copy so caller can't mutate our buffer
        else:
            # rare case: buffer was too small. Grow and re-run once.
            self._ensure_out_capacity(cnt)
            cnt2 = _grid_radius_query3_numba(
                self._vx, self._vy, self._vz,
                self._order, self._offsets,
                self._vmin, self._h, self._nx, self._ny, self._nz,
                self.vx_scale, self.vy_scale, self.vz_scale,
                q[0], q[1], q[2],
                float(delta),
                self._out
            )
            out = self._out[:cnt2].copy()

        if self._return_ids:
            return self.ids[out]
        return out

    # Optional: if you need this for compatibility. This is NOT optimized.
    def knn_query(self, query, k=1):
        """
        Slow fallback: brute-force KNN in scaled space.
        If you rely on knn heavily, we can add a faster grid-based KNN too.
        """
        q = self._embed_query(query)
        dx = (self._vx - q[0]) / self.vx_scale
        dy = (self._vy - q[1]) / self.vy_scale
        dz = (self._vz - q[2]) / self.vz_scale
        d2 = dx * dx + dy * dy + dz * dz
        k = int(k)
        idx = np.argpartition(d2, k - 1)[:k]
        # return in sorted order by distance
        ord2 = np.argsort(d2[idx])
        idx = idx[ord2]
        dist = np.sqrt(d2[idx])
        if self._return_ids:
            return self.ids[idx], dist
        return idx.astype(np.int64), dist


# ---------------------------
# Timing helper
# ---------------------------

def benchmark_vxyz_tree():
    import time

    N = 100_000
    max_speed = 0.5
    rng = np.random.default_rng(0)

    vx = rng.uniform(-max_speed, max_speed, size=N)
    vy = rng.uniform(-max_speed, max_speed, size=N)
    vz = rng.uniform(-max_speed, max_speed, size=N)

    delta = 0.1
    # Important: pick cell_size close to your typical delta.
    # delta or delta/2 are usually best.
    cell_size = delta  # try delta/2 as well

    tree = VxyzTree(
        vx, vy, vz,
        ids=np.arange(N, dtype=np.int64),
        scales=(1.0, 1.0, 1.0),
        vmin=-max_speed,
        vmax= max_speed,
        cell_size=cell_size,
        initial_out_capacity=4096,
        return_ids=True
    )

    # warmup (compilation + cache effects)
    for _ in range(200):
        q = (rng.uniform(-max_speed, max_speed),
             rng.uniform(-max_speed, max_speed),
             rng.uniform(-max_speed, max_speed))
        tree.radius_query(q, delta)

    # timing
    num_tries = 10_000
    qx = rng.uniform(-max_speed, max_speed, size=num_tries)
    qy = rng.uniform(-max_speed, max_speed, size=num_tries)
    qz = rng.uniform(-max_speed, max_speed, size=num_tries)

    t0 = time.perf_counter()
    total_found = 0
    for i in range(num_tries):
        # Avoid per-iter np.array allocations: pass tuple
        out = tree.radius_query((qx[i], qy[i], qz[i]), delta)
        total_found += out.size
    t1 = time.perf_counter()

    avg_us = (t1 - t0) / num_tries * 1e6
    avg_neighbors = total_found / num_tries

    print(f"Grid radius_query: {avg_us:.2f} µs/call, avg neighbors {avg_neighbors:.2f}, "
          f"cell_size={cell_size}, delta={delta}, N={N}")


# if __name__ == "__main__":
#     benchmark_vxyz_tree()



"""
import time

N = 100_000
max_speed = 0.5
rng = np.random.default_rng(0)

vx = rng.uniform(-max_speed, max_speed, size=N)
vy = rng.uniform(-max_speed, max_speed, size=N)
vz = rng.uniform(-max_speed, max_speed, size=N)
edge_ids = np.arange(N, dtype=np.int64)

delta = 0.1
# Important: pick cell_size close to your typical delta.
# delta or delta/2 are usually best.
cell_size = delta/2  # try delta/2 as well

tree = VxyzTree(
    vx, vy, vz,
    ids=edge_ids,
    scales=(1.0, 1.0, 1.0),
    vmin=-max_speed,
    vmax= max_speed,
    cell_size=cell_size,
    initial_out_capacity=4096,
    return_ids=False
)

# warmup (compilation + cache effects)
for _ in range(200):
    q = (rng.uniform(-max_speed, max_speed),
            rng.uniform(-max_speed, max_speed),
            rng.uniform(-max_speed, max_speed))
    tree.radius_query(q, delta)

# timing
vxq, vyq, vzq, δ = 0.1, 0.2, -0.3, 0.1
vxq, vyq, vzq, δ = 0.5, -0.2, 0.3, 0.1
ids = tree.radius_query([vxq, vyq, vzq], δ)
print("Query:", vxq, vyq, vzq, "delta =", δ)
print("Found", len(ids), "neighbors\n")

for i in range(min(len(ids), 10)):  # print only a few
    idx = ids[i]
    print(f"ID: {idx}, "
        f"vx: {vx[idx]:.4f}, "
        f"vy: {vy[idx]:.4f}, "
        f"vz: {vz[idx]:.4f}"
    )

#calculate distance for verification
def calculate_distance(vx1, vy1, vz1, vx2, vy2, vz2, sx, sy, sz):
    dvx = (vx1 - vx2) / sx
    dvy = (vy1 - vy2) / sy
    dvz = (vz1 - vz2) / sz
    return np.sqrt(dvx * dvx + dvy * dvy + dvz * dvz)
print("\nVerifying distances:")
for i in range(min(len(ids), 10)):  # print only a few
    idx = ids[i]
    dist = calculate_distance(vxq, vyq, vzq, vx[idx], vy[idx], vz[idx],
                            tree.vx_scale, tree.vy_scale, tree.vz_scale)
    print(f"ID: {idx}, Distance: {dist:.4f}")



t0 = time.perf_counter()
near_ids = tree.radius_query([vxq, vyq, vzq], δ)
t1 = time.perf_counter()
print(
    f"\nSingle radius_query: {(t1 - t0)*1e6:.2f} µs, "
    f"found {near_ids.size} neighbors")


# Average timing over many calls
δ = 0.1
num_tries = 10_000
vxq_array = rng.uniform(-max_speed, max_speed, size=num_tries)
vyq_array = rng.uniform(-max_speed, max_speed, size=num_tries)
vzq_array = rng.uniform(-max_speed, max_speed, size=num_tries)
start = time.perf_counter()
for i in range(num_tries):
    q = (vxq_array[i], vyq_array[i], vzq_array[i])
    tree.radius_query(q,δ)
end = time.perf_counter()

print(f"Average time: {(end - start)/num_tries*1e6:.2f} µs per call")

"""