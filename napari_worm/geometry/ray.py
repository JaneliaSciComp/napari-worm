import numpy as np
from scipy.interpolate import CubicSpline


def sample_ray(data, start, end, n_samples=None):
    """Walk a ray through the volume, sampling via trilinear interpolation.

    Matches MIPAV's accurate mode (getFloatTriLinearBounds) — samples at
    continuous float coordinates instead of snapping to nearest voxel.
    """
    if n_samples is None:
        n_samples = int(np.linalg.norm(end - start)) + 1
    positions = np.linspace(start, end, n_samples)
    from scipy.ndimage import map_coordinates
    # map_coordinates expects shape (ndim, npts) with coords in z,y,x order
    coords = positions.T  # (3, n_samples)
    values = map_coordinates(data.astype(np.float64), coords, order=1,
                             mode='constant', cval=0.0)
    return positions, values


def find_peak_along_ray(data, start, end):
    positions, values = sample_ray(data, start, end)
    if len(values) == 0 or np.max(values) == 0:
        return (start + end) / 2
    return positions[np.argmax(values)]


def gradient_ascent_3d(data, seed, max_steps=50):
    shape = np.array(data.shape)
    pos = np.clip(np.round(seed).astype(int), 0, shape - 1)
    for _ in range(max_steps):
        best_val = data[tuple(pos)]
        best_pos = pos.copy()
        for delta in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
            nb = pos + np.array(delta)
            if np.all(nb >= 0) and np.all(nb < shape):
                val = data[tuple(nb)]
                if val > best_val:
                    best_val = val
                    best_pos = nb.copy()
        if np.all(best_pos == pos):
            break
        pos = best_pos
    return pos.astype(float)


def find_nucleus_centroid(data, seed, radius=7, threshold_fraction=0.5):
    local_max = gradient_ascent_3d(data, seed)
    local_max_int = np.clip(local_max.astype(int), 0, np.array(data.shape) - 1)
    peak_value = float(data[tuple(local_max_int)])
    if peak_value == 0:
        return seed
    threshold = peak_value * threshold_fraction
    z0 = max(0, local_max_int[0]-radius);  z1 = min(data.shape[0], local_max_int[0]+radius+1)
    y0 = max(0, local_max_int[1]-radius);  y1 = min(data.shape[1], local_max_int[1]+radius+1)
    x0 = max(0, local_max_int[2]-radius);  x1 = min(data.shape[2], local_max_int[2]+radius+1)
    zz, yy, xx = np.mgrid[z0:z1, y0:y1, x0:x1]
    sub = data[z0:z1, y0:y1, x0:x1].astype(float)
    dist_sq = (zz-local_max[0])**2 + (yy-local_max[1])**2 + (xx-local_max[2])**2
    mask = (dist_sq <= radius**2) & (sub >= threshold)
    if not np.any(mask):
        return local_max
    weights = sub[mask]
    total = weights.sum()
    return np.array([
        (zz[mask]*weights).sum()/total,
        (yy[mask]*weights).sum()/total,
        (xx[mask]*weights).sum()/total,
    ])


def _point_to_ray_distance(point, ray_start, ray_end):
    """Minimum distance from a 3D point to a line segment (ray_start→ray_end)."""
    d = ray_end - ray_start
    length_sq = np.dot(d, d)
    if length_sq < 1e-12:
        return np.linalg.norm(point - ray_start)
    t = np.clip(np.dot(point - ray_start, d) / length_sq, 0, 1)
    projection = ray_start + t * d
    return np.linalg.norm(point - projection)


def _find_closest_lattice_point_by_ray(near, far, left_pts, right_pts,
                                        threshold=12.0):
    """Find the closest existing lattice point to the camera ray.

    Uses ray-to-point distance so it works regardless of where the peak/centroid
    lands. If you click visually "on" a point, the ray passes near it.
    Returns (side, index, distance) or None.
    """
    best = None
    for side_char, pts in [('L', left_pts), ('R', right_pts)]:
        if len(pts) == 0:
            continue
        for i, pt in enumerate(pts):
            dist = _point_to_ray_distance(pt, near, far)
            if dist <= threshold and (best is None or dist < best[2]):
                best = (side_char, i, dist)
    return best


def _find_closest_annotation_point_by_ray(near, far, pts, threshold=12.0):
    """Closest annotation point to the camera ray. Returns (index, distance) or None."""
    if len(pts) == 0:
        return None
    best = None
    for i, pt in enumerate(pts):
        dist = _point_to_ray_distance(pt, near, far)
        if dist <= threshold and (best is None or dist < best[1]):
            best = (i, dist)
    return best


def _segment_to_segment_distance(a0, a1, b0, b1):
    """Minimum distance between two 3D line segments a0→a1 and b0→b1.

    Matches MIPAV's DistanceSegment3Segment3.Get() used in addInsertionPoint().
    """
    d1 = a1 - a0  # direction of segment A
    d2 = b1 - b0  # direction of segment B
    r = a0 - b0
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    if a < 1e-12 and e < 1e-12:
        return np.linalg.norm(r)
    if a < 1e-12:
        s, t = 0.0, np.clip(f / e, 0, 1)
    else:
        c = np.dot(d1, r)
        if e < 1e-12:
            t, s = 0.0, np.clip(-c / a, 0, 1)
        else:
            b_val = np.dot(d1, d2)
            denom = a * e - b_val * b_val
            s = np.clip((b_val * f - c * e) / denom, 0, 1) if abs(denom) > 1e-12 else 0.0
            t = (b_val * s + f) / e
            if t < 0:
                t, s = 0.0, np.clip(-c / a, 0, 1)
            elif t > 1:
                t, s = 1.0, np.clip((b_val - c) / a, 0, 1)
    closest = r + s * d1 - t * d2
    return np.linalg.norm(closest)


def _closest_polyline_segment_to_ray(near: np.ndarray, far: np.ndarray,
                                      pts: np.ndarray,
                                      samples_per_segment: int = 20):
    """Build the natural cubic spline through ``pts`` (same parametrization as
    the rendered curve), sample it finely, and return ``(bracket_idx, dist)``
    for the polyline segment closest to the camera ray. ``bracket_idx`` is the
    control-point pair the closest sample falls between. Returns ``None`` if
    fewer than 2 control points or degenerate geometry.
    """
    n = len(pts)
    if n < 2:
        return None
    dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cumlen = np.concatenate([[0.0], np.cumsum(dists)])
    total = cumlen[-1]
    if total < 1e-12:
        return None
    t_ctrl = cumlen / total
    if n == 2:
        polyline = pts
        t_fine = t_ctrl
    else:
        cs = CubicSpline(t_ctrl, pts, axis=0, bc_type='natural')
        n_samples = max((n - 1) * samples_per_segment + 1, 2)
        t_fine = np.linspace(0.0, 1.0, n_samples)
        polyline = cs(t_fine)

    best_dist = float('inf')
    best_seg = -1
    for k in range(len(polyline) - 1):
        d = _segment_to_segment_distance(near, far, polyline[k], polyline[k + 1])
        if d < best_dist:
            best_dist = d
            best_seg = k
    if best_seg < 0:
        return None
    t_mid = 0.5 * (t_fine[best_seg] + t_fine[best_seg + 1])
    bracket = int(np.searchsorted(t_ctrl, t_mid, side='right')) - 1
    bracket = max(0, min(bracket, n - 2))
    return bracket, best_dist


def _find_insertion_index(near: np.ndarray, far: np.ndarray,
                          left_pts: np.ndarray, right_pts: np.ndarray,
                          threshold: float = 12.0) -> tuple[int, str] | None:
    """Find where to insert a new point between existing pairs by testing the
    camera ray against the *displayed* L/R cubic-spline polylines (not just
    the chord between adjacent control points). Returns (insert_index, side)
    or None.

    Diverges from MIPAV's addInsertionPoint (LatticeModel.java:6248-6300),
    which only tests chords and has the same latent tail-curve bug.
    """
    n = min(len(left_pts), len(right_pts))
    if n < 2:
        return None

    left_hit = _closest_polyline_segment_to_ray(near, far, left_pts[:n])
    right_hit = _closest_polyline_segment_to_ray(near, far, right_pts[:n])

    best_idx_l, best_dist_l = (-1, float('inf'))
    if left_hit is not None and left_hit[1] <= threshold:
        best_idx_l, best_dist_l = left_hit

    best_idx_r, best_dist_r = (-1, float('inf'))
    if right_hit is not None and right_hit[1] <= threshold:
        best_idx_r, best_dist_r = right_hit

    if best_idx_l != -1 and best_idx_r != -1:
        return (best_idx_l, 'L') if best_dist_l <= best_dist_r else (best_idx_r, 'R')
    if best_idx_l != -1:
        return (best_idx_l, 'L')
    if best_idx_r != -1:
        return (best_idx_r, 'R')
    return None


def _pick_ring_vertex(near: np.ndarray, far: np.ndarray,
                      ellipse_rings: list[np.ndarray],
                      threshold: float = 8.0):
    """Find the ring vertex closest to the mouse ray.

    Iterates all K rings × 32 vertices. Returns ``(ring_idx, vertex_idx, dist)``
    or None if nothing is within ``threshold`` voxels.
    """
    best = None
    for ring_idx, ring in enumerate(ellipse_rings):
        for v_idx in range(len(ring)):
            d = _point_to_ray_distance(ring[v_idx], near, far)
            if d <= threshold and (best is None or d < best[2]):
                best = (ring_idx, v_idx, d)
    return best


def _project_point_on_ray(p: np.ndarray, near: np.ndarray,
                          far: np.ndarray) -> np.ndarray:
    """Return the point on the infinite ray through ``near→far`` closest to ``p``."""
    d = far - near
    denom = np.dot(d, d)
    if denom < 1e-12:
        return near.copy()
    t = np.dot(p - near, d) / denom
    return near + t * d
