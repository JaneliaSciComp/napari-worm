"""
napari_worm: 3D Cell Annotation Tool for C. elegans

Usage:
    python napari_worm.py                                         # opens directory picker dialog
    python napari_worm.py /path/to/volume.tif                    # single file
    python napari_worm.py /path/to/directory/                     # dual view (MIPAV-style)
    python napari_worm.py /path/to/directory/ --start 100         # start at timepoint 100
    python napari_worm.py /path/to/directory/ --no-grid           # dask 4D slider (fallback)
    python napari_worm.py /path/to/volume.tif -a existing.csv     # load existing annotations
    python napari_worm.py /path/to/RegB/ --channels RegA,RegB    # explicit multi-channel
    python napari_worm.py /path/to/RegB/ --start 100             # auto-discovers RegA if present
"""

import argparse
from pathlib import Path

import dask.array as da
import napari
from napari.utils.notifications import show_info
import numpy as np
import pandas as pd
import tifffile
from scipy.interpolate import CubicSpline
from dask import delayed
from qtpy.QtCore import QEvent, QObject, Qt, QTimer
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QHBoxLayout, QHeaderView, QLabel,
    QMessageBox, QPushButton, QShortcut, QSlider, QSpinBox, QSplitter,
    QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)
import pyqtgraph as pg


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_volume(path: str | Path) -> np.ndarray:
    """Load a 3D TIFF volume."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Volume not found: {path}")
    data = tifffile.imread(str(path))
    print(f"Loaded volume: {path.name}  shape={data.shape}  dtype={data.dtype}")
    return data


def _numeric_sort_key(path: Path) -> list:
    import re
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', path.name)]


def scan_time_series(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    filenames = sorted(directory.glob("*.tif"), key=_numeric_sort_key)
    if not filenames:
        raise FileNotFoundError(f"No .tif files found in {directory}")
    print(f"Found {len(filenames)} timepoints in {directory.name}")
    return filenames


def _channel_colormap(name: str) -> str:
    """Assign colormap by channel directory name (matching MIPAV conventions)."""
    n = name.lower()
    if '405' in n or 'blue' in n:
        return 'blue'
    if '488' in n or 'green' in n:
        return 'green'
    if '561' in n or 'red' in n:
        return 'red'
    if n == 'rega':
        return 'red'
    if n == 'regb':
        return 'green'
    return 'gray'


def discover_channels(volume_path: Path, explicit: str | None = None) -> list[tuple[Path, str, str]]:
    """Discover channel directories.

    Returns list of (directory, name, colormap) tuples.
    Single channel returns one entry with 'gray' colormap.
    """
    if explicit:
        parent = volume_path.parent
        channels = []
        for ch_name in explicit.split(','):
            ch_name = ch_name.strip()
            ch_dir = parent / ch_name
            if ch_dir.is_dir() and list(ch_dir.glob("*.tif"))[:1]:
                channels.append((ch_dir, ch_name, _channel_colormap(ch_name)))
            else:
                print(f"Warning: channel dir not found or empty: {ch_dir}")
        if channels:
            return channels

    # Auto-discover sibling directories that contain .tif files
    parent = volume_path.parent
    siblings = sorted([
        d for d in parent.iterdir()
        if d.is_dir() and d.name.startswith('Reg') and list(d.glob("*.tif"))[:1]
    ], key=lambda d: d.name)

    if len(siblings) >= 2:
        channels = [(d, d.name, _channel_colormap(d.name)) for d in siblings]
        print(f"Discovered {len(channels)} channels: {[c[1] for c in channels]}")
        return channels

    # Single channel — gray
    return [(volume_path, volume_path.name, 'gray')]


def load_time_series_dask(filenames: list[Path]) -> da.Array:
    sample = tifffile.imread(str(filenames[0]))
    print(f"  Each volume: shape={sample.shape}, dtype={sample.dtype}")
    lazy_imread = delayed(tifffile.imread)
    lazy_arrays = [
        da.from_delayed(lazy_imread(str(fn)), shape=sample.shape, dtype=sample.dtype)
        for fn in filenames
    ]
    stack = da.stack(lazy_arrays, axis=0)
    print(f"  Time series shape: {stack.shape} (T, Z, Y, X)")
    return stack


# ---------------------------------------------------------------------------
# Ray picking helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Annotation I/O
# ---------------------------------------------------------------------------

def save_annotations(points, filepath, names=None, segments=None):
    """Save annotations in MIPAV format: name,x_voxels,y_voxels,z_voxels,R,G,B[,lattice_segment]"""
    filepath = Path(filepath)
    n = len(points)
    if names is None:
        names = [f"A{i}" for i in range(n)]
    elif len(names) < n:
        names = list(names) + [f"A{i}" for i in range(len(names), n)]
    df = pd.DataFrame({
        'name':     names,
        'x_voxels': points[:, 2],
        'y_voxels': points[:, 1],
        'z_voxels': points[:, 0],
        'R': 255, 'G': 255, 'B': 255,
    })
    # Add lattice_segment column only if any annotation has a segment assigned
    if segments is not None and any(s != -1 for s in segments):
        seg_vals = list(segments) + [-1] * (n - len(segments))
        df['lattice_segment'] = [s if s != -1 else '' for s in seg_vals[:n]]
    import time
    for attempt in range(5):
        try:
            df.to_csv(filepath, index=False)
            return True
        except BlockingIOError:
            time.sleep(1.0)
    print(f"  WARNING: could not write {filepath} (network drive busy after 5 retries)")
    return False


def _lattice_pair_name(pair_idx: int) -> str:
    """Return name prefix for the i-th L/R lattice pair (matches MIPAV a0/a1/... convention)."""
    return f"a{pair_idx}"


# Seam cell naming: MIPAV uses uppercase H0, H1, H2, V1-V6, T
# Each seam cell is also an L/R pair (H0L/H0R, V1L/V1R, TL/TR)
_SEAM_CELL_SEQUENCE = ['H0', 'H1', 'H2', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'T']


def _renumber_lattice_pairs(pair_infos: list[dict]) -> list[dict]:
    """Re-label all lattice pair names based on positional order.

    Matches MIPAV's updateSeamCount() (LatticeModel.java:8859).
    Walks through pairs nose→tail, numbering seam and non-seam independently:
      - Non-seam:  a0, a1, a2, ...
      - Seam (≤10 total): H0, H1, H2, V1, V2, V3, V4, V5, V6, T
    """
    seam_count = 0
    lattice_count = 0
    for info in pair_infos:
        if info['type'] == 'seam':
            if seam_count < len(_SEAM_CELL_SEQUENCE):
                info['name'] = _SEAM_CELL_SEQUENCE[seam_count]
            else:
                info['name'] = f'S{seam_count}'
            seam_count += 1
        else:
            info['name'] = f'a{lattice_count}'
            lattice_count += 1
    return pair_infos


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


def _find_insertion_index(near: np.ndarray, far: np.ndarray,
                          left_pts: np.ndarray, right_pts: np.ndarray,
                          threshold: float = 12.0) -> tuple[int, str] | None:
    """Find where to insert a new point between existing pairs.

    Matches MIPAV's addInsertionPoint(): measures the 3D distance from the
    mouse RAY (near→far) to each left-curve segment and each right-curve
    segment, then picks the closest segment within the threshold.

    Returns
    -------
    (index, side) where index i means insert at position i+1, and side is
    'L' or 'R' indicating which curve was clicked (closer to ray).
    The clicked side gets the actual click position; the other side is
    interpolated from neighbors. Returns None if no segment is close enough.
    """
    n = min(len(left_pts), len(right_pts))
    if n < 2:
        return None

    best_dist_l = float('inf')
    best_idx_l = -1
    for i in range(n - 1):
        dist = _segment_to_segment_distance(near, far, left_pts[i], left_pts[i + 1])
        if dist < best_dist_l and dist <= threshold:
            best_dist_l = dist
            best_idx_l = i

    best_dist_r = float('inf')
    best_idx_r = -1
    for i in range(n - 1):
        dist = _segment_to_segment_distance(near, far, right_pts[i], right_pts[i + 1])
        if dist < best_dist_r and dist <= threshold:
            best_dist_r = dist
            best_idx_r = i

    # Match MIPAV: if both sides have a hit, pick the closer one
    if best_idx_l != -1 and best_idx_r != -1:
        if best_dist_l <= best_dist_r:
            return (best_idx_l, 'L')
        else:
            return (best_idx_r, 'R')
    elif best_idx_l != -1:
        return (best_idx_l, 'L')
    elif best_idx_r != -1:
        return (best_idx_r, 'R')
    return None


def _smooth_midline_spline(midpoints: np.ndarray, samples_per_segment: int = 20) -> np.ndarray:
    """Fit a natural cubic spline through midpoints (matching MIPAV's NaturalSpline3 BT_FREE).

    Uses arc-length parametrization: parameter t is based on cumulative distance
    between consecutive control points, normalized to [0, 1].

    Parameters
    ----------
    midpoints : (N, 3) array of 3D control points (midpoints between L/R pairs).
    samples_per_segment : number of interpolated points per segment for display.

    Returns
    -------
    (M, 3) array of smoothly interpolated 3D points along the spline.
    """
    n = len(midpoints)
    if n < 2:
        return midpoints
    if n == 2:
        # Only two points — spline would just be a line; return as-is
        return midpoints

    # Arc-length parametrization (matches MIPAV's smoothCurve)
    dists = np.linalg.norm(np.diff(midpoints, axis=0), axis=1)
    cumlen = np.concatenate([[0.0], np.cumsum(dists)])
    total = cumlen[-1]
    if total < 1e-12:
        return midpoints
    t = cumlen / total  # normalized to [0, 1]

    # Natural cubic spline through control points (bc_type='natural' = free boundaries = BT_FREE)
    cs = CubicSpline(t, midpoints, axis=0, bc_type='natural')

    # Sample uniformly along the curve
    n_samples = max((n - 1) * samples_per_segment, 2)
    t_fine = np.linspace(0, 1, n_samples)
    return cs(t_fine)


def _make_spline_with_derivative(points: np.ndarray):
    """Create a natural cubic spline and return (spline, derivative_spline).

    Returns a tuple (cs, cs_deriv) where cs(t) gives positions and
    cs_deriv(t) gives tangent vectors, both in arc-length parametrization.
    """
    n = len(points)
    if n < 2:
        return None, None
    dists = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumlen = np.concatenate([[0.0], np.cumsum(dists)])
    total = cumlen[-1]
    if total < 1e-12:
        return None, None
    t = cumlen / total
    cs = CubicSpline(t, points, axis=0, bc_type='natural')
    cs_deriv = cs.derivative()
    return cs, cs_deriv


def generate_wireframe_mesh(left_pts: np.ndarray, right_pts: np.ndarray,
                            num_ellipse_pts: int = 32,
                            num_samples: int = 0) -> list[np.ndarray]:
    """Generate wireframe mesh from lattice L/R points (MIPAV generateCurves + generateEllipses).

    Algorithm (matching LatticeModel.java):
    1. Compute center spline as midpoint of L/R pairs
    2. Sample center spline uniformly (1-voxel steps)
    3. At each sample: build orthogonal frame (tangent, right, up)
    4. Draw ellipse in the (right, up) plane
    5. Fit num_ellipse_pts longitudinal splines through corresponding ellipse points
    6. Return paths for wireframe rendering

    Parameters
    ----------
    left_pts : (N, 3) array of left lattice points (z, y, x)
    right_pts : (N, 3) array of right lattice points (z, y, x)
    num_ellipse_pts : points per cross-section circle (MIPAV default: 32)
    num_samples : number of uniformly-spaced cross-sections along center spline.
                  0 = auto (1-voxel step size, matching MIPAV).

    Returns
    -------
    List of (M, 3) arrays — each is a smooth longitudinal path for wireframe display.
    Also includes cross-section rings.
    """
    n = min(len(left_pts), len(right_pts))
    if n < 3:
        return []

    # --- Step 1: Center, left, right splines ---
    centers = (left_pts[:n] + right_pts[:n]) / 2.0
    center_cs, center_deriv = _make_spline_with_derivative(centers)
    left_cs, _ = _make_spline_with_derivative(left_pts[:n])
    right_cs, _ = _make_spline_with_derivative(right_pts[:n])
    if center_cs is None or left_cs is None or right_cs is None:
        return []

    # --- Step 2: Uniform sampling along center spline (MIPAV: 1-voxel steps) ---
    # Estimate total arc length
    t_dense = np.linspace(0, 1, 500)
    pts_dense = center_cs(t_dense)
    arc_lengths = np.concatenate([[0], np.cumsum(
        np.linalg.norm(np.diff(pts_dense, axis=0), axis=1))])
    total_length = arc_lengths[-1]
    if total_length < 1e-6:
        return []

    if num_samples <= 0:
        num_samples = max(int(np.ceil(total_length)), 10)

    # Find parameter values for uniformly-spaced arc-length positions
    target_arcs = np.linspace(0, total_length, num_samples)
    t_uniform = np.interp(target_arcs, arc_lengths, t_dense)

    # --- Step 3 & 4: Build orthogonal frames and ellipses at each cross-section ---
    angles = np.linspace(0, 2 * np.pi, num_ellipse_pts, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    # ellipse_rings[i] = (num_ellipse_pts, 3) — points around i-th cross-section
    ellipse_rings = []

    for t_val in t_uniform:
        center = center_cs(t_val)
        tangent = center_deriv(t_val)
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-12:
            tangent = np.array([1.0, 0.0, 0.0])
        else:
            tangent = tangent / tangent_norm

        # Right vector: from left curve to right curve at this parameter
        left_pt = left_cs(t_val)
        right_pt = right_cs(t_val)
        right_vec = right_pt - left_pt
        right_norm = np.linalg.norm(right_vec)
        if right_norm < 1e-12:
            # Fallback: pick an arbitrary perpendicular
            if abs(tangent[0]) < 0.9:
                right_vec = np.cross(tangent, np.array([1, 0, 0]))
            else:
                right_vec = np.cross(tangent, np.array([0, 1, 0]))
            right_norm = np.linalg.norm(right_vec)

        right_vec = right_vec / right_norm

        # Remove tangent component from right_vec (Gram-Schmidt) to ensure orthogonality
        right_vec = right_vec - np.dot(right_vec, tangent) * tangent
        rn = np.linalg.norm(right_vec)
        if rn < 1e-12:
            if abs(tangent[0]) < 0.9:
                right_vec = np.cross(tangent, np.array([1, 0, 0]))
            else:
                right_vec = np.cross(tangent, np.array([0, 1, 0]))
            right_vec = right_vec / np.linalg.norm(right_vec)
        else:
            right_vec = right_vec / rn

        # Up vector: cross product (matches MIPAV: upDir = cross(normal, rightDir))
        up_vec = np.cross(tangent, right_vec)
        up_vec = up_vec / np.linalg.norm(up_vec)

        # Worm diameter at this point (distance between L and R curves)
        diameter = np.linalg.norm(right_pt - left_pt)
        # Radius = half L/R distance (circular cross-section, matching MIPAV
        # updateEllipseModel → makeEllipse2DA with single radius)
        radius = diameter / 2.0

        # Generate circle points in the (right_vec, up_vec) plane
        ring = np.empty((num_ellipse_pts, 3))
        for j in range(num_ellipse_pts):
            ring[j] = center + radius * cos_a[j] * right_vec + radius * sin_a[j] * up_vec
        ellipse_rings.append(ring)

    # --- Step 5: Fit longitudinal splines through corresponding ellipse points ---
    paths = []

    # Longitudinal curves (32 splines running nose-to-tail)
    for j in range(num_ellipse_pts):
        control_pts = np.array([ring[j] for ring in ellipse_rings])
        spline_pts = _smooth_midline_spline(control_pts, samples_per_segment=5)
        paths.append(spline_pts)

    # Cross-section rings (every few samples for wireframe look)
    ring_step = max(1, num_samples // 20)  # ~20 cross-section rings
    for i in range(0, num_samples, ring_step):
        ring = ellipse_rings[i]
        # Close the ring by appending the first point
        closed_ring = np.vstack([ring, ring[0:1]])
        paths.append(closed_ring)

    return paths


def generate_surface_mesh(left_pts: np.ndarray, right_pts: np.ndarray,
                          num_ellipse_pts: int = 32,
                          num_samples: int = 0):
    """Generate a triangle mesh from lattice L/R points (MIPAV generateTriMesh).

    Uses the same ellipse cross-section computation as generate_wireframe_mesh(),
    then packs the rings into (vertices, faces, values) for napari add_surface().

    Vertex layout (matches MIPAV LatticeModel.java:1454-1481):
        [head_center, ring0_pt0..ring0_pt31, ring1_pt0..ring1_pt31, ..., tail_center]

    Face construction (matches MIPAV LatticeModel.java:1483-1517):
        - Head cap: triangle fan from head_center to first ring
        - Body: adjacent rings connected by quads (2 triangles each)
        - Tail cap: triangle fan from tail_center to last ring

    Returns (vertices, faces, values) or None if insufficient data.
    """
    n = min(len(left_pts), len(right_pts))
    if n < 3:
        return None

    # --- Reuse the same spline + ellipse computation as wireframe ---
    centers = (left_pts[:n] + right_pts[:n]) / 2.0
    center_cs, center_deriv = _make_spline_with_derivative(centers)
    left_cs, _ = _make_spline_with_derivative(left_pts[:n])
    right_cs, _ = _make_spline_with_derivative(right_pts[:n])
    if center_cs is None or left_cs is None or right_cs is None:
        return None

    # Uniform sampling along center spline (1-voxel steps)
    t_dense = np.linspace(0, 1, 500)
    pts_dense = center_cs(t_dense)
    arc_lengths = np.concatenate([[0], np.cumsum(
        np.linalg.norm(np.diff(pts_dense, axis=0), axis=1))])
    total_length = arc_lengths[-1]
    if total_length < 1e-6:
        return None

    if num_samples <= 0:
        num_samples = max(int(np.ceil(total_length)), 10)

    target_arcs = np.linspace(0, total_length, num_samples)
    t_uniform = np.interp(target_arcs, arc_lengths, t_dense)

    # Build ellipse rings at each cross-section
    angles = np.linspace(0, 2 * np.pi, num_ellipse_pts, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    ellipse_rings = []
    for t_val in t_uniform:
        center = center_cs(t_val)
        tangent = center_deriv(t_val)
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-12:
            tangent = np.array([1.0, 0.0, 0.0])
        else:
            tangent = tangent / tangent_norm

        left_pt = left_cs(t_val)
        right_pt = right_cs(t_val)
        right_vec = right_pt - left_pt
        right_norm = np.linalg.norm(right_vec)
        if right_norm < 1e-12:
            if abs(tangent[0]) < 0.9:
                right_vec = np.cross(tangent, np.array([1, 0, 0]))
            else:
                right_vec = np.cross(tangent, np.array([0, 1, 0]))
            right_norm = np.linalg.norm(right_vec)

        right_vec = right_vec / right_norm
        right_vec = right_vec - np.dot(right_vec, tangent) * tangent
        rn = np.linalg.norm(right_vec)
        if rn < 1e-12:
            if abs(tangent[0]) < 0.9:
                right_vec = np.cross(tangent, np.array([1, 0, 0]))
            else:
                right_vec = np.cross(tangent, np.array([0, 1, 0]))
            right_vec = right_vec / np.linalg.norm(right_vec)
        else:
            right_vec = right_vec / rn

        up_vec = np.cross(tangent, right_vec)
        up_vec = up_vec / np.linalg.norm(up_vec)

        diameter = np.linalg.norm(right_pt - left_pt)
        radius = diameter / 2.0

        ring = np.empty((num_ellipse_pts, 3))
        for j in range(num_ellipse_pts):
            ring[j] = center + radius * cos_a[j] * right_vec + radius * sin_a[j] * up_vec
        ellipse_rings.append(ring)

    # --- Pack into (vertices, faces, values) matching MIPAV generateTriMesh ---
    S = len(ellipse_rings)  # number of cross-sections
    E = num_ellipse_pts     # points per ring

    # Vertices: [head_center, ring0..ringS-1, tail_center]
    head_center = ellipse_rings[0].mean(axis=0)
    tail_center = ellipse_rings[-1].mean(axis=0)
    vertices = np.empty((2 + S * E, 3))
    vertices[0] = head_center
    for i, ring in enumerate(ellipse_rings):
        vertices[1 + i * E: 1 + (i + 1) * E] = ring
    vertices[-1] = tail_center

    # Faces
    face_list = []

    # Head cap: triangle fan from vertex 0 to first ring (MIPAV lines 1486-1492)
    for j in range(E):
        face_list.append([0, 1 + (j + 1) % E, 1 + j])

    # Body: connect adjacent rings with quads → 2 triangles each (MIPAV lines 1494-1506)
    for i in range(S - 1):
        for j in range(E):
            sj   = 1 + i * E + j
            sj1  = 1 + i * E + (j + 1) % E
            nj   = 1 + (i + 1) * E + j
            nj1  = 1 + (i + 1) * E + (j + 1) % E
            face_list.append([sj, nj1, nj])
            face_list.append([sj, sj1, nj1])

    # Tail cap: triangle fan from last vertex to last ring (MIPAV lines 1508-1514)
    tail_idx = len(vertices) - 1
    last_ring_start = 1 + (S - 1) * E
    for j in range(E):
        face_list.append([tail_idx, last_ring_start + j, last_ring_start + (j + 1) % E])

    faces = np.array(face_list, dtype=np.int32)

    # Values: arc-length parameter for colormap (head=0, tail=1)
    values = np.empty(len(vertices))
    values[0] = 0.0  # head center
    for i in range(S):
        values[1 + i * E: 1 + (i + 1) * E] = i / max(S - 1, 1)
    values[-1] = 1.0  # tail center

    return vertices, faces, values


def load_annotations(filepath):
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Annotations not found: {filepath}")
    df = pd.read_csv(filepath)
    segments = None
    if 'x_voxels' in df.columns:
        points = df[['z_voxels','y_voxels','x_voxels']].values
        names = df['name'].tolist() if 'name' in df.columns else None
        if 'lattice_segment' in df.columns:
            segments = [int(s) if str(s).strip() not in ('', 'nan') else -1
                        for s in df['lattice_segment']]
        print(f"Loaded {len(points)} MIPAV annotations from {filepath}")
    else:
        points = df[['z','y','x']].values
        names = None
        print(f"Loaded {len(points)} annotations from {filepath}")
    return points, names, segments


# ---------------------------------------------------------------------------
# MIPAV-style dual-view window
# ---------------------------------------------------------------------------

class _CommitOnEnterSpinBox(QSpinBox):
    """SpinBox that fires immediately on arrow clicks but waits for Enter when typing.

    - Arrow buttons / keyboard arrows: ``committed`` emitted on every step
    - Typing digits: ``committed`` emitted only on Enter or focus-out
    """
    from qtpy.QtCore import Signal
    committed = Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._typing = False
        self.lineEdit().textEdited.connect(self._on_text_edited)
        self.valueChanged.connect(self._on_value_changed)
        self.editingFinished.connect(self._on_editing_finished)

    def _on_text_edited(self, _text):
        self._typing = True

    def _on_value_changed(self, _val):
        if not self._typing:
            self.committed.emit()

    def _on_editing_finished(self):
        if self._typing:
            self._typing = False
            self.committed.emit()


class _ArrowKeyFilter(QObject):
    """Intercept arrow keys before napari's canvas to enable lattice nudging.

    Matches MIPAV behavior: when a lattice point is selected (hasSelectedPoint),
    arrow keys move the point; otherwise they navigate timepoints.
    """
    _KEY_MAP = {
        Qt.Key_Up: 'up', Qt.Key_Down: 'down',
        Qt.Key_Left: 'left', Qt.Key_Right: 'right',
    }

    def __init__(self, annotator: 'WormAnnotator', parent):
        super().__init__(parent)
        self._annotator = annotator

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress and event.key() in self._KEY_MAP:
            direction = self._KEY_MAP[event.key()]
            ann = self._annotator
            if ann.lattice_mode and ann.lattice_last_placed is not None:
                ann._nudge_last_point(direction)
                return True  # consume event — don't let napari rotate
            else:
                # Navigate timepoints
                if direction == 'right':
                    ann._grid_next(ann.viewer_left)
                elif direction == 'left':
                    ann._grid_prev(ann.viewer_left)
                return True  # consume event
        return False  # pass other events through


class _TableDeleteFilter(QObject):
    """Intercept Delete/Backspace key on table widgets to remove selected row."""
    def __init__(self, annotator, side: int, table_type: str):
        super().__init__()
        self._ann = annotator
        self._side = side
        self._type = table_type  # 'annotation' or 'lattice'

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress and event.key() in (
                Qt.Key_Delete, Qt.Key_Backspace):
            if self._type == 'annotation':
                self._ann._delete_annotation_row(self._side)
            else:
                self._ann._delete_lattice_row(self._side)
            return True
        return False


class _CanvasClickFilter(QObject):
    """Switch the active control panel when a canvas receives a mouse click.

    Mirrors MIPAV's setActiveRenderer(this) callback — clicking either
    renderer canvas activates it and switches the left-side dock widgets.
    """
    def __init__(self, dual_window: 'DualViewWindow', side: int):
        super().__init__()
        self._dual_window = dual_window
        self._side = side

    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseButtonPress:
            # Defer dock switch to next event-loop iteration so it doesn't
            # run inside the GL canvas's own mouse-press handler (which causes
            # a freeze on macOS when the dock resize triggers a GL repaint).
            QTimer.singleShot(0, lambda: self._dual_window.set_active_side(self._side))
            # Clear table highlights when clicking on the canvas
            QTimer.singleShot(0, lambda: self._dual_window.clear_all_highlights())
        return False  # pass event through


class HistogramLUTWidget(QWidget):
    """MIPAV-style 2D histogram + transfer function widget.

    Mirrors MIPAV's ViewJComponentHistoLUT.java:
    - 2D plot: X = image intensity, Y = output brightness (0–255)
    - Histogram bars drawn as background (log scale, bottom-up like MIPAV)
    - Yellow transfer function line with 4 draggable control points:
      * Endpoints (first/last): move Y only (like MIPAV setLinearIndex)
      * Middle points: move freely in X and Y (gamma/windowing)
    - Right-side count labels on histogram (log scale, like MIPAV)
    - Dragging endpoints updates napari contrast_limits; middle points
      adjust the transfer curve shape (gamma-like)
    - One widget per viewer — embedded in dockLayerList, switches with side

    MIPAV reference: ViewJComponentHistoLUT.java, TransferFunction.java
    Default 4 points match MIPAV's linearMode():
      (min, 0), (min+range/3, 85), (min+2*range/3, 170), (max, 255)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layer = None
        self._updating = False
        self._hist_counts = None
        self._hist_edges = None
        self._data_min = 0.0
        self._data_max = 65535.0
        self._bar_item = None
        self._count_labels = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # --- Layout: vertical color bar (left) + 2D plot (right) ---
        plot_row = QHBoxLayout()
        plot_row.setSpacing(2)

        # Vertical LUT color bar (MIPAV's ViewJComponentLUT — 20px wide).
        # Uses an ImageItem for a smooth gradient that reflects the layer's
        # colormap (gray for gray, red for red channel, etc.).
        self._lut_bar = pg.GraphicsLayoutWidget()
        self._lut_bar.setFixedWidth(30)
        self._lut_bar.setMinimumHeight(180)
        self._lut_bar.setMaximumHeight(250)
        self._lut_bar.setBackground((40, 40, 40))
        self._lut_bar_plot = self._lut_bar.addPlot()
        self._lut_bar_plot.hideAxis('bottom')
        self._lut_bar_plot.hideAxis('left')
        self._lut_bar_plot.setMouseEnabled(x=False, y=False)
        self._lut_bar_plot.hideButtons()
        self._lut_bar_plot.setYRange(0, 255)
        self._lut_bar_plot.setXRange(0, 1)
        self._lut_bar_plot.getViewBox().setDefaultPadding(0)
        # Gradient image: 1 pixel wide × 256 tall, will be colored per-layer
        self._lut_gradient = np.linspace(0, 255, 256).reshape(256, 1).astype(np.uint8)
        self._lut_bar_img = pg.ImageItem(self._lut_gradient)
        self._lut_bar_img.setRect(0, 0, 1, 255)
        self._lut_bar_plot.addItem(self._lut_bar_img)
        plot_row.addWidget(self._lut_bar)

        # Main 2D plot
        self._graphics = pg.GraphicsLayoutWidget()
        self._graphics.setMinimumHeight(180)
        self._graphics.setMaximumHeight(250)
        self._graphics.setBackground((40, 40, 40))
        self._plot = self._graphics.addPlot()
        self._plot.setLabel('bottom', 'Image Intensities',
                            color='#CCCCCC', **{'font-size': '9pt'})
        self._plot.setLabel('left', 'Output Brightness',
                            color='#CCCCCC', **{'font-size': '9pt'})
        self._plot.setMouseEnabled(x=False, y=False)
        self._plot.hideButtons()
        self._plot.setYRange(0, 255)
        self._plot.getViewBox().setBackgroundColor((50, 50, 60, 255))
        # Extra bottom margin so axis labels don't get cut off
        self._plot.setContentsMargins(0, 0, 0, 8)
        for axis_name in ('bottom', 'left'):
            axis = self._plot.getAxis(axis_name)
            axis.setStyle(tickFont=pg.QtGui.QFont('', 8))
            axis.setTextPen(pg.mkPen('#CCCCCC'))
            axis.setPen(pg.mkPen('#999999'))

        # Transfer function line (yellow, like MIPAV)
        self._tf_line = pg.PlotCurveItem(pen=pg.mkPen('y', width=2))
        self._tf_line.setZValue(10)
        self._plot.addItem(self._tf_line)

        # 4 draggable control points (MIPAV linearMode default)
        # Points 0,3 = endpoints (Y only); Points 1,2 = middle (free X+Y)
        self._points = []
        for _ in range(4):
            pt = pg.TargetItem(
                pos=(0, 0), size=10, symbol='s',
                pen=pg.mkPen('y', width=2),
                brush=pg.mkBrush(0, 0, 0, 180),
                movable=True)
            pt.setZValue(20)
            self._plot.addItem(pt)
            self._points.append(pt)
        for i, pt in enumerate(self._points):
            pt.sigPositionChanged.connect(
                lambda _p, idx=i: self._on_point_moved(idx))

        plot_row.addWidget(self._graphics, stretch=1)
        layout.addLayout(plot_row)

        # --- Controls row ---
        ctrl_row = QHBoxLayout()
        self._log_cb = QCheckBox("Log scale")
        self._log_cb.setChecked(True)
        self._log_cb.stateChanged.connect(self._redraw_histogram)
        ctrl_row.addWidget(self._log_cb)
        ctrl_row.addStretch()
        self._range_label = QLabel("")
        self._range_label.setStyleSheet("font-size: 9px;")
        ctrl_row.addWidget(self._range_label)
        layout.addLayout(ctrl_row)

    def set_layer(self, layer):
        """Bind to a single napari Image layer."""
        if self._layer is not None:
            try:
                self._layer.events.contrast_limits.disconnect(
                    self._on_layer_contrast_changed)
            except (TypeError, RuntimeError):
                pass
            try:
                self._layer.events.colormap.disconnect(
                    self._on_layer_colormap_changed)
            except (TypeError, RuntimeError):
                pass
        self._layer = layer
        if layer is not None:
            layer.events.contrast_limits.connect(self._on_layer_contrast_changed)
            layer.events.colormap.connect(self._on_layer_colormap_changed)
            self._update_lut_bar_colormap()
            self._compute_histogram()
            self._redraw_histogram()
            self._sync_points_from_layer()

    def _update_lut_bar_colormap(self):
        """Update the vertical LUT color bar to match the layer's colormap."""
        if self._layer is None:
            return
        # Build a 256×1 RGBA image using the layer's colormap
        cmap = self._layer.colormap
        # Sample the colormap at 256 positions
        positions = np.linspace(0, 1, 256)
        colors = cmap.map(positions)  # (256, 4) RGBA float 0–1
        # Convert to uint8 RGBA image (256 rows × 1 col × 4 channels)
        rgba = (colors * 255).astype(np.uint8).reshape(256, 1, 4)
        self._lut_bar_img.setImage(rgba)

    def _compute_histogram(self):
        """Compute histogram bins from layer data."""
        if self._layer is None:
            return
        data = np.asarray(self._layer.data)
        self._data_min = float(data.min())
        self._data_max = float(data.max())
        if self._data_min == self._data_max:
            self._data_max = self._data_min + 1
        self._hist_counts, self._hist_edges = np.histogram(
            data.ravel(), bins=256, range=(self._data_min, self._data_max))

    def _redraw_histogram(self):
        """Redraw histogram bars and count labels."""
        if self._hist_counts is None:
            return

        if hasattr(self, '_hist_bar_items'):
            for item in self._hist_bar_items:
                self._plot.removeItem(item)
            self._hist_bar_items = []
        for lbl in self._count_labels:
            self._plot.removeItem(lbl)
        self._count_labels.clear()

        counts = self._hist_counts.astype(float)
        raw_max = float(self._hist_counts.max()) if len(counts) > 0 else 1.0
        use_log = self._log_cb.isChecked()
        if use_log:
            counts = np.log1p(counts)

        c_max = counts.max() if counts.max() > 0 else 1.0
        norm_counts = counts / c_max * 255.0

        bin_centers = (self._hist_edges[:-1] + self._hist_edges[1:]) / 2.0
        bin_width = self._hist_edges[1] - self._hist_edges[0]

        # Draw individual bars colored by LUT position (MIPAV: each bar uses
        # its transfer function mapped color — dark red for low intensity,
        # bright red for high intensity).
        if hasattr(self, '_hist_bar_items'):
            for item in self._hist_bar_items:
                self._plot.removeItem(item)
        self._hist_bar_items = []
        # Batch bars into groups for performance (every 4 bins)
        step = max(1, len(bin_centers) // 64)
        for i in range(0, len(bin_centers), step):
            end_i = min(i + step, len(bin_centers))
            h = norm_counts[i:end_i].max()
            cx = bin_centers[i:end_i].mean()
            w = bin_width * step
            # Color: map bin intensity position through channel colormap
            t = (cx - self._data_min) / (self._data_max - self._data_min)
            cmap = self._layer.colormap if self._layer is not None else None
            if cmap is not None:
                rgba = cmap.map(np.array([max(0.15, t)]))[0]  # floor at 0.15 so dark end stays visible
                r, g, b = int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)
            else:
                r, g, b = int(40 + t * 200), 10, 10
            bar = pg.BarGraphItem(
                x=[cx], height=[h], width=w,
                brush=pg.mkBrush(r, g, b, 180),
                pen=pg.mkPen(None))
            self._plot.addItem(bar)
            self._hist_bar_items.append(bar)
        self._bar_item = None  # no longer a single item

        # Right-side count labels (like MIPAV's log/linear tick marks)
        x_pos = self._data_max
        if use_log and raw_max > 1:
            log_max = np.log(raw_max)
            for frac in [1.0, 0.5, 0.25, 0.125]:
                count_val = int(np.exp(log_max * frac) + 0.5)
                y_pos = frac * 255.0
                lbl = pg.TextItem(str(count_val), color=(100, 150, 255),
                                  anchor=(0, 0.5))
                lbl.setPos(x_pos, y_pos)
                lbl.setZValue(5)
                font = lbl.textItem.font()
                font.setPointSize(7)
                lbl.setFont(font)
                self._plot.addItem(lbl)
                self._count_labels.append(lbl)
        elif raw_max > 0:
            for frac in [1.0, 0.75, 0.5, 0.25]:
                count_val = int(raw_max * frac + 0.5)
                y_pos = frac * 255.0
                lbl = pg.TextItem(str(count_val), color=(100, 150, 255),
                                  anchor=(0, 0.5))
                lbl.setPos(x_pos, y_pos)
                lbl.setZValue(5)
                font = lbl.textItem.font()
                font.setPointSize(7)
                lbl.setFont(font)
                self._plot.addItem(lbl)
                self._count_labels.append(lbl)

        self._tf_line.setZValue(10)
        for pt in self._points:
            pt.setZValue(20)

        self._plot.setXRange(self._data_min, self._data_max, padding=0.02)
        self._plot.setYRange(0, 255, padding=0.02)

    def _sync_points_from_layer(self):
        """Set 4 control points from layer's contrast_limits (MIPAV linearMode)."""
        if self._layer is None or self._updating:
            return
        self._updating = True
        try:
            lo, hi = self._layer.contrast_limits
            rng = hi - lo if hi > lo else 1.0
            # MIPAV default: 4 points evenly spaced along the ramp
            self._points[0].setPos(lo, 0)
            self._points[1].setPos(lo + rng / 3, 85)
            self._points[2].setPos(lo + 2 * rng / 3, 170)
            self._points[3].setPos(hi, 255)
            self._update_tf_line()
            self._range_label.setText(f"min: {lo:.0f}  max: {hi:.0f}")
        finally:
            self._updating = False

    def _update_tf_line(self):
        """Redraw the yellow transfer function line through all 4 points."""
        xs = [self._points[i].pos().x() for i in range(4)]
        ys = [self._points[i].pos().y() for i in range(4)]
        # Flat extensions beyond endpoints (like MIPAV)
        self._tf_line.setData(
            x=[self._data_min] + xs + [self._data_max],
            y=[ys[0]] + ys + [ys[3]])

    def _on_point_moved(self, idx):
        """Called when user drags a control point."""
        if self._updating or self._layer is None:
            return
        self._updating = True
        try:
            pt = self._points[idx]
            px, py = pt.pos().x(), pt.pos().y()
            py = max(0.0, min(255.0, py))

            if idx == 0 or idx == 3:
                # Endpoints: Y moves freely, X fixed at contrast limit
                # (MIPAV: endpoints only move in Y direction)
                fixed_x = self._points[idx].pos().x()
                pt.setPos(fixed_x, py)
            else:
                # Middle points: free X+Y, but X stays ordered
                # (MIPAV setLinearIndex constraint)
                x_lo = self._points[idx - 1].pos().x()
                x_hi = self._points[idx + 1].pos().x()
                px = max(x_lo + 1, min(x_hi - 1, px))
                pt.setPos(px, py)

            self._update_tf_line()

            # Update contrast_limits from endpoint X positions
            lo_x = self._points[0].pos().x()
            hi_x = self._points[3].pos().x()
            if lo_x < hi_x:
                self._layer.contrast_limits = (lo_x, hi_x)

            # Approximate gamma from BOTH middle points' curvature.
            # MIPAV's piecewise-linear transfer function controls each
            # segment independently, but napari only has a single gamma.
            # We average both middle points' deviation from the linear ramp.
            lo_y = self._points[0].pos().y()
            hi_y = self._points[3].pos().y()
            if hi_x > lo_x and hi_y > lo_y:
                gammas = []
                for ref in (1, 2):
                    mid_x = self._points[ref].pos().x()
                    mid_y = self._points[ref].pos().y()
                    t = (mid_x - lo_x) / (hi_x - lo_x)
                    t = max(0.01, min(0.99, t))
                    norm_y = (mid_y - lo_y) / (hi_y - lo_y)
                    norm_y = max(0.01, min(0.99, norm_y))
                    g = np.log(norm_y) / np.log(t)
                    gammas.append(max(0.1, min(5.0, g)))
                self._layer.gamma = (gammas[0] + gammas[1]) / 2.0

            self._range_label.setText(
                f"min: {lo_x:.0f}  max: {hi_x:.0f}")
        finally:
            self._updating = False

    def _on_layer_contrast_changed(self, event=None):
        """Called when napari's built-in slider changes contrast_limits."""
        self._sync_points_from_layer()

    def _on_layer_colormap_changed(self, event=None):
        """Called when the layer's colormap changes — redraw bars and LUT bar."""
        self._update_lut_bar_colormap()
        self._redraw_histogram()


class DualViewWindow:
    """Modifies viewer_left's native NapariQtMainWindow to host two independent
    canvases side-by-side with switchable left-side dock panels.

    Layout (mirrors MIPAV's PlugInDialogVolumeRenderDual):

      ┌─────────────────────┬──────────────────────────────────────┐
      │  dockLayerControls  │  qt_viewer_left  │  qt_viewer_right  │
      │  dockLayerList      │  (canvas, dims)  │  (canvas, dims)   │
      │  (only one viewer's │                                       │
      │   docks visible at  │  (QSplitter — two independent views) │
      │   a time)           │                                       │
      └─────────────────────┴──────────────────────────────────────┘
      │  Navigation: [Left t= ___] [Right t= ___]                  │
      └────────────────────────────────────────────────────────────┘

    Key design: we reuse viewer_left's NapariQtMainWindow as the host so that
    all napari styling, icons, menus, console, and dock infrastructure come for
    free.  The central widget is replaced to add the second canvas; the right
    viewer's dock widgets are moved into this same window via addDockWidget().
    """

    def __init__(self, viewer_left, viewer_right, nav_widget):
        self._qt_left  = viewer_left.window._qt_viewer
        self._qt_right = viewer_right.window._qt_viewer
        self._active_side = 0  # track current side to avoid redundant show/hide
        self._annotator = None  # set by WormAnnotator after construction
        # Reuse viewer_left's native napari window — correct stylesheet, menubar,
        # status bar, console button, and dock area all work out of the box.
        self._host = viewer_left.window._qt_window

        # Reparent both canvases into a horizontal splitter BEFORE calling
        # setCentralWidget so Qt doesn't destroy _qt_left when the old central
        # widget is replaced.
        canvas_splitter = QSplitter(Qt.Horizontal)
        canvas_splitter.addWidget(self._qt_left)
        canvas_splitter.addWidget(self._qt_right)
        canvas_splitter.setSizes([10000, 10000])

        central = QWidget()
        vbox = QVBoxLayout(central)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)
        vbox.addWidget(canvas_splitter, stretch=1)
        vbox.addWidget(nav_widget, stretch=0)
        self._host.setCentralWidget(central)

        # Per-viewer tab widgets inside dockLayerList:
        #   Tab 0 "Layers": existing layer list + histogram (as before)
        #   Tab 1 "Tables": annotation table + lattice table
        self.histogram_left = HistogramLUTWidget()
        self.histogram_right = HistogramLUTWidget()
        self.annotation_tables: list[QTableWidget] = [None, None]
        self.lattice_tables: list[QTableWidget] = [None, None]
        self.tab_widgets: list[QTabWidget] = [None, None]

        for side, (qt_viewer, hist_widget) in enumerate([
            (self._qt_left, self.histogram_left),
            (self._qt_right, self.histogram_right),
        ]):
            dock_widget = qt_viewer.dockLayerList.widget()
            dock_layout = dock_widget.layout()

            # --- Build single combined panel: controls + threshold + histogram + layers/tables ---
            combined = QVBoxLayout()
            combined.setContentsMargins(0, 0, 0, 0)
            combined.setSpacing(0)

            # Layer controls (opacity, blending, etc.)
            controls_widget = qt_viewer.dockLayerControls.widget()
            combined.addWidget(controls_widget)

            # Threshold slider
            from superqt import QLabeledDoubleSlider
            thresh_row = QHBoxLayout()
            thresh_row.setContentsMargins(8, 2, 8, 2)
            thresh_lbl = QLabel("threshold:")
            thresh_lbl.setFixedWidth(62)
            thresh_row.addWidget(thresh_lbl)
            thresh_slider = QLabeledDoubleSlider(Qt.Horizontal)
            thresh_slider.setRange(0, 65535)
            thresh_slider.setValue(0)
            thresh_slider.setToolTip(
                "Minimum intensity threshold — voxels below this value\n"
                "appear dark on both viewers")
            thresh_slider.valueChanged.connect(
                lambda v: self._annotator._apply_threshold(int(v)))
            thresh_row.addWidget(thresh_slider)
            combined.addLayout(thresh_row)

            if not hasattr(self, '_thresh_sliders'):
                self._thresh_sliders = [None, None]
            self._thresh_sliders[side] = thresh_slider

            # Histogram
            combined.addWidget(hist_widget)

            # Layers tab: reparent existing layer list children
            layers_tab = QWidget()
            layers_layout = QVBoxLayout(layers_tab)
            layers_layout.setContentsMargins(0, 0, 0, 0)
            layers_layout.setSpacing(0)
            while dock_layout.count():
                item = dock_layout.takeAt(0)
                w = item.widget()
                if w:
                    layers_layout.addWidget(w)

            # Tables tab: annotation + lattice tables
            tables_tab = QWidget()
            tables_layout = QVBoxLayout(tables_tab)
            tables_layout.setContentsMargins(0, 0, 0, 0)
            tables_layout.setSpacing(4)

            ann_label = QLabel("Annotations")
            ann_label.setStyleSheet("font-weight: bold; padding: 4px;")
            tables_layout.addWidget(ann_label)
            ann_table = QTableWidget(0, 6)
            ann_table.setHorizontalHeaderLabels(["Name", "X", "Y", "Z", "Intensity", "Seg"])
            ann_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            ann_table.setEditTriggers(QTableWidget.DoubleClicked | QTableWidget.SelectedClicked)
            ann_table.setSelectionBehavior(QTableWidget.SelectRows)
            ann_table.verticalHeader().setVisible(False)
            tables_layout.addWidget(ann_table, stretch=1)

            lat_label = QLabel("Lattice")
            lat_label.setStyleSheet("font-weight: bold; padding: 4px;")
            tables_layout.addWidget(lat_label)
            lat_table = QTableWidget(0, 7)
            lat_table.setHorizontalHeaderLabels(
                ["Pair", "L_X", "L_Y", "L_Z", "R_X", "R_Y", "R_Z"])
            lat_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            lat_table.setEditTriggers(QTableWidget.DoubleClicked | QTableWidget.SelectedClicked)
            lat_table.setSelectionBehavior(QTableWidget.SelectRows)
            lat_table.verticalHeader().setVisible(False)
            tables_layout.addWidget(lat_table, stretch=1)

            # Assemble Layers/Tables tabs
            tab_widget = QTabWidget()
            tab_widget.addTab(layers_tab, "Layers")
            tab_widget.addTab(tables_tab, "Tables")
            combined.addWidget(tab_widget)

            # Put combined layout into the dock
            combined_widget = QWidget()
            combined_widget.setLayout(combined)
            dock_layout.addWidget(combined_widget)

            # Hide the now-empty controls dock
            qt_viewer.dockLayerControls.hide()

            self.annotation_tables[side] = ann_table
            self.lattice_tables[side] = lat_table
            self.tab_widgets[side] = tab_widget

        # Tabify the two layer list docks (only one pair of tabs now)
        self._host.tabifyDockWidget(self._qt_left.dockLayerList,
                                    self._qt_right.dockLayerList)
        self._qt_left.dockLayerList.raise_()

        # Canvas click → switch active dock panel (MIPAV's setActiveRenderer)
        self._filter_left  = _CanvasClickFilter(self, 0)
        self._filter_right = _CanvasClickFilter(self, 1)
        self._qt_left.canvas.native.installEventFilter(self._filter_left)
        self._qt_right.canvas.native.installEventFilter(self._filter_right)

        # Continuous canvas refresh (mirrors MIPAV's startAnimator)
        self._refresh_timer = QTimer(self._host)
        self._refresh_timer.timeout.connect(self._qt_left.canvas.native.update)
        self._refresh_timer.timeout.connect(self._qt_right.canvas.native.update)
        self._refresh_timer.start(100)  # 10 fps — enough for smooth property updates

        # Mirror viewer_right's status text into the host window's status bar
        # so the cursor position/value is always visible regardless of active canvas.
        if hasattr(viewer_right, 'status'):
            viewer_right.events.status.connect(
                lambda e: self._host.statusBar().showMessage(e.value)
                if self._active_side == 1 else None
            )

    # Proxy methods so call-sites in _init_dual_window_mode stay unchanged
    def showMaximized(self):
        self._host.showMaximized()

    def setWindowTitle(self, title: str):
        self._host.setWindowTitle(title)

    def update_dock_titles(self, t_left: int, t_right: int):
        """Update dock tab labels to show timepoint numbers."""
        self._qt_left.dockLayerList.setWindowTitle(f"t={t_left}")
        self._qt_right.dockLayerList.setWindowTitle(f"t={t_right}")

    def resizeDocks(self, docks, sizes, orientation):
        self._host.resizeDocks(docks, sizes, orientation)

    def set_active_side(self, side: int):
        """Bring the active viewer's dock tabs to the front."""
        if side == self._active_side:
            return
        self._active_side = side
        if side == 0:
            self._qt_left.dockLayerList.raise_()
        else:
            self._qt_right.dockLayerList.raise_()

    def clear_all_highlights(self):
        """Clear table-driven highlights on all point layers."""
        if self._annotator is None:
            return
        ann = self._annotator
        for side in range(2):
            ann._clear_highlight(ann.grid_points_layers[side], 'yellow', 5)
            ann._clear_highlight(ann.lattice_left_layers[side], 'cyan', 7)
            ann._clear_highlight(ann.lattice_right_layers[side], 'magenta', 7)
        # Also deselect table rows
        for table in self.annotation_tables + self.lattice_tables:
            if table is not None:
                table.clearSelection()


# ---------------------------------------------------------------------------
# Main annotator class
# ---------------------------------------------------------------------------

class WormAnnotator:
    """Main annotation tool class."""

    def __init__(self, volume_path, annotations_path=None, grid_mode=True, start_t=0,
                 channels=None):
        self.volume_path = Path(volume_path)
        self.annotations_path = annotations_path
        self.is_time_series = self.volume_path.is_dir()
        self.use_grid = grid_mode and self.is_time_series
        self.start_t = start_t
        self.channels_arg = channels

        if self.use_grid:
            self._init_dual_window_mode()
        elif self.is_time_series:
            self.viewer = napari.Viewer(ndisplay=3)
            self._init_dask_mode()
        else:
            self.viewer = napari.Viewer(ndisplay=3)
            self._init_single_mode()

        self.cell_names = None
        if annotations_path and Path(annotations_path).exists():
            points, names, _segments = load_annotations(annotations_path)
            if not self.use_grid:
                self.points_layer.data = points
            self.cell_names = names

        if not self.use_grid:
            self.viewer.bind_key('s', self._save_annotations)

        print("\n--- napari_worm ---")
        print("  Cmd+Click         : annotate at peak intensity")
        print("  L                 : toggle lattice mode")
        print("  (lattice mode)    Cmd+Click       = place Left  lattice point")
        print("  (lattice mode)    Cmd+Shift+Click = place Right lattice point")
        print("  Right / ]         : next timepoint pair")
        print("  Left  / [         : previous timepoint pair")
        print("  Cmd+Z             : undo last annotation/lattice point")
        print("  S                 : save annotations + lattice")
        print("  W                 : toggle wireframe mesh")
        print("  Shift+W           : toggle surface mesh")
        print("  Histogram widget  : drag min/max slider for contrast")
        if hasattr(self, 'multi_channel') and self.multi_channel:
            ch_names = [c[1] for c in self.channels]
            print(f"  Channels          : {ch_names} (click layer to switch histogram)")
        print("-------------------\n")

    # ------------------------------------------------------------------ #
    # Single-file mode                                                     #
    # ------------------------------------------------------------------ #

    def _init_single_mode(self):
        self.data = load_volume(self.volume_path)
        self.image_layer = self.viewer.add_image(
            self.data, name='Volume', colormap='gray', rendering='mip')
        self.points_layer = self.viewer.add_points(
            ndim=3, name='Annotations', size=5, face_color='yellow')
        self.image_layer.mouse_drag_callbacks.append(self._on_click_single)

    def _on_click_single(self, layer, event):
        if 'Control' not in event.modifiers:
            return
        if self.viewer.dims.ndisplay != 3:
            return
        near, far = layer.get_ray_intersections(
            event.position, event.view_direction, event.dims_displayed)
        if near is None:
            return
        pos = find_nucleus_centroid(self.data, find_peak_along_ray(self.data, near, far))
        self.points_layer.add(pos)
        print(f"Added at z={pos[0]:.1f} y={pos[1]:.1f} x={pos[2]:.1f}")

    # ------------------------------------------------------------------ #
    # Dask 4D slider mode  (--no-grid)                                     #
    # ------------------------------------------------------------------ #

    def _init_dask_mode(self):
        self.tiff_files = scan_time_series(self.volume_path)
        self.data = load_time_series_dask(self.tiff_files)
        sample = self.data[0].compute()
        self.image_layer = self.viewer.add_image(
            self.data, name='Volume', colormap='gray', rendering='mip',
            contrast_limits=[float(sample.min()), float(sample.max())],
            multiscale=False)
        self.points_layer = self.viewer.add_points(
            ndim=4, name='Annotations', size=5, face_color='yellow')
        self.image_layer.mouse_drag_callbacks.append(self._on_click_dask)

    def _on_click_dask(self, layer, event):
        if 'Control' not in event.modifiers or self.viewer.dims.ndisplay != 3:
            return
        near, far = layer.get_ray_intersections(
            event.position, event.view_direction, event.dims_displayed)
        if near is None:
            return
        t = int(self.viewer.dims.current_step[0])
        vol = np.asarray(self.data[t])
        pos = find_nucleus_centroid(vol, find_peak_along_ray(vol, near[-3:], far[-3:]))
        self.points_layer.add(np.concatenate([[t], pos]))
        print(f"Added at z={pos[0]:.1f} y={pos[1]:.1f} x={pos[2]:.1f}")

    # ------------------------------------------------------------------ #
    # Dual-window mode  (MIPAV-style, default for directories)             #
    # ------------------------------------------------------------------ #

    def _init_dual_window_mode(self):
        """One window, shared left-side controls, two independent 3D canvases.

        Two napari viewers are created with show=False.  Their layer-list and
        control widgets are extracted and placed in a QStackedWidget on the left
        (only the active viewer's panel is visible).  Their canvas+dims widgets
        fill a horizontal QSplitter on the right.  Clicking either canvas
        switches the stacked widget via _CanvasClickFilter.
        """
        # Channel discovery: find sibling Reg* directories
        self.channels = discover_channels(self.volume_path, self.channels_arg)
        self.channel_tiff_files: list[list[Path]] = []
        for ch_dir, ch_name, _ in self.channels:
            self.channel_tiff_files.append(scan_time_series(ch_dir))
        self.tiff_files = self.channel_tiff_files[0]  # primary channel for navigation
        self.multi_channel = len(self.channels) > 1

        max_t = len(self.tiff_files) - 1
        start = min(self.start_t, max_t - 1)

        self.grid_annotations: dict[int, np.ndarray] = {}
        self.grid_annotation_segments: dict[int, list[int]] = {}  # parallel to grid_annotations, -1 = no lattice segment
        self.undo_stack: list[tuple[int, object]] = []
        # For multi-channel: grid_image_layers[side] is a list of image layers (one per channel)
        self.grid_image_layers: list = [None, None]
        self.grid_points_layers: list = [None, None]
        self.grid_timepoints: list[int] = [start, min(start + 1, max_t)]

        # Lattice state
        self.lattice_mode = False
        self.lattice_annotations: dict[int, dict] = {}  # {t: {'left': arr, 'right': arr}}
        self.lattice_left_layers:  list = [None, None]
        self.lattice_right_layers: list = [None, None]
        self.lattice_line_layers:  list = [None, None]
        self.lattice_mid_layers:   list = [None, None]
        self.lattice_left_curve_layers:  list = [None, None]
        self.lattice_right_curve_layers: list = [None, None]
        self.wireframe_layers: list = [None, None]
        self.wireframe_visible: bool = False
        self.surface_layers: list = [None, None]
        self.surface_visible: bool = False
        self.lattice_next_side: str = 'L'  # alternates L→R→L→R (nose to tail)
        self.lattice_last_placed: dict | None = None  # {'side_idx', 'timepoint', 'layer', 'char'}
        # Ordered list of pair metadata per timepoint:
        # [{name: 'a0', type: 'lattice'}, {name: 'H0', type: 'seam'}, ...]
        # The i-th entry corresponds to i-th L point and i-th R point in the layers
        self.lattice_pair_names: dict[int, list[dict]] = {}  # {timepoint: [pair_info, ...]}
        self.lattice_seam_counter: dict[int, list[str]] = {}  # {timepoint: [used seam names]}
        self.lattice_undo_stack: list = []  # ('L'|'R', timepoint, layer)

        # Per-channel contrast limits — initialized lazily in _load_dual_pair
        # on first load to avoid redundant volume reads at startup
        self.channel_contrast_limits: dict[str, list[float]] = {}
        self.contrast_limits = None

        # Two independent napari viewers — hidden until embedded in DualViewWindow
        self.viewer_left  = napari.Viewer(ndisplay=3, show=False)
        self.viewer_right = napari.Viewer(ndisplay=3, show=False)
        self.viewer = self.viewer_left  # primary reference for save/print logic

        # Navigation spinboxes
        self._nav_updating = False
        nav_widget = self._build_nav_widget(max_t, start)

        # Build the single main window (MIPAV's JFrame equivalent)
        self.dual_window = DualViewWindow(self.viewer_left, self.viewer_right, nav_widget)
        self.dual_window._annotator = self

        # Connect annotation table edits (Seg column) and row selection
        for side in range(2):
            ann_table = self.dual_window.annotation_tables[side]
            ann_table.cellChanged.connect(
                lambda row, col, s=side: self._on_annotation_table_edited(s, row, col))
            ann_table.currentCellChanged.connect(
                lambda row, col, prev_row, prev_col, s=side:
                    self._on_annotation_row_selected(s, row))
            lat_table = self.dual_window.lattice_tables[side]
            lat_table.cellChanged.connect(
                lambda row, col, s=side: self._on_lattice_table_edited(s, row, col))
            lat_table.currentCellChanged.connect(
                lambda row, col, prev_row, prev_col, s=side:
                    self._on_lattice_row_selected(s, row, col))
            # Delete key on tables
            ann_del = _TableDeleteFilter(self, side, 'annotation')
            ann_table.installEventFilter(ann_del)
            lat_del = _TableDeleteFilter(self, side, 'lattice')
            lat_table.installEventFilter(lat_del)
            # Keep references so they aren't garbage collected
            if not hasattr(self, '_table_filters'):
                self._table_filters = []
            self._table_filters.extend([ann_del, lat_del])

        # Qt-level shortcuts on the host window so they fire regardless of which
        # child widget has keyboard focus (napari bind_key only fires when the
        # vispy canvas is focused, which it loses after layer-list interactions).
        host = self.dual_window._host
        def _sc(key, fn):
            s = QShortcut(QKeySequence(key), host)
            s.activated.connect(fn)
        _sc('S',       lambda: self._save_annotations(self.viewer_left))
        _sc('Ctrl+S',  lambda: self._save_annotations(self.viewer_left))
        _sc('Ctrl+Z',  lambda: self._undo_last_point(self.viewer_left))
        _sc(']',       lambda: self._grid_next(self.viewer_left))
        _sc('[',       lambda: self._grid_prev(self.viewer_left))
        _sc('L',       self._toggle_lattice_mode)
        _sc('D',       self._on_lattice_done)
        _sc('W',       self._toggle_wireframe)
        _sc('Shift+W', self._toggle_surface)

        # Event filter intercepts arrow keys BEFORE napari's canvas consumes them.
        # In lattice mode with a selected point → nudge; otherwise → navigate.
        # Matches MIPAV: arrow keys move selected point when hasSelectedPoint() is true,
        # otherwise navigate the camera.
        # Install arrow key filter on ALL widgets that might receive key events:
        # the host window AND both canvas native widgets (vispy consumes keys directly)
        self._arrow_filter = _ArrowKeyFilter(self, host)
        host.installEventFilter(self._arrow_filter)
        qt_left = self.dual_window._qt_left
        qt_right = self.dual_window._qt_right
        self._arrow_filter_l = _ArrowKeyFilter(self, qt_left.canvas.native)
        self._arrow_filter_r = _ArrowKeyFilter(self, qt_right.canvas.native)
        qt_left.canvas.native.installEventFilter(self._arrow_filter_l)
        qt_right.canvas.native.installEventFilter(self._arrow_filter_r)

        # Show window FIRST so Qt initialises the GL context for both canvases,
        # then load layers.  If layers are added before the GL context exists,
        # vispy creates visuals without a valid context and the event→repaint
        # pipeline is broken (controls don't visually update the canvas).
        self.dual_window.showMaximized()
        QApplication.processEvents()   # flush Qt events → GL context ready

        # Set left dock area to 370 px so control labels are fully visible
        # (resizeDocks must be called after the window is shown)
        self.dual_window.resizeDocks(
            [self.dual_window._qt_left.dockLayerList],
            [370],
            Qt.Horizontal,
        )

        self._load_dual_pair(start, min(start + 1, max_t))

    def _build_nav_widget(self, max_t, start) -> QWidget:
        nav = QWidget()
        layout = QHBoxLayout(nav)
        layout.setContentsMargins(4, 4, 4, 4)

        layout.addStretch()
        layout.addWidget(QLabel("Left t="))
        self._left_spin = _CommitOnEnterSpinBox()
        self._left_spin.setRange(0, max_t)
        self._left_spin.setValue(start)
        self._left_spin.setFixedWidth(80)
        self._left_spin.committed.connect(self._on_spinbox_changed)
        layout.addWidget(self._left_spin)

        layout.addStretch()
        layout.addWidget(QLabel("Right t="))
        self._right_spin = _CommitOnEnterSpinBox()
        self._right_spin.setRange(0, max_t)
        self._right_spin.setValue(min(start + 1, max_t))
        self._right_spin.setFixedWidth(80)
        self._right_spin.committed.connect(self._on_spinbox_changed)
        layout.addWidget(self._right_spin)

        layout.addStretch()

        save_btn = QPushButton("Save")
        save_btn.setToolTip(
            "Save annotations and lattice for all visited timepoints (S)")
        save_btn.clicked.connect(
            lambda: self._save_annotations(self.viewer_left))
        layout.addWidget(save_btn)

        help_btn = QPushButton("?")
        help_btn.setFixedWidth(30)
        help_btn.setToolTip("Show keyboard shortcuts and help")
        help_btn.clicked.connect(self._show_help)
        layout.addWidget(help_btn)
        layout.addSpacing(20)

        return nav

    def _show_help(self):
        """Show a floating help window with shortcuts and usage info."""
        # Reuse existing window if still open
        if hasattr(self, '_help_window') and self._help_window.isVisible():
            self._help_window.raise_()
            return

        from qtpy.QtWidgets import QTextBrowser
        win = QWidget(None, Qt.Window)
        win.setWindowTitle("napari-worm Help")
        win.resize(480, 520)
        layout = QVBoxLayout(win)
        layout.setContentsMargins(10, 10, 10, 10)
        text = QTextBrowser()
        text.setOpenExternalLinks(True)
        text.setHtml("""
<h2>napari-worm Shortcuts</h2>

<h3>Annotation Mode (default)</h3>
<table cellpadding="4">
<tr><td><b>Cmd+Click</b></td><td>Place annotation at peak intensity</td></tr>
<tr><td><b>Cmd+Z</b></td><td>Undo last annotation</td></tr>
<tr><td><b>S</b> / <b>Cmd+S</b></td><td>Save annotations + lattice</td></tr>
<tr><td><b>L</b></td><td>Switch to Lattice mode</td></tr>
<tr><td><b>Delete</b></td><td>Remove selected table row</td></tr>
</table>

<h3>Lattice Mode (press L)</h3>
<table cellpadding="4">
<tr><td><b>Cmd+Click</b></td><td>Place lattice point (alternates L/R)</td></tr>
<tr><td><b>Cmd+Shift+Click</b></td><td>Place seam cell (L/R pair)</td></tr>
<tr><td><b>Cmd+Click on point</b></td><td>Select point, then drag to move</td></tr>
<tr><td><b>Cmd+Click on curve</b></td><td>Insert new pair between existing</td></tr>
<tr><td><b>Arrow keys</b></td><td>Nudge selected point by 1 voxel</td></tr>
<tr><td><b>Cmd+Z</b></td><td>Undo last lattice operation</td></tr>
<tr><td><b>D</b></td><td>Done — save and exit lattice mode</td></tr>
<tr><td><b>L</b></td><td>Back to annotation mode</td></tr>
</table>

<h3>Navigation</h3>
<table cellpadding="4">
<tr><td><b>Right</b> / <b>]</b></td><td>Next timepoint pair</td></tr>
<tr><td><b>Left</b> / <b>[</b></td><td>Previous timepoint pair</td></tr>
<tr><td><b>Spinboxes</b></td><td>Type number + Enter to jump</td></tr>
</table>

<h3>Visualization</h3>
<table cellpadding="4">
<tr><td><b>W</b></td><td>Toggle wireframe mesh</td></tr>
<tr><td><b>Shift+W</b></td><td>Toggle surface mesh</td></tr>
<tr><td><b>Eye icon</b></td><td>Toggle channel visibility</td></tr>
<tr><td><b>Click layer</b></td><td>Switch histogram to that channel</td></tr>
</table>

<h3>Tips</h3>
<ul>
<li>Click on either canvas to switch the left panel to that viewer's controls</li>
<li>Click a row in the Tables tab to highlight that point in 3D</li>
<li>Edit X/Y/Z coordinates directly in the table (double-click)</li>
<li>Multi-channel: RegA (red) + RegB (green) auto-discovered from sibling folders</li>
<li><b>Threshold slider</b> (below layer controls): filters dim voxels on both viewers</li>
<li><b>Contrast limits</b>: adjusts the selected layer/channel only</li>
</ul>
""")
        layout.addWidget(text)
        win.show()
        self._help_window = win

    def _find_peak_multi_channel(self, near, far, side):
        """Find peak along ray by blending all channels (MIPAV accurate mode).

        MIPAV blends channels at each ray sample point, then finds one peak on
        the combined signal. This avoids the problem of each channel finding a
        peak at a different z-depth and the wrong channel winning.
        """
        img_layers = self.grid_image_layers[side]
        if not isinstance(img_layers, list):
            img_layers = [img_layers]
        layers = [l for l in img_layers if l is not None]
        if not layers:
            return (near + far) / 2

        # Sample ray positions (same for all channels)
        n_samples = int(np.linalg.norm(far - near)) + 1
        positions = np.linspace(near, far, n_samples)
        from scipy.ndimage import map_coordinates
        coords = positions.T  # (3, n_samples)

        # Blend: sum normalized intensities across all visible channels
        blended = np.zeros(n_samples)
        for img_lyr in layers:
            if not img_lyr.visible:
                continue
            vol = np.asarray(img_lyr.data).astype(np.float64)
            vals = map_coordinates(vol, coords, order=1, mode='constant', cval=0.0)
            # Normalize each channel to [0,1] by its contrast range so channels
            # with different intensity ranges contribute equally
            cmin, cmax = img_lyr.contrast_limits
            if cmax > cmin:
                vals = (vals - cmin) / (cmax - cmin)
            blended += vals

        if len(blended) == 0 or np.max(blended) == 0:
            return (near + far) / 2
        return positions[np.argmax(blended)]

    def _get_blended_volume(self, side):
        """Return a blended volume for centroid refinement (multi-channel aware).

        Each visible channel is normalized by its contrast limits and summed,
        so gradient_ascent and centroid operate on the combined signal.
        """
        img_layers = self.grid_image_layers[side]
        if not isinstance(img_layers, list):
            return np.asarray(img_layers.data).astype(np.float64)
        blended = None
        for img_lyr in img_layers:
            if img_lyr is None or not img_lyr.visible:
                continue
            vol = np.asarray(img_lyr.data).astype(np.float64)
            cmin, cmax = img_lyr.contrast_limits
            if cmax > cmin:
                vol = (vol - cmin) / (cmax - cmin)
            if blended is None:
                blended = vol
            else:
                blended += vol
        if blended is None:
            return np.asarray(img_layers[0].data).astype(np.float64)
        return blended

    def _load_dual_pair(self, t_left: int, t_right: int):
        """Load two timepoints into their respective viewers."""
        self._save_grid_annotations_to_cache()
        self._save_lattice_to_cache()

        self.viewer_left.layers.clear()
        self.viewer_right.layers.clear()
        self.grid_timepoints = [t_left, t_right]

        for side, (viewer_ref, ti) in enumerate(
                [(self.viewer_left, t_left), (self.viewer_right, t_right)]):

            # Load all channels as separate image layers with additive blending
            channel_img_layers = []
            all_volumes = []
            for ch_idx, (ch_files, (_, ch_name, ch_cmap)) in enumerate(
                    zip(self.channel_tiff_files, self.channels)):
                vol = load_volume(ch_files[ti])
                all_volumes.append(vol)
                # Single channel → gray, no blending change; multi → colored + additive
                if self.multi_channel:
                    cmap = ch_cmap
                    blending = 'additive'
                    layer_name = f'{ch_name} t={ti}'
                else:
                    cmap = 'gray'
                    blending = 'translucent'
                    layer_name = f'Volume t={ti}'
                # Initialize contrast limits lazily on first load
                if ch_name not in self.channel_contrast_limits:
                    self.channel_contrast_limits[ch_name] = [
                        float(vol.min()), float(vol.max())]
                    if self.contrast_limits is None:
                        self.contrast_limits = self.channel_contrast_limits[ch_name]
                clim = self.channel_contrast_limits[ch_name]
                img = viewer_ref.add_image(
                    vol, name=layer_name, colormap=cmap,
                    rendering='mip', contrast_limits=clim,
                    blending=blending, multiscale=False)
                channel_img_layers.append(img)

            # Surface layer: sits below all interactive layers so it never steals clicks
            empty_verts = np.zeros((3, 3))
            empty_faces = np.array([[0, 1, 2]])
            empty_vals  = np.zeros(3)
            surface = viewer_ref.add_surface(
                (empty_verts, empty_faces, empty_vals),
                name=f'Surface t={ti}', shading='smooth',
                colormap='turbo', opacity=0.7)
            surface.interactive = False
            surface.visible = self.surface_visible
            pts = viewer_ref.add_points(
                ndim=3, name=f'Annotations t={ti}', size=5, face_color='yellow')
            pts.mode = 'pan_zoom'  # prevent napari's native add-on-click

            if ti in self.grid_annotations and len(self.grid_annotations[ti]) > 0:
                pts.data = self.grid_annotations[ti]

            # Lattice layers: left (cyan squares), right (magenta squares), lines (yellow)
            lat_l = viewer_ref.add_points(
                ndim=3, name=f'Lattice Left t={ti}',  size=7,
                face_color='cyan',    symbol='square')
            lat_r = viewer_ref.add_points(
                ndim=3, name=f'Lattice Right t={ti}', size=7,
                face_color='magenta', symbol='square')
            lat_lines = viewer_ref.add_shapes(
                ndim=3, name=f'Lattice Lines t={ti}',
                edge_color='yellow', edge_width=1, face_color='transparent')
            lat_mid = viewer_ref.add_shapes(
                ndim=3, name=f'Lattice Mid t={ti}',
                edge_color='red', edge_width=1, face_color='transparent')
            lat_left_curve = viewer_ref.add_shapes(
                ndim=3, name=f'Lattice Left Curve t={ti}',
                edge_color='magenta', edge_width=1, face_color='transparent')
            lat_right_curve = viewer_ref.add_shapes(
                ndim=3, name=f'Lattice Right Curve t={ti}',
                edge_color='green', edge_width=1, face_color='transparent')
            wireframe = viewer_ref.add_shapes(
                ndim=3, name=f'Wireframe t={ti}',
                edge_color='white', edge_width=0.5, face_color='transparent')
            wireframe.visible = self.wireframe_visible
            # Lock lattice layers to pan_zoom — our handler does all adding
            for lyr in (lat_l, lat_r, lat_lines, lat_mid,
                        lat_left_curve, lat_right_curve, wireframe):
                lyr.mode = 'pan_zoom'
            # Lock pts too if already in lattice mode (e.g. after timepoint change)
            if self.lattice_mode:
                pts.mode = 'pan_zoom'

            cached_lat = self.lattice_annotations.get(ti, {})
            if len(cached_lat.get('left',  [])) > 0:
                lat_l.data = cached_lat['left']
            if len(cached_lat.get('right', [])) > 0:
                lat_r.data = cached_lat['right']
            self._update_lattice_visuals(lat_l, lat_r, lat_lines, lat_mid,
                                        lat_left_curve, lat_right_curve)

            # Use first channel's image layer as the ray-intersection reference
            img_ref = channel_img_layers[0]

            # Each viewer handles only its own canvas — no click-routing needed
            def make_handler(side_idx, timepoint,
                             img_ref, pts_ref,
                             lat_l_ref, lat_r_ref, lat_lines_ref, lat_mid_ref,
                             lat_lc_ref, lat_rc_ref):
                def handler(layer, event):
                    if 'Control' not in event.modifiers:
                        return
                    if not self.lattice_mode:
                        self._on_click_dual(img_ref, event,
                                            side_idx, timepoint, pts_ref)
                        return

                    near, far = img_ref.get_ray_intersections(
                        event.position, event.view_direction, event.dims_displayed)
                    if near is None or far is None:
                        return
                    peak = self._find_peak_multi_channel(near, far, side_idx)
                    pos = find_nucleus_centroid(
                        self._get_blended_volume(side_idx), peak)

                    n_l = len(lat_l_ref.data)
                    n_r = len(lat_r_ref.data)
                    n_complete = min(n_l, n_r)

                    # --- Check for nearby existing point (select / drag) ---
                    drag_target = None
                    if n_l > 0 or n_r > 0:
                        l_data = np.asarray(lat_l_ref.data) if n_l > 0 else np.empty((0, 3))
                        r_data = np.asarray(lat_r_ref.data) if n_r > 0 else np.empty((0, 3))
                        closest = _find_closest_lattice_point_by_ray(
                            near, far, l_data, r_data, threshold=12.0)
                        if closest is not None:
                            drag_target = closest  # (side_char, pt_idx, dist)

                    if drag_target is not None:
                        pass  # fall through to drag/select below
                    elif n_complete >= 2:
                        # --- Check for INSERTION (click on curve, MIPAV addInsertionPoint) ---
                        result = _find_insertion_index(
                            near, far,
                            np.asarray(lat_l_ref.data),
                            np.asarray(lat_r_ref.data))
                        if result is not None:
                            insert_idx, clicked_side = result
                            is_seam = 'Shift' in event.modifiers
                            self._do_lattice_insert(
                                insert_idx, clicked_side, pos, is_seam,
                                side_idx, timepoint,
                                lat_l_ref, lat_r_ref,
                                lat_lines_ref, lat_mid_ref,
                                lat_lc_ref, lat_rc_ref)
                            return
                        else:
                            self._on_lattice_click(img_ref, event, None,
                                                   side_idx, timepoint,
                                                   lat_l_ref, lat_r_ref,
                                                   lat_lines_ref, lat_mid_ref,
                                                   lat_lc_ref, lat_rc_ref)
                            return
                    else:
                        self._on_lattice_click(img_ref, event, None,
                                               side_idx, timepoint,
                                               lat_l_ref, lat_r_ref,
                                               lat_lines_ref, lat_mid_ref,
                                               lat_lc_ref, lat_rc_ref)
                        return

                    # --- Select / drag existing point (MIPAV modifyLattice) ---
                    side_char, pt_idx, dist = drag_target
                    drag_layer = lat_l_ref if side_char == 'L' else lat_r_ref
                    pair_names = self.lattice_pair_names.get(timepoint, [])
                    name = pair_names[pt_idx]['name'] + side_char if pt_idx < len(pair_names) else f'?{pt_idx}{side_char}'
                    self.lattice_last_placed = {
                        'side_idx': side_idx, 'timepoint': timepoint,
                        'layer': drag_layer, 'char': side_char, 'index': pt_idx}
                    print(f"[t={timepoint}] Selected {name} (drag to move, arrows to nudge)")

                    yield  # wait for mouse move events

                    dragged = False
                    while event.type == 'mouse_move':
                        if 'Control' not in event.modifiers:
                            break
                        dragged = True
                        near2, far2 = img_ref.get_ray_intersections(
                            event.position, event.view_direction, event.dims_displayed)
                        if near2 is not None and far2 is not None:
                            new_pos = self._find_peak_multi_channel(near2, far2, side_idx)
                            pts = drag_layer.data.copy()
                            pts[pt_idx] = new_pos
                            drag_layer.data = pts
                            self._update_lattice_visuals(
                                lat_l_ref, lat_r_ref, lat_lines_ref,
                                lat_mid_ref, lat_lc_ref, lat_rc_ref)
                        yield

                    # Update cache on release
                    entry = self.lattice_annotations.setdefault(timepoint, {})
                    entry['left'] = lat_l_ref.data.copy() if len(lat_l_ref.data) > 0 else np.empty((0, 3))
                    entry['right'] = lat_r_ref.data.copy() if len(lat_r_ref.data) > 0 else np.empty((0, 3))
                    if dragged:
                        print(f"[t={timepoint}] Released {name}")
                        self._refresh_tables()

                return handler

            cb = make_handler(side, ti, img_ref, pts, lat_l, lat_r, lat_lines, lat_mid,
                              lat_left_curve, lat_right_curve)
            # Register on ALL layers so Cmd+Click works regardless of sidebar selection
            all_interactive = list(channel_img_layers) + [
                pts, lat_l, lat_r, lat_lines, lat_mid,
                lat_left_curve, lat_right_curve, wireframe]
            for lyr in all_interactive:
                lyr.mouse_drag_callbacks.append(cb)

            # Store image layers: list for multi-channel, single for compat
            self.grid_image_layers[side] = channel_img_layers if self.multi_channel else channel_img_layers[0]
            self.grid_points_layers[side] = pts
            self.lattice_left_layers[side]  = lat_l
            self.lattice_right_layers[side] = lat_r
            self.lattice_line_layers[side]  = lat_lines
            self.lattice_mid_layers[side]   = lat_mid
            self.lattice_left_curve_layers[side]  = lat_left_curve
            self.lattice_right_curve_layers[side] = lat_right_curve
            self.wireframe_layers[side] = wireframe
            self.surface_layers[side] = surface

        self._nav_updating = True
        self._left_spin.setValue(t_left)
        self._right_spin.setValue(t_right)
        self._nav_updating = False

        ch_info = f"  channels: {[c[1] for c in self.channels]}" if self.multi_channel else ""
        self.dual_window.setWindowTitle(
            f"napari_worm  —  t={t_left} (left)  |  t={t_right} (right)  "
            f"[of {len(self.tiff_files)-1}]{ch_info}")
        self.dual_window.update_dock_titles(t_left, t_right)
        print(f"Showing t={t_left} (left) and t={t_right} (right) of {len(self.tiff_files)-1}")

        # Update histogram LUT widgets — bind to first channel's image layer,
        # or connect to active layer selection for multi-channel switching
        self._update_histograms()
        self._refresh_tables()

        # Auto-select the first image layer on both sides so layer controls
        # default to volume (otherwise Wireframe is selected and controls
        # for opacity/blending don't visually affect the volume)
        for side_viewer, side_idx in [(self.viewer_left, 0), (self.viewer_right, 1)]:
            img_layers = self.grid_image_layers[side_idx]
            if img_layers:
                first_img = img_layers[0] if isinstance(img_layers, list) else img_layers
                side_viewer.layers.selection.active = first_img

    def _apply_threshold(self, tmin: int):
        """Set lower contrast limit on all image layers (hides dim voxels).

        Uses contrast_limits rather than zeroing data, so it's non-destructive
        and works with the histogram transfer function.
        """
        for side in range(2):
            img_layers = self.grid_image_layers[side]
            if img_layers is None:
                continue
            if not isinstance(img_layers, list):
                img_layers = [img_layers]
            for img in img_layers:
                cmax = img.contrast_limits[1]
                img.contrast_limits = (tmin, cmax)
        # Sync both sliders
        for sl in self.dual_window._thresh_sliders:
            if sl is not None:
                sl.blockSignals(True)
                sl.setValue(tmin)
                sl.blockSignals(False)
        pass  # no print/notification — slider fires continuously

    def _save_grid_annotations_to_cache(self):
        for side, pts in enumerate(self.grid_points_layers):
            if pts is not None and side < len(self.grid_timepoints):
                ti = self.grid_timepoints[side]
                if len(pts.data) > 0:
                    self.grid_annotations[ti] = pts.data.copy()

    def _save_lattice_to_cache(self):
        for side, (lat_l, lat_r) in enumerate(
                zip(self.lattice_left_layers, self.lattice_right_layers)):
            if lat_l is None or side >= len(self.grid_timepoints):
                continue
            ti = self.grid_timepoints[side]
            entry = self.lattice_annotations.setdefault(ti, {})
            if len(lat_l.data) > 0:
                entry['left']  = lat_l.data.copy()
            if lat_r is not None and len(lat_r.data) > 0:
                entry['right'] = lat_r.data.copy()

    def _refresh_tables(self):
        """Refresh both annotation and lattice tables for both sides."""
        for side in range(2):
            self._refresh_annotation_table(side)
            self._refresh_lattice_table(side)
            self._refresh_point_labels(side)

    def _refresh_point_labels(self, side: int):
        """Update text labels on annotation and lattice point layers.

        Sets ``layer.properties`` + ``layer.text`` so that napari renders a
        label next to each point.  Must be called only *after* the layer's
        ``.data`` is already up-to-date (i.e. after table refresh).
        """
        _text_style = dict(color='white', anchor='upper_left', size=8,
                           translation=[0, 0, 15])

        # --- Annotation labels: A0, A1, ... ---
        pts = self.grid_points_layers[side]
        if pts is not None:
            n = len(pts.data)
            if n > 0:
                pts.properties = {'label': [f'A{i}' for i in range(n)]}
                pts.text = {**_text_style, 'string': 'label', 'color': 'white'}
            else:
                pts.properties = {'label': []}
                pts.text = None

        # --- Lattice labels: pair_nameL / pair_nameR ---
        ti = self.grid_timepoints[side] if side < len(self.grid_timepoints) else None
        pair_names = self.lattice_pair_names.get(ti, []) if ti is not None else []

        for layer, suffix, color in [
            (self.lattice_left_layers[side],  'L', 'cyan'),
            (self.lattice_right_layers[side], 'R', 'magenta'),
        ]:
            if layer is None:
                continue
            n = len(layer.data)
            if n > 0:
                labels = [
                    f"{pair_names[i]['name']}{suffix}" if i < len(pair_names)
                    else f"a{i}{suffix}"
                    for i in range(n)
                ]
                layer.properties = {'label': labels}
                layer.text = {**_text_style, 'string': 'label', 'color': color}
            else:
                layer.properties = {'label': []}
                layer.text = None

    def _refresh_annotation_table(self, side: int):
        """Populate annotation table from current points layer data."""
        table = self.dual_window.annotation_tables[side]
        if table is None:
            return
        # Block signals while rebuilding to avoid spurious cellChanged
        table.blockSignals(True)
        pts_layer = self.grid_points_layers[side]
        if pts_layer is None or len(pts_layer.data) == 0:
            table.setRowCount(0)
            table.blockSignals(False)
            return
        ti = self.grid_timepoints[side]
        data = np.asarray(pts_layer.data)
        segments = self.grid_annotation_segments.get(ti, [])
        table.setRowCount(len(data))
        # Get raw volume for intensity lookup (first image layer)
        img_layers = self.grid_image_layers[side]
        if isinstance(img_layers, list):
            raw_vol = np.asarray(img_layers[0].data)
        elif img_layers is not None:
            raw_vol = np.asarray(img_layers.data)
        else:
            raw_vol = None
        vol_shape = np.array(raw_vol.shape) if raw_vol is not None else None
        for i, (z, y, x) in enumerate(data):
            # Name — read-only
            name_item = QTableWidgetItem(f"A{i}")
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            table.setItem(i, 0, name_item)
            # X, Y, Z — editable
            table.setItem(i, 1, QTableWidgetItem(f"{x:.1f}"))
            table.setItem(i, 2, QTableWidgetItem(f"{y:.1f}"))
            table.setItem(i, 3, QTableWidgetItem(f"{z:.1f}"))
            # Intensity — read-only, raw value from primary channel
            if raw_vol is not None:
                idx = np.clip(np.round([z, y, x]).astype(int), 0, vol_shape - 1)
                val = int(raw_vol[tuple(idx)])
            else:
                val = 0
            val_item = QTableWidgetItem(str(val))
            val_item.setFlags(val_item.flags() & ~Qt.ItemIsEditable)
            table.setItem(i, 4, val_item)
            # Seg — editable (now column 5)
            seg_val = segments[i] if i < len(segments) else -1
            table.setItem(i, 5, QTableWidgetItem("" if seg_val == -1 else str(seg_val)))
        table.blockSignals(False)

    def _refresh_lattice_table(self, side: int):
        """Populate lattice table from current lattice layer data."""
        table = self.dual_window.lattice_tables[side]
        if table is None:
            return
        table.blockSignals(True)
        ti = self.grid_timepoints[side] if side < len(self.grid_timepoints) else None
        lat_l = self.lattice_left_layers[side]
        lat_r = self.lattice_right_layers[side]
        if lat_l is None or (len(lat_l.data) == 0 and (lat_r is None or len(lat_r.data) == 0)):
            table.setRowCount(0)
            table.blockSignals(False)
            return
        l_data = np.asarray(lat_l.data) if len(lat_l.data) > 0 else np.empty((0, 3))
        r_data = np.asarray(lat_r.data) if lat_r is not None and len(lat_r.data) > 0 else np.empty((0, 3))
        pair_names = self.lattice_pair_names.get(ti, []) if ti is not None else []
        n_rows = max(len(l_data), len(r_data))
        table.setRowCount(n_rows)
        for i in range(n_rows):
            # Pair name — read-only
            name = pair_names[i]['name'] if i < len(pair_names) else f"a{i}"
            name_item = QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            table.setItem(i, 0, name_item)
            # L coords — editable
            if i < len(l_data):
                z, y, x = l_data[i]
                table.setItem(i, 1, QTableWidgetItem(f"{x:.1f}"))
                table.setItem(i, 2, QTableWidgetItem(f"{y:.1f}"))
                table.setItem(i, 3, QTableWidgetItem(f"{z:.1f}"))
            # R coords — editable
            if i < len(r_data):
                z, y, x = r_data[i]
                table.setItem(i, 4, QTableWidgetItem(f"{x:.1f}"))
                table.setItem(i, 5, QTableWidgetItem(f"{y:.1f}"))
                table.setItem(i, 6, QTableWidgetItem(f"{z:.1f}"))
        table.blockSignals(False)

    def _on_annotation_table_edited(self, side: int, row: int, col: int):
        """Handle user editing X/Y/Z or Seg columns in the annotation table."""
        if col in (0, 4):  # Name and Intensity are read-only
            return
        table = self.dual_window.annotation_tables[side]
        item = table.item(row, col)
        if item is None:
            return
        text = item.text().strip()

        if col == 5:  # Seg column
            ti = self.grid_timepoints[side]
            segments = self.grid_annotation_segments.setdefault(ti, [])
            while len(segments) <= row:
                segments.append(-1)
            try:
                segments[row] = int(text) if text else -1
            except ValueError:
                segments[row] = -1
                table.blockSignals(True)
                item.setText("")
                table.blockSignals(False)
            return

        # X/Y/Z coordinate edit (cols 1-3)
        pts_layer = self.grid_points_layers[side]
        if pts_layer is None or row >= len(pts_layer.data):
            return
        try:
            val = float(text)
        except ValueError:
            return
        pts = pts_layer.data.copy()
        old_val = pts[row].copy()
        # Table cols: 1=X, 2=Y, 3=Z → data indices: X=col2, Y=col1, Z=col0
        if col == 1:
            pts[row][2] = val  # X
        elif col == 2:
            pts[row][1] = val  # Y
        elif col == 3:
            pts[row][0] = val  # Z
        pts_layer.data = pts
        ti = self.grid_timepoints[side]
        self.undo_stack.append(('TABLE_ANN', ti, side, row, old_val))

    def _on_lattice_table_edited(self, side: int, row: int, col: int):
        """Handle user editing coordinate columns in the lattice table."""
        if col == 0:  # Pair name is read-only
            return
        table = self.dual_window.lattice_tables[side]
        item = table.item(row, col)
        if item is None:
            return
        try:
            val = float(item.text().strip())
        except ValueError:
            return
        # Cols 1-3 = left (L_X, L_Y, L_Z), cols 4-6 = right (R_X, R_Y, R_Z)
        if col <= 3:
            layer = self.lattice_left_layers[side]
            coord_col = col  # 1=X, 2=Y, 3=Z
        else:
            layer = self.lattice_right_layers[side]
            coord_col = col - 3  # 4→1=X, 5→2=Y, 6→3=Z
        if layer is None or row >= len(layer.data):
            return
        pts = layer.data.copy()
        old_val = pts[row].copy()
        if coord_col == 1:
            pts[row][2] = val  # X
        elif coord_col == 2:
            pts[row][1] = val  # Y
        elif coord_col == 3:
            pts[row][0] = val  # Z
        layer.data = pts
        ti = self.grid_timepoints[side]
        lr_char = 'L' if col <= 3 else 'R'
        self.lattice_undo_stack.append(('TABLE_LAT', ti, (layer, row, old_val)))
        # Update lattice visuals and cache
        ti = self.grid_timepoints[side]
        for ll, lr, llines, lmid, llc, lrc in zip(
                self.lattice_left_layers, self.lattice_right_layers,
                self.lattice_line_layers, self.lattice_mid_layers,
                self.lattice_left_curve_layers, self.lattice_right_curve_layers):
            if ll is self.lattice_left_layers[side]:
                self._update_lattice_visuals(ll, lr, llines, lmid, llc, lrc)
                break
        entry = self.lattice_annotations.setdefault(ti, {})
        lat_l = self.lattice_left_layers[side]
        lat_r = self.lattice_right_layers[side]
        if lat_l is not None and len(lat_l.data) > 0:
            entry['left'] = lat_l.data.copy()
        if lat_r is not None and len(lat_r.data) > 0:
            entry['right'] = lat_r.data.copy()

    def _delete_annotation_row(self, side: int):
        """Delete the selected annotation row."""
        table = self.dual_window.annotation_tables[side]
        row = table.currentRow()
        pts_layer = self.grid_points_layers[side]
        if row < 0 or pts_layer is None or row >= len(pts_layer.data):
            return
        ti = self.grid_timepoints[side]
        old_data = pts_layer.data.copy()
        pts_layer.data = np.delete(old_data, row, axis=0)
        # Remove segment
        segs = self.grid_annotation_segments.get(ti, [])
        if row < len(segs):
            segs.pop(row)
        # Push undo
        self.undo_stack.append(('DELETE_ANN', ti, side, row, old_data[row].copy()))
        print(f"[t={ti}] Deleted annotation A{row}")
        self._refresh_annotation_table(side)

    def _delete_lattice_row(self, side: int):
        """Delete the selected lattice pair (both L and R)."""
        table = self.dual_window.lattice_tables[side]
        row = table.currentRow()
        lat_l = self.lattice_left_layers[side]
        lat_r = self.lattice_right_layers[side]
        if row < 0 or lat_l is None:
            return
        ti = self.grid_timepoints[side]
        old_l = lat_l.data.copy() if len(lat_l.data) > 0 else np.empty((0, 3))
        old_r = lat_r.data.copy() if lat_r is not None and len(lat_r.data) > 0 else np.empty((0, 3))
        if row < len(old_l):
            lat_l.data = np.delete(old_l, row, axis=0) if len(old_l) > 1 else np.empty((0, 3))
        if row < len(old_r):
            lat_r.data = np.delete(old_r, row, axis=0) if len(old_r) > 1 else np.empty((0, 3))
        # Remove pair name and renumber
        pair_names = self.lattice_pair_names.get(ti, [])
        removed_name = None
        if row < len(pair_names):
            removed_name = pair_names.pop(row)
            _renumber_lattice_pairs(pair_names)
            self.lattice_seam_counter[ti] = [
                p['name'] for p in pair_names if p['type'] == 'seam']
        # Update visuals and cache
        for ll, lr, llines, lmid, llc, lrc in zip(
                self.lattice_left_layers, self.lattice_right_layers,
                self.lattice_line_layers, self.lattice_mid_layers,
                self.lattice_left_curve_layers, self.lattice_right_curve_layers):
            if ll is lat_l:
                self._update_lattice_visuals(ll, lr, llines, lmid, llc, lrc)
                break
        entry = self.lattice_annotations.setdefault(ti, {})
        entry['left'] = lat_l.data.copy() if len(lat_l.data) > 0 else np.empty((0, 3))
        entry['right'] = lat_r.data.copy() if lat_r is not None and len(lat_r.data) > 0 else np.empty((0, 3))
        self.lattice_last_placed = None
        # Push undo
        old_l_pt = old_l[row].copy() if row < len(old_l) else None
        old_r_pt = old_r[row].copy() if row < len(old_r) else None
        self.lattice_undo_stack.append(('DELETE_LAT', ti, (row, old_l_pt, old_r_pt, removed_name)))
        name = removed_name['name'] if removed_name else f"a{row}"
        print(f"[t={ti}] Deleted lattice pair {name}")
        self._refresh_tables()

    def _highlight_point(self, layer, index, default_color, default_size):
        """Make one point stand out with a white edge ring and larger size."""
        n = len(layer.data)
        if n == 0 or index < 0 or index >= n:
            return
        sizes = np.full(n, float(default_size))
        sizes[index] = default_size * 2.0
        layer.size = sizes
        edge_widths = np.zeros(n)
        edge_widths[index] = 2.0
        layer.edge_width = edge_widths
        edge_colors = np.array(['transparent'] * n)
        edge_colors[index] = 'white'
        layer.edge_color = edge_colors

    def _clear_highlight(self, layer, default_color, default_size):
        """Reset all points to default appearance."""
        if layer is None or len(layer.data) == 0:
            return
        layer.size = default_size
        layer.edge_width = 0
        layer.edge_color = 'transparent'

    def _on_annotation_row_selected(self, side: int, row: int):
        """Highlight the selected annotation point in the 3D viewer."""
        pts_layer = self.grid_points_layers[side]
        if pts_layer is None or row < 0 or row >= len(pts_layer.data):
            return
        self._highlight_point(pts_layer, row, 'yellow', 5)

    def _on_lattice_row_selected(self, side: int, row: int, col: int):
        """Highlight both L and R lattice points for the selected pair row."""
        lat_l = self.lattice_left_layers[side]
        lat_r = self.lattice_right_layers[side]
        if row < 0:
            return
        # Clear previous highlights
        self._clear_highlight(lat_l, 'cyan', 7)
        self._clear_highlight(lat_r, 'magenta', 7)
        # Highlight both L and R for the pair
        if lat_l is not None and row < len(lat_l.data):
            self._highlight_point(lat_l, row, 'cyan', 7)
        if lat_r is not None and row < len(lat_r.data):
            self._highlight_point(lat_r, row, 'magenta', 7)

    def _update_histograms(self):
        """Bind histogram widgets to image layers after loading a timepoint pair."""
        if self.multi_channel:
            # Multi-channel: bind to first channel initially, then switch on
            # active layer selection (clicking a channel in the layer list)
            for side, (viewer_ref, hist_widget) in enumerate([
                (self.viewer_left, self.dual_window.histogram_left),
                (self.viewer_right, self.dual_window.histogram_right),
            ]):
                img_layers = self.grid_image_layers[side]
                if isinstance(img_layers, list) and img_layers:
                    hist_widget.set_layer(img_layers[0])
                    # Connect active layer selection to histogram switching
                    self._connect_layer_selection(viewer_ref, hist_widget, img_layers)
        else:
            # Single channel: same as before
            img_l = self.grid_image_layers[0]
            img_r = self.grid_image_layers[1]
            if img_l is not None:
                self.dual_window.histogram_left.set_layer(img_l)
            if img_r is not None:
                self.dual_window.histogram_right.set_layer(img_r)

    def _connect_layer_selection(self, viewer, hist_widget, img_layers):
        """When user selects a different image layer, update histogram."""
        img_layer_set = set(id(lyr) for lyr in img_layers)

        def on_active_changed(event):
            active = viewer.layers.selection.active
            if active is not None and id(active) in img_layer_set:
                hist_widget.set_layer(active)

        # Disconnect any previous handler to avoid stacking
        if not hasattr(self, '_ch_hist_handlers'):
            self._ch_hist_handlers = {}
        prev = self._ch_hist_handlers.get(id(viewer))
        if prev is not None:
            try:
                viewer.layers.selection.events.active.disconnect(prev)
            except (TypeError, RuntimeError):
                pass
        self._ch_hist_handlers[id(viewer)] = on_active_changed
        viewer.layers.selection.events.active.connect(on_active_changed)

    def _toggle_lattice_mode(self):
        self.lattice_mode = not self.lattice_mode
        # pts layers are always pan_zoom — our handler does all point adding.
        # No mode change needed here; the toggle is purely a routing flag.
        if self.lattice_mode:
            self.lattice_next_side = 'L'
            self.lattice_last_placed = None
            print("=" * 50)
            print("  MODE: LATTICE  (nose → tail, left → right)")
            print("  Cmd+Click       → add point (alternates L/R)")
            print("  Cmd+Click curve → insert pair (click on magenta/green curve)")
            print("  Cmd+Click point → select point, then drag or arrow-nudge")
            print("  Cmd+Shift+Click → seam cell (also L/R pair)")
            print("  Arrow keys      → nudge selected point")
            print("  Cmd+Z           → undo (point, pair, or insertion)")
            print("  D               → done with lattice")
            print("  L               → back to annotation mode")
            print("=" * 50)
            show_info("LATTICE MODE")
        else:
            print("=" * 50)
            print("  MODE: ANNOTATION  (Cmd+Click = annotate)")
            print("=" * 50)
            show_info("ANNOTATION MODE")

    def _on_lattice_click(self, img_layer, event, _volume_data_unused,
                          side_idx, timepoint,
                          lat_left_layer, lat_right_layer,
                          lat_lines_layer, lat_mid_layer,
                          lat_left_curve_layer, lat_right_curve_layer):
        """Append a new lattice point (L or R). Insertion is handled separately."""
        try:
            near, far = img_layer.get_ray_intersections(
                event.position, event.view_direction, event.dims_displayed)
            if near is None or far is None:
                print("[lattice] ray missed volume — click closer to the worm")
                return
            peak = self._find_peak_multi_channel(near, far, side_idx)
            pos = find_nucleus_centroid(self._get_blended_volume(side_idx), peak)
            canvas_label = "left" if side_idx == 0 else "right"

            is_seam = 'Shift' in event.modifiers
            pair_type = 'seam' if is_seam else 'lattice'
            pair_names = self.lattice_pair_names.setdefault(timepoint, [])

            if self.lattice_next_side == 'L':
                lat_left_layer.add(pos)
                new_info = {'name': '', 'type': pair_type}
                pair_names.append(new_info)
                _renumber_lattice_pairs(pair_names)
                self.lattice_seam_counter[timepoint] = [
                    p['name'] for p in pair_names if p['type'] == 'seam']

                name = pair_names[-1]['name'] + 'L'
                print(f"[t={timepoint} {canvas_label}] {name}  "
                      f"z={pos[0]:.1f} y={pos[1]:.1f} x={pos[2]:.1f}  (next: R)")
                if is_seam:
                    show_info(f"Seam cell: {pair_names[-1]['name']}L")
                self.lattice_undo_stack.append(('L', timepoint, lat_left_layer))
                self.lattice_last_placed = {
                    'side_idx': side_idx, 'timepoint': timepoint,
                    'layer': lat_left_layer, 'char': 'L',
                    'index': len(lat_left_layer.data) - 1}
                self.lattice_next_side = 'R'
            else:
                lat_right_layer.add(pos)
                pair_idx = len(lat_right_layer.data) - 1
                if pair_idx < len(pair_names):
                    name = pair_names[pair_idx]['name'] + 'R'
                else:
                    name = f'?{pair_idx}R'
                print(f"[t={timepoint} {canvas_label}] {name}  "
                      f"z={pos[0]:.1f} y={pos[1]:.1f} x={pos[2]:.1f}  (next: L)")
                if pair_idx < len(pair_names) and pair_names[pair_idx]['type'] == 'seam':
                    show_info(f"Seam cell: {name}")
                self.lattice_undo_stack.append(('R', timepoint, lat_right_layer))
                self.lattice_last_placed = {
                    'side_idx': side_idx, 'timepoint': timepoint,
                    'layer': lat_right_layer, 'char': 'R',
                    'index': len(lat_right_layer.data) - 1}
                self.lattice_next_side = 'L'

            self._update_lattice_visuals(lat_left_layer, lat_right_layer,
                                         lat_lines_layer, lat_mid_layer,
                                         lat_left_curve_layer, lat_right_curve_layer)
            entry = self.lattice_annotations.setdefault(timepoint, {})
            entry['left']  = lat_left_layer.data.copy()  if len(lat_left_layer.data)  > 0 else np.empty((0, 3))
            entry['right'] = lat_right_layer.data.copy() if len(lat_right_layer.data) > 0 else np.empty((0, 3))
            self._refresh_lattice_table(side_idx)
            self._refresh_point_labels(side_idx)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"[lattice] click error: {exc}")

    def _do_lattice_insert(self, seg_idx, clicked_side, pos, is_seam,
                           side_idx, timepoint,
                           lat_left_layer, lat_right_layer,
                           lat_lines_layer, lat_mid_layer,
                           lat_left_curve_layer, lat_right_curve_layer):
        """Insert a new L+R pair between existing pairs (MIPAV addInsertionPoint).

        The clicked side gets the actual click position; the other side is
        interpolated as the midpoint of its neighbors.
        """
        try:
            idx = seg_idx + 1  # insert AFTER the matched segment
            pair_type = 'seam' if is_seam else 'lattice'
            canvas_label = "left" if side_idx == 0 else "right"

            l_pts = np.asarray(lat_left_layer.data)
            r_pts = np.asarray(lat_right_layer.data)

            if clicked_side == 'L':
                new_left = pos
                new_right = (r_pts[seg_idx] + r_pts[seg_idx + 1]) / 2.0
            else:
                new_right = pos
                new_left = (l_pts[seg_idx] + l_pts[seg_idx + 1]) / 2.0

            left_data = np.insert(l_pts, idx, new_left, axis=0)
            right_data = np.insert(r_pts, idx, new_right, axis=0)
            lat_left_layer.data = left_data
            lat_right_layer.data = right_data

            pair_names = self.lattice_pair_names.setdefault(timepoint, [])
            pair_names.insert(idx, {'name': '', 'type': pair_type})
            _renumber_lattice_pairs(pair_names)
            self.lattice_seam_counter[timepoint] = [
                p['name'] for p in pair_names if p['type'] == 'seam']

            inserted_name = pair_names[idx]['name']
            label = "Seam cell" if is_seam else "Inserted"
            print(f"[t={timepoint} {canvas_label}] {label} {inserted_name}L + {inserted_name}R "
                  f"at index {idx} (clicked {clicked_side} curve)")
            names_str = ', '.join(p['name'] for p in pair_names)
            print(f"  Renumbered: {names_str}")
            if is_seam:
                show_info(f"Seam cell inserted: {inserted_name}")

            self.lattice_undo_stack.append(('INSERT', timepoint,
                                            (lat_left_layer, lat_right_layer, idx)))
            clicked_layer = lat_left_layer if clicked_side == 'L' else lat_right_layer
            self.lattice_last_placed = {
                'side_idx': side_idx, 'timepoint': timepoint,
                'layer': clicked_layer, 'char': clicked_side, 'index': idx}

            self._update_lattice_visuals(lat_left_layer, lat_right_layer,
                                         lat_lines_layer, lat_mid_layer,
                                         lat_left_curve_layer, lat_right_curve_layer)
            entry = self.lattice_annotations.setdefault(timepoint, {})
            entry['left']  = lat_left_layer.data.copy()
            entry['right'] = lat_right_layer.data.copy()
            self._refresh_lattice_table(side_idx)
            self._refresh_point_labels(side_idx)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"[lattice] insert error: {exc}")

    def _update_lattice_visuals(self, lat_l, lat_r, lat_lines, lat_mid,
                               lat_left_curve, lat_right_curve):
        n = min(len(lat_l.data), len(lat_r.data))

        # --- L/R connecting lines (yellow cross-rungs) ---
        if len(lat_lines.data) > 0:
            lat_lines.selected_data = set(range(len(lat_lines.data)))
            lat_lines.remove_selected()
        for i in range(n):
            lat_lines.add(
                np.array([lat_l.data[i], lat_r.data[i]]),
                shape_type='line',
            )

        # --- Helper: clear a Shapes layer ---
        def _clear_shapes(layer):
            if len(layer.data) > 0:
                layer.selected_data = set(range(len(layer.data)))
                layer.remove_selected()

        # --- Center midline spline (red, matching MIPAV) ---
        _clear_shapes(lat_mid)
        if n >= 2:
            mids = np.array([(lat_l.data[i] + lat_r.data[i]) / 2 for i in range(n)])
            lat_mid.add(_smooth_midline_spline(mids), shape_type='path')

        # --- Left curve spline (magenta, matching MIPAV) ---
        _clear_shapes(lat_left_curve)
        if len(lat_l.data) >= 2:
            lat_left_curve.add(_smooth_midline_spline(np.asarray(lat_l.data)),
                               shape_type='path')

        # --- Right curve spline (green, matching MIPAV) ---
        _clear_shapes(lat_right_curve)
        if len(lat_r.data) >= 2:
            lat_right_curve.add(_smooth_midline_spline(np.asarray(lat_r.data)),
                                shape_type='path')

        # --- Wireframe mesh (32 longitudinal splines + cross-section rings) ---
        # Only compute when wireframe is visible (W key) — expensive operation
        if self.wireframe_visible:
            wf_layer = None
            for side, ll in enumerate(self.lattice_left_layers):
                if ll is lat_l:
                    wf_layer = self.wireframe_layers[side]
                    break
            if wf_layer is not None:
                _clear_shapes(wf_layer)
                if n >= 3:
                    left_data = np.asarray(lat_l.data)
                    right_data = np.asarray(lat_r.data)
                    paths = generate_wireframe_mesh(left_data, right_data)
                    for path in paths:
                        if len(path) >= 2:
                            wf_layer.add(path, shape_type='path')

        # --- Surface mesh (triangle mesh, Shift+W) ---
        if self.surface_visible:
            surf_layer = None
            for side, ll in enumerate(self.lattice_left_layers):
                if ll is lat_l:
                    surf_layer = self.surface_layers[side]
                    break
            if surf_layer is not None and n >= 3:
                left_data = np.asarray(lat_l.data)
                right_data = np.asarray(lat_r.data)
                result = generate_surface_mesh(left_data, right_data)
                if result is not None:
                    surf_layer.data = result

    @staticmethod
    def _save_csv_retry(df, path, retries=3):
        """Save DataFrame to CSV with retry for network drives (BlockingIOError)."""
        import time
        for attempt in range(retries):
            try:
                df.to_csv(path, index=False)
                return True
            except BlockingIOError:
                if attempt < retries - 1:
                    time.sleep(0.5)
                else:
                    print(f"  WARNING: could not write {path} (network drive busy)")
                    return False

    def _save_lattice(self):
        self._save_lattice_to_cache()
        saved = 0
        saved_paths = []
        for ti, entry in sorted(self.lattice_annotations.items()):
            left  = entry.get('left',  np.empty((0, 3)))
            right = entry.get('right', np.empty((0, 3)))
            if len(left) == 0 and len(right) == 0:
                continue
            pair_names = self.lattice_pair_names.get(ti, [])
            # Interleave L/R pairs with proper names (matches MIPAV order)
            n_pairs = min(len(left), len(right))
            rows = []
            for i in range(n_pairs):
                # Use pair metadata name if available, else fall back to a{i}
                if i < len(pair_names):
                    prefix = pair_names[i]['name']
                else:
                    prefix = _lattice_pair_name(i)
                for pt, side in ((left[i], 'L'), (right[i], 'R')):
                    rows.append({'name': prefix + side,
                                 'x_voxels': pt[2], 'y_voxels': pt[1], 'z_voxels': pt[0],
                                 'R': 255, 'G': 255, 'B': 255})
            # Append any unpaired trailing L points
            for i in range(n_pairs, len(left)):
                if i < len(pair_names):
                    prefix = pair_names[i]['name']
                else:
                    prefix = _lattice_pair_name(i)
                pt = left[i]
                rows.append({'name': prefix + 'L',
                             'x_voxels': pt[2], 'y_voxels': pt[1], 'z_voxels': pt[0],
                             'R': 255, 'G': 255, 'B': 255})
            stem = self.tiff_files[ti].stem
            lat_dir = self.volume_path / stem / f"{stem}_results" / "lattice_final"
            lat_dir.mkdir(parents=True, exist_ok=True)
            save_path = lat_dir / "lattice_test.csv"
            self._save_csv_retry(pd.DataFrame(rows), save_path)
            print(f"  Lattice t={ti}: {len(rows)} points → {save_path}")
            saved_paths.append(str(save_path))
            saved += 1
        if saved:
            msg = f"Lattice saved for {saved} timepoint(s):\n" + "\n".join(saved_paths)
            print(msg)
            show_info(msg)

    def _on_spinbox_changed(self):
        if self._nav_updating:
            return
        new_left, new_right = self._left_spin.value(), self._right_spin.value()
        # Skip if timepoints haven't actually changed
        if (new_left, new_right) == self.grid_timepoints:
            return
        # Prevent loading same timepoint on both sides
        if new_left == new_right:
            QMessageBox.warning(
                self.dual_window._host,
                "Duplicate timepoint",
                f"Left and right panels cannot both show timepoint {new_left}.\n"
                "Please select different timepoints for each side.",
            )
            # Revert spinboxes to current values
            self._nav_updating = True
            self._left_spin.setValue(self.grid_timepoints[0])
            self._right_spin.setValue(self.grid_timepoints[1])
            self._nav_updating = False
            return
        self._load_dual_pair(new_left, new_right)

    def _grid_next(self, viewer):
        t_left, t_right = self.grid_timepoints
        if t_right >= len(self.tiff_files) - 1:
            print("Already at last timepoint")
            return
        self._load_dual_pair(t_left + 1, t_right + 1)

    def _grid_prev(self, viewer):
        t_left, t_right = self.grid_timepoints
        if t_left <= 0:
            print("Already at first timepoint")
            return
        self._load_dual_pair(t_left - 1, t_right - 1)

    def _on_click_dual(self, layer, event, side_idx, timepoint, points_layer):
        near, far = layer.get_ray_intersections(
            event.position, event.view_direction, event.dims_displayed)
        if near is None or far is None:
            return
        peak = self._find_peak_multi_channel(near, far, side_idx)
        pos = find_nucleus_centroid(self._get_blended_volume(side_idx), peak)
        points_layer.add(pos)
        self.undo_stack.append((timepoint, points_layer))
        self.grid_annotation_segments.setdefault(timepoint, []).append(-1)
        label = "left" if side_idx == 0 else "right"
        print(f"[t={timepoint} {label}] Added at z={pos[0]:.1f} y={pos[1]:.1f} x={pos[2]:.1f}")
        self._refresh_annotation_table(side_idx)
        self._refresh_point_labels(side_idx)

    def _undo_last_point(self, viewer):
        if self.lattice_mode:
            self._undo_last_lattice_point()
            return
        if not self.undo_stack:
            print("Nothing to undo")
            return
        entry = self.undo_stack.pop()

        # Handle table coordinate edit undo
        if isinstance(entry, tuple) and len(entry) >= 4 and entry[0] == 'TABLE_ANN':
            _, ti, side, row, old_val = entry
            pts_layer = self.grid_points_layers[side]
            if pts_layer is not None and row < len(pts_layer.data):
                pts = pts_layer.data.copy()
                pts[row] = old_val
                pts_layer.data = pts
                print(f"[t={ti}] Undid table edit on A{row}")
                self._refresh_tables()
            return

        # Handle delete annotation undo — re-insert the deleted point
        if isinstance(entry, tuple) and len(entry) >= 4 and entry[0] == 'DELETE_ANN':
            _, ti, side, row, old_val = entry
            pts_layer = self.grid_points_layers[side]
            if pts_layer is not None:
                pts = pts_layer.data.copy() if len(pts_layer.data) > 0 else np.empty((0, 3))
                pts_layer.data = np.insert(pts, row, old_val, axis=0)
                segs = self.grid_annotation_segments.setdefault(ti, [])
                segs.insert(row, -1)
                print(f"[t={ti}] Undid delete of A{row}")
                self._refresh_tables()
            return

        timepoint, pts = entry
        active = [p for p in self.grid_points_layers if p is not None]
        if any(p is pts for p in active) and len(pts.data) > 0:
            pts.data = pts.data[:-1]
            segs = self.grid_annotation_segments.get(timepoint, [])
            if segs:
                segs.pop()
            print(f"[t={timepoint}] Undid last annotation")
        elif timepoint in self.grid_annotations and len(self.grid_annotations[timepoint]) > 0:
            self.grid_annotations[timepoint] = self.grid_annotations[timepoint][:-1]
            segs = self.grid_annotation_segments.get(timepoint, [])
            if segs:
                segs.pop()
            for side, ti in enumerate(self.grid_timepoints):
                if ti == timepoint and side < len(self.grid_points_layers):
                    dp = self.grid_points_layers[side]
                    if dp is not None:
                        dp.data = (self.grid_annotations[timepoint].copy()
                                   if len(self.grid_annotations[timepoint]) > 0
                                   else np.empty((0, 3)))
            print(f"[t={timepoint}] Undid last annotation (cached)")
        else:
            print(f"[t={timepoint}] Nothing to undo")
            return
        self._refresh_tables()

    def _undo_last_lattice_point(self):
        if not self.lattice_undo_stack:
            print("[lattice] Nothing to undo")
            return
        undo_entry = self.lattice_undo_stack.pop()
        side_char = undo_entry[0]
        timepoint = undo_entry[1]

        if side_char == 'DELETE_LAT':
            # Undo a lattice pair deletion — re-insert
            row, old_l_pt, old_r_pt, removed_name = undo_entry[2]
            # Find the side that has this timepoint
            for side in range(2):
                if side < len(self.grid_timepoints) and self.grid_timepoints[side] == timepoint:
                    lat_l = self.lattice_left_layers[side]
                    lat_r = self.lattice_right_layers[side]
                    if lat_l is not None and old_l_pt is not None:
                        l_data = lat_l.data.copy() if len(lat_l.data) > 0 else np.empty((0, 3))
                        lat_l.data = np.insert(l_data, row, old_l_pt, axis=0)
                    if lat_r is not None and old_r_pt is not None:
                        r_data = lat_r.data.copy() if len(lat_r.data) > 0 else np.empty((0, 3))
                        lat_r.data = np.insert(r_data, row, old_r_pt, axis=0)
                    pair_names = self.lattice_pair_names.setdefault(timepoint, [])
                    if removed_name is not None:
                        pair_names.insert(row, removed_name)
                        _renumber_lattice_pairs(pair_names)
                        self.lattice_seam_counter[timepoint] = [
                            p['name'] for p in pair_names if p['type'] == 'seam']
                    for ll, lr, llines, lmid, llc, lrc in zip(
                            self.lattice_left_layers, self.lattice_right_layers,
                            self.lattice_line_layers, self.lattice_mid_layers,
                            self.lattice_left_curve_layers, self.lattice_right_curve_layers):
                        if ll is lat_l:
                            self._update_lattice_visuals(ll, lr, llines, lmid, llc, lrc)
                            break
                    entry = self.lattice_annotations.setdefault(timepoint, {})
                    entry['left'] = lat_l.data.copy() if len(lat_l.data) > 0 else np.empty((0, 3))
                    entry['right'] = lat_r.data.copy() if lat_r is not None and len(lat_r.data) > 0 else np.empty((0, 3))
                    break
            print(f"[t={timepoint}] Undid delete of lattice pair at index {row}")
            self._refresh_tables()
            return

        if side_char == 'TABLE_LAT':
            # Undo a lattice table coordinate edit
            layer, row, old_val = undo_entry[2]
            if layer is not None and row < len(layer.data):
                pts = layer.data.copy()
                pts[row] = old_val
                layer.data = pts
                for ll, lr, llines, lmid, llc, lrc in zip(
                        self.lattice_left_layers, self.lattice_right_layers,
                        self.lattice_line_layers, self.lattice_mid_layers,
                        self.lattice_left_curve_layers, self.lattice_right_curve_layers):
                    if ll is layer or lr is layer:
                        self._update_lattice_visuals(ll, lr, llines, lmid, llc, lrc)
                        break
                entry = self.lattice_annotations.setdefault(timepoint, {})
                for side, lat_l in enumerate(self.lattice_left_layers):
                    if lat_l is layer:
                        entry['left'] = layer.data.copy()
                        break
                for side, lat_r in enumerate(self.lattice_right_layers):
                    if lat_r is layer:
                        entry['right'] = layer.data.copy()
                        break
                print(f"[t={timepoint}] Undid lattice table edit")
                self._refresh_tables()
            return

        if side_char == 'INSERT':
            # Undo a full pair insertion
            lat_left_layer, lat_right_layer, idx = undo_entry[2]
            left_data = np.delete(np.asarray(lat_left_layer.data), idx, axis=0)
            right_data = np.delete(np.asarray(lat_right_layer.data), idx, axis=0)
            lat_left_layer.data = left_data if len(left_data) > 0 else np.empty((0, 3))
            lat_right_layer.data = right_data if len(right_data) > 0 else np.empty((0, 3))
            # Remove pair name and renumber
            pair_names = self.lattice_pair_names.get(timepoint, [])
            if idx < len(pair_names):
                pair_names.pop(idx)
                _renumber_lattice_pairs(pair_names)
                self.lattice_seam_counter[timepoint] = [
                    p['name'] for p in pair_names if p['type'] == 'seam']
            self.lattice_last_placed = None
            # Update visuals
            for ll, lr, llines, lmid, llc, lrc in zip(
                    self.lattice_left_layers, self.lattice_right_layers,
                    self.lattice_line_layers, self.lattice_mid_layers,
                    self.lattice_left_curve_layers, self.lattice_right_curve_layers):
                if ll is lat_left_layer or lr is lat_right_layer:
                    self._update_lattice_visuals(ll, lr, llines, lmid, llc, lrc)
                    break
            print(f"[t={timepoint}] Undid insertion at index {idx}")
            names_str = ', '.join(p['name'] for p in pair_names)
            print(f"  Renumbered: {names_str}")
            self._refresh_tables()
            return

        layer = undo_entry[2]
        if len(layer.data) > 0:
            layer.data = layer.data[:-1]
            # Reverse the L/R alternation
            if side_char in ('L', 'R'):
                self.lattice_next_side = side_char
            # Remove pair name metadata when undoing L (the pair creator)
            if side_char == 'L':
                pair_names = self.lattice_pair_names.get(timepoint, [])
                if pair_names:
                    pair_names.pop()
                    _renumber_lattice_pairs(pair_names)
                    self.lattice_seam_counter[timepoint] = [
                        p['name'] for p in pair_names if p['type'] == 'seam']
            self.lattice_last_placed = None
            for ll, lr, llines, lmid, llc, lrc in zip(
                    self.lattice_left_layers,
                    self.lattice_right_layers,
                    self.lattice_line_layers,
                    self.lattice_mid_layers,
                    self.lattice_left_curve_layers,
                    self.lattice_right_curve_layers):
                if ll is layer or lr is layer:
                    self._update_lattice_visuals(ll, lr, llines, lmid, llc, lrc)
                    break
            print(f"[t={timepoint}] Undid last lattice {side_char} point")
            self._refresh_tables()
        else:
            print(f"[t={timepoint}] Nothing to undo in lattice")

    # ------------------------------------------------------------------ #
    # Arrow-key nudge & lattice done                                       #
    # ------------------------------------------------------------------ #

    def _nudge_last_point(self, direction: str, step: float = 1.0):
        """Move the selected/last-placed lattice point by 1 voxel.

        Matches MIPAV's moveSelectedPoint() — adds direction vector to point.
        """
        if not self.lattice_mode:
            return
        info = self.lattice_last_placed
        if info is None:
            return
        layer = info['layer']
        if len(layer.data) == 0:
            return
        # Use specific index if set (selected existing point), else last point
        idx = info.get('index', len(layer.data) - 1)
        if idx >= len(layer.data):
            return
        pts = layer.data.copy()
        # directions map to z/y/x axes: up/down = z, left/right = x
        if direction == 'up':
            pts[idx][0] -= step   # z decreases (visually "up")
        elif direction == 'down':
            pts[idx][0] += step
        elif direction == 'left':
            pts[idx][2] -= step   # x decreases
        elif direction == 'right':
            pts[idx][2] += step
        layer.data = pts
        # Update visuals
        ti = info['timepoint']
        for ll, lr, llines, lmid, llc, lrc in zip(
                self.lattice_left_layers, self.lattice_right_layers,
                self.lattice_line_layers, self.lattice_mid_layers,
                self.lattice_left_curve_layers, self.lattice_right_curve_layers):
            if ll is layer or lr is layer:
                self._update_lattice_visuals(ll, lr, llines, lmid, llc, lrc)
                break
        # Update cache
        entry = self.lattice_annotations.setdefault(ti, {})
        for side, lat_l in enumerate(self.lattice_left_layers):
            if lat_l is layer:
                entry['left'] = layer.data.copy()
                break
        for side, lat_r in enumerate(self.lattice_right_layers):
            if lat_r is layer:
                entry['right'] = layer.data.copy()
                break
        self._refresh_tables()

    def _toggle_wireframe(self):
        """Toggle wireframe mesh visibility. Rebuilds mesh when turning on."""
        self.wireframe_visible = not self.wireframe_visible
        if self.wireframe_visible:
            # Rebuild wireframe to catch up with current lattice state
            for ll, lr, llines, lmid, llc, lrc in zip(
                    self.lattice_left_layers, self.lattice_right_layers,
                    self.lattice_line_layers, self.lattice_mid_layers,
                    self.lattice_left_curve_layers, self.lattice_right_curve_layers):
                if ll is not None and lr is not None:
                    self._update_lattice_visuals(ll, lr, llines, lmid, llc, lrc)
        for wf in self.wireframe_layers:
            if wf is not None:
                wf.visible = self.wireframe_visible
        state = "ON" if self.wireframe_visible else "OFF"
        print(f"Wireframe: {state}")
        show_info(f"Wireframe: {state}")

    def _toggle_surface(self):
        """Toggle surface mesh visibility. Rebuilds mesh when turning on."""
        self.surface_visible = not self.surface_visible
        if self.surface_visible:
            for ll, lr, llines, lmid, llc, lrc in zip(
                    self.lattice_left_layers, self.lattice_right_layers,
                    self.lattice_line_layers, self.lattice_mid_layers,
                    self.lattice_left_curve_layers, self.lattice_right_curve_layers):
                if ll is not None and lr is not None:
                    self._update_lattice_visuals(ll, lr, llines, lmid, llc, lrc)
        for sf in self.surface_layers:
            if sf is not None:
                sf.visible = self.surface_visible
        if self.surface_visible:
            print("=" * 50)
            print("  SURFACE MESH: ON")
            print("  Shift+W → toggle off")
            print("  Rendered with smooth shading, turbo colormap")
            print("=" * 50)
            show_info("Surface: ON")
        else:
            print("=" * 50)
            print("  SURFACE MESH: OFF")
            print("=" * 50)
            show_info("Surface: OFF")

    def _on_lattice_done(self):
        """Exit lattice mode (use S to save)."""
        if not self.lattice_mode:
            return
        self._save_annotations(self.viewer_left)
        self.lattice_mode = False
        self.lattice_last_placed = None
        print("=" * 50)
        print("  LATTICE DONE — saved. Back to annotation mode.")
        print("=" * 50)

    # ------------------------------------------------------------------ #
    # Save / Run                                                           #
    # ------------------------------------------------------------------ #

    def _save_annotations(self, viewer):
        if self.use_grid:
            self._save_grid_annotations_to_cache()
            if not self.grid_annotations:
                print("No annotations to save")
                show_info("No annotations to save")
                return
            total = 0
            ann_paths = []
            for ti, pts in sorted(self.grid_annotations.items()):
                if len(pts) == 0:
                    continue
                stem = self.tiff_files[ti].stem
                ann_dir = (self.volume_path / stem / f"{stem}_results"
                           / "integrated_annotation")
                ann_dir.mkdir(parents=True, exist_ok=True)
                save_path = ann_dir / "annotations_test.csv"
                segs = self.grid_annotation_segments.get(ti, [])
                if save_annotations(pts, save_path, segments=segs):
                    total += len(pts)
                    ann_paths.append(str(save_path))
                    print(f"  Saved {len(pts)} to {save_path}")
            msg = f"Saved {total} annotations across {len(self.grid_annotations)} timepoint(s):\n" + "\n".join(ann_paths)
            print(msg)
            show_info(msg)
            self._save_lattice()
        else:
            if not hasattr(self, 'points_layer') or len(self.points_layer.data) == 0:
                print("No annotations to save")
                show_info("No annotations to save")
                return
            save_path = (Path(self.annotations_path) if self.annotations_path
                         else self.volume_path.with_suffix('.csv'))
            save_annotations(self.points_layer.data, save_path)
            show_info(f"Saved {len(self.points_layer.data)} annotations to {save_path}")

    def run(self):
        napari.run()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="napari_worm: 3D Cell Annotation Tool for C. elegans")
    parser.add_argument("volume", nargs="?", default=None,
                        help="TIFF file or directory of TIFFs "
                             "(opens file dialog if omitted)")
    parser.add_argument("--annotations", "-a", default=None,
                        help="Existing annotations CSV (optional)")
    parser.add_argument("--no-grid", action="store_true",
                        help="Use dask 4D slider instead of dual window mode")
    parser.add_argument("--start", "-s", type=int, default=0,
                        help="Starting timepoint index (default: 0)")
    parser.add_argument("--channels", "-c", default=None,
                        help="Comma-separated channel dir names (e.g. RegA,RegB). "
                             "Auto-discovered if omitted.")
    args = parser.parse_args()

    volume = args.volume
    if volume is None:
        # No path on command line — show a GUI dialog for path selection
        import json
        app = QApplication.instance() or QApplication([])

        # Load last-used path from settings file
        settings_file = Path.home() / ".napari_worm_settings.json"
        last_dir = ""
        last_start = 100
        if settings_file.exists():
            try:
                settings = json.loads(settings_file.read_text())
                last_dir = settings.get("last_dir", "")
                last_start = settings.get("last_start", 100)
            except Exception:
                pass

        # Build a small dialog with path text field + Browse button + start timepoint
        from qtpy.QtWidgets import (QDialog, QDialogButtonBox, QLineEdit,
                                     QPushButton)
        dlg = QDialog()
        dlg.setWindowTitle("napari-worm — Select Volume Directory")
        dlg.setMinimumWidth(600)
        layout = QVBoxLayout(dlg)

        layout.addWidget(QLabel("Enter or browse to a volume directory (folder of TIFFs):"))
        path_row = QHBoxLayout()
        path_edit = QLineEdit(last_dir)
        path_edit.setPlaceholderText("/path/to/RegB  or  /Volumes/shroff/.../For_Tracking/RegB")
        browse_btn = QPushButton("Browse...")
        path_row.addWidget(path_edit)
        path_row.addWidget(browse_btn)
        layout.addLayout(path_row)

        start_row = QHBoxLayout()
        start_row.addWidget(QLabel("Start timepoint:"))
        start_spin = QSpinBox()
        start_spin.setRange(0, 9999)
        start_spin.setValue(last_start)
        start_spin.setFixedWidth(80)
        start_row.addWidget(start_spin)
        start_row.addStretch()
        layout.addLayout(start_row)

        def _browse():
            start = path_edit.text() or last_dir or "/Volumes"
            d = QFileDialog.getExistingDirectory(dlg, "Select directory", start,
                                                  QFileDialog.ShowDirsOnly)
            if d:
                path_edit.setText(d)

        browse_btn.clicked.connect(_browse)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        # Accept on Enter in the text field
        path_edit.returnPressed.connect(dlg.accept)

        if not dlg.exec():
            print("No directory selected — exiting.")
            return
        volume = path_edit.text().strip()
        if not volume:
            print("No directory selected — exiting.")
            return
        args.start = start_spin.value()

        # Save the selected path and start timepoint for next time
        try:
            settings_file.write_text(json.dumps({
                "last_dir": volume,
                "last_start": args.start,
            }))
        except Exception:
            pass

    # If the selected directory has no TIFFs but contains Reg* subdirectories,
    # auto-resolve to the first channel dir (e.g. For_Tracking/ → RegB/)
    vp = Path(volume)
    if vp.is_dir() and not list(vp.glob("*.tif"))[:1]:
        reg_dirs = sorted([
            d for d in vp.iterdir()
            if d.is_dir() and d.name.startswith('Reg') and list(d.glob("*.tif"))[:1]
        ], key=lambda d: d.name)
        if reg_dirs:
            volume = str(reg_dirs[-1])  # prefer RegB (last alphabetically)
            print(f"No TIFFs in selected directory — using {reg_dirs[-1].name}/")
            print(f"  (sibling channels will be auto-discovered)")

    WormAnnotator(volume, args.annotations,
                  grid_mode=not args.no_grid, start_t=args.start,
                  channels=args.channels).run()


if __name__ == "__main__":
    main()
