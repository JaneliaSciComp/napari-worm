import numpy as np
from scipy.interpolate import CubicSpline


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


def _build_cross_section_rings(left_pts: np.ndarray, right_pts: np.ndarray,
                               num_ellipse_pts: int,
                               num_samples: int,
                               lattice_aligned: bool,
                               overrides: dict | None = None):
    """Build ellipse cross-section rings along the center spline.

    Returns (ellipse_rings, centers_on_rings) where:
      ellipse_rings : list of (num_ellipse_pts, 3) arrays in world coords
      centers_on_rings : (K, 3) array of ring center points (used for radial ops)
    Or ([], []) if insufficient data.

    When lattice_aligned=True, K == N (one ring per lattice pair), centers and
    L/R refs come from the actual lattice points (no spline drift), and ring
    index == lattice slice index — matching MIPAV's displayContours layout and
    latticeCrossSection_<i>.csv naming.

    When overrides is provided (dict[ring_idx, (num_ellipse_pts, 3) array]),
    matching rings are replaced with `center + override[j]` for each vertex j
    (center-relative world-frame offsets — matches MIPAV relativeCrossSections,
    LatticeModel.java:4806-4830). Only applied when lattice_aligned=True, since
    ring-indexed overrides only make sense against the lattice-slice layout.
    """
    n = min(len(left_pts), len(right_pts))
    if n < 3:
        return [], np.empty((0, 3))

    centers = (left_pts[:n] + right_pts[:n]) / 2.0
    center_cs, center_deriv = _make_spline_with_derivative(centers)
    left_cs, _ = _make_spline_with_derivative(left_pts[:n])
    right_cs, _ = _make_spline_with_derivative(right_pts[:n])
    if center_cs is None or left_cs is None or right_cs is None:
        return [], np.empty((0, 3))

    if lattice_aligned:
        dists = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        cumlen = np.concatenate([[0.0], np.cumsum(dists)])
        total = cumlen[-1]
        if total < 1e-12:
            return [], np.empty((0, 3))
        t_samples = cumlen / total
        # Use lattice control points directly — no spline-evaluation drift.
        sample_centers = centers
        sample_left = left_pts[:n]
        sample_right = right_pts[:n]
    else:
        t_dense = np.linspace(0, 1, 500)
        pts_dense = center_cs(t_dense)
        arc_lengths = np.concatenate([[0], np.cumsum(
            np.linalg.norm(np.diff(pts_dense, axis=0), axis=1))])
        total_length = arc_lengths[-1]
        if total_length < 1e-6:
            return [], np.empty((0, 3))
        if num_samples <= 0:
            num_samples = max(int(np.ceil(total_length)), 10)
        target_arcs = np.linspace(0, total_length, num_samples)
        t_samples = np.interp(target_arcs, arc_lengths, t_dense)
        sample_centers = center_cs(t_samples)
        sample_left = left_cs(t_samples)
        sample_right = right_cs(t_samples)

    angles = np.linspace(0, 2 * np.pi, num_ellipse_pts, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    ellipse_rings = []
    for i, t_val in enumerate(t_samples):
        center = sample_centers[i]
        left_pt = sample_left[i]
        right_pt = sample_right[i]

        tangent = center_deriv(t_val)
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-12:
            tangent = np.array([1.0, 0.0, 0.0])
        else:
            tangent = tangent / tangent_norm

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

        override_ring = None
        if lattice_aligned and overrides is not None:
            override_ring = overrides.get(i)

        ring = np.empty((num_ellipse_pts, 3))
        if override_ring is not None and override_ring.shape == (num_ellipse_pts, 3):
            for j in range(num_ellipse_pts):
                ring[j] = center + override_ring[j]
        else:
            for j in range(num_ellipse_pts):
                ring[j] = center + radius * cos_a[j] * right_vec + radius * sin_a[j] * up_vec
        ellipse_rings.append(ring)

    return ellipse_rings, np.asarray(sample_centers)
