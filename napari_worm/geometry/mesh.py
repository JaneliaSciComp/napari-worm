import numpy as np

from napari_worm.geometry.spline import _build_cross_section_rings, _smooth_midline_spline


def generate_wireframe_mesh(left_pts: np.ndarray, right_pts: np.ndarray,
                            num_ellipse_pts: int = 32,
                            num_samples: int = 0,
                            lattice_aligned: bool = False,
                            overrides: dict | None = None) -> list[np.ndarray]:
    """Generate wireframe mesh from lattice L/R points (MIPAV generateCurves + generateEllipses).

    Algorithm (matching LatticeModel.java):
    1. Compute center spline as midpoint of L/R pairs
    2. Sample center spline (uniform steps, or one ring per lattice pair when lattice_aligned)
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
                  0 = auto (1-voxel step size). Ignored when lattice_aligned=True.
    lattice_aligned : if True, place exactly one anchor ring per lattice pair at
                      that pair's position. Ring index == lattice slice index,
                      matching MIPAV's latticeCrossSection_<i>.csv format.

    Returns
    -------
    List of (M, 3) arrays — each is a smooth longitudinal path for wireframe display.
    Also includes cross-section rings.
    """
    ellipse_rings, _ = _build_cross_section_rings(
        left_pts, right_pts, num_ellipse_pts, num_samples, lattice_aligned,
        overrides=overrides)
    if not ellipse_rings:
        return []

    paths = []

    # Longitudinal curves (32 splines running nose-to-tail)
    for j in range(num_ellipse_pts):
        control_pts = np.array([ring[j] for ring in ellipse_rings])
        spline_pts = _smooth_midline_spline(control_pts, samples_per_segment=5)
        paths.append(spline_pts)

    # Cross-section rings — show every anchor ring when lattice-aligned (they
    # are the editable handles); else ~20 display rings for the dense wireframe.
    K = len(ellipse_rings)
    ring_step = 1 if lattice_aligned else max(1, K // 20)
    for i in range(0, K, ring_step):
        ring = ellipse_rings[i]
        closed_ring = np.vstack([ring, ring[0:1]])
        paths.append(closed_ring)

    return paths


def generate_surface_mesh(left_pts: np.ndarray, right_pts: np.ndarray,
                          num_ellipse_pts: int = 32,
                          num_samples: int = 0,
                          lattice_aligned: bool = False,
                          overrides: dict | None = None):
    """Generate a triangle mesh from lattice L/R points (MIPAV generateTriMesh).

    Uses the same ellipse cross-section computation as generate_wireframe_mesh(),
    then packs the rings into (vertices, faces, values) for napari add_surface().

    Vertex layout (matches MIPAV LatticeModel.java:1454-1481):
        [head_center, ring0_pt0..ring0_pt31, ring1_pt0..ring1_pt31, ..., tail_center]

    Face construction (matches MIPAV LatticeModel.java:1483-1517):
        - Head cap: triangle fan from head_center to first ring
        - Body: adjacent rings connected by quads (2 triangles each)
        - Tail cap: triangle fan from tail_center to last ring

    When lattice_aligned=True, one ring per lattice pair (ring index == lattice
    slice index), matching MIPAV's displayContours layout.

    Returns (vertices, faces, values) or None if insufficient data.
    """
    ellipse_rings, _ = _build_cross_section_rings(
        left_pts, right_pts, num_ellipse_pts, num_samples, lattice_aligned,
        overrides=overrides)
    if not ellipse_rings:
        return None

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
