import numpy as np

# Caroline Malin-Mayor's worm-space coordinate-transform package.
# Provides PythonCelegansModel.{straighten_volume, retwist, get_basis_vectors,
# get_best_candidate}. We delegate all (ML, DV, AP) coordinate math to this
# package; napari-worm is a UI shell that drives it.
from celegans_model import PythonCelegansModel


def _make_celegans_model(left_pts_zyx: np.ndarray, right_pts_zyx: np.ndarray,
                         names: list[str] | None = None) -> PythonCelegansModel | None:
    """Build a PythonCelegansModel from napari-worm lattice arrays.

    napari-worm stores lattice points as (N, 3) (z, y, x) arrays. Caroline's
    PythonCelegansModel takes a (N, 2, 3) array where ``[:, 0]=right`` and
    ``[:, 1]=left``. Coordinate convention is consistent within a single
    model instance, so we keep (z, y, x) end-to-end — model splines, retwist
    output, and straighten_volume input all live in (z, y, x).

    `names` are pair base-names like ``a0``, ``H0``, ``H1``, ... If they match
    the canonical 11 standard cells in the standard order we use
    `parameterization='uniform'`; otherwise we fall back to `'arc_length'`
    (which doesn't require canonical names).

    Returns None if there are fewer than 3 paired lattice points.
    """
    n = min(len(left_pts_zyx), len(right_pts_zyx))
    if n < 3:
        return None
    lattice = np.stack([right_pts_zyx[:n], left_pts_zyx[:n]], axis=1).astype(float)
    standard = ['a0', 'h0', 'h1', 'h2', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 't']
    if names is not None:
        names_lower = [str(s).lower() for s in names[:n]]
        if names_lower == standard[:n]:
            return PythonCelegansModel(
                lattice, parameterization='uniform', spacing=1.0,
                lattice_point_names=names_lower)
    # Fallback: arc-length parameterization works without canonical names.
    return PythonCelegansModel(lattice, parameterization='arc_length')
