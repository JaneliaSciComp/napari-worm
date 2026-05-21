from pathlib import Path

import numpy as np


def _save_cross_section_csv(path, ring_offsets: np.ndarray) -> bool:
    """Write a single cross-section override to MIPAV latticeCrossSection_<i>.csv.

    Format (matches LatticeModel.saveContourAsCSV, LatticeModel.java:6796-6835):
        Contour
        x_voxels,y_voxels,z_voxels
        <x>,<y>,<z>
        ... (num_ellipse_pts rows)
        <trailing blank line>

    `ring_offsets` is a (num_ellipse_pts, 3) array of CENTER-RELATIVE offsets in
    napari (z, y, x) order; they are written as (x, y, z) to match MIPAV.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import time
    for attempt in range(5):
        try:
            with open(path, 'w') as f:
                f.write("Contour\n")
                f.write("x_voxels,y_voxels,z_voxels\n")
                for off in ring_offsets:
                    dz, dy, dx = float(off[0]), float(off[1]), float(off[2])
                    f.write(f"{dx},{dy},{dz}\n")
                f.write("\n")
            return True
        except BlockingIOError:
            if attempt < 4:
                time.sleep(1.0)
    print(f"  WARNING: could not write {path} (network drive busy after 5 retries)")
    return False


def _load_cross_section_csv(path) -> np.ndarray | None:
    """Read a MIPAV latticeCrossSection_<i>.csv into (num_ellipse_pts, 3) array
    of center-relative offsets in napari (z, y, x) order.

    Returns None if the file is missing, empty, or malformed.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        offsets = []
        with open(path, 'r') as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        # Skip the two header lines: "Contour" and "x_voxels,y_voxels,z_voxels"
        for line in lines[2:]:
            parts = line.split(',')
            if len(parts) != 3:
                continue
            dx, dy, dz = float(parts[0]), float(parts[1]), float(parts[2])
            offsets.append((dz, dy, dx))
        if not offsets:
            return None
        return np.asarray(offsets, dtype=float)
    except (OSError, ValueError) as exc:
        print(f"  WARNING: failed to read {path}: {exc}")
        return None
