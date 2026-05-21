from pathlib import Path

import numpy as np
import pandas as pd


def save_annotations(points, filepath, names=None, segments=None):
    """Save annotations in MIPAV format: name,x_voxels,y_voxels,z_voxels,R,G,B[,lattice_segment]"""
    filepath = Path(filepath)
    n = len(points)
    # MIPAV-compatible 1-indexed naming (A1, A2, ...). Validated against
    # straightened_annotations.csv from RW10752_NU/Decon_reg_15_results.
    if names is None:
        names = [f"A{i + 1}" for i in range(n)]
    elif len(names) < n:
        names = list(names) + [f"A{i + 1}" for i in range(len(names), n)]
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
