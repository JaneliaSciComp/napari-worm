"""
napari_worm: 3D Cell Annotation Tool for C. elegans

Usage:
    python napari_worm.py /path/to/volume.tif                    # single file
    python napari_worm.py /path/to/directory/                     # dual view (MIPAV-style)
    python napari_worm.py /path/to/directory/ --start 100         # start at timepoint 100
    python napari_worm.py /path/to/directory/ --no-grid           # dask 4D slider (fallback)
    python napari_worm.py /path/to/volume.tif -a existing.csv     # load existing annotations
"""

import argparse
from pathlib import Path

import dask.array as da
import napari
import numpy as np
import pandas as pd
import tifffile
from scipy.interpolate import CubicSpline
from dask import delayed
from qtpy.QtCore import QEvent, QObject, Qt, QTimer
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (
    QHBoxLayout, QLabel, QShortcut, QSpinBox,
    QSplitter, QVBoxLayout, QWidget,
)


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
    if n_samples is None:
        n_samples = int(np.linalg.norm(end - start)) + 1
    positions = np.linspace(start, end, n_samples)
    values = []
    for pos in positions:
        idx = np.round(pos).astype(int)
        if np.all(idx >= 0) and np.all(idx < data.shape):
            values.append(data[tuple(idx)])
        else:
            values.append(0)
    return positions, np.array(values)


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

def save_annotations(points, filepath, names=None):
    """Save annotations in MIPAV format: name,x_voxels,y_voxels,z_voxels,R,G,B"""
    filepath = Path(filepath)
    n = len(points)
    if names is None:
        names = [f"A{i}" for i in range(n)]
    df = pd.DataFrame({
        'name':     names,
        'x_voxels': points[:, 2],
        'y_voxels': points[:, 1],
        'z_voxels': points[:, 0],
        'R': 255, 'G': 255, 'B': 255,
    })
    import time
    for attempt in range(3):
        try:
            df.to_csv(filepath, index=False)
            return
        except BlockingIOError:
            if attempt < 2:
                time.sleep(0.5)
            else:
                print(f"  WARNING: could not write {filepath} (network drive busy)")


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


def _find_closest_lattice_point(pos: np.ndarray, left_pts: np.ndarray,
                                right_pts: np.ndarray, threshold: float = 8.0):
    """Find the closest existing lattice point to pos.

    Returns (side, index, distance) where side is 'L' or 'R', or None if none within threshold.
    Matches MIPAV's selectLattice() proximity check (12-unit threshold in MIPAV).
    """
    best = None
    for side_char, pts in [('L', left_pts), ('R', right_pts)]:
        if len(pts) == 0:
            continue
        dists = np.linalg.norm(pts - pos, axis=1)
        min_idx = int(np.argmin(dists))
        min_dist = dists[min_idx]
        if min_dist <= threshold and (best is None or min_dist < best[2]):
            best = (side_char, min_idx, min_dist)
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
                          threshold: float = 12.0) -> int | None:
    """Find where to insert a new point between existing pairs.

    Matches MIPAV's addInsertionPoint(): measures the 3D distance from the
    mouse RAY (near→far) to each left-curve segment and each right-curve
    segment, then picks the closest segment within the threshold.

    Parameters
    ----------
    near, far : start/end of the camera ray through the volume.
    left_pts, right_pts : existing L/R point arrays.
    threshold : max distance from ray to a segment (MIPAV uses 12 voxels).

    Returns
    -------
    Index i such that the new point should be inserted at position i+1,
    or None if no segment is close enough (append instead).
    """
    n = min(len(left_pts), len(right_pts))
    if n < 2:
        return None

    min_dist = float('inf')
    min_idx = None

    # Check both L and R curve segments (like MIPAV checks both)
    for pts in (left_pts[:n], right_pts[:n]):
        for i in range(n - 1):
            dist = _segment_to_segment_distance(near, far, pts[i], pts[i + 1])
            if dist < min_dist:
                min_dist = dist
                if dist <= threshold:
                    min_idx = i

    return min_idx


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


def load_annotations(filepath):
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Annotations not found: {filepath}")
    df = pd.read_csv(filepath)
    if 'x_voxels' in df.columns:
        points = df[['z_voxels','y_voxels','x_voxels']].values
        names = df['name'].tolist() if 'name' in df.columns else None
        print(f"Loaded {len(points)} MIPAV annotations from {filepath}")
    else:
        points = df[['z','y','x']].values
        names = None
        print(f"Loaded {len(points)} annotations from {filepath}")
    return points, names


# ---------------------------------------------------------------------------
# MIPAV-style dual-view window
# ---------------------------------------------------------------------------

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
        return False  # pass event through


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

        # Tabify right viewer's docks behind left viewer's docks.
        # raise_() to switch tabs is purely cosmetic — no layout recalculation,
        # no GL resize, no freeze (unlike setVisible which forces a full relayout).
        self._host.tabifyDockWidget(self._qt_left.dockLayerControls,
                                    self._qt_right.dockLayerControls)
        self._host.tabifyDockWidget(self._qt_left.dockLayerList,
                                    self._qt_right.dockLayerList)
        # Start with left viewer's tabs on top
        self._qt_left.dockLayerControls.raise_()
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

    def resizeDocks(self, docks, sizes, orientation):
        self._host.resizeDocks(docks, sizes, orientation)

    def set_active_side(self, side: int):
        """Bring the active viewer's dock tabs to the front."""
        if side == self._active_side:
            return
        self._active_side = side
        if side == 0:
            self._qt_left.dockLayerControls.raise_()
            self._qt_left.dockLayerList.raise_()
        else:
            self._qt_right.dockLayerControls.raise_()
            self._qt_right.dockLayerList.raise_()


# ---------------------------------------------------------------------------
# Main annotator class
# ---------------------------------------------------------------------------

class WormAnnotator:
    """Main annotation tool class."""

    def __init__(self, volume_path, annotations_path=None, grid_mode=True, start_t=0):
        self.volume_path = Path(volume_path)
        self.annotations_path = annotations_path
        self.is_time_series = self.volume_path.is_dir()
        self.use_grid = grid_mode and self.is_time_series
        self.start_t = start_t

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
            points, names = load_annotations(annotations_path)
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
        self.tiff_files = scan_time_series(self.volume_path)
        max_t = len(self.tiff_files) - 1
        start = min(self.start_t, max_t - 1)

        self.grid_annotations: dict[int, np.ndarray] = {}
        self.undo_stack: list[tuple[int, object]] = []
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
        self.lattice_next_side: str = 'L'  # alternates L→R→L→R (nose to tail)
        self.lattice_last_placed: dict | None = None  # {'side_idx', 'timepoint', 'layer', 'char'}
        # Ordered list of pair metadata per timepoint:
        # [{name: 'a0', type: 'lattice'}, {name: 'H0', type: 'seam'}, ...]
        # The i-th entry corresponds to i-th L point and i-th R point in the layers
        self.lattice_pair_names: dict[int, list[dict]] = {}  # {timepoint: [pair_info, ...]}
        self.lattice_seam_counter: dict[int, list[str]] = {}  # {timepoint: [used seam names]}
        self.lattice_undo_stack: list = []  # ('L'|'R', timepoint, layer)

        first_vol = load_volume(self.tiff_files[start])
        self.contrast_limits = [float(first_vol.min()), float(first_vol.max())]

        # Two independent napari viewers — hidden until embedded in DualViewWindow
        self.viewer_left  = napari.Viewer(ndisplay=3, show=False)
        self.viewer_right = napari.Viewer(ndisplay=3, show=False)
        self.viewer = self.viewer_left  # primary reference for save/print logic

        # Navigation spinboxes
        self._nav_updating = False
        nav_widget = self._build_nav_widget(max_t, start)

        # Build the single main window (MIPAV's JFrame equivalent)
        self.dual_window = DualViewWindow(self.viewer_left, self.viewer_right, nav_widget)

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
        from qtpy.QtWidgets import QApplication
        QApplication.processEvents()   # flush Qt events → GL context ready

        # Set left dock area to 370 px so control labels are fully visible
        # (resizeDocks must be called after the window is shown)
        self.dual_window.resizeDocks(
            [self.dual_window._qt_left.dockLayerControls],
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
        self._left_spin = QSpinBox()
        self._left_spin.setRange(0, max_t)
        self._left_spin.setValue(start)
        self._left_spin.setFixedWidth(80)
        self._left_spin.valueChanged.connect(self._on_spinbox_changed)
        layout.addWidget(self._left_spin)

        layout.addStretch()
        layout.addWidget(QLabel("Right t="))
        self._right_spin = QSpinBox()
        self._right_spin.setRange(0, max_t)
        self._right_spin.setValue(min(start + 1, max_t))
        self._right_spin.setFixedWidth(80)
        self._right_spin.valueChanged.connect(self._on_spinbox_changed)
        layout.addWidget(self._right_spin)

        layout.addStretch()
        return nav

    def _load_dual_pair(self, t_left: int, t_right: int):
        """Load two timepoints into their respective viewers."""
        self._save_grid_annotations_to_cache()
        self._save_lattice_to_cache()

        self.viewer_left.layers.clear()
        self.viewer_right.layers.clear()
        self.grid_timepoints = [t_left, t_right]

        for side, (viewer_ref, ti) in enumerate(
                [(self.viewer_left, t_left), (self.viewer_right, t_right)]):
            vol = load_volume(self.tiff_files[ti])

            img = viewer_ref.add_image(
                vol, name=f'Volume t={ti}', colormap='gray',
                rendering='mip', contrast_limits=self.contrast_limits, multiscale=False)
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
            # Lock lattice layers to pan_zoom — our handler does all adding
            for lyr in (lat_l, lat_r, lat_lines, lat_mid,
                        lat_left_curve, lat_right_curve):
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

            # Each viewer handles only its own canvas — no click-routing needed
            def make_handler(volume_data, side_idx, timepoint,
                             img_ref, pts_ref,
                             lat_l_ref, lat_r_ref, lat_lines_ref, lat_mid_ref,
                             lat_lc_ref, lat_rc_ref):
                def handler(layer, event):
                    if 'Control' not in event.modifiers:
                        return
                    if not self.lattice_mode:
                        self._on_click_dual(img_ref, event, volume_data,
                                            side_idx, timepoint, pts_ref)
                        return

                    # --- Lattice mode: check if dragging an existing point ---
                    near, far = img_ref.get_ray_intersections(
                        event.position, event.view_direction, event.dims_displayed)
                    if near is None or far is None:
                        return
                    pos = find_nucleus_centroid(
                        volume_data, find_peak_along_ray(volume_data, near, far))

                    n_l = len(lat_l_ref.data)
                    n_r = len(lat_r_ref.data)
                    drag_target = None
                    if n_l > 0 or n_r > 0:
                        closest = _find_closest_lattice_point(
                            pos,
                            np.asarray(lat_l_ref.data) if n_l > 0 else np.empty((0, 3)),
                            np.asarray(lat_r_ref.data) if n_r > 0 else np.empty((0, 3)))
                        if closest is not None:
                            drag_target = closest  # (side_char, pt_idx, dist)

                    if drag_target is None:
                        # No nearby point — add new point (click, not drag)
                        self._on_lattice_click(img_ref, event, volume_data,
                                               side_idx, timepoint,
                                               lat_l_ref, lat_r_ref,
                                               lat_lines_ref, lat_mid_ref,
                                               lat_lc_ref, lat_rc_ref)
                        return

                    # --- Drag existing point (matches MIPAV modifyLattice) ---
                    side_char, pt_idx, dist = drag_target
                    drag_layer = lat_l_ref if side_char == 'L' else lat_r_ref
                    pair_names = self.lattice_pair_names.get(timepoint, [])
                    name = pair_names[pt_idx]['name'] + side_char if pt_idx < len(pair_names) else f'?{pt_idx}{side_char}'
                    self.lattice_last_placed = {
                        'side_idx': side_idx, 'timepoint': timepoint,
                        'layer': drag_layer, 'char': side_char, 'index': pt_idx}
                    print(f"[t={timepoint}] Dragging {name}")

                    yield  # wait for mouse move events

                    while event.type == 'mouse_move':
                        if 'Control' not in event.modifiers:
                            break
                        near2, far2 = img_ref.get_ray_intersections(
                            event.position, event.view_direction, event.dims_displayed)
                        if near2 is not None and far2 is not None:
                            # Use peak only during drag for responsiveness
                            # (MIPAV's modifyLattice uses raw ray position)
                            new_pos = find_peak_along_ray(volume_data, near2, far2)
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
                    print(f"[t={timepoint}] Released {name}")

                return handler

            cb = make_handler(vol, side, ti, img, pts, lat_l, lat_r, lat_lines, lat_mid,
                              lat_left_curve, lat_right_curve)
            # Register on ALL layers so Cmd+Click works regardless of sidebar selection
            for lyr in (img, pts, lat_l, lat_r, lat_lines, lat_mid,
                        lat_left_curve, lat_right_curve):
                lyr.mouse_drag_callbacks.append(cb)

            self.grid_image_layers[side]  = img
            self.grid_points_layers[side] = pts
            self.lattice_left_layers[side]  = lat_l
            self.lattice_right_layers[side] = lat_r
            self.lattice_line_layers[side]  = lat_lines
            self.lattice_mid_layers[side]   = lat_mid
            self.lattice_left_curve_layers[side]  = lat_left_curve
            self.lattice_right_curve_layers[side] = lat_right_curve

        self._nav_updating = True
        self._left_spin.setValue(t_left)
        self._right_spin.setValue(t_right)
        self._nav_updating = False

        self.dual_window.setWindowTitle(
            f"napari_worm  —  t={t_left} (left)  |  t={t_right} (right)  "
            f"[of {len(self.tiff_files)-1}]")
        print(f"Showing t={t_left} (left) and t={t_right} (right) of {len(self.tiff_files)-1}")

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

    def _toggle_lattice_mode(self):
        self.lattice_mode = not self.lattice_mode
        # pts layers are always pan_zoom — our handler does all point adding.
        # No mode change needed here; the toggle is purely a routing flag.
        if self.lattice_mode:
            self.lattice_next_side = 'L'
            self.lattice_last_placed = None
            print("=" * 50)
            print("  MODE: LATTICE  (nose → tail, left → right)")
            print("  Cmd+Click       → add lattice point (alternates L/R)")
            print("  Cmd+Drag        → drag existing point (near point)")
            print("  Cmd+Click       → insert pair (near existing segment)")
            print("  Cmd+Shift+Click → seam cell (also L/R pair)")
            print("  Arrow keys      → nudge selected point")
            print("  Cmd+Z           → undo (point, pair, or insertion)")
            print("  D               → done with lattice")
            print("  L               → back to annotation mode")
            print("=" * 50)
        else:
            print("=" * 50)
            print("  MODE: ANNOTATION  (Cmd+Click = annotate)")
            print("=" * 50)

    def _on_lattice_click(self, img_layer, event, volume_data,
                          side_idx, timepoint,
                          lat_left_layer, lat_right_layer,
                          lat_lines_layer, lat_mid_layer,
                          lat_left_curve_layer, lat_right_curve_layer):
        try:
            near, far = img_layer.get_ray_intersections(
                event.position, event.view_direction, event.dims_displayed)
            if near is None or far is None:
                print("[lattice] ray missed volume — click closer to the worm")
                return
            pos = find_nucleus_centroid(volume_data, find_peak_along_ray(volume_data, near, far))
            canvas_label = "left" if side_idx == 0 else "right"

            n_left = len(lat_left_layer.data)
            n_right = len(lat_right_layer.data)

            # Determine if this is a seam cell (Cmd+Shift) or regular lattice (Cmd)
            is_seam = 'Shift' in event.modifiers
            pair_type = 'seam' if is_seam else 'lattice'
            n_complete = min(n_left, n_right)
            pair_names = self.lattice_pair_names.setdefault(timepoint, [])

            # --- Check for INSERTION between existing pairs ---
            # Uses ray-to-segment distance (matches MIPAV addInsertionPoint)
            # Only when starting a new pair (next_side == 'L') and ≥2 complete pairs exist
            insert_idx = None
            if self.lattice_next_side == 'L' and n_complete >= 2:
                insert_idx = _find_insertion_index(
                    near, far,
                    np.asarray(lat_left_layer.data),
                    np.asarray(lat_right_layer.data))

            if insert_idx is not None:
                # --- INSERT both L and R (matching MIPAV addInsertionPoint) ---
                idx = insert_idx + 1  # insert AFTER the matched segment
                new_left = pos
                # Interpolate R from neighbors (average of surrounding R points)
                r_pts = np.asarray(lat_right_layer.data)
                new_right = (r_pts[insert_idx] + r_pts[insert_idx + 1]) / 2.0

                # Insert into layer data arrays
                left_data = np.insert(np.asarray(lat_left_layer.data), idx, new_left, axis=0)
                right_data = np.insert(np.asarray(lat_right_layer.data), idx, new_right, axis=0)
                lat_left_layer.data = left_data
                lat_right_layer.data = right_data

                # Insert pair name metadata and renumber
                pair_names.insert(idx, {'name': '', 'type': pair_type})
                _renumber_lattice_pairs(pair_names)
                # Rebuild seam counter from pair_names
                self.lattice_seam_counter[timepoint] = [
                    p['name'] for p in pair_names if p['type'] == 'seam']

                inserted_name = pair_names[idx]['name']
                print(f"[t={timepoint} {canvas_label}] INSERTED {inserted_name}L + {inserted_name}R "
                      f"between pair {insert_idx} and {insert_idx + 1}")
                # Print renumbered names
                names_str = ', '.join(p['name'] for p in pair_names)
                print(f"  Renumbered: {names_str}")

                self.lattice_undo_stack.append(('INSERT', timepoint,
                                                (lat_left_layer, lat_right_layer, idx)))
                self.lattice_last_placed = {
                    'side_idx': side_idx, 'timepoint': timepoint,
                    'layer': lat_left_layer, 'char': 'L'}
                # next_side stays 'L' since we inserted a complete pair

            elif self.lattice_next_side == 'L':
                # --- APPEND new L point (start of a new pair) ---
                lat_left_layer.add(pos)
                new_info = {'name': '', 'type': pair_type}
                pair_names.append(new_info)
                _renumber_lattice_pairs(pair_names)
                # Rebuild seam counter
                self.lattice_seam_counter[timepoint] = [
                    p['name'] for p in pair_names if p['type'] == 'seam']

                name = pair_names[-1]['name'] + 'L'
                print(f"[t={timepoint} {canvas_label}] {name}  "
                      f"z={pos[0]:.1f} y={pos[1]:.1f} x={pos[2]:.1f}  (next: R)")
                self.lattice_undo_stack.append(('L', timepoint, lat_left_layer))
                self.lattice_last_placed = {
                    'side_idx': side_idx, 'timepoint': timepoint,
                    'layer': lat_left_layer, 'char': 'L'}
                self.lattice_next_side = 'R'
            else:
                # --- APPEND R point (complete the current pair) ---
                lat_right_layer.add(pos)
                pair_idx = len(lat_right_layer.data) - 1
                if pair_idx < len(pair_names):
                    name = pair_names[pair_idx]['name'] + 'R'
                else:
                    name = f'?{pair_idx}R'
                print(f"[t={timepoint} {canvas_label}] {name}  "
                      f"z={pos[0]:.1f} y={pos[1]:.1f} x={pos[2]:.1f}  (next: L)")
                self.lattice_undo_stack.append(('R', timepoint, lat_right_layer))
                self.lattice_last_placed = {
                    'side_idx': side_idx, 'timepoint': timepoint,
                    'layer': lat_right_layer, 'char': 'R'}
                self.lattice_next_side = 'L'

            self._update_lattice_visuals(lat_left_layer, lat_right_layer,
                                         lat_lines_layer, lat_mid_layer,
                                         lat_left_curve_layer, lat_right_curve_layer)
            entry = self.lattice_annotations.setdefault(timepoint, {})
            entry['left']  = lat_left_layer.data.copy()  if len(lat_left_layer.data)  > 0 else np.empty((0, 3))
            entry['right'] = lat_right_layer.data.copy() if len(lat_right_layer.data) > 0 else np.empty((0, 3))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"[lattice] click error: {exc}")

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
            saved += 1
        if saved:
            print(f"Lattice saved for {saved} timepoint(s)")

    def _on_spinbox_changed(self, _):
        if self._nav_updating:
            return
        self._load_dual_pair(self._left_spin.value(), self._right_spin.value())

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

    def _on_click_dual(self, layer, event, volume_data, side_idx, timepoint, points_layer):
        near, far = layer.get_ray_intersections(
            event.position, event.view_direction, event.dims_displayed)
        if near is None or far is None:
            return
        pos = find_nucleus_centroid(volume_data, find_peak_along_ray(volume_data, near, far))
        points_layer.add(pos)
        self.undo_stack.append((timepoint, points_layer))
        label = "left" if side_idx == 0 else "right"
        print(f"[t={timepoint} {label}] Added at z={pos[0]:.1f} y={pos[1]:.1f} x={pos[2]:.1f}")

    def _undo_last_point(self, viewer):
        if self.lattice_mode:
            self._undo_last_lattice_point()
            return
        if not self.undo_stack:
            print("Nothing to undo")
            return
        timepoint, pts = self.undo_stack.pop()
        active = [p for p in self.grid_points_layers if p is not None]
        if any(p is pts for p in active) and len(pts.data) > 0:
            pts.data = pts.data[:-1]
            print(f"[t={timepoint}] Undid last annotation")
        elif timepoint in self.grid_annotations and len(self.grid_annotations[timepoint]) > 0:
            self.grid_annotations[timepoint] = self.grid_annotations[timepoint][:-1]
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

    def _undo_last_lattice_point(self):
        if not self.lattice_undo_stack:
            print("[lattice] Nothing to undo")
            return
        undo_entry = self.lattice_undo_stack.pop()
        side_char = undo_entry[0]
        timepoint = undo_entry[1]

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

    def _on_lattice_done(self):
        """Finalize the lattice — save and exit lattice mode."""
        if not self.lattice_mode:
            return
        self._save_lattice()
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
                return
            total = 0
            for ti, pts in sorted(self.grid_annotations.items()):
                if len(pts) == 0:
                    continue
                stem = self.tiff_files[ti].stem
                ann_dir = (self.volume_path / stem / f"{stem}_results"
                           / "integrated_annotation")
                ann_dir.mkdir(parents=True, exist_ok=True)
                save_path = ann_dir / "annotations_test.csv"
                save_annotations(pts, save_path)
                total += len(pts)
                print(f"  Saved {len(pts)} to {save_path}")
            print(f"Total: {total} annotations across {len(self.grid_annotations)} timepoints")
            self._save_lattice()
        else:
            if not hasattr(self, 'points_layer') or len(self.points_layer.data) == 0:
                print("No annotations to save")
                return
            save_path = (Path(self.annotations_path) if self.annotations_path
                         else self.volume_path.with_suffix('.csv'))
            save_annotations(self.points_layer.data, save_path)

    def run(self):
        napari.run()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="napari_worm: 3D Cell Annotation Tool for C. elegans")
    parser.add_argument("volume", help="TIFF file or directory of TIFFs")
    parser.add_argument("--annotations", "-a", default=None,
                        help="Existing annotations CSV (optional)")
    parser.add_argument("--no-grid", action="store_true",
                        help="Use dask 4D slider instead of dual window mode")
    parser.add_argument("--start", "-s", type=int, default=0,
                        help="Starting timepoint index (default: 0)")
    args = parser.parse_args()

    WormAnnotator(args.volume, args.annotations,
                  grid_mode=not args.no_grid, start_t=args.start).run()


if __name__ == "__main__":
    main()
