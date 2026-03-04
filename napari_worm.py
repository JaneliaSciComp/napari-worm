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
from dask import delayed
from qtpy.QtCore import QEvent, QObject, Qt, QTimer
from qtpy.QtWidgets import (
    QHBoxLayout, QLabel, QSpinBox,
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

def save_annotations(points, filepath):
    filepath = Path(filepath)
    pd.DataFrame(points, columns=['z','y','x']).to_csv(filepath, index=False)
    print(f"Saved {len(points)} annotations to {filepath}")


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
            self._dual_window.set_active_side(self._side)
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

        # Move right viewer's docks into the host window; start hidden.
        # Left viewer's docks are already registered with _host.
        self._host.addDockWidget(Qt.LeftDockWidgetArea,
                                 self._qt_right.dockLayerControls)
        self._host.addDockWidget(Qt.LeftDockWidgetArea,
                                 self._qt_right.dockLayerList)
        self._qt_right.dockLayerControls.hide()
        self._qt_right.dockLayerList.hide()

        # Canvas click → switch active dock panel (MIPAV's setActiveRenderer)
        self._filter_left  = _CanvasClickFilter(self, 0)
        self._filter_right = _CanvasClickFilter(self, 1)
        self._qt_left.canvas.native.installEventFilter(self._filter_left)
        self._qt_right.canvas.native.installEventFilter(self._filter_right)

        # Continuous canvas refresh (mirrors MIPAV's startAnimator)
        self._refresh_timer = QTimer(self._host)
        self._refresh_timer.timeout.connect(self._qt_left.canvas.native.update)
        self._refresh_timer.timeout.connect(self._qt_right.canvas.native.update)
        self._refresh_timer.start(33)  # ~30 fps

    # Proxy methods so call-sites in _init_dual_window_mode stay unchanged
    def showMaximized(self):
        self._host.showMaximized()

    def setWindowTitle(self, title: str):
        self._host.setWindowTitle(title)

    def resizeDocks(self, docks, sizes, orientation):
        self._host.resizeDocks(docks, sizes, orientation)

    def set_active_side(self, side: int):
        """Show one viewer's dock panels and hide the other's."""
        self._qt_left.dockLayerControls.setVisible(side == 0)
        self._qt_left.dockLayerList.setVisible(side == 0)
        self._qt_right.dockLayerControls.setVisible(side == 1)
        self._qt_right.dockLayerList.setVisible(side == 1)


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
        print("  Ctrl+Click : annotate at peak intensity")
        print("  Right / ]  : next timepoint pair")
        print("  Left  / [  : previous timepoint pair")
        print("  Ctrl+Z     : undo last annotation")
        print("  S          : save annotations")
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

        # Key bindings on both viewers so shortcuts work regardless of active canvas
        for v in [self.viewer_left, self.viewer_right]:
            v.bind_key('Right',     self._grid_next)
            v.bind_key(']',         self._grid_next)
            v.bind_key('Left',      self._grid_prev)
            v.bind_key('[',         self._grid_prev)
            v.bind_key('Control-z', self._undo_last_point)
            v.bind_key('s',         self._save_annotations)

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

            if ti in self.grid_annotations and len(self.grid_annotations[ti]) > 0:
                pts.data = self.grid_annotations[ti]

            # Each viewer handles only its own canvas — no click-routing needed
            def make_handler(volume_data, side_idx, timepoint, img_ref, pts_ref):
                def handler(layer, event):
                    if 'Control' not in event.modifiers:
                        return
                    self._on_click_dual(img_ref, event, volume_data, side_idx, timepoint, pts_ref)
                return handler

            cb = make_handler(vol, side, ti, img, pts)
            img.mouse_drag_callbacks.append(cb)
            pts.mouse_drag_callbacks.append(cb)

            self.grid_image_layers[side] = img
            self.grid_points_layers[side] = pts

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
                pd.DataFrame(pts, columns=['z','y','x']).to_csv(save_path, index=False)
                total += len(pts)
                print(f"  Saved {len(pts)} to {save_path}")
            print(f"Total: {total} annotations across {len(self.grid_annotations)} timepoints")
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
