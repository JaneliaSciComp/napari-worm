"""
napari_worm: 3D Cell Annotation Tool for C. elegans

Usage:
    python napari_worm.py /path/to/volume.tif                    # single file
    python napari_worm.py /path/to/directory/                     # dual view (grid mode)
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
from qtpy.QtWidgets import QWidget, QHBoxLayout, QLabel, QSpinBox


def load_volume(path: str | Path) -> np.ndarray:
    """Load a 3D TIFF volume."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Volume not found: {path}")

    data = tifffile.imread(str(path))
    print(f"Loaded volume: {path.name}")
    print(f"  Shape: {data.shape}, dtype: {data.dtype}")
    return data


def _numeric_sort_key(path: Path) -> list:
    """Sort key that handles numbers naturally: Decon_reg_0, 1, 2, ..., 10, 11."""
    import re
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', path.name)]


def scan_time_series(directory: str | Path) -> list[Path]:
    """Find all TIFF files in a directory, sorted numerically."""
    directory = Path(directory)
    filenames = sorted(directory.glob("*.tif"), key=_numeric_sort_key)
    if not filenames:
        raise FileNotFoundError(f"No .tif files found in {directory}")
    print(f"Found {len(filenames)} timepoints in {directory.name}")
    return filenames


def load_time_series_dask(filenames: list[Path]) -> da.Array:
    """Lazily load TIFF files as a 4D dask array (T, Z, Y, X).

    Only the currently viewed timepoint is read from disk.
    """
    sample = tifffile.imread(str(filenames[0]))
    print(f"  Each volume: shape={sample.shape}, dtype={sample.dtype}")

    lazy_imread = delayed(tifffile.imread)
    lazy_arrays = [
        da.from_delayed(
            lazy_imread(str(fn)),
            shape=sample.shape,
            dtype=sample.dtype,
        )
        for fn in filenames
    ]
    stack = da.stack(lazy_arrays, axis=0)
    print(f"  Time series shape: {stack.shape} (T, Z, Y, X)")
    return stack


def sample_ray(data: np.ndarray, start: np.ndarray, end: np.ndarray, n_samples: int | None = None):
    """Sample intensity values along a ray through the volume.

    Args:
        data: 3D volume array
        start: Starting point (z, y, x)
        end: Ending point (z, y, x)
        n_samples: Number of samples (default: ray length in voxels)

    Returns:
        positions: (N, 3) array of sample positions
        values: (N,) array of intensity values
    """
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


def find_peak_along_ray(data: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Find position of maximum intensity along a ray.

    Args:
        data: 3D volume array
        start: Ray start point (z, y, x)
        end: Ray end point (z, y, x)

    Returns:
        Position of peak intensity (z, y, x)
    """
    positions, values = sample_ray(data, start, end)

    if len(values) == 0 or np.max(values) == 0:
        # No valid samples, return midpoint
        return (start + end) / 2

    peak_idx = np.argmax(values)
    return positions[peak_idx]


def save_annotations(points: np.ndarray, filepath: str | Path):
    """Save annotation points to CSV.

    Args:
        points: (N, 3) array of points in (z, y, x) order
        filepath: Output CSV path
    """
    filepath = Path(filepath)
    df = pd.DataFrame(points, columns=['z', 'y', 'x'])
    df.to_csv(filepath, index=False)
    print(f"Saved {len(points)} annotations to {filepath}")


def load_annotations(filepath: str | Path) -> tuple[np.ndarray, list[str] | None]:
    """Load annotation points from CSV.

    Supports two formats:
    1. Simple: z,y,x columns
    2. MIPAV: name,x_voxels,y_voxels,z_voxels,R,G,B

    Args:
        filepath: Input CSV path

    Returns:
        points: (N, 3) array of points in (z, y, x) order
        names: List of cell names (if MIPAV format), or None
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Annotations not found: {filepath}")

    df = pd.read_csv(filepath)

    # Detect format
    if 'x_voxels' in df.columns:
        # MIPAV format: name,x_voxels,y_voxels,z_voxels,R,G,B
        points = df[['z_voxels', 'y_voxels', 'x_voxels']].values
        names = df['name'].tolist() if 'name' in df.columns else None
        print(f"Loaded {len(points)} MIPAV annotations from {filepath}")
    else:
        # Simple format: z,y,x
        points = df[['z', 'y', 'x']].values
        names = None
        print(f"Loaded {len(points)} annotations from {filepath}")

    return points, names


class WormAnnotator:
    """Main annotation tool class."""

    def __init__(self, volume_path: str | Path, annotations_path: str | Path | None = None,
                 grid_mode: bool = True, start_t: int = 0):
        self.volume_path = Path(volume_path)
        self.annotations_path = annotations_path
        self.is_time_series = self.volume_path.is_dir()
        self.use_grid = grid_mode and self.is_time_series
        self.start_t = start_t

        # Create viewer
        self.viewer = napari.Viewer(ndisplay=3)

        if self.use_grid:
            self._init_grid_mode()
        elif self.is_time_series:
            self._init_dask_mode()
        else:
            self._init_single_mode()

        # Load existing annotations if provided
        self.cell_names = None
        if annotations_path and Path(annotations_path).exists():
            points, names = load_annotations(annotations_path)
            if not self.use_grid:
                self.points_layer.data = points
            self.cell_names = names

        # Bind save key
        self.viewer.bind_key('s', self._save_annotations)

        # Print instructions
        print("\n--- napari_worm ---")
        print("Controls:")
        print("  Ctrl+Click  : Place annotation at peak intensity")
        print("  3D rotate   : Click and drag")
        print("  Zoom        : Scroll wheel")
        print("  Delete      : Select point, press Delete key")
        print("  Save        : Press 'S' to save annotations")
        if self.use_grid:
            print("  Right / ]   : Next timepoint pair")
            print("  Left  / [   : Previous timepoint pair")
            print("  Ctrl+Z      : Undo last annotation (any side)")
        print("-------------------\n")

    # ---- Single file mode ----

    def _init_single_mode(self):
        """Load a single TIFF volume."""
        self.data = load_volume(self.volume_path)
        self.image_layer = self.viewer.add_image(
            self.data, name='Volume', colormap='gray', rendering='mip',
        )
        self.points_layer = self.viewer.add_points(
            ndim=3, name='Annotations', size=5, face_color='yellow',
        )
        self.image_layer.mouse_drag_callbacks.append(self._on_click_single)

    def _on_click_single(self, layer, event):
        """Ctrl+Click handler for single volume mode."""
        if 'Control' not in event.modifiers:
            return
        if self.viewer.dims.ndisplay != 3:
            print("Switch to 3D view for annotation (click '3D' button)")
            return

        near_point, far_point = layer.get_ray_intersections(
            event.position, event.view_direction, event.dims_displayed,
        )
        if near_point is None or far_point is None:
            return

        peak_pos = find_peak_along_ray(self.data, near_point, far_point)
        self.points_layer.add(peak_pos)
        print(f"Added annotation at z={peak_pos[0]:.1f}, y={peak_pos[1]:.1f}, x={peak_pos[2]:.1f}")

    # ---- Dask 4D slider mode (--no-grid) ----

    def _init_dask_mode(self):
        """Load time series as a 4D dask array with time slider."""
        self.tiff_files = scan_time_series(self.volume_path)
        self.data = load_time_series_dask(self.tiff_files)

        sample_slice = self.data[0].compute()
        self.image_layer = self.viewer.add_image(
            self.data, name='Volume', colormap='gray', rendering='mip',
            contrast_limits=[float(sample_slice.min()), float(sample_slice.max())],
            multiscale=False,
        )
        self.points_layer = self.viewer.add_points(
            ndim=4, name='Annotations', size=5, face_color='yellow',
        )
        self.image_layer.mouse_drag_callbacks.append(self._on_click_dask)

    def _on_click_dask(self, layer, event):
        """Ctrl+Click handler for dask 4D mode."""
        if 'Control' not in event.modifiers:
            return
        if self.viewer.dims.ndisplay != 3:
            print("Switch to 3D view for annotation (click '3D' button)")
            return

        near_point, far_point = layer.get_ray_intersections(
            event.position, event.view_direction, event.dims_displayed,
        )
        if near_point is None or far_point is None:
            return

        t = int(self.viewer.dims.current_step[0])
        volume_3d = np.asarray(self.data[t])
        peak_pos = find_peak_along_ray(volume_3d, near_point[-3:], far_point[-3:])
        peak_pos = np.concatenate([[t], peak_pos])

        self.points_layer.add(peak_pos)
        coords = peak_pos[-3:]
        print(f"Added annotation at z={coords[0]:.1f}, y={coords[1]:.1f}, x={coords[2]:.1f}")

    # ---- Grid mode (default for directories) ----

    def _init_grid_mode(self):
        """Load two timepoints side by side in grid mode."""
        self.tiff_files = scan_time_series(self.volume_path)
        max_t = len(self.tiff_files) - 1
        start = min(self.start_t, max_t - 1)
        # {timepoint_index: (N, 3) array of annotations}
        self.grid_annotations: dict[int, np.ndarray] = {}
        # Global undo stack: list of (timepoint, points_layer) in order of actions
        self.undo_stack: list[tuple[int, object]] = []

        # Load first pair and get contrast limits
        first_vol = load_volume(self.tiff_files[start])
        self.contrast_limits = [float(first_vol.min()), float(first_vol.max())]

        # Store references: index 0 = left (t), index 1 = right (t+1)
        self.grid_image_layers: list = []
        self.grid_points_layers: list = []
        self.grid_timepoints: list[int] = []

        # Create navigation widget with two spinboxes
        self._nav_updating = False  # flag to prevent feedback loops
        nav_widget = QWidget()
        layout = QHBoxLayout(nav_widget)
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

        self.viewer.window.add_dock_widget(nav_widget, name='Navigation', area='bottom')

        self._load_grid_pair(start, min(start + 1, max_t))

        # Enable grid mode
        self.viewer.grid.enabled = True
        self.viewer.grid.stride = 2  # 2 layers per grid cell (image + points)

        # Key bindings for navigation
        self.viewer.bind_key('Right', self._grid_next)
        self.viewer.bind_key(']', self._grid_next)
        self.viewer.bind_key('Left', self._grid_prev)
        self.viewer.bind_key('[', self._grid_prev)
        self.viewer.bind_key('Control-z', self._undo_last_point)

    def _load_grid_pair(self, t_left: int, t_right: int):
        """Load two timepoints into the grid panels.

        Napari grid with stride=2: bottom of layer list = left cell.
        So we add left side first, then right side.
        grid_image_layers/grid_points_layers are indexed [0=left, 1=right].
        """
        # Save current annotations before switching
        self._save_grid_annotations_to_cache()

        # Remove old layers
        for layer in self.grid_image_layers + self.grid_points_layers:
            if layer in self.viewer.layers:
                self.viewer.layers.remove(layer)
        self.grid_image_layers.clear()
        self.grid_points_layers.clear()

        self.grid_timepoints = [t_left, t_right]

        # Add left first, then right
        # Bottom of layer list = left grid cell in napari
        image_layers_tmp = [None, None]
        points_layers_tmp = [None, None]

        for side, ti in [(0, t_left), (1, t_right)]:
            vol = load_volume(self.tiff_files[ti])
            side_label = "left" if side == 0 else "right"
            img_layer = self.viewer.add_image(
                vol, name=f'Volume t={ti} ({side_label})', colormap='gray',
                rendering='mip', contrast_limits=self.contrast_limits,
                multiscale=False,
            )
            pts_layer = self.viewer.add_points(
                ndim=3, name=f'Annotations t={ti} ({side_label})', size=5,
                face_color='yellow',
            )

            # Restore cached annotations for this timepoint
            if ti in self.grid_annotations and len(self.grid_annotations[ti]) > 0:
                pts_layer.data = self.grid_annotations[ti]

            # Click handler: any click sets active side + selects image layer;
            # Ctrl+Click also adds annotation.
            # Registered on BOTH image and points layers so clicking anywhere
            # on either side works without needing to select in the sidebar.
            def make_click_handler(volume_data, side_idx, timepoint, img_layer_ref, points_layer):
                def handler(layer, event):
                    if 'Control' in event.modifiers:
                        self._on_click_grid(
                            img_layer_ref, event, volume_data, side_idx, timepoint, points_layer,
                        )
                return handler

            click_handler = make_click_handler(vol, side, ti, img_layer, pts_layer)
            img_layer.mouse_drag_callbacks.append(click_handler)
            pts_layer.mouse_drag_callbacks.append(click_handler)

            image_layers_tmp[side] = img_layer
            points_layers_tmp[side] = pts_layer

        self.grid_image_layers = image_layers_tmp
        self.grid_points_layers = points_layers_tmp

        # Select the left image layer so layer controls show up
        self.viewer.layers.selection.clear()
        self.viewer.layers.selection.add(self.grid_image_layers[0])

        # Update spinboxes (without triggering reload)
        self._nav_updating = True
        self._left_spin.setValue(t_left)
        self._right_spin.setValue(t_right)
        self._nav_updating = False

        # Update window title and text overlay with current timepoints
        label = f"t={t_left}  |  t={t_right}   (of {len(self.tiff_files) - 1})"
        self.viewer.title = f"napari_worm — {label}"
        self.viewer.text_overlay.visible = True
        self.viewer.text_overlay.text = label
        self.viewer.text_overlay.font_size = 20
        print(f"Showing timepoints t={t_left} (left) and t={t_right} (right) of {len(self.tiff_files) - 1}")

    def _save_grid_annotations_to_cache(self):
        """Cache annotations from current grid points layers."""
        for side, pts_layer in enumerate(self.grid_points_layers):
            if pts_layer is not None and side < len(self.grid_timepoints):
                ti = self.grid_timepoints[side]
                if len(pts_layer.data) > 0:
                    self.grid_annotations[ti] = pts_layer.data.copy()

    def _on_spinbox_changed(self, _value):
        """Handle spinbox value changes — reload the grid pair."""
        if self._nav_updating:
            return
        t_left = self._left_spin.value()
        t_right = self._right_spin.value()
        self._load_grid_pair(t_left, t_right)

    def _grid_next(self, viewer):
        """Advance both timepoints by 1."""
        t_left, t_right = self.grid_timepoints
        max_t = len(self.tiff_files) - 1
        if t_right >= max_t:
            print("Already at last timepoint")
            return
        self._load_grid_pair(t_left + 1, t_right + 1)

    def _grid_prev(self, viewer):
        """Go back both timepoints by 1."""
        t_left, t_right = self.grid_timepoints
        if t_left <= 0:
            print("Already at first timepoint")
            return
        self._load_grid_pair(t_left - 1, t_right - 1)

    def _on_click_grid(self, layer, event, volume_data, side_idx, timepoint, points_layer):
        """Ctrl+Click handler for grid mode — adds annotation on the clicked side."""
        if self.viewer.dims.ndisplay != 3:
            print("Switch to 3D view for annotation (click '3D' button)")
            return

        near_point, far_point = layer.get_ray_intersections(
            event.position, event.view_direction, event.dims_displayed,
        )
        if near_point is None or far_point is None:
            return

        peak_pos = find_peak_along_ray(volume_data, near_point, far_point)
        points_layer.add(peak_pos)
        self.undo_stack.append((timepoint, points_layer))
        side_label = "left" if side_idx == 0 else "right"
        print(f"[t={timepoint} {side_label}] Added annotation at z={peak_pos[0]:.1f}, y={peak_pos[1]:.1f}, x={peak_pos[2]:.1f}")

    def _undo_last_point(self, viewer):
        """Remove the last added annotation (Ctrl+Z). Works globally across sides."""
        if not self.undo_stack:
            print("Nothing to undo")
            return

        timepoint, pts_layer = self.undo_stack.pop()

        # If the points layer is still in the viewer (current pair), remove directly
        if pts_layer in self.viewer.layers and len(pts_layer.data) > 0:
            pts_layer.data = pts_layer.data[:-1]
            print(f"[t={timepoint}] Undid last annotation")
        # If we navigated away, undo from the cached annotations
        elif timepoint in self.grid_annotations and len(self.grid_annotations[timepoint]) > 0:
            self.grid_annotations[timepoint] = self.grid_annotations[timepoint][:-1]
            # Also update currently displayed layer if this timepoint is showing
            for side, ti in enumerate(self.grid_timepoints):
                if ti == timepoint and side < len(self.grid_points_layers):
                    displayed_pts = self.grid_points_layers[side]
                    if displayed_pts in self.viewer.layers:
                        if len(self.grid_annotations[timepoint]) > 0:
                            displayed_pts.data = self.grid_annotations[timepoint].copy()
                        else:
                            displayed_pts.data = np.empty((0, 3))
            print(f"[t={timepoint}] Undid last annotation (cached)")
        else:
            print(f"[t={timepoint}] Nothing to undo")

    # ---- Save / Run ----

    def _save_annotations(self, viewer):
        """Save annotations to CSV, one file per timepoint (like MIPAV).

        MIPAV layout:
          RegB/Decon_reg_X/Decon_reg_X_results/integrated_annotation/annotations.csv
        We save to the same structure so annotations are compatible.
        """
        if self.use_grid:
            # Cache current annotations first
            self._save_grid_annotations_to_cache()
            if not self.grid_annotations:
                print("No annotations to save")
                return
            total_saved = 0
            for ti, pts in sorted(self.grid_annotations.items()):
                if len(pts) == 0:
                    continue
                # Build MIPAV-compatible path
                tiff_name = self.tiff_files[ti].stem  # e.g. "Decon_reg_100"
                ann_dir = self.volume_path / tiff_name / f"{tiff_name}_results" / "integrated_annotation"
                ann_dir.mkdir(parents=True, exist_ok=True)
                save_path = ann_dir / "annotations_test.csv"
                df = pd.DataFrame(pts, columns=['z', 'y', 'x'])
                df.to_csv(save_path, index=False)
                total_saved += len(pts)
                print(f"  Saved {len(pts)} annotations to {save_path}")
            print(f"Saved {total_saved} total annotations across {len(self.grid_annotations)} timepoints")
        else:
            if not hasattr(self, 'points_layer') or len(self.points_layer.data) == 0:
                print("No annotations to save")
                return
            if self.annotations_path:
                save_path = Path(self.annotations_path)
            else:
                save_path = self.volume_path.with_suffix('.csv')
            save_annotations(self.points_layer.data, save_path)

    def run(self):
        """Start the viewer."""
        napari.run()


def main():
    parser = argparse.ArgumentParser(
        description="napari_worm: 3D Cell Annotation Tool for C. elegans"
    )
    parser.add_argument(
        "volume",
        type=str,
        help="Path to a single TIFF file or a directory of TIFFs (time series)",
    )
    parser.add_argument(
        "--annotations", "-a",
        type=str,
        default=None,
        help="Path to existing annotations CSV (optional)",
    )
    parser.add_argument(
        "--no-grid",
        action="store_true",
        default=False,
        help="Use dask 4D slider instead of grid mode for directories",
    )
    parser.add_argument(
        "--start", "-s",
        type=int,
        default=0,
        help="Starting timepoint index for grid mode (default: 0)",
    )

    args = parser.parse_args()

    annotator = WormAnnotator(args.volume, args.annotations, grid_mode=not args.no_grid,
                              start_t=args.start)
    annotator.run()


if __name__ == "__main__":
    main()
