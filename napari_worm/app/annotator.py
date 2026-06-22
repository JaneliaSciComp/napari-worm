import re
from pathlib import Path

import napari
import numpy as np
import pandas as pd
from napari.utils.notifications import show_info
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QShortcut, QSplitter, QTableWidgetItem, QVBoxLayout, QWidget,
)

from napari_worm.app.dual_view import DualViewWindow
from napari_worm.geometry.cross_section import (
    _apply_fourier_falloff, _apply_gaussian_falloff, _GAUSSIAN_MODE,
)
from napari_worm.geometry.lattice import (
    _lattice_pair_name, _renumber_lattice_pairs, _SEAM_CELL_SEQUENCE,
)
from napari_worm.geometry.mesh import generate_surface_mesh, generate_wireframe_mesh
from napari_worm.geometry.ray import (
    _find_closest_annotation_point_by_ray, _find_closest_lattice_point_by_ray,
    _find_insertion_index, _pick_ring_vertex, _project_point_on_ray,
    find_nucleus_centroid, find_peak_along_ray,
)
from napari_worm.geometry.spline import (
    _build_cross_section_rings, _smooth_midline_spline,
)
from napari_worm.io.annotations import load_annotations, save_annotations
from napari_worm.io.cross_section import _load_cross_section_csv, _save_cross_section_csv
from napari_worm.io.volume import (
    discover_channels, load_time_series_dask, load_volume, scan_time_series,
)
from napari_worm.model_adapter import _make_celegans_model
from napari_worm.widgets.event_filters import (
    _ArrowKeyFilter, _CanvasClickFilter, _ClipDragFilter, _TableDeleteFilter,
)
from napari_worm.widgets.spinbox import _CommitOnEnterSpinBox


# Register 'voxel' as a pint alias so napari's scale-bar overlay can parse
# it when re-created on subsequent add_image calls (else UndefinedUnitError).
try:
    import pint as _pint
    _ureg = _pint.get_application_registry()
    if 'voxel' not in _ureg:
        _ureg.define('voxel = pixel')
except Exception:
    pass


# ---------------------------------------------------------------------------
# Main annotator class
# ---------------------------------------------------------------------------

class WormAnnotator:
    """Main annotation tool class."""

    def __init__(self, volume_path, annotations_path=None, grid_mode=True, start_t=0,
                 channels=None, downsample=None):
        self.volume_path = Path(volume_path)
        self.annotations_path = annotations_path
        self.is_time_series = self.volume_path.is_dir()
        self.use_grid = grid_mode and self.is_time_series
        self.start_t = start_t
        self.channels_arg = channels
        self.downsample = downsample  # (fZ, fY, fX) or None
        # napari scale for all layers: makes each downsampled voxel display at
        # the correct physical size so annotations overlay the image correctly.
        self.scale = tuple(downsample) if (downsample and any(f > 1 for f in downsample)) else None
        if self.scale:
            print(f"Downsampling Z×{downsample[0]} Y×{downsample[1]} X×{downsample[2]}"
                  f" — axis order detected per TIFF metadata")

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
        print("  Cmd+Z / Ctrl+Z    : undo last annotation/lattice point")
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
        self.data = load_volume(self.volume_path, downsample=self.downsample)
        self.image_layer = self.viewer.add_image(
            self.data, name='Volume', colormap='gray', rendering='mip',
            scale=self.scale)
        self.points_layer = self.viewer.add_points(
            ndim=3, name='Annotations', size=5, face_color='yellow',
            scale=self.scale)
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
        # User-editable annotation names, parallel to grid_annotations. Default
        # is f"A{i+1}" auto-numbered; Ryan groups by first letter (A*, B*, C*).
        # Persisted to CSV `name` column; read back on cold reload.
        self.grid_annotation_names: dict[int, list[str]] = {}
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
        self.annotation_last_placed: dict | None = None

        # Cross-section overrides — per timepoint, per ring-index, center-relative
        # (num_ellipse_pts, 3) offsets. Mirrors MIPAV's relativeCrossSections
        # (LatticeModel.java:4806-4830). Saved per-ring to
        # model_crossSections/latticeCrossSection_<i>.csv — only edited rings
        # get a CSV; others fall back to the default circle.
        self.cross_section_overrides: dict[int, dict[int, np.ndarray]] = {}

        # Cross-section editing mode state (MIPAV editingCrossSections +
        # crossSectionSamples). When True, Cmd+Click+Drag picks a ring vertex
        # and reshapes the ring via Fourier falloff of width `cross_section_n_samples`.
        self.cross_section_edit_mode: bool = False
        self.cross_section_n_samples: int = 8  # MIPAV default: Medium

        # Arbitrary clip plane state — per timepoint, mirrors MIPAV's
        # IntegratedWormData.clipArb + clipArbOn (PlugInDialogVolumeRenderDual.java:273).
        # Key = timepoint. Each side's Clip UI reflects its current timepoint.
        # Normal = rotation @ (0, 0, 1)  (default = +x world axis in napari (z,y,x) order;
        # the actual direction doesn't matter — rotation makes it arbitrary).
        self.clip_state: dict[int, dict] = {}
        # Per-side frame-outline Shapes layers (populated in _load_dual_pair)
        self.clip_frame_layers: list = [None, None]
        # Per-side throttle timers for clip-plane updates during Shift+Drag.
        # Coalesces expensive per-mouse-move layer updates to ~30fps.
        self._clip_throttle_timers: list = [None, None]
        # Ordered list of pair metadata per timepoint:
        # [{name: 'a0', type: 'lattice'}, {name: 'H0', type: 'seam'}, ...]
        # The i-th entry corresponds to i-th L point and i-th R point in the layers
        self.lattice_pair_names: dict[int, list[dict]] = {}  # {timepoint: [pair_info, ...]}
        self.lattice_seam_counter: dict[int, list[str]] = {}  # {timepoint: [used seam names]}
        self.lattice_undo_stack: list = []  # ('L'|'R', timepoint, layer)

        # Preview-mode state (per side) — see _enter_preview_mode.
        self.preview_mode_active: list[bool] = [False, False]
        self.preview_models: list = [None, None]
        self.preview_ap_values: list = [None, None]
        self.preview_extents: list = [None, None]
        self.preview_image_layers: list = [None, None]
        self.preview_points_layers: list = [None, None]
        self.preview_overlay_layers: list = [[], []]   # midline + lattice splines per side
        self.preview_hidden_layers: list = [[], []]
        self.preview_click_callbacks: list = [None, None]

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

        # Shift+Drag for arbitrary clip plane rotation. Qt event filter installed
        # on the canvas BEFORE vispy — consumes the event so vispy's camera
        # rotation is suppressed during plane rotation. Matches MIPAV's behavior
        # where Ctrl+Drag rotates only the plane (VolumeTriPlanarRenderBase.java
        # :1921-1926). Using Shift instead of Ctrl because napari on Mac remaps
        # Cmd↔Ctrl, making 'Control' collide with Cmd+Click annotation.

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

        # Shift+Drag clip plane rotation — Qt-level filters on canvas widgets so
        # vispy's camera rotation is suppressed during plane rotation (see
        # _ClipDragFilter docstring for rationale vs viewer.mouse_drag_callbacks).
        self._clip_drag_l = _ClipDragFilter(self, 0)
        self._clip_drag_r = _ClipDragFilter(self, 1)
        qt_left.canvas.native.installEventFilter(self._clip_drag_l)
        qt_right.canvas.native.installEventFilter(self._clip_drag_r)

        # Eye-aligned clip planes follow viewer.camera.view_direction.
        # Re-apply them on every camera angle change (gated to no-op if
        # no eye-plane is enabled, so cost is one dict lookup per rotate).
        for s, vw in ((0, self.viewer_left), (1, self.viewer_right)):
            vw.camera.events.angles.connect(
                lambda _ev, side=s: self._on_camera_rotated(side))

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

        layout.addSpacing(10)

        back_btn = QPushButton("◀ BACK")
        back_btn.setToolTip(
            "Save current timepoints and step back by 1 ([)")
        back_btn.clicked.connect(lambda: self._grid_prev(self.viewer_left))
        layout.addWidget(back_btn)

        next_btn = QPushButton("NEXT ▶")
        next_btn.setToolTip(
            "Save current timepoints and advance by 1 (])")
        next_btn.clicked.connect(lambda: self._grid_next(self.viewer_left))
        layout.addWidget(next_btn)

        layout.addSpacing(10)

        layout.addWidget(QLabel("Right t="))
        self._right_spin = _CommitOnEnterSpinBox()
        self._right_spin.setRange(0, max_t)
        self._right_spin.setValue(min(start + 1, max_t))
        self._right_spin.setFixedWidth(80)
        self._right_spin.committed.connect(self._on_spinbox_changed)
        layout.addWidget(self._right_spin)

        layout.addStretch()

        save_btn = QPushButton("Save All")
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

    # ------------------------------------------------------------------ #
    # Notification helpers — thin wrappers over napari's native toast       #
    # system so multi-line messages get the compact "^" expand caret.       #
    # Transient uses INFO severity (auto-dismiss); persistent uses WARNING  #
    # severity which napari leaves up until the user dismisses it.          #
    # ------------------------------------------------------------------ #
    def _toast(self, msg: str, ms: int = 2500):
        """Short-lived mode-change notification (napari auto-dismisses)."""
        show_info(msg)

    def _toast_persistent(self, msg: str):
        """Sticky notification for save paths / errors. Uses WARNING severity
        so napari doesn't auto-dismiss — user clicks × or ^ to expand."""
        try:
            from napari.utils.notifications import show_warning
            show_warning(msg)
        except Exception:
            show_info(msg)

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
<tr><td><b>Cmd+Z</b> (Mac) / <b>Ctrl+Z</b> (Win)</td><td>Undo last annotation</td></tr>
<tr><td><b>S</b> / <b>Cmd+S</b></td><td>Save annotations + lattice</td></tr>
<tr><td><b>L</b></td><td>Switch to Lattice mode</td></tr>
<tr><td><b>Delete</b></td><td>Remove selected table row</td></tr>
<tr><td><b>Rename in tables</b></td><td>Double-click the <b>Name</b> column (annotations) or <b>Pair</b> column (lattice) to rename. Non-blank and unique within the timepoint. Cmd+Z (macOS) / Ctrl+Z (Win) undoes. Custom lattice names persist across add/delete renumber.</td></tr>
<tr><td><b>New annotations</b> button</td><td>Delete all annotation points for this timepoint <i>and</i> the corresponding <code>annotations.csv</code> on disk (confirm dialog)</td></tr>
<tr><td><b>New lattice</b> button</td><td>Delete all lattice points + ring overrides for this timepoint <i>and</i> the corresponding <code>lattice.csv</code> + <code>latticeCrossSection_*.csv</code> on disk (confirm dialog)</td></tr>
</table>
<p style="font-size: 11px; color: #888; margin-left: 4px;">
Saves write to <code>integrated_annotation/annotations.csv</code> and
<code>lattice_final/lattice.csv</code> (MIPAV-canonical names).
</p>

<h3>Lattice Mode (press L)</h3>
<table cellpadding="4">
<tr><td><b>Cmd+Click</b></td><td>Place lattice point (alternates L/R)</td></tr>
<tr><td><b>Cmd+Shift+Click</b></td><td>Place seam cell (L/R pair)</td></tr>
<tr><td><b>Cmd+Click on point</b></td><td>Select point, then drag to move</td></tr>
<tr><td><b>Cmd+Click on curve</b></td><td>Insert new pair between existing</td></tr>
<tr><td><b>Arrow keys</b></td><td>Nudge selected point by 1 voxel; <b>Shift+Arrow</b> = 5 voxels</td></tr>
<tr><td><b>Cmd+Z</b> (Mac) / <b>Ctrl+Z</b> (Win)</td><td>Undo last lattice operation</td></tr>
<tr><td><b>D</b></td><td>Done — save and exit lattice mode</td></tr>
<tr><td><b>L</b></td><td>Back to annotation mode</td></tr>
</table>

<h3>Navigation</h3>
<table cellpadding="4">
<tr><td><b>NEXT ▶</b> / <b>Right</b> / <b>]</b></td><td>Auto-save current + step forward 1 (old right becomes new left)</td></tr>
<tr><td><b>◀ BACK</b> / <b>Left</b> / <b>[</b></td><td>Auto-save current + step back 1</td></tr>
<tr><td><b>Spinboxes</b></td><td>Type number + Enter to jump (no auto-save)</td></tr>
<tr><td><b>Save All</b> / <b>S</b> / <b>Cmd+S</b></td><td>Save all visited timepoints to disk</td></tr>
</table>

<h3>Visualization</h3>
<table cellpadding="4">
<tr><td><b>W</b></td><td>Toggle wireframe mesh (one anchor ring per lattice pair + 32 longitudinal splines)</td></tr>
<tr><td><b>Shift+W</b></td><td>Toggle surface mesh (solid tube, quad-strip between rings, head/tail caps)</td></tr>
<tr><td><b>Eye icon</b></td><td>Toggle channel visibility</td></tr>
<tr><td><b>Click layer</b></td><td>Switch histogram to that channel</td></tr>
</table>
<p style="font-size: 11px; color: #888; margin-left: 4px;">
Wireframe rings are lattice-aligned: ring <i>i</i> sits at lattice pair <i>i</i>, matching MIPAV.
Any cross-section edits save per-ring to
<code>&lt;results&gt;/model_crossSections/latticeCrossSection_&lt;i&gt;.csv</code>
and reload automatically on next launch.
</p>

<h3>Ring editing (Rings tab)</h3>
<table cellpadding="4">
<tr><td><b>Enable ring editing</b></td><td>Turn on Cmd+Click+Drag on wireframe ring vertices</td></tr>
<tr><td><b>Single (32)</b></td><td>Only the clicked vertex moves (MIPAV "direct")</td></tr>
<tr><td><b>Narrow (16) / Medium (8) / Wide (4)</b></td><td>Fourier-kernel bulge: wider = more vertices affected</td></tr>
<tr><td><b>Gaussian</b></td><td>5-vertex weighted bulge (clicked + ±1 at 0.37× + ±2 at 0.14×); smoother than narrow Fourier kernels</td></tr>
<tr><td><b>Reset rings</b></td><td>Wipe overrides for both currently-displayed timepoints</td></tr>
<tr><td><b>Cmd+Click+Drag</b></td><td>On a wireframe ring vertex: reshape that cross-section radially</td></tr>
</table>
<p style="font-size: 11px; color: #888; margin-left: 4px;">
Enabling "Enable ring editing" auto-enables Lattice mode (L) and the wireframe (W)
since both are required. Each drag moves all 32 vertices of the selected ring
purely radially from its center — matches MIPAV's editingCrossSections behavior
(<code>LatticeModel.java:8492-8665</code>). While "Enable ring editing" is on,
Cmd+Click is captured for ring editing only (a click that misses every vertex
does nothing — matches MIPAV, which ignores lattice add calls while in
editingCrossSections mode; see <code>LatticeModel.java:1001-1004</code>).
Uncheck to return to lattice placement. Save with <b>S</b> or <b>Save All</b>.
</p>

<h3>Clipping (Clip tab)</h3>
<table cellpadding="4">
<tr><td><b>Enable arbitrary clip plane</b></td><td>Cut the volume with a user-rotatable plane</td></tr>
<tr><td><b>Show plane frame</b></td><td>Display red rectangle at the plane position</td></tr>
<tr><td><b>Position slider</b></td><td>Move plane along its normal (from volume center)</td></tr>
<tr><td><b>Thickness slider</b></td><td>0 = single plane; &gt;0 = slab (show only region between two parallel planes)</td></tr>
<tr><td><b>Shift+Drag on canvas</b></td><td>Rotate the arbitrary plane (only when its frame is shown)</td></tr>
<tr><td><b>Reset orientation</b></td><td>Restore X-axis default at volume center; also zeros eye-plane positions</td></tr>
<tr><td><b>Enable Near clip plane</b></td><td>Camera-aligned plane that clips between observer and plane (MIPAV CLIP_EYE)</td></tr>
<tr><td><b>Enable Far clip plane</b></td><td>Camera-aligned plane that clips behind the plane (MIPAV CLIP_EYE_INV)</td></tr>
<tr><td><b>Near/Far Pos sliders</b></td><td>Independent positions along the camera view direction</td></tr>
</table>
<p style="font-size: 11px; color: #888; margin-left: 4px;">
Eye-aligned planes auto-follow the camera as you rotate the view; their normals
re-sync continuously. Combine Near + Far to get a camera-aligned slab. Clip state
is remembered per timepoint — each timepoint can have its own plane configuration.
</p>

<h3>Preview (Preview tab)</h3>
<table cellpadding="4">
<tr><td><b>Enable straightened view</b></td><td>Replaces the twisted volume with a straightened tube; Z=AP, Y=DV, X=ML</td></tr>
<tr><td><b>Cmd+Click</b></td><td>Place an annotation in straightened space; retwisted to twisted volume so it persists when preview toggles off</td></tr>
</table>
<p style="font-size: 11px; color: #888; margin-left: 4px;">
Requires lattice with ≥3 pairs. Lattice / Wireframe / Mesh / Ring-edit controls
are greyed out while preview is on (read-mostly). When cross-section ring overrides
exist for the timepoint, each AP slice is sampled out to its own per-ring outer
radius (max distance of the 32 cross-section vertices from the midline) — narrow
regions stay tight, wide regions get more room. Auto-exits when timepoint changes.
Powered by Caroline Malin-Mayor's <code>celegans_model</code> package — straighten
+ retwist run via <code>PythonCelegansModel</code>.
</p>

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

        # Auto-exit preview mode — its straightened volume + retwist frames
        # are tied to the current timepoint's lattice and would be stale.
        for side in (0, 1):
            if self.preview_mode_active[side]:
                self._reset_preview_checkbox(side)
                self._exit_preview_mode(side)

        self.viewer_left.layers.clear()
        self.viewer_right.layers.clear()
        self.grid_timepoints = [t_left, t_right]
        self.annotation_last_placed = None
        self.lattice_last_placed = None

        # Repopulate in-memory caches from disk for any timepoint we haven't
        # touched this session. Matches MIPAV's behavior of picking up previously
        # saved annotation/lattice CSVs on re-open.
        for ti in (t_left, t_right):
            self._load_annotations_from_disk(ti)
            self._load_lattice_from_disk(ti)

        for side, (viewer_ref, ti) in enumerate(
                [(self.viewer_left, t_left), (self.viewer_right, t_right)]):

            # Load all channels as separate image layers with additive blending
            channel_img_layers = []
            all_volumes = []
            for ch_idx, (ch_files, (_, ch_name, ch_cmap)) in enumerate(
                    zip(self.channel_tiff_files, self.channels)):
                vol = load_volume(ch_files[ti], downsample=self.downsample)
                all_volumes.append(vol)
                # Single channel → gray, no blending change; multi → colored + additive
                if self.multi_channel:
                    cmap = ch_cmap
                    blending = 'additive'
                    layer_name = ch_name
                else:
                    cmap = 'gray'
                    blending = 'translucent'
                    layer_name = 'Volume'
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
                    blending=blending, multiscale=False,
                    scale=self.scale)
                channel_img_layers.append(img)

            # Surface layer: sits below all interactive layers so it never steals clicks
            empty_verts = np.zeros((3, 3))
            empty_faces = np.array([[0, 1, 2]])
            empty_vals  = np.zeros(3)
            surface = viewer_ref.add_surface(
                (empty_verts, empty_faces, empty_vals),
                name='Surface', shading='smooth',
                colormap='turbo', opacity=0.7,
                scale=self.scale)
            surface.interactive = False
            surface.visible = self.surface_visible
            pts = viewer_ref.add_points(
                ndim=3, name='Annotations', size=5, face_color='yellow',
                scale=self.scale)
            pts.mode = 'pan_zoom'  # prevent napari's native add-on-click

            if ti in self.grid_annotations and len(self.grid_annotations[ti]) > 0:
                pts.data = self.grid_annotations[ti]

            # Lattice layers: left (cyan squares), right (magenta squares), lines (yellow)
            lat_l = viewer_ref.add_points(
                ndim=3, name='Lattice Left',  size=7,
                face_color='cyan',    symbol='square',
                scale=self.scale)
            lat_r = viewer_ref.add_points(
                ndim=3, name='Lattice Right', size=7,
                face_color='magenta', symbol='square',
                scale=self.scale)
            lat_lines = viewer_ref.add_shapes(
                ndim=3, name='Lattice Lines',
                edge_color='yellow', edge_width=1, face_color='transparent',
                scale=self.scale)
            lat_mid = viewer_ref.add_shapes(
                ndim=3, name='Lattice Mid',
                edge_color='red', edge_width=1, face_color='transparent',
                scale=self.scale)
            lat_left_curve = viewer_ref.add_shapes(
                ndim=3, name='Lattice Left Curve',
                edge_color='magenta', edge_width=1, face_color='transparent',
                scale=self.scale)
            lat_right_curve = viewer_ref.add_shapes(
                ndim=3, name='Lattice Right Curve',
                edge_color='green', edge_width=1, face_color='transparent',
                scale=self.scale)
            wireframe = viewer_ref.add_shapes(
                ndim=3, name='Wireframe',
                edge_color='white', edge_width=0.5, face_color='transparent',
                scale=self.scale)
            wireframe.visible = self.wireframe_visible
            clip_frame = viewer_ref.add_shapes(
                ndim=3, name='Clip Frame',
                edge_color='red', edge_width=1.0, face_color='transparent',
                scale=self.scale)
            clip_frame.interactive = False
            try:
                clip_frame.current_shape_type = 'polygon'
            except Exception:
                pass
            self.clip_frame_layers[side] = clip_frame
            # Lock lattice layers to pan_zoom — our handler does all adding
            for lyr in (lat_l, lat_r, lat_lines, lat_mid,
                        lat_left_curve, lat_right_curve, wireframe, clip_frame):
                lyr.mode = 'pan_zoom'
            # Lock pts too if already in lattice mode (e.g. after timepoint change)
            if self.lattice_mode:
                pts.mode = 'pan_zoom'

            cached_lat = self.lattice_annotations.get(ti, {})
            if len(cached_lat.get('left',  [])) > 0:
                lat_l.data = cached_lat['left']
            if len(cached_lat.get('right', [])) > 0:
                lat_r.data = cached_lat['right']
            # Pull any existing cross-section overrides from disk (only once —
            # if already loaded this session, keep the in-memory state).
            if ti not in self.cross_section_overrides:
                self._load_cross_sections_for_timepoint(ti)
            # Register lattice/wireframe layers *before* the visual rebuild —
            # _update_lattice_visuals looks up the active side via
            # self.lattice_left_layers, so the new layers must be visible there
            # first, otherwise the wireframe/surface rebuild silently skips.
            self.lattice_left_layers[side]  = lat_l
            self.lattice_right_layers[side] = lat_r
            self.wireframe_layers[side]     = wireframe
            self.surface_layers[side]       = surface
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
                        yield from self._on_click_dual(
                            img_ref, event, side_idx, timepoint, pts_ref)
                        return

                    near, far = img_ref.get_ray_intersections(
                        event.position, event.view_direction, event.dims_displayed)
                    if near is None or far is None:
                        return

                    # --- Cross-section ring editing (MIPAV editingCrossSections) ---
                    # When Edit-tab's "Enable ring editing" is on, treat it as a
                    # pure mode: Cmd+Click only edits rings and a miss is
                    # swallowed (no accidental lattice points). Uncheck the box
                    # to return to normal lattice placement.
                    if self.cross_section_edit_mode:
                        l_tmp = np.asarray(lat_l_ref.data)
                        r_tmp = np.asarray(lat_r_ref.data)
                        picked = None
                        if min(len(l_tmp), len(r_tmp)) >= 3:
                            overrides = self.cross_section_overrides.get(timepoint)
                            rings, centers = _build_cross_section_rings(
                                l_tmp, r_tmp, 32, 0, True, overrides=overrides)
                            if rings:
                                picked = _pick_ring_vertex(near, far, rings, threshold=12.0)
                        if picked is not None:
                            yield from self._handle_cross_section_drag(
                                event, img_ref, side_idx, timepoint,
                                lat_l_ref, lat_r_ref, lat_lines_ref, lat_mid_ref,
                                lat_lc_ref, lat_rc_ref,
                                pre_picked=picked, rings=rings, centers=centers)
                        else:
                            self._toast("Ring edit miss — click closer to a wireframe vertex "
                                        "(or uncheck Enable ring editing to place lattice).")
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
                        # Final rebuild so the wireframe/surface reflect the
                        # exact released lattice position (not the last
                        # mouse_move frame, which can lag by a throttle tick).
                        self._update_lattice_visuals(
                            lat_l_ref, lat_r_ref, lat_lines_ref,
                            lat_mid_ref, lat_lc_ref, lat_rc_ref)
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

        # Restore clip-plane state per-timepoint (matches MIPAV's updateClipPanel)
        for side_idx in range(2):
            self._refresh_clip_ui(side_idx)
            self._apply_clip_planes(side_idx)
            self._update_clip_frame(side_idx)

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

    # ──────────────────────── Arbitrary clip plane ────────────────────────
    # Mirrors MIPAV's CLIP_A (JPanelClip_WM.java + VolumeTriPlanarRenderBase.java).
    # State stored per timepoint (like MIPAV's IntegratedWormData.clipArb).

    @staticmethod
    def _default_clip_state() -> dict:
        return {
            'enabled': False,
            'rotation': np.eye(3),   # 3×3 matrix applied to default normal (0,0,1)
            'position': 0.0,         # offset along normal from volume center, voxels
            'thickness': 0.0,        # 0 = single plane; >0 = slab (pair of planes)
            'frame_visible': False,
            # Eye-aligned dual planes (camera-following). Mirrors MIPAV's
            # CLIP_EYE + CLIP_EYE_INV (JPanelClip_WM.java:50-51): two planes
            # whose normals follow viewer.camera.view_direction, with
            # independent positions. Near = clips between observer and plane;
            # Far = clips behind plane. Together = camera-aligned slab.
            'eye_near_enabled': False,
            'eye_near_position': 0.0,
            'eye_near_frame': False,
            'eye_far_enabled': False,
            'eye_far_position': 0.0,
            'eye_far_frame': False,
        }

    def _get_clip_state(self, side: int) -> dict:
        ti = self.grid_timepoints[side]
        if ti not in self.clip_state:
            self.clip_state[ti] = self._default_clip_state()
        return self.clip_state[ti]

    def _volume_shape(self, side: int):
        img_layers = self.grid_image_layers[side]
        if img_layers is None:
            return None
        layer = img_layers[0] if isinstance(img_layers, list) else img_layers
        return layer.data.shape   # (z, y, x)

    def _compute_clip_plane(self, state: dict, shape) -> tuple:
        """Return (position, normal) in napari (z, y, x) coordinates.

        Default normal is (0, 0, 1) pointing along +x; state['rotation']
        rotates it to any orientation. Position is measured along the normal
        from the volume center.
        """
        default_normal = np.array([0.0, 0.0, 1.0])
        normal = state['rotation'] @ default_normal
        normal = normal / np.linalg.norm(normal)
        center = np.array(shape) / 2.0
        position = center + normal * state['position']
        return position, normal

    def _plane_frame_corners(self, position, normal, shape):
        """Four corners of a square centered at `position`, lying in the plane."""
        size = float(max(shape))  # matches MIPAV m_fMax
        half = size / 2.0
        # Orthonormal basis (right, up) in the plane
        temp = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(normal, temp)) > 0.95:
            temp = np.array([1.0, 0.0, 0.0])
        right = np.cross(normal, temp)
        right = right / np.linalg.norm(right)
        up = np.cross(right, normal)  # unit
        c0 = position - right * half - up * half
        c1 = position + right * half - up * half
        c2 = position + right * half + up * half
        c3 = position - right * half + up * half
        return np.stack([c0, c1, c2, c3])

    def _apply_clip_planes(self, side: int):
        """Push clip state to all Image and Surface layers on this side's viewer."""
        state = self._get_clip_state(side)
        shape = self._volume_shape(side)
        if shape is None:
            return
        viewer = self.viewer_left if side == 0 else self.viewer_right
        planes = []
        if state['enabled']:
            position, normal = self._compute_clip_plane(state, shape)
            planes.append({'position': tuple(position), 'normal': tuple(normal),
                           'enabled': True})
            if state['thickness'] > 0:
                pos2 = position + normal * state['thickness']
                planes.append({'position': tuple(pos2),
                               'normal': tuple(-normal), 'enabled': True})
        # Eye-aligned planes (camera-following) — MIPAV CLIP_EYE / CLIP_EYE_INV.
        # Independent from the arbitrary plane; can be active simultaneously.
        if state['eye_near_enabled'] or state['eye_far_enabled']:
            view_dir, center = self._eye_plane_basis(viewer, shape)
            if view_dir is not None:
                if state['eye_near_enabled']:
                    p = center + view_dir * state['eye_near_position']
                    planes.append({'position': tuple(p),
                                   'normal': tuple(view_dir), 'enabled': True})
                if state['eye_far_enabled']:
                    p = center + view_dir * state['eye_far_position']
                    planes.append({'position': tuple(p),
                                   'normal': tuple(-view_dir), 'enabled': True})
        for layer in viewer.layers:
            if hasattr(layer, 'experimental_clipping_planes'):
                try:
                    layer.experimental_clipping_planes = planes
                except Exception:
                    pass  # some layer types may reject — silently skip

    @staticmethod
    def _eye_plane_basis(viewer, shape):
        """Return (unit view_direction, volume center) or (None, None) if invalid.

        view_direction is in napari (z, y, x); the eye-aligned plane normals
        are derived from it. Center = shape/2 (volume midpoint).
        """
        try:
            view_dir = np.asarray(viewer.camera.view_direction, dtype=float)
        except Exception:
            return None, None
        norm = np.linalg.norm(view_dir)
        if not np.isfinite(norm) or norm == 0:
            return None, None
        return view_dir / norm, np.array(shape) / 2.0

    def _set_volume_quality(self, side: int, hi: bool):
        """Swap volume ray-cast step size between drag (coarse) and idle (fine).

        Mirrors MIPAV's VolumeTriPlanarRenderBase.java:1933-1934 / 1985-1986
        where fSample_mouseDragged=0.5 and fSample_mouseReleased=1.0 swap
        the volume sample rate. Vispy's VolumeVisual exposes the equivalent
        as `relative_step_size` (bigger = coarser = faster).

        Uses private `canvas.layer_to_visual` from napari to reach the vispy
        node — may break on napari upgrades.
        """
        qt_viewer = self.dual_window._qt_left if side == 0 else self.dual_window._qt_right
        canvas = getattr(qt_viewer, 'canvas', None)
        layer_to_visual = getattr(canvas, 'layer_to_visual', None)
        if layer_to_visual is None:
            return
        step = 0.8 if hi else 2.0  # 0.8 = napari default; 2.0 ≈ 2.5× faster
        viewer = self.viewer_left if side == 0 else self.viewer_right
        for layer in viewer.layers:
            vispy_layer = layer_to_visual.get(layer)
            if vispy_layer is None:
                continue
            node = getattr(vispy_layer, 'node', None)
            if node is not None and hasattr(node, 'relative_step_size'):
                try:
                    node.relative_step_size = step
                except Exception:
                    pass

    def _update_clip_frame(self, side: int):
        """Update the red frame rectangle Shapes layer for this side.

        Uses in-place `data = [...]` assignment (relies on
        current_shape_type='polygon' set at layer creation) to avoid the
        remove_selected+add cycle which rebuilds vispy buffers.
        Draws frames for the arbitrary plane (and its slab partner) plus
        any enabled eye-aligned planes whose "Show frame" is on.
        """
        frame_layer = self.clip_frame_layers[side]
        if frame_layer is None:
            return
        state = self._get_clip_state(side)
        shape = self._volume_shape(side)
        if shape is None:
            return
        show_arb = state['enabled'] and state['frame_visible']
        show_eye_n = state['eye_near_enabled'] and state['eye_near_frame']
        show_eye_f = state['eye_far_enabled'] and state['eye_far_frame']
        if not (show_arb or show_eye_n or show_eye_f):
            if len(frame_layer.data) > 0:
                frame_layer.data = []
            return
        shapes_data = []
        if show_arb:
            position, normal = self._compute_clip_plane(state, shape)
            shapes_data.append(self._plane_frame_corners(position, normal, shape))
            if state['thickness'] > 0:
                pos2 = position + normal * state['thickness']
                shapes_data.append(self._plane_frame_corners(pos2, normal, shape))
        if show_eye_n or show_eye_f:
            viewer = self.viewer_left if side == 0 else self.viewer_right
            view_dir, center = self._eye_plane_basis(viewer, shape)
            if view_dir is not None:
                if show_eye_n:
                    p = center + view_dir * state['eye_near_position']
                    shapes_data.append(self._plane_frame_corners(p, view_dir, shape))
                if show_eye_f:
                    p = center + view_dir * state['eye_far_position']
                    shapes_data.append(self._plane_frame_corners(p, -view_dir, shape))
        frame_layer.data = shapes_data

    def _refresh_clip_ui(self, side: int):
        """Sync the Clip tab controls to the side's current state (no signals)."""
        ctrls = self.dual_window._clip_controls[side]
        if ctrls is None:
            return
        state = self._get_clip_state(side)
        shape = self._volume_shape(side)
        max_extent = float(max(shape)) if shape else 500.0
        for key, w in ctrls.items():
            if hasattr(w, 'blockSignals'):
                w.blockSignals(True)
        ctrls['enable'].setChecked(state['enabled'])
        ctrls['frame'].setChecked(state['frame_visible'])
        ctrls['position'].setRange(-max_extent / 2, max_extent / 2)
        ctrls['position'].setValue(state['position'])
        ctrls['thickness'].setRange(0, max_extent)
        ctrls['thickness'].setValue(state['thickness'])
        # Eye-aligned controls (camera-following dual planes)
        if 'eye_near_enable' in ctrls:
            ctrls['eye_near_enable'].setChecked(state['eye_near_enabled'])
            ctrls['eye_near_frame'].setChecked(state['eye_near_frame'])
            ctrls['eye_near_position'].setRange(-max_extent / 2, max_extent / 2)
            ctrls['eye_near_position'].setValue(state['eye_near_position'])
            ctrls['eye_far_enable'].setChecked(state['eye_far_enabled'])
            ctrls['eye_far_frame'].setChecked(state['eye_far_frame'])
            ctrls['eye_far_position'].setRange(-max_extent / 2, max_extent / 2)
            ctrls['eye_far_position'].setValue(state['eye_far_position'])
        for key, w in ctrls.items():
            if hasattr(w, 'blockSignals'):
                w.blockSignals(False)

    def _on_clip_control_changed(self, side: int, key: str, value):
        state = self._get_clip_state(side)
        state[key] = value
        self._apply_clip_planes(side)
        self._update_clip_frame(side)

    def _on_camera_rotated(self, side: int):
        """Re-apply eye-aligned planes when the camera rotates.

        Cheap no-op if no eye plane is active. Calls the flush path
        directly (not the throttle) so the frame stays camera-perpendicular
        during the drag instead of trailing by the throttle interval —
        the throttle existed for Shift+Drag plane rotation, where we
        coalesce 100+ mouse-moves per second; camera angles events fire
        at a sane rate already, so the throttle just introduces visible
        lag here.
        """
        state = self._get_clip_state(side)
        if (state['eye_near_enabled'] or state['eye_far_enabled']
                or state['eye_near_frame'] or state['eye_far_frame']):
            self._flush_clip_refresh(side)

    def _reset_clip_plane(self, side: int):
        state = self._get_clip_state(side)
        state['rotation'] = np.eye(3)
        state['position'] = 0.0
        state['thickness'] = 0.0
        state['eye_near_position'] = 0.0
        state['eye_far_position'] = 0.0
        self._refresh_clip_ui(side)
        self._apply_clip_planes(side)
        self._update_clip_frame(side)

    def _rotate_clip_by_mouse(self, side: int, dx: float, dy: float):
        """Virtual-trackball rotation from a mouse delta (pixels).

        Matches MIPAV's VolumeTriPlanarRenderBase.java:1921-1926 where
        Ctrl+Drag rotates m_kArbRotate while frame is displayed.
        Simplified axis mapping: dx → rotate around world y (axis 1),
        dy → rotate around world x (axis 2) in napari (z,y,x) order.

        State updates immediately (so no rotation is lost); the expensive
        layer/frame refresh is coalesced via a throttle timer.
        """
        sensitivity = 0.5  # degrees per pixel
        ay = np.radians(dx * sensitivity)
        ax = np.radians(-dy * sensitivity)
        Ry = np.array([[ np.cos(ay), 0, np.sin(ay)],
                       [ 0,          1, 0         ],
                       [-np.sin(ay), 0, np.cos(ay)]])
        Rx = np.array([[ np.cos(ax), -np.sin(ax), 0],
                       [ np.sin(ax),  np.cos(ax), 0],
                       [ 0,           0,          1]])
        state = self._get_clip_state(side)
        state['rotation'] = Rx @ Ry @ state['rotation']
        self._schedule_clip_refresh(side)

    def _schedule_clip_refresh(self, side: int):
        """Schedule a throttled refresh of both planes and frame.

        Both must update together — if only the frame moves, it gets
        depth-occluded by the volume that's still clipped at the old
        orientation. Coalesces rapid-fire mouse_move updates to ~30fps.
        """
        t = self._clip_throttle_timers[side]
        if t is None:
            from qtpy.QtCore import QTimer
            t = QTimer()
            t.setSingleShot(True)
            t.setInterval(33)  # ~30fps
            t.timeout.connect(lambda s=side: self._flush_clip_refresh(s))
            self._clip_throttle_timers[side] = t
        if not t.isActive():
            t.start()

    def _flush_clip_refresh(self, side: int):
        """Apply planes + frame together so they stay visually in sync."""
        self._apply_clip_planes(side)
        self._update_clip_frame(side)

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

    def _cross_section_dir(self, ti: int) -> Path:
        """MIPAV path: <sharedOutputDir>/model_crossSections/ for timepoint ti."""
        stem = self.tiff_files[ti].stem
        return self.volume_path / stem / f"{stem}_results" / "model_crossSections"

    def _save_cross_sections_for_timepoint(self, ti: int) -> list[str]:
        """Write all cross-section overrides for ti to
        model_crossSections/latticeCrossSection_<i>.csv. Any CSV in the dir
        whose ring index is NOT in the current overrides dict is deleted —
        mirrors MIPAV LatticeModel.saveCrossSections (line 2718) which
        deletes the CSV for rings that are no longer edited.
        """
        overrides = self.cross_section_overrides.get(ti, {})
        csdir = self._cross_section_dir(ti)
        saved_paths = []

        # MIPAV file naming is 1-indexed (latticeCrossSection_1.csv =
        # first ring), validated against
        # RW10752_NU/Decon_reg_15_results/model_crossSections (files 1..11,
        # no _0). Our in-memory ring_idx is 0-based; map to 1-based on disk.
        if overrides:
            csdir.mkdir(parents=True, exist_ok=True)
            for ring_idx, offsets in overrides.items():
                path = csdir / f"latticeCrossSection_{ring_idx + 1}.csv"
                if _save_cross_section_csv(path, offsets):
                    saved_paths.append(str(path))

        # Clean up stale CSVs — any file in the dir whose ring index isn't
        # in the current overrides dict should be removed (matches MIPAV).
        if csdir.exists():
            import re
            for f in csdir.glob("latticeCrossSection_*.csv"):
                m = re.match(r"latticeCrossSection_(\d+)\.csv", f.name)
                if m and (int(m.group(1)) - 1) not in overrides:
                    try:
                        f.unlink()
                    except OSError:
                        pass
        return saved_paths

    def _handle_cross_section_drag(self, event, img_ref, side_idx, timepoint,
                                   lat_l_ref, lat_r_ref,
                                   lat_lines_ref, lat_mid_ref,
                                   lat_lc_ref, lat_rc_ref,
                                   pre_picked, rings, centers):
        """Cmd+Click+Drag on a ring vertex → reshape the ring with radial
        Fourier falloff. Mirrors MIPAV LatticeModel.updateLattice
        (LatticeModel.java:8492-8665).

        Uses a generator (yield) pattern so it integrates with napari's
        mouse_drag_callbacks protocol. `pre_picked`/`rings`/`centers` come
        from the caller's pick-first check so the picking work isn't duplicated.
        """
        ring_idx, vert_idx, _dist = pre_picked
        center = centers[ring_idx]
        current_ring = rings[ring_idx].copy()
        n_samples = self.cross_section_n_samples
        mode_name = {32: 'Single', 16: 'Narrow', 8: 'Medium', 4: 'Wide',
                     _GAUSSIAN_MODE: 'Gaussian'}.get(
            n_samples, f'n={n_samples}')
        print(f"[ring-edit t={timepoint}] Ring {ring_idx}, vertex {vert_idx}, "
              f"mode={mode_name}")

        yield  # first mouse_move

        while event.type == 'mouse_move':
            if 'Control' not in event.modifiers:
                break
            near2, far2 = img_ref.get_ray_intersections(
                event.position, event.view_direction, event.dims_displayed)
            if near2 is not None and far2 is not None:
                current_vertex = current_ring[vert_idx]
                radial = current_vertex - center
                r_norm = np.linalg.norm(radial)
                if r_norm >= 1e-9:
                    radial_unit = radial / r_norm
                    # Project mouse position onto the radial axis through the
                    # current vertex (MIPAV: radialDirection, line 2376-2395).
                    ray_pt = _project_point_on_ray(current_vertex, near2, far2)
                    new_radius = float(np.dot(ray_pt - center, radial_unit))
                    delta_length = new_radius - r_norm
                    if n_samples == _GAUSSIAN_MODE:
                        current_ring = _apply_gaussian_falloff(
                            current_ring, center, vert_idx, delta_length)
                    else:
                        current_ring = _apply_fourier_falloff(
                            current_ring, center, vert_idx, delta_length, n_samples)
                    # Persist as center-relative override
                    self.cross_section_overrides.setdefault(
                        timepoint, {})[ring_idx] = current_ring - center
                    self._update_lattice_visuals(
                        lat_l_ref, lat_r_ref, lat_lines_ref,
                        lat_mid_ref, lat_lc_ref, lat_rc_ref)
            yield

        # Mouse up: one final rebuild so the released state is authoritative.
        self._update_lattice_visuals(
            lat_l_ref, lat_r_ref, lat_lines_ref,
            lat_mid_ref, lat_lc_ref, lat_rc_ref)
        print(f"[ring-edit t={timepoint}] Released ring {ring_idx}")

    def _compute_per_ring_max_radii(self, ti: int, left_pts: np.ndarray,
                                    right_pts: np.ndarray) -> np.ndarray | None:
        """One outer-bound radius per lattice anchor (Mark, 2026-04-27).

        Per-ring: max distance of the 32 cross-section vertices from the
        midline center. Falls back to L/R half-distance when no override.
        Returns None if there are no overrides for ``ti`` (preserves the
        old global-extent path).
        """
        overrides = self.cross_section_overrides.get(ti)
        if not overrides:
            return None
        n = len(left_pts)
        radii = np.empty(n, dtype=np.float64)
        for i in range(n):
            off = overrides.get(i)
            if off is not None:
                radii[i] = float(np.max(np.linalg.norm(off, axis=1)))
            else:
                radii[i] = 0.5 * float(np.linalg.norm(right_pts[i] - left_pts[i]))
        return radii

    def _on_preview_toggle(self, side: int, enabled: bool):
        """UI checkbox → enter/exit preview mode for one side."""
        if enabled:
            self._enter_preview_mode(side)
        else:
            self._exit_preview_mode(side)

    def _enter_preview_mode(self, side: int):
        """Build straightened volume for `side`, swap layers."""
        # Idempotent: if already active, do nothing. Re-entering would
        # overwrite preview_hidden_layers (= original twisted layers) with
        # the previous preview's own layers, which leaks the originals.
        if self.preview_mode_active[side]:
            return
        lat_l = self.lattice_left_layers[side]
        lat_r = self.lattice_right_layers[side]
        if lat_l is None or lat_r is None:
            self._toast_persistent("Preview: no lattice layer for this side.")
            self._reset_preview_checkbox(side)
            return
        left_pts = np.asarray(lat_l.data)
        right_pts = np.asarray(lat_r.data)
        n = min(len(left_pts), len(right_pts))
        if n < 3:
            self._toast_persistent(
                f"Preview: need ≥3 lattice pairs (have {n}).")
            self._reset_preview_checkbox(side)
            return

        ti = self.grid_timepoints[side]
        pair_names = self.lattice_pair_names.get(ti, [])
        names = [p['name'] for p in pair_names[:n]] if pair_names else None

        # Gather visible channels (multi-channel-aware). Each visible channel
        # gets its own straightened layer so colors are preserved in preview
        # mode — matches the twisted view's additive overlay pattern.
        img_layers = self.grid_image_layers[side]
        if isinstance(img_layers, list):
            channel_layers = [lyr for lyr in img_layers
                              if lyr is not None and lyr.visible]
            if not channel_layers:
                channel_layers = [lyr for lyr in img_layers if lyr is not None]
        else:
            channel_layers = [img_layers] if img_layers is not None else []
        if not channel_layers:
            self._toast_persistent("Preview: no image layers to straighten.")
            self._reset_preview_checkbox(side)
            return

        try:
            model = _make_celegans_model(left_pts, right_pts, names)
            if model is None:
                raise ValueError("model construction returned None")
            self._toast(f"Building straightened volume "
                        f"({len(channel_layers)} channel"
                        f"{'s' if len(channel_layers) > 1 else ''})…")
            QApplication.processEvents()
            per_ring_radii = self._compute_per_ring_max_radii(
                ti, left_pts[:n], right_pts[:n])
            straight_channels = []
            ap_values = None
            for ch_layer in channel_layers:
                vol = np.asarray(ch_layer.data).astype(np.float32)
                straight, ap = model.straighten_volume(
                    vol, per_ring_max_radii=per_ring_radii)
                straight_channels.append(straight)
                if ap_values is None:
                    ap_values = ap
        except Exception as exc:
            import traceback; traceback.print_exc()
            self._toast_persistent(f"Preview: straighten failed — {exc}")
            self._reset_preview_checkbox(side)
            return

        # Use first channel for layout — all channels share the same AP/DV/ML grid
        straight = straight_channels[0]
        extent = (straight.shape[1] - 1) // 2
        self.preview_models[side] = model
        self.preview_ap_values[side] = ap_values
        self.preview_extents[side] = extent

        viewer = self.viewer_left if side == 0 else self.viewer_right
        hidden = []
        for layer in viewer.layers:
            if layer.visible:
                hidden.append(layer)
                layer.visible = False
        self.preview_hidden_layers[side] = hidden

        # Add one preview image layer per channel. Multi-channel: use each
        # channel's colormap + additive blending (matches twisted view).
        # Single-channel: gray + translucent (unchanged from before).
        preview_imgs = []
        multi = len(straight_channels) > 1
        for ch_layer, straight_vol in zip(channel_layers, straight_channels):
            cmap = ch_layer.colormap.name if multi else 'gray'
            blending = 'additive' if multi else 'translucent'
            ch_name = ch_layer.name
            contrast = (float(np.percentile(straight_vol, 1.0)),
                        float(np.percentile(straight_vol, 99.5)))
            preview_imgs.append(viewer.add_image(
                straight_vol,
                name=(f'Straightened {ch_name} t={ti}' if multi
                      else f'Straightened t={ti}'),
                colormap=cmap, blending=blending,
                contrast_limits=list(contrast)))
        img = preview_imgs[0]
        self.preview_image_layers[side] = preview_imgs if multi else img

        # Project the lattice splines (L, R, center) into straightened space
        # so the user has the same midline + lattice overlays they're used to
        # in the twisted view. Each lattice point becomes (ap_idx, dv+e, ml+e)
        # in the straightened frame; in straightened space ML/DV are pure
        # linear projections, so the splines render as nearly-straight lines
        # along Z (AP) at the worm's midline.
        overlays = []
        try:
            ap_dense = np.linspace(*model.internal_range, num=200)
            center_pts = model.center_spline.interpolate(ap_dense)
            left_pts_spline = model.left_spline.interpolate(ap_dense)
            right_pts_spline = model.right_spline.interpolate(ap_dense)
            ap_idx_dense = np.interp(ap_dense, ap_values,
                                     np.arange(len(ap_values)))

            def _project(world_pts):
                proj = np.empty((len(world_pts), 3))
                for i, (p, ap_i) in enumerate(zip(world_pts, ap_idx_dense)):
                    delta = p - center_pts[i]
                    ml_b, dv_b, _ = model.get_basis_vectors(float(ap_dense[i]))
                    ml = float(np.dot(delta, ml_b))
                    dv = float(np.dot(delta, dv_b))
                    proj[i] = [ap_i, dv + extent, ml + extent]
                return proj

            center_proj = _project(center_pts)
            left_proj = _project(left_pts_spline)
            right_proj = _project(right_pts_spline)

            mid_layer = viewer.add_shapes(
                [center_proj], shape_type='path', edge_color='red',
                edge_width=1.5, name='Midline (straightened)', opacity=0.9)
            left_layer = viewer.add_shapes(
                [left_proj], shape_type='path', edge_color='magenta',
                edge_width=1.0, name='Left spline (straightened)', opacity=0.9)
            right_layer = viewer.add_shapes(
                [right_proj], shape_type='path', edge_color='green',
                edge_width=1.0, name='Right spline (straightened)', opacity=0.9)
            overlays = [mid_layer, left_layer, right_layer]

            # Also overlay the lattice control points so the user sees the
            # H0/H1/V1.. structure laid out along AP.
            n_lat = min(len(left_pts), len(right_pts))
            lat_proj_pts = []
            for i in range(n_lat):
                for p in (left_pts[i], right_pts[i]):
                    cands = model.get_best_candidate(np.asarray(p))
                    if cands is None:
                        continue
                    ml_v, dv_v, ap_v = cands
                    ap_i = float(np.interp(ap_v, ap_values,
                                           np.arange(len(ap_values))))
                    lat_proj_pts.append([ap_i, dv_v + extent, ml_v + extent])
            if lat_proj_pts:
                lat_layer = viewer.add_points(
                    np.asarray(lat_proj_pts), name='Lattice (straightened)',
                    size=8, face_color='cyan', border_color='black',
                    border_width=0.15, ndim=3, opacity=0.9)
                overlays.append(lat_layer)
        except Exception as exc:
            print(f"[preview] overlay projection skipped: {exc}")

        # Project existing twisted-space annotations into straightened space.
        # Always create the layer (even when empty) so click-time `add` can
        # show a dot for points placed in preview mode.
        pts_layer = self.grid_points_layers[side]
        proj = []
        if pts_layer is not None and len(pts_layer.data) > 0:
            for p in pts_layer.data:
                try:
                    coords = model.get_best_candidate(np.asarray(p))
                except Exception:
                    coords = None
                if coords is None:
                    continue
                ml_v, dv_v, ap_v = coords
                ap_idx = float(np.interp(ap_v, ap_values, np.arange(len(ap_values))))
                proj.append([ap_idx, dv_v + extent, ml_v + extent])
        ppts = viewer.add_points(
            np.asarray(proj) if proj else np.empty((0, 3)),
            name='Annotations (straightened)',
            size=10, face_color='yellow', border_color='red',
            border_width=0.2, ndim=3, opacity=1.0)
        # Text labels offset above each point — same trick the twisted-space
        # layer uses so points buried inside bright nuclei stay findable.
        n = len(ppts.data)
        if n > 0:
            ppts.properties = {'label': [f'P{i + 1}' for i in range(n)]}
        else:
            ppts.properties = {'label': []}
        ppts.text = {
            'string': 'label', 'color': 'yellow', 'size': 10,
            'anchor': 'upper_left', 'translation': [0, 0, 15],
        }
        self.preview_points_layers[side] = ppts

        # Click handler — register on every layer that could receive the click
        # (the straightened image and the projection-points layer). Matches
        # the multi-layer registration pattern used in _load_dual_pair.
        side_idx = side

        def _preview_click_handler(layer, event):
            if 'Control' not in event.modifiers:
                return
            if not self.preview_mode_active[side_idx]:
                return
            preview_layer = self.preview_image_layers[side_idx]
            if isinstance(preview_layer, list):
                preview_layer = preview_layer[0] if preview_layer else None
            if preview_layer is None:
                return
            self._on_preview_click(side_idx, preview_layer, event)

        # Layer-level only — viewer-level was a fallback that ended up firing
        # the same event twice. Register on EVERY preview channel layer so the
        # click works regardless of which channel the user has selected in the
        # layer list.
        for ch_img in preview_imgs:
            ch_img.mouse_drag_callbacks.append(_preview_click_handler)
        self.preview_click_callbacks[side] = _preview_click_handler

        try:
            viewer.layers.selection.active = img
        except Exception:
            pass

        # Custom AP/DV/ML axis triad pinned to the volume's near-corner. napari's
        # built-in axes indicator renders labels too small to read; a Shapes
        # layer + Points-with-text gives full control over size and placement.
        try:
            arrow_len = max(20.0, extent * 0.5)
            origin = np.array([0.0, 2.0 * extent, 0.0])  # bottom-front corner
            ap_tip = origin + np.array([arrow_len, 0.0, 0.0])  # along AP (axis 0)
            dv_tip = origin + np.array([0.0, -arrow_len, 0.0])  # along DV up
            ml_tip = origin + np.array([0.0, 0.0, arrow_len])  # along ML
            tri_layer = viewer.add_shapes(
                [np.array([origin, ap_tip]),
                 np.array([origin, dv_tip]),
                 np.array([origin, ml_tip])],
                shape_type='line',
                edge_color=['red', 'green', 'blue'],
                edge_width=2.5, name='Axes triad', opacity=1.0)
            tri_lbl = viewer.add_points(
                np.array([
                    ap_tip + np.array([6, 0, 0]),
                    dv_tip + np.array([0, -6, 0]),
                    ml_tip + np.array([0, 0, 6]),
                ]),
                name='Axes labels', size=0.5, face_color='transparent',
                properties={'label': ['AP', 'DV', 'ML']},
                text={'string': 'label', 'color': 'white', 'size': 14,
                      'anchor': 'center'},
                ndim=3, opacity=1.0)
            overlays.extend([tri_layer, tri_lbl])
            viewer.scale_bar.visible = True
            viewer.scale_bar.unit = 'voxel'
        except Exception as exc:
            print(f"[preview] axis triad skipped: {exc}")
        self.preview_overlay_layers[side] = overlays

        self.preview_mode_active[side] = True
        self._update_preview_status(side)

        # Conflict resolution: preview is read-mostly. Force-disable lattice
        # mode + ring editing if they were on, hide the wireframe, and grey
        # out the conflicting controls on this side's panel.
        if self.lattice_mode:
            self._toggle_lattice_mode()
        if self.cross_section_edit_mode:
            self._on_cross_section_enable_changed(False)
        if self.wireframe_visible:
            self._toggle_wireframe()
        self._set_preview_conflicts_enabled(side, False)

        viewer.reset_view()
        # Default view: AP axis runs horizontal across screen, DV vertical,
        # ML into/out of screen. Camera looks along ML (axis 2 = napari x).
        # set_view_direction takes (z, y, x) world-coord vectors.
        try:
            viewer.camera.set_view_direction(
                view_direction=(0.0, 0.0, 1.0),   # look along +ML
                up_direction=(0.0, 1.0, 0.0),     # DV is up; AP becomes horizontal
            )
            viewer.camera.zoom *= 0.5
        except Exception as exc:
            print(f"[preview] camera setup skipped: {exc}")
        self._toast_persistent(
            f"Preview ON (side {'L' if side == 0 else 'R'} t={ti}): "
            f"AP={straight.shape[0]}, extent=±{extent}")

    def _exit_preview_mode(self, side: int):
        """Tear down straightened-volume display, restore twisted layers."""
        print(f"[preview] _exit_preview_mode(side={side}) called")
        viewer = self.viewer_left if side == 0 else self.viewer_right

        cb = self.preview_click_callbacks[side]
        if cb is not None:
            preview_layer = self.preview_image_layers[side]
            layers_to_unhook = (preview_layer if isinstance(preview_layer, list)
                                else [preview_layer])
            for lyr in layers_to_unhook:
                if lyr is None:
                    continue
                try:
                    lyr.mouse_drag_callbacks.remove(cb)
                except (ValueError, AttributeError):
                    pass
            self.preview_click_callbacks[side] = None

        # preview_image_layers may be a list (multi-channel) or a single
        # layer (single-channel); remove every entry.
        img_entry = self.preview_image_layers[side]
        img_list = (img_entry if isinstance(img_entry, list)
                    else ([img_entry] if img_entry is not None else []))
        for layer in img_list:
            try:
                if layer in viewer.layers:
                    viewer.layers.remove(layer)
            except Exception:
                pass
        self.preview_image_layers[side] = None

        pts_layer = self.preview_points_layers[side]
        if pts_layer is not None:
            try:
                if pts_layer in viewer.layers:
                    viewer.layers.remove(pts_layer)
            except Exception:
                pass
        self.preview_points_layers[side] = None

        for layer in self.preview_overlay_layers[side]:
            try:
                if layer in viewer.layers:
                    viewer.layers.remove(layer)
            except Exception:
                pass
        self.preview_overlay_layers[side] = []

        for layer in self.preview_hidden_layers[side]:
            try:
                layer.visible = True
            except Exception:
                pass
        self.preview_hidden_layers[side] = []

        self.preview_models[side] = None
        self.preview_ap_values[side] = None
        self.preview_extents[side] = None
        self.preview_mode_active[side] = False
        self._update_preview_status(side)
        self._set_preview_conflicts_enabled(side, True)

        # Restore active layer selection — preview set selection.active to
        # the (now-removed) straightened image layer; if we don't reset it,
        # click routing for ring editing on the wireframe layer can break.
        try:
            img_layers = self.grid_image_layers[side]
            primary = (img_layers[0] if isinstance(img_layers, list)
                       else img_layers)
            if primary is not None:
                viewer.layers.selection.active = primary
        except Exception:
            pass

        # Restore default axis labels and hide preview-only indicators.
        try:
            viewer.axes.visible = False
            viewer.scale_bar.visible = False
            viewer.dims.axis_labels = ('z', 'y', 'x')
        except Exception:
            pass

        try:
            viewer.reset_view()
        except Exception:
            pass
        print(f"[preview] _exit done — viewer has {len(viewer.layers)} layers, "
              f"active={viewer.layers.selection.active.name if viewer.layers.selection.active else None}")

    def _on_preview_click(self, side: int, layer, event):
        """Cmd+Click in straightened view → retwist → add to twisted points."""
        # Dedupe: viewer-level + layer-level callbacks fire for the same click;
        # tag the event so we only handle once.
        if getattr(event, '_napari_worm_preview_handled', False):
            return
        try:
            event._napari_worm_preview_handled = True
        except Exception:
            pass

        if not self.preview_mode_active[side]:
            return
        if self.lattice_mode or self.cross_section_edit_mode:
            self._toast("Preview mode is read-mostly — disable lattice/ring edit.")
            return

        try:
            try:
                near, far = layer.get_ray_intersections(
                    event.position, event.view_direction, event.dims_displayed)
            except Exception as exc:
                print(f"[preview-click] ray_intersections raised: {exc}")
                return
            if near is None or far is None:
                print(f"[preview-click] ray missed layer")
                return

            print(f"[preview-click side={side}] ray near={near}, far={far}")

            peak = find_peak_along_ray(layer.data, near, far)
            print(f"  peak={peak}")
            pos_straight = find_nucleus_centroid(layer.data, peak)
            print(f"  pos_straight={pos_straight} (z=AP, y=DV-shifted, x=ML-shifted)")

            model = self.preview_models[side]
            ap_values = self.preview_ap_values[side]
            extent = self.preview_extents[side]
            ap_idx = float(pos_straight[0])
            dv_v = float(pos_straight[1]) - extent
            ml_v = float(pos_straight[2]) - extent

            i0 = int(np.floor(np.clip(ap_idx, 0, len(ap_values) - 1)))
            i1 = min(i0 + 1, len(ap_values) - 1)
            f = float(np.clip(ap_idx - i0, 0.0, 1.0))
            ap_param = float((1 - f) * ap_values[i0] + f * ap_values[i1])
            print(f"  worm-coords: ml={ml_v:.2f} dv={dv_v:.2f} ap_param={ap_param:.3f}")

            twisted_pos = np.asarray(model.retwist(ml_v, dv_v, ap_param), dtype=float)
            print(f"  twisted_pos (z,y,x) = {twisted_pos}")

            pts_layer = self.grid_points_layers[side]
            if pts_layer is None:
                print("  [skip] grid_points_layers[side] is None")
                return
            pts_layer.add(twisted_pos)
            ti = self.grid_timepoints[side]
            segs = self.grid_annotation_segments.setdefault(ti, [])
            segs.append(-1)
            self.grid_annotations[ti] = pts_layer.data.copy()
            self.undo_stack.append((ti, ('preview_add', pts_layer)))

            ppts = self.preview_points_layers[side]
            if ppts is not None:
                ppts.add(np.asarray(pos_straight, dtype=float))
                n = len(ppts.data)
                ppts.properties = {'label': [f'P{i + 1}' for i in range(n)]}
                print(f"  preview layer points: {n}, "
                      f"visible={ppts.visible}, last={ppts.data[-1]}")

            self._refresh_annotation_table(side)
            self._refresh_point_labels(side)
            print(f"  [ok] annotation #{len(pts_layer.data)} added at t={ti}")
        except Exception as exc:
            import traceback
            print(f"[preview-click] EXCEPTION: {exc}")
            traceback.print_exc()

    def _set_preview_conflicts_enabled(self, side: int, enabled: bool):
        """Grey out controls that conflict with preview on the given side.

        Preview is read-mostly (the straightened volume is built once from
        the current lattice + cross-sections); changing lattice / wireframe
        / ring-edit state while looking at it would either silently desync
        or crash the click routing. Disable those controls on this side's
        panel for the duration of preview, and re-enable on exit.
        """
        dw = getattr(self, 'dual_window', None)
        if dw is None:
            return
        targets: list = []
        for attr in ('_mode_ann_btns', '_mode_lat_btns',
                     '_mode_wf_btns', '_mode_mesh_btns'):
            btns = getattr(dw, attr, None)
            if btns and side < len(btns) and btns[side] is not None:
                targets.append(btns[side])
        cs = getattr(dw, '_cs_controls', None)
        if cs and side < len(cs) and cs[side] is not None:
            targets.append(cs[side]['enable'])
            for btn in cs[side]['modes'].values():
                targets.append(btn)
            targets.append(cs[side]['reset'])
        for t in targets:
            t.setEnabled(enabled)

    def _reset_preview_checkbox(self, side: int):
        """Uncheck the preview checkbox without re-triggering the toggle."""
        if not hasattr(self.dual_window, '_preview_controls'):
            return
        ctrls = self.dual_window._preview_controls[side]
        if ctrls is None:
            return
        cb = ctrls['enable']
        cb.blockSignals(True)
        cb.setChecked(False)
        cb.blockSignals(False)
        self._update_preview_status(side)

    def _update_preview_status(self, side: int):
        """Refresh the 'Status: …' label in the Preview tab."""
        if not hasattr(self.dual_window, '_preview_controls'):
            return
        ctrls = self.dual_window._preview_controls[side]
        if ctrls is None:
            return
        if self.preview_mode_active[side]:
            n_ap = len(self.preview_ap_values[side]) if self.preview_ap_values[side] is not None else 0
            ext = self.preview_extents[side] or 0
            ctrls['info'].setText(f"Status: ON · AP={n_ap}, extent=±{ext}")
        else:
            ctrls['info'].setText("Status: off")

    def _on_cross_section_enable_changed(self, enabled: bool):
        """UI checkbox → state. Mirror to both sides' checkboxes.

        Ring editing only works on top of Lattice mode + a visible Wireframe
        (vertices live on the wireframe; Cmd+Click routing flows through
        lattice mode). Auto-enable both when the user turns ring editing on
        so they don't have to chase three separate buttons. When disabling,
        we leave lattice/wireframe alone — the user may still want to keep
        inspecting or placing lattice points.
        """
        self.cross_section_edit_mode = bool(enabled)
        if hasattr(self, '_cs_controls'):
            for ctrls in self._cs_controls:
                if ctrls is None:
                    continue
                cb = ctrls['enable']
                if cb.isChecked() != self.cross_section_edit_mode:
                    cb.blockSignals(True)
                    cb.setChecked(self.cross_section_edit_mode)
                    cb.blockSignals(False)
        if enabled:
            if not self.lattice_mode:
                self._toggle_lattice_mode()
            if not self.wireframe_visible:
                self._toggle_wireframe()
        self._toast(f"Ring editing: {'ON' if enabled else 'OFF'}")

    def _on_cross_section_mode_changed(self, n_samples: int):
        """Mode button clicked → state. Mirror to both sides."""
        self.cross_section_n_samples = int(n_samples)
        if hasattr(self, '_cs_controls'):
            for ctrls in self._cs_controls:
                if ctrls is None:
                    continue
                btn = ctrls['modes'].get(self.cross_section_n_samples)
                if btn is not None and not btn.isChecked():
                    btn.blockSignals(True)
                    btn.setChecked(True)
                    btn.blockSignals(False)

    def _new_annotations(self, side: int):
        """Delete all annotation points for `side`'s current timepoint,
        both in memory and on disk (integrated_annotation/annotations.csv).
        Confirms before deleting.
        """
        if side >= len(self.grid_timepoints):
            return
        ti = self.grid_timepoints[side]
        stem = self.tiff_files[ti].stem
        ann_csv = (self.volume_path / stem / f"{stem}_results"
                   / "integrated_annotation" / "annotations.csv")

        reply = QMessageBox.question(
            self.dual_window._host,
            "New annotations",
            f"Delete all annotation points for t={ti}?\n\n"
            f"Will also delete on disk:\n"
            f"  {ann_csv}\n\n"
            "This cannot be undone.",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if reply != QMessageBox.Yes:
            return

        pts_layer = self.grid_points_layers[side]
        if pts_layer is not None:
            pts_layer.data = np.empty((0, 3))

        self.grid_annotations.pop(ti, None)
        self.grid_annotation_segments.pop(ti, None)
        self.grid_annotation_names.pop(ti, None)
        self.undo_stack = [
            e for e in self.undo_stack if not (len(e) >= 1 and e[0] == ti)]

        deleted = []
        try:
            if ann_csv.exists():
                ann_csv.unlink()
                deleted.append(str(ann_csv))
        except OSError as e:
            print(f"  WARNING: could not delete {ann_csv}: {e}")

        self._refresh_annotation_table(side)
        self._refresh_point_labels(side)
        msg = (f"New annotations t={ti}: cleared in memory, "
               f"deleted {len(deleted)} file(s)")
        print(msg)
        for p in deleted:
            print(f"  removed {p}")
        self._toast(msg)

    def _new_lattice(self, side: int):
        """Delete all lattice points + ring overrides for `side`'s current
        timepoint, both in memory and the corresponding CSV files on disk
        (lattice_final/lattice.csv and model_crossSections/
        latticeCrossSection_*.csv). Confirms before deleting.
        """
        if side >= len(self.grid_timepoints):
            return
        ti = self.grid_timepoints[side]
        stem = self.tiff_files[ti].stem
        results_base = self.volume_path / stem / f"{stem}_results"
        lat_csv = results_base / "lattice_final" / "lattice.csv"
        cs_dir = results_base / "model_crossSections"

        reply = QMessageBox.question(
            self.dual_window._host,
            "New lattice",
            f"Delete all lattice points and ring overrides for t={ti}?\n\n"
            f"Will also delete on disk:\n"
            f"  {lat_csv}\n"
            f"  {cs_dir}/latticeCrossSection_*.csv\n\n"
            "This cannot be undone.",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if reply != QMessageBox.Yes:
            return

        lat_l = self.lattice_left_layers[side]
        lat_r = self.lattice_right_layers[side]
        if lat_l is not None:
            lat_l.data = np.empty((0, 3))
        if lat_r is not None:
            lat_r.data = np.empty((0, 3))

        self.lattice_annotations.pop(ti, None)
        self.lattice_pair_names.pop(ti, None)
        self.lattice_seam_counter.pop(ti, None)
        self.cross_section_overrides.pop(ti, None)
        self.lattice_last_placed = None
        self.lattice_undo_stack = [
            e for e in self.lattice_undo_stack if not (len(e) >= 2 and e[1] == ti)]

        if lat_l is not None and lat_r is not None:
            self._update_lattice_visuals(
                lat_l, lat_r,
                self.lattice_line_layers[side],
                self.lattice_mid_layers[side],
                self.lattice_left_curve_layers[side],
                self.lattice_right_curve_layers[side])

        deleted = []
        try:
            if lat_csv.exists():
                lat_csv.unlink()
                deleted.append(str(lat_csv))
        except OSError as e:
            print(f"  WARNING: could not delete {lat_csv}: {e}")
        if cs_dir.exists():
            import re
            for f in cs_dir.glob("latticeCrossSection_*.csv"):
                if re.match(r"latticeCrossSection_\d+\.csv", f.name):
                    try:
                        f.unlink()
                        deleted.append(str(f))
                    except OSError as e:
                        print(f"  WARNING: could not delete {f}: {e}")

        self._refresh_tables()
        self._refresh_point_labels(side)
        msg = (f"New lattice t={ti}: cleared in memory, "
               f"deleted {len(deleted)} file(s)")
        print(msg)
        for p in deleted:
            print(f"  removed {p}")
        self._toast(msg)

    def _reset_cross_sections_current(self):
        """Wipe cross-section overrides for both currently-displayed
        timepoints and rebuild wireframe/surface. Stale CSVs on disk will
        be deleted on next save via _save_cross_sections_for_timepoint."""
        changed = False
        for ti in self.grid_timepoints:
            if ti in self.cross_section_overrides:
                del self.cross_section_overrides[ti]
                changed = True
        if changed:
            for side, lat_l in enumerate(self.lattice_left_layers):
                lat_r = self.lattice_right_layers[side]
                if lat_l is None or lat_r is None:
                    continue
                lat_lines = self.lattice_line_layers[side]
                lat_mid   = self.lattice_mid_layers[side]
                lat_lc    = self.lattice_left_curve_layers[side]
                lat_rc    = self.lattice_right_curve_layers[side]
                self._update_lattice_visuals(
                    lat_l, lat_r, lat_lines, lat_mid, lat_lc, lat_rc)
            self._toast("Cross-section rings reset (save to persist delete)")

    def _load_cross_sections_for_timepoint(self, ti: int) -> int:
        """Read any latticeCrossSection_*.csv files for ti into
        self.cross_section_overrides[ti]. Returns the number of rings loaded.
        Safe to call even when no CSVs exist.
        """
        csdir = self._cross_section_dir(ti)
        if not csdir.exists():
            return 0
        # MIPAV file naming is 1-indexed; map back to our 0-based ring_idx.
        # Be tolerant of older napari-worm output that wrote 0-indexed files
        # (latticeCrossSection_0.csv has never existed in MIPAV output).
        import re
        loaded = {}
        for f in csdir.glob("latticeCrossSection_*.csv"):
            m = re.match(r"latticeCrossSection_(\d+)\.csv", f.name)
            if not m:
                continue
            file_idx = int(m.group(1))
            ring_idx = file_idx - 1 if file_idx >= 1 else file_idx
            offsets = _load_cross_section_csv(f)
            if offsets is not None:
                loaded[ring_idx] = offsets
        if loaded:
            self.cross_section_overrides[ti] = loaded
        return len(loaded)

    def _load_lattice_from_disk(self, ti: int) -> bool:
        """Read lattice_final/lattice.csv for ti and repopulate
        self.lattice_annotations[ti] + self.lattice_pair_names[ti].
        No-op if ti already has an in-memory entry or no file exists.
        """
        if ti in self.lattice_annotations:
            return False
        if ti < 0 or ti >= len(self.tiff_files):
            return False
        stem = self.tiff_files[ti].stem
        path = (self.volume_path / stem / f"{stem}_results"
                / "lattice_final" / "lattice.csv")
        if not path.exists():
            return False
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"  WARNING: could not read {path}: {e}")
            return False
        pairs: dict[str, dict[str, np.ndarray]] = {}
        order: list[str] = []
        for _, row in df.iterrows():
            name = str(row['name'])
            if len(name) < 2 or name[-1] not in ('L', 'R'):
                continue
            prefix, side = name[:-1], name[-1]
            if prefix not in pairs:
                pairs[prefix] = {}
                order.append(prefix)
            pairs[prefix][side] = np.array([
                float(row['z_voxels']),
                float(row['y_voxels']),
                float(row['x_voxels']),
            ])
        left_pts, right_pts, pair_infos = [], [], []
        for prefix in order:
            p = pairs[prefix]
            if 'L' in p:
                left_pts.append(p['L'])
            if 'R' in p:
                right_pts.append(p['R'])
            if prefix in _SEAM_CELL_SEQUENCE:
                pair_infos.append({'name': prefix, 'type': 'seam'})
            elif re.match(r'^a\d+$', prefix):
                pair_infos.append({'name': prefix, 'type': 'lattice'})
            else:
                # User-renamed pair: preserve across future renumber calls
                pair_infos.append({'name': prefix, 'type': 'lattice',
                                   'custom_name': True})
        if not left_pts and not right_pts:
            return False
        self.lattice_annotations[ti] = {
            'left':  np.asarray(left_pts)  if left_pts  else np.empty((0, 3)),
            'right': np.asarray(right_pts) if right_pts else np.empty((0, 3)),
        }
        self.lattice_pair_names[ti] = pair_infos
        self.lattice_seam_counter[ti] = [
            p['name'] for p in pair_infos if p['type'] == 'seam']
        print(f"  Lattice t={ti}: loaded {len(left_pts)}L + {len(right_pts)}R from disk")
        return True

    def _load_annotations_from_disk(self, ti: int) -> bool:
        """Read integrated_annotation/annotations.csv for ti and repopulate
        self.grid_annotations[ti] + segments. No-op if already cached or absent.
        """
        if ti in self.grid_annotations and len(self.grid_annotations[ti]) > 0:
            return False
        if ti < 0 or ti >= len(self.tiff_files):
            return False
        stem = self.tiff_files[ti].stem
        path = (self.volume_path / stem / f"{stem}_results"
                / "integrated_annotation" / "annotations.csv")
        if not path.exists():
            return False
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"  WARNING: could not read {path}: {e}")
            return False
        if len(df) == 0:
            return False
        pts = np.stack([
            df['z_voxels'].to_numpy(dtype=float),
            df['y_voxels'].to_numpy(dtype=float),
            df['x_voxels'].to_numpy(dtype=float),
        ], axis=1)
        self.grid_annotations[ti] = pts
        if 'lattice_segment' in df.columns:
            segs = []
            for v in df['lattice_segment']:
                try:
                    segs.append(int(v))
                except (ValueError, TypeError):
                    segs.append(-1)
            self.grid_annotation_segments[ti] = segs
        else:
            self.grid_annotation_segments[ti] = [-1] * len(pts)
        if 'name' in df.columns:
            self.grid_annotation_names[ti] = [
                str(v) if pd.notna(v) and str(v).strip() else f"A{i + 1}"
                for i, v in enumerate(df['name'].tolist())
            ]
        else:
            self.grid_annotation_names[ti] = [f"A{i + 1}" for i in range(len(pts))]
        print(f"  Annotations t={ti}: loaded {len(pts)} points from disk")
        return True

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

        # --- Annotation labels: user-editable, default A1, A2, ... ---
        pts = self.grid_points_layers[side]
        if pts is not None:
            n = len(pts.data)
            if n > 0:
                ti = self.grid_timepoints[side] if side < len(self.grid_timepoints) else None
                if ti is not None:
                    labels = list(self._annotation_names(ti, n))
                else:
                    labels = [f'A{i + 1}' for i in range(n)]
                pts.properties = {'label': labels}
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

    def _annotation_names(self, ti: int, n: int) -> list[str]:
        """Return the list of annotation names for ti, padded to length n.

        Missing slots default to ``f"A{i+1}"``. Mutates the stored list in
        place so subsequent reads see the same defaults.
        """
        names = self.grid_annotation_names.setdefault(ti, [])
        if len(names) < n:
            names.extend(f"A{i + 1}" for i in range(len(names), n))
        return names

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
        names = self._annotation_names(ti, len(data))
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
            # Name — editable; user can override default A1/A2/… per Ryan's
            # tissue-grouping workflow (A*, B*, C* by region).
            name_item = QTableWidgetItem(names[i])
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

    def _on_group_filter_changed(self, side: int, text: str):
        """Re-filter lattice table rows when group dropdown changes."""
        self._refresh_lattice_table(side)

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

        # Get active group filter
        group_filter = ""
        if hasattr(self.dual_window, '_group_filters') and self.dual_window._group_filters[side]:
            group_text = self.dual_window._group_filters[side].currentText()
            if group_text.startswith("a"):
                group_filter = "a"
            elif group_text.startswith("H"):
                group_filter = "H"
            elif group_text.startswith("V"):
                group_filter = "V"
            elif group_text.startswith("T"):
                group_filter = "T"

        n_rows = max(len(l_data), len(r_data))
        table.setRowCount(n_rows)
        for i in range(n_rows):
            # Pair name — editable; user-renamed pairs keep their name across
            # add/delete renumbers (marked custom_name=True). Default scheme
            # stays a0/a1/H0/H1/V1.. for auto-numbered slots.
            name = pair_names[i]['name'] if i < len(pair_names) else f"a{i}"
            name_item = QTableWidgetItem(name)
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
            # Show/hide row based on group filter
            if group_filter:
                table.setRowHidden(i, not name.startswith(group_filter))
            else:
                table.setRowHidden(i, False)
        table.blockSignals(False)

    def _on_annotation_table_edited(self, side: int, row: int, col: int):
        """Handle user editing Name, X/Y/Z, or Seg columns in the annotation table."""
        if col == 4:  # Intensity is read-only
            return
        table = self.dual_window.annotation_tables[side]
        item = table.item(row, col)
        if item is None:
            return
        text = item.text().strip()

        if col == 0:  # Name column — validate non-blank, non-duplicate
            ti = self.grid_timepoints[side]
            pts_layer = self.grid_points_layers[side]
            n = len(pts_layer.data) if pts_layer is not None else 0
            names = self._annotation_names(ti, n)
            old_name = names[row] if row < len(names) else f"A{row + 1}"
            if text == old_name:
                return
            if not text:
                self._toast("Annotation name cannot be blank")
                table.blockSignals(True)
                item.setText(old_name)
                table.blockSignals(False)
                return
            if text in names[:row] + names[row + 1:]:
                self._toast(f"Duplicate annotation name: {text}")
                table.blockSignals(True)
                item.setText(old_name)
                table.blockSignals(False)
                return
            names[row] = text
            self.undo_stack.append(('RENAME_ANN', ti, side, row, old_name))
            self._refresh_point_labels(side)
            return

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
        """Handle user editing Pair name or coordinate columns in the lattice table."""
        table = self.dual_window.lattice_tables[side]
        item = table.item(row, col)
        if item is None:
            return

        if col == 0:  # Pair name — validate non-blank, non-duplicate; mark custom
            ti = self.grid_timepoints[side]
            pair_names = self.lattice_pair_names.setdefault(ti, [])
            if row >= len(pair_names):
                return
            text = item.text().strip()
            old_info = pair_names[row]
            old_name = old_info['name']
            old_custom = old_info.get('custom_name', False)
            if text == old_name:
                return
            if not text:
                self._toast("Lattice pair name cannot be blank")
                table.blockSignals(True)
                item.setText(old_name)
                table.blockSignals(False)
                return
            other_names = [p['name'] for j, p in enumerate(pair_names) if j != row]
            if text in other_names:
                self._toast(f"Duplicate lattice pair name: {text}")
                table.blockSignals(True)
                item.setText(old_name)
                table.blockSignals(False)
                return
            old_info['name'] = text
            old_info['custom_name'] = True
            self.lattice_undo_stack.append(
                ('RENAME_LAT', ti, (row, old_name, old_custom)))
            self._refresh_point_labels(side)
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
        # Remove segment + name
        segs = self.grid_annotation_segments.get(ti, [])
        if row < len(segs):
            segs.pop(row)
        names = self.grid_annotation_names.get(ti, [])
        deleted_name = names[row] if row < len(names) else f"A{row + 1}"
        if row < len(names):
            names.pop(row)
        self.undo_stack.append(
            ('DELETE_ANN', ti, side, row, old_data[row].copy(), deleted_name))
        self.annotation_last_placed = None
        print(f"[t={ti}] Deleted annotation {deleted_name}")
        self._refresh_annotation_table(side)
        self._refresh_point_labels(side)

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
            self._toast("LATTICE MODE")
        else:
            print("=" * 50)
            print("  MODE: ANNOTATION  (Cmd+Click = annotate)")
            print("=" * 50)
            self._toast("ANNOTATION MODE")
        # Sync GUI buttons
        if hasattr(self, 'dual_window'):
            self.dual_window._update_mode_buttons()

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
                    self._toast(f"Seam cell: {pair_names[-1]['name']}L")
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
                    self._toast(f"Seam cell: {name}")
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
                self._toast(f"Seam cell inserted: {inserted_name}")

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
        # Only compute when wireframe is visible (W key) — expensive operation.
        # Use lattice-aligned mode + per-timepoint cross-section overrides so
        # ring index == lattice slice index (MIPAV parity).
        active_side = None
        for side, ll in enumerate(self.lattice_left_layers):
            if ll is lat_l:
                active_side = side
                break
        ti = (self.grid_timepoints[active_side]
              if active_side is not None and active_side < len(self.grid_timepoints)
              else None)
        overrides = self.cross_section_overrides.get(ti) if ti is not None else None

        if self.wireframe_visible and active_side is not None:
            wf_layer = self.wireframe_layers[active_side]
            if wf_layer is not None:
                _clear_shapes(wf_layer)
                if n >= 3:
                    left_data = np.asarray(lat_l.data)
                    right_data = np.asarray(lat_r.data)
                    paths = generate_wireframe_mesh(
                        left_data, right_data,
                        lattice_aligned=True, overrides=overrides)
                    for path in paths:
                        if len(path) >= 2:
                            wf_layer.add(path, shape_type='path')

        # --- Surface mesh (triangle mesh, Shift+W) ---
        if self.surface_visible and active_side is not None:
            surf_layer = self.surface_layers[active_side]
            if surf_layer is not None and n >= 3:
                left_data = np.asarray(lat_l.data)
                right_data = np.asarray(lat_r.data)
                result = generate_surface_mesh(
                    left_data, right_data,
                    lattice_aligned=True, overrides=overrides)
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
            save_path = lat_dir / "lattice.csv"
            self._save_csv_retry(pd.DataFrame(rows), save_path)
            print(f"  Lattice t={ti}: {len(rows)} points → {save_path}")
            saved_paths.append(str(save_path))
            saved += 1

        # Cross-section overrides — save for every timepoint with edits.
        # Also handles the "user deleted all overrides" case via stale-CSV cleanup.
        for ti in set(list(self.cross_section_overrides.keys())
                      + [t for t in self.lattice_annotations.keys()]):
            if ti < 0 or ti >= len(self.tiff_files):
                continue
            cs_saved = self._save_cross_sections_for_timepoint(ti)
            if cs_saved:
                print(f"  Cross-sections t={ti}: {len(cs_saved)} ring(s)")
                saved_paths.extend(cs_saved)

        if saved:
            msg = f"Lattice saved for {saved} timepoint(s):\n" + "\n".join(saved_paths)
            print(msg)
            self._toast_persistent(msg)

    def _save_current_timepoints(self):
        """Save annotations + lattice for the 2 currently displayed timepoints.

        Matches MIPAV's saveAll() in PlugInDialogVolumeRenderDual.java:2729-2740,
        which saves only imageIndex and imageIndex+1 (not the full visited history).
        Called on NEXT/BACK so persistence happens as part of the advance.
        """
        self._save_grid_annotations_to_cache()
        self._save_lattice_to_cache()

        saved_ts = []
        for ti in self.grid_timepoints:
            stem = self.tiff_files[ti].stem
            results_base = self.volume_path / stem / f"{stem}_results"

            pts = self.grid_annotations.get(ti, np.empty((0, 3)))
            ann_saved = False
            if len(pts) > 0:
                ann_dir = results_base / "integrated_annotation"
                ann_dir.mkdir(parents=True, exist_ok=True)
                segs = self.grid_annotation_segments.get(ti, [])
                ann_saved = save_annotations(
                    pts, ann_dir / "annotations.csv", segments=segs)

            entry = self.lattice_annotations.get(ti, {})
            left  = entry.get('left',  np.empty((0, 3)))
            right = entry.get('right', np.empty((0, 3)))
            lat_saved = False
            if len(left) > 0 or len(right) > 0:
                pair_names = self.lattice_pair_names.get(ti, [])
                n_pairs = min(len(left), len(right))
                rows = []
                for i in range(n_pairs):
                    prefix = (pair_names[i]['name'] if i < len(pair_names)
                              else _lattice_pair_name(i))
                    for pt, side in ((left[i], 'L'), (right[i], 'R')):
                        rows.append({'name': prefix + side,
                                     'x_voxels': pt[2], 'y_voxels': pt[1],
                                     'z_voxels': pt[0],
                                     'R': 255, 'G': 255, 'B': 255})
                for i in range(n_pairs, len(left)):
                    prefix = (pair_names[i]['name'] if i < len(pair_names)
                              else _lattice_pair_name(i))
                    pt = left[i]
                    rows.append({'name': prefix + 'L',
                                 'x_voxels': pt[2], 'y_voxels': pt[1],
                                 'z_voxels': pt[0],
                                 'R': 255, 'G': 255, 'B': 255})
                lat_dir = results_base / "lattice_final"
                lat_dir.mkdir(parents=True, exist_ok=True)
                lat_saved = self._save_csv_retry(
                    pd.DataFrame(rows), lat_dir / "lattice.csv")

            # Cross-section overrides (MIPAV model_crossSections/)
            cs_saved = self._save_cross_sections_for_timepoint(ti)
            if cs_saved:
                print(f"  Cross-sections t={ti}: {len(cs_saved)} ring(s)")

            if ann_saved or lat_saved or cs_saved:
                saved_ts.append(ti)

        if saved_ts:
            msg = "Saved " + ", ".join(f"t={t}" for t in saved_ts)
            print(msg)
            self._toast(msg)

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
        self._save_current_timepoints()
        self._load_dual_pair(t_left + 1, t_right + 1)

    def _grid_prev(self, viewer):
        t_left, t_right = self.grid_timepoints
        if t_left <= 0:
            print("Already at first timepoint")
            return
        self._save_current_timepoints()
        self._load_dual_pair(t_left - 1, t_right - 1)

    def _on_click_dual(self, layer, event, side_idx, timepoint, points_layer):
        near, far = layer.get_ray_intersections(
            event.position, event.view_direction, event.dims_displayed)
        if near is None or far is None:
            return

        if points_layer is not None and len(points_layer.data) > 0:
            closest = _find_closest_annotation_point_by_ray(
                near, far, np.asarray(points_layer.data), threshold=12.0)
            if closest is not None:
                pt_idx, _ = closest
                self.annotation_last_placed = {
                    'side_idx': side_idx, 'timepoint': timepoint,
                    'layer': points_layer, 'index': pt_idx}
                print(f"[t={timepoint}] Selected A{pt_idx + 1} "
                      "(drag to move, arrows to nudge)")

                yield

                dragged = False
                while event.type == 'mouse_move':
                    if 'Control' not in event.modifiers:
                        break
                    dragged = True
                    near2, far2 = layer.get_ray_intersections(
                        event.position, event.view_direction, event.dims_displayed)
                    if near2 is not None and far2 is not None:
                        new_peak = self._find_peak_multi_channel(near2, far2, side_idx)
                        new_pos = find_nucleus_centroid(
                            self._get_blended_volume(side_idx), new_peak)
                        pts = points_layer.data.copy()
                        pts[pt_idx] = new_pos
                        points_layer.data = pts
                        self._refresh_point_labels(side_idx)
                    yield

                self.grid_annotations[timepoint] = points_layer.data.copy()
                if dragged:
                    print(f"[t={timepoint}] Released A{pt_idx + 1}")
                    self._refresh_annotation_table(side_idx)
                return

        peak = self._find_peak_multi_channel(near, far, side_idx)
        pos = find_nucleus_centroid(self._get_blended_volume(side_idx), peak)
        points_layer.add(pos)
        self.undo_stack.append((timepoint, points_layer))
        self.grid_annotation_segments.setdefault(timepoint, []).append(-1)
        new_name = f"A{len(points_layer.data)}"
        self.grid_annotation_names.setdefault(timepoint, []).append(new_name)
        self.annotation_last_placed = {
            'side_idx': side_idx, 'timepoint': timepoint,
            'layer': points_layer,
            'index': len(points_layer.data) - 1}
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
                print(f"[t={ti}] Undid table edit on A{row + 1}")
                self._refresh_tables()
            return

        # Handle rename undo
        if isinstance(entry, tuple) and len(entry) >= 5 and entry[0] == 'RENAME_ANN':
            _, ti, side, row, old_name = entry
            names = self.grid_annotation_names.setdefault(ti, [])
            if row < len(names):
                names[row] = old_name
            print(f"[t={ti}] Undid rename → {old_name}")
            self._refresh_tables()
            return

        # Handle delete annotation undo — re-insert the deleted point
        if isinstance(entry, tuple) and len(entry) >= 4 and entry[0] == 'DELETE_ANN':
            _, ti, side, row, old_val = entry[:5]
            old_name = entry[5] if len(entry) >= 6 else f"A{row + 1}"
            pts_layer = self.grid_points_layers[side]
            if pts_layer is not None:
                pts = pts_layer.data.copy() if len(pts_layer.data) > 0 else np.empty((0, 3))
                pts_layer.data = np.insert(pts, row, old_val, axis=0)
                segs = self.grid_annotation_segments.setdefault(ti, [])
                segs.insert(row, -1)
                names = self.grid_annotation_names.setdefault(ti, [])
                names.insert(row, old_name)
                print(f"[t={ti}] Undid delete of {old_name}")
                self._refresh_tables()
            return

        timepoint, pts = entry
        active = [p for p in self.grid_points_layers if p is not None]
        if any(p is pts for p in active) and len(pts.data) > 0:
            pts.data = pts.data[:-1]
            segs = self.grid_annotation_segments.get(timepoint, [])
            if segs:
                segs.pop()
            names = self.grid_annotation_names.get(timepoint, [])
            if names:
                names.pop()
            print(f"[t={timepoint}] Undid last annotation")
        elif timepoint in self.grid_annotations and len(self.grid_annotations[timepoint]) > 0:
            self.grid_annotations[timepoint] = self.grid_annotations[timepoint][:-1]
            segs = self.grid_annotation_segments.get(timepoint, [])
            if segs:
                segs.pop()
            names = self.grid_annotation_names.get(timepoint, [])
            if names:
                names.pop()
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

        if side_char == 'RENAME_LAT':
            row, old_name, old_custom = undo_entry[2]
            pair_names = self.lattice_pair_names.setdefault(timepoint, [])
            if row < len(pair_names):
                pair_names[row]['name'] = old_name
                if old_custom:
                    pair_names[row]['custom_name'] = True
                else:
                    pair_names[row].pop('custom_name', None)
            print(f"[t={timepoint}] Undid lattice rename → {old_name}")
            self._refresh_tables()
            return

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

    def _screen_aligned_delta(self, viewer, direction: str,
                              step: float = 1.0) -> np.ndarray:
        """Return a (z, y, x) world-space delta of length ``step`` aligned to
        the active viewer's screen axes:
            up/down   → ± camera.up_direction
            right/left → ± cross(up_direction, view_direction) (screen-right)

        Falls back to world (z, x) axes if camera vectors are degenerate.
        """
        try:
            view_dir = np.asarray(viewer.camera.view_direction, dtype=float)
            up_dir = np.asarray(viewer.camera.up_direction, dtype=float)
        except Exception:
            view_dir = np.array([1.0, 0.0, 0.0])
            up_dir = np.array([-1.0, 0.0, 0.0])
        n_up = np.linalg.norm(up_dir)
        n_view = np.linalg.norm(view_dir)
        if n_up < 1e-9 or n_view < 1e-9:
            # Fallback: legacy world-axis behavior
            if direction == 'up':    return np.array([-step, 0.0, 0.0])
            if direction == 'down':  return np.array([+step, 0.0, 0.0])
            if direction == 'left':  return np.array([0.0, 0.0, -step])
            if direction == 'right': return np.array([0.0, 0.0, +step])
            return np.zeros(3)
        up_dir = up_dir / n_up
        view_dir = view_dir / n_view
        right_dir = np.cross(view_dir, up_dir)
        n_right = np.linalg.norm(right_dir)
        if n_right < 1e-9:
            right_dir = np.array([0.0, 0.0, 1.0])
        else:
            right_dir = right_dir / n_right
        if direction == 'up':    return  step * up_dir
        if direction == 'down':  return -step * up_dir
        if direction == 'right': return  step * right_dir
        if direction == 'left':  return -step * right_dir
        return np.zeros(3)

    def _nudge_last_point(self, direction: str, step: float = 1.0):
        """Move the selected/last-placed lattice point by 1 voxel along the
        active viewer's screen axes (so Up always moves visually up regardless
        of view rotation).
        """
        if not self.lattice_mode:
            return
        info = self.lattice_last_placed
        if info is None:
            return
        layer = info['layer']
        if len(layer.data) == 0:
            return
        idx = info.get('index', len(layer.data) - 1)
        if idx >= len(layer.data):
            return
        side_idx = info.get('side_idx', 0)
        viewer = self.viewer_left if side_idx == 0 else self.viewer_right
        delta = self._screen_aligned_delta(viewer, direction, step)
        pts = layer.data.copy()
        pts[idx] = pts[idx] + delta
        layer.data = pts
        ti = info['timepoint']
        self._nudge_visuals_after_lattice(ti, layer)
        return

    def _nudge_visuals_after_lattice(self, ti, layer):
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

    def _nudge_annotation_point(self, direction: str, step: float = 1.0):
        if self.lattice_mode:
            return
        info = self.annotation_last_placed
        if info is None:
            return
        layer = info['layer']
        if layer is None or len(layer.data) == 0:
            return
        idx = info.get('index', len(layer.data) - 1)
        if idx >= len(layer.data):
            return
        side_idx = info.get('side_idx', 0)
        viewer = self.viewer_left if side_idx == 0 else self.viewer_right
        delta = self._screen_aligned_delta(viewer, direction, step)
        pts = layer.data.copy()
        pts[idx] = pts[idx] + delta
        layer.data = pts
        ti = info['timepoint']
        self.grid_annotations[ti] = layer.data.copy()
        self._refresh_point_labels(side_idx)
        self._refresh_annotation_table(side_idx)

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
        self._toast(f"Wireframe: {state}")
        if hasattr(self, 'dual_window') and self.dual_window is not None:
            self.dual_window._update_mode_buttons()

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
            self._toast("Surface: ON")
        else:
            print("=" * 50)
            print("  SURFACE MESH: OFF")
            print("=" * 50)
            self._toast("Surface: OFF")
        if hasattr(self, 'dual_window') and self.dual_window is not None:
            self.dual_window._update_mode_buttons()

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
            self._save_lattice_to_cache()
            has_annotations = any(len(pts) > 0 for pts in self.grid_annotations.values())
            has_lattice = any(
                (len(e.get('left', [])) + len(e.get('right', []))) > 0
                for e in self.lattice_annotations.values()
            )
            has_rings = bool(self.cross_section_overrides)
            if not (has_annotations or has_lattice or has_rings):
                print("Nothing to save")
                self._toast_persistent("Nothing to save")
                return
            total = 0
            ann_paths = []
            if has_annotations:
                for ti, pts in sorted(self.grid_annotations.items()):
                    if len(pts) == 0:
                        continue
                    stem = self.tiff_files[ti].stem
                    ann_dir = (self.volume_path / stem / f"{stem}_results"
                               / "integrated_annotation")
                    ann_dir.mkdir(parents=True, exist_ok=True)
                    save_path = ann_dir / "annotations.csv"
                    segs = self.grid_annotation_segments.get(ti, [])
                    nms = self._annotation_names(ti, len(pts))
                    if save_annotations(pts, save_path, names=nms, segments=segs):
                        total += len(pts)
                        ann_paths.append(str(save_path))
                        print(f"  Saved {len(pts)} to {save_path}")
                msg = f"Saved {total} annotations across {len(self.grid_annotations)} timepoint(s):\n" + "\n".join(ann_paths)
                print(msg)
                self._toast_persistent(msg)
            self._save_lattice()
        else:
            if not hasattr(self, 'points_layer') or len(self.points_layer.data) == 0:
                print("No annotations to save")
                self._toast_persistent("No annotations to save")
                return
            save_path = (Path(self.annotations_path) if self.annotations_path
                         else self.volume_path.with_suffix('.csv'))
            save_annotations(self.points_layer.data, save_path)
            self._toast_persistent(
                f"Saved {len(self.points_layer.data)} annotations to {save_path}")

    def run(self):
        napari.run()
