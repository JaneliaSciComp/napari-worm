from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QHeaderView, QLabel, QPushButton,
    QSplitter, QTabWidget, QTableWidget, QVBoxLayout, QWidget,
)

from napari_worm.geometry.cross_section import _GAUSSIAN_MODE
from napari_worm.widgets.event_filters import _CanvasClickFilter
from napari_worm.widgets.histogram import HistogramLUTWidget


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

            ann_header = QHBoxLayout()
            ann_label = QLabel("Annotations")
            ann_label.setStyleSheet("font-weight: bold; padding: 4px;")
            ann_header.addWidget(ann_label)
            ann_header.addStretch()
            new_ann_btn = QPushButton("New annotations")
            new_ann_btn.setToolTip(
                "Delete all annotation points for this timepoint, plus the "
                "corresponding annotations.csv on disk.")
            new_ann_btn.setStyleSheet(
                "QPushButton { background-color: #c62828; color: white; "
                "padding: 2px 6px; }")
            new_ann_btn.clicked.connect(
                lambda _, s=side: self._annotator._new_annotations(s))
            ann_header.addWidget(new_ann_btn)
            tables_layout.addLayout(ann_header)
            ann_table = QTableWidget(0, 6)
            ann_table.setHorizontalHeaderLabels(["Name", "X", "Y", "Z", "Intensity", "Seg"])
            ann_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            ann_table.setEditTriggers(QTableWidget.DoubleClicked | QTableWidget.SelectedClicked)
            ann_table.setSelectionBehavior(QTableWidget.SelectRows)
            ann_table.verticalHeader().setVisible(False)
            tables_layout.addWidget(ann_table, stretch=1)

            lat_header = QHBoxLayout()
            lat_label = QLabel("Lattice")
            lat_label.setStyleSheet("font-weight: bold; padding: 4px;")
            lat_header.addWidget(lat_label)
            lat_header.addStretch()
            lat_header.addWidget(QLabel("Group:"))
            group_filter = QComboBox()
            group_filter.addItems(["All", "a (body)", "H (seam)", "V (seam)", "T (tail)"])
            group_filter.setFixedWidth(90)
            group_filter.currentTextChanged.connect(
                lambda text, s=side: self._annotator._on_group_filter_changed(s, text))
            lat_header.addWidget(group_filter)
            new_lat_btn = QPushButton("New lattice")
            new_lat_btn.setToolTip(
                "Delete all lattice points and ring overrides for this "
                "timepoint, plus the corresponding lattice.csv and "
                "latticeCrossSection_*.csv files on disk.")
            new_lat_btn.setStyleSheet(
                "QPushButton { background-color: #c62828; color: white; "
                "padding: 2px 6px; }")
            new_lat_btn.clicked.connect(
                lambda _, s=side: self._annotator._new_lattice(s))
            lat_header.addWidget(new_lat_btn)
            tables_layout.addLayout(lat_header)

            if not hasattr(self, '_group_filters'):
                self._group_filters = [None, None]
            self._group_filters[side] = group_filter
            lat_table = QTableWidget(0, 7)
            lat_table.setHorizontalHeaderLabels(
                ["Pair", "L_X", "L_Y", "L_Z", "R_X", "R_Y", "R_Z"])
            lat_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            lat_table.setEditTriggers(QTableWidget.DoubleClicked | QTableWidget.SelectedClicked)
            lat_table.setSelectionBehavior(QTableWidget.SelectRows)
            lat_table.verticalHeader().setVisible(False)
            tables_layout.addWidget(lat_table, stretch=1)

            # Mode toggle buttons: Annotation vs Lattice
            mode_row = QHBoxLayout()
            mode_row.setContentsMargins(4, 4, 4, 4)
            ann_btn = QPushButton("Annotation")
            ann_btn.setCheckable(True)
            ann_btn.setChecked(True)
            lat_btn = QPushButton("Lattice")
            lat_btn.setCheckable(True)
            lat_btn.setChecked(False)
            ann_btn.setStyleSheet(
                "QPushButton:checked { background-color: #b8a000; color: white; font-weight: bold; }"
                "QPushButton { padding: 4px 8px; }")
            lat_btn.setStyleSheet(
                "QPushButton:checked { background-color: #0097a7; color: white; font-weight: bold; }"
                "QPushButton { padding: 4px 8px; }")

            def _on_ann_btn():
                if self._annotator.lattice_mode:
                    self._annotator._toggle_lattice_mode()
                self._update_mode_buttons()
            def _on_lat_btn():
                if not self._annotator.lattice_mode:
                    self._annotator._toggle_lattice_mode()
                self._update_mode_buttons()
            ann_btn.clicked.connect(_on_ann_btn)
            lat_btn.clicked.connect(_on_lat_btn)
            mode_row.addWidget(ann_btn)
            mode_row.addWidget(lat_btn)

            # Wireframe + Mesh visibility toggles — same pattern (checkable + styled).
            # Init unchecked; _update_mode_buttons() syncs them to the annotator's
            # actual state once the annotator is attached.
            wf_btn = QPushButton("Wireframe")
            wf_btn.setCheckable(True)
            wf_btn.setStyleSheet(
                "QPushButton:checked { background-color: #546e7a; color: white; font-weight: bold; }"
                "QPushButton { padding: 4px 8px; }")
            mesh_btn = QPushButton("Mesh")
            mesh_btn.setCheckable(True)
            mesh_btn.setStyleSheet(
                "QPushButton:checked { background-color: #6a1b9a; color: white; font-weight: bold; }"
                "QPushButton { padding: 4px 8px; }")
            wf_btn.clicked.connect(
                lambda _=False: (self._annotator._toggle_wireframe(),
                                 self._update_mode_buttons()))
            mesh_btn.clicked.connect(
                lambda _=False: (self._annotator._toggle_surface(),
                                 self._update_mode_buttons()))
            mode_row.addWidget(wf_btn)
            mode_row.addWidget(mesh_btn)
            combined.addLayout(mode_row)

            if not hasattr(self, '_mode_ann_btns'):
                self._mode_ann_btns = [None, None]
                self._mode_lat_btns = [None, None]
                self._mode_wf_btns  = [None, None]
                self._mode_mesh_btns = [None, None]
            self._mode_ann_btns[side] = ann_btn
            self._mode_lat_btns[side] = lat_btn
            self._mode_wf_btns[side]  = wf_btn
            self._mode_mesh_btns[side] = mesh_btn

            # Clip tab — mirrors MIPAV's "Clip" tab in PlugInDialogVolumeRenderDual
            # (JPanelClip_WM.java arbitrary plane: enable, position, thickness, frame)
            clip_tab = QWidget()
            clip_layout = QVBoxLayout(clip_tab)
            clip_layout.setContentsMargins(8, 8, 8, 8)
            clip_layout.setSpacing(6)

            clip_enable = QCheckBox("Enable arbitrary clip plane")
            clip_layout.addWidget(clip_enable)

            clip_frame_cb = QCheckBox("Show plane frame (red outline)")
            clip_layout.addWidget(clip_frame_cb)

            pos_row = QHBoxLayout()
            pos_lbl = QLabel("Position:")
            pos_lbl.setFixedWidth(70)
            pos_row.addWidget(pos_lbl)
            clip_pos = QLabeledDoubleSlider(Qt.Horizontal)
            pos_row.addWidget(clip_pos)
            clip_layout.addLayout(pos_row)

            thk_row = QHBoxLayout()
            thk_lbl = QLabel("Thickness:")
            thk_lbl.setFixedWidth(70)
            thk_row.addWidget(thk_lbl)
            clip_thk = QLabeledDoubleSlider(Qt.Horizontal)
            thk_row.addWidget(clip_thk)
            clip_layout.addLayout(thk_row)

            clip_reset = QPushButton("Reset orientation (X-axis, center)")
            clip_layout.addWidget(clip_reset)

            clip_hint = QLabel(
                "Rotate plane: Shift+Drag on canvas while frame is shown")
            clip_hint.setStyleSheet("color: #888; font-size: 10px;")
            clip_hint.setWordWrap(True)
            clip_layout.addWidget(clip_hint)

            # Eye-aligned dual planes (MIPAV CLIP_EYE + CLIP_EYE_INV) —
            # camera-following slab with independent near/far positions.
            eye_hdr = QLabel("<b>Eye-aligned clipping</b> (follows camera)")
            clip_layout.addWidget(eye_hdr)
            eye_hint = QLabel(
                "Near clips between you and the plane; "
                "Far clips behind the plane. Both follow the camera as you rotate.")
            eye_hint.setStyleSheet("color: #888; font-size: 10px;")
            eye_hint.setWordWrap(True)
            clip_layout.addWidget(eye_hint)

            eye_near_enable = QCheckBox("Enable Near clip plane (toward observer)")
            clip_layout.addWidget(eye_near_enable)
            eye_near_frame = QCheckBox("Show Near plane frame")
            clip_layout.addWidget(eye_near_frame)
            near_row = QHBoxLayout()
            near_lbl = QLabel("Near pos:")
            near_lbl.setFixedWidth(70)
            near_row.addWidget(near_lbl)
            eye_near_pos = QLabeledDoubleSlider(Qt.Horizontal)
            near_row.addWidget(eye_near_pos)
            clip_layout.addLayout(near_row)

            eye_far_enable = QCheckBox("Enable Far clip plane (away from observer)")
            clip_layout.addWidget(eye_far_enable)
            eye_far_frame = QCheckBox("Show Far plane frame")
            clip_layout.addWidget(eye_far_frame)
            far_row = QHBoxLayout()
            far_lbl = QLabel("Far pos:")
            far_lbl.setFixedWidth(70)
            far_row.addWidget(far_lbl)
            eye_far_pos = QLabeledDoubleSlider(Qt.Horizontal)
            far_row.addWidget(eye_far_pos)
            clip_layout.addLayout(far_row)

            clip_layout.addStretch()

            if not hasattr(self, '_clip_controls'):
                self._clip_controls = [None, None]
            self._clip_controls[side] = {
                'enable': clip_enable, 'frame': clip_frame_cb,
                'position': clip_pos, 'thickness': clip_thk,
                'reset': clip_reset,
                'eye_near_enable': eye_near_enable,
                'eye_near_frame': eye_near_frame,
                'eye_near_position': eye_near_pos,
                'eye_far_enable': eye_far_enable,
                'eye_far_frame': eye_far_frame,
                'eye_far_position': eye_far_pos,
            }
            # Wire controls → annotator state. Lambdas capture side via default arg.
            clip_enable.toggled.connect(
                lambda v, s=side: self._annotator._on_clip_control_changed(s, 'enabled', v))
            clip_frame_cb.toggled.connect(
                lambda v, s=side: self._annotator._on_clip_control_changed(s, 'frame_visible', v))
            clip_pos.valueChanged.connect(
                lambda v, s=side: self._annotator._on_clip_control_changed(s, 'position', float(v)))
            clip_thk.valueChanged.connect(
                lambda v, s=side: self._annotator._on_clip_control_changed(s, 'thickness', float(v)))
            clip_reset.clicked.connect(
                lambda _, s=side: self._annotator._reset_clip_plane(s))
            # Eye-aligned signals
            eye_near_enable.toggled.connect(
                lambda v, s=side: self._annotator._on_clip_control_changed(s, 'eye_near_enabled', v))
            eye_near_frame.toggled.connect(
                lambda v, s=side: self._annotator._on_clip_control_changed(s, 'eye_near_frame', v))
            eye_near_pos.valueChanged.connect(
                lambda v, s=side: self._annotator._on_clip_control_changed(s, 'eye_near_position', float(v)))
            eye_far_enable.toggled.connect(
                lambda v, s=side: self._annotator._on_clip_control_changed(s, 'eye_far_enabled', v))
            eye_far_frame.toggled.connect(
                lambda v, s=side: self._annotator._on_clip_control_changed(s, 'eye_far_frame', v))
            eye_far_pos.valueChanged.connect(
                lambda v, s=side: self._annotator._on_clip_control_changed(s, 'eye_far_position', float(v)))

            # --- Edit tab: cross-section ring editing (MIPAV parity) ---
            edit_tab = QWidget()
            edit_layout = QVBoxLayout(edit_tab)
            edit_layout.setContentsMargins(8, 8, 8, 8)
            edit_layout.setSpacing(6)

            edit_title = QLabel("<b>Cross-section editing</b>")
            edit_layout.addWidget(edit_title)

            cs_enable = QCheckBox("Enable ring editing (Cmd+Click+Drag ring vertex)")
            edit_layout.addWidget(cs_enable)

            mode_lbl = QLabel("Falloff width:")
            edit_layout.addWidget(mode_lbl)

            # 4 mode buttons — checkable + exclusive (QButtonGroup)
            from qtpy.QtWidgets import QButtonGroup
            mode_row = QHBoxLayout()
            cs_mode_group = QButtonGroup(edit_tab)
            cs_mode_group.setExclusive(True)
            cs_mode_buttons = {}
            for label, ns_val in (("Single (32)", 32), ("Narrow (16)", 16),
                                  ("Medium (8)", 8), ("Wide (4)", 4),
                                  ("Gaussian", _GAUSSIAN_MODE)):
                btn = QPushButton(label)
                btn.setCheckable(True)
                if ns_val == 8:  # MIPAV default
                    btn.setChecked(True)
                cs_mode_group.addButton(btn)
                mode_row.addWidget(btn)
                cs_mode_buttons[ns_val] = btn
                btn.clicked.connect(
                    lambda _=False, n=ns_val: self._annotator._on_cross_section_mode_changed(n))
            edit_layout.addLayout(mode_row)

            cs_reset = QPushButton("Reset rings for current timepoints")
            edit_layout.addWidget(cs_reset)
            cs_reset.clicked.connect(
                lambda: self._annotator._reset_cross_sections_current())

            cs_hint = QLabel(
                "Single: only clicked vertex · Narrow/Medium/Wide: Fourier "
                "bulge around the click · Gaussian: 5-vertex weighted bulge "
                "(clicked + ±1 at 0.37× + ±2 at 0.14×; smoother than narrow "
                "Fourier kernels) · all vertices move purely radially from "
                "the ring center (MIPAV parity).")
            cs_hint.setStyleSheet("color: #888; font-size: 10px;")
            cs_hint.setWordWrap(True)
            edit_layout.addWidget(cs_hint)
            edit_layout.addStretch()

            if not hasattr(self, '_cs_controls'):
                self._cs_controls = [None, None]
            self._cs_controls[side] = {
                'enable': cs_enable, 'modes': cs_mode_buttons, 'reset': cs_reset}

            cs_enable.toggled.connect(
                lambda v: self._annotator._on_cross_section_enable_changed(v))

            # --- Preview tab: straightened-volume rendering (MIPAV previewMode) ---
            preview_tab = QWidget()
            preview_layout = QVBoxLayout(preview_tab)
            preview_layout.setContentsMargins(8, 8, 8, 8)
            preview_layout.setSpacing(6)

            preview_title = QLabel("<b>Preview mode (straightening)</b>")
            preview_layout.addWidget(preview_title)

            preview_enable = QCheckBox("Enable straightened view")
            preview_layout.addWidget(preview_enable)

            preview_info = QLabel("Status: off")
            preview_info.setStyleSheet("color: #888; font-size: 10px;")
            preview_info.setWordWrap(True)
            preview_layout.addWidget(preview_info)

            preview_hint = QLabel(
                "Requires lattice with ≥3 pairs. Straightened axes: "
                "Z=AP (head→tail), Y=DV, X=ML. Cmd+Click in this view "
                "places an annotation; the position is retwisted to the "
                "twisted volume so it persists when preview is toggled "
                "off. Lattice + ring editing are disabled while preview "
                "is on. Powered by Caroline Malin-Mayor's "
                "<code>celegans_model</code> package "
                "(<code>PythonCelegansModel.straighten_volume</code> + "
                "<code>retwist</code>).")
            preview_hint.setStyleSheet("color: #888; font-size: 10px;")
            preview_hint.setWordWrap(True)
            preview_layout.addWidget(preview_hint)
            preview_layout.addStretch()

            if not hasattr(self, '_preview_controls'):
                self._preview_controls = [None, None]
            self._preview_controls[side] = {
                'enable': preview_enable, 'info': preview_info}

            preview_enable.toggled.connect(
                lambda v, s=side: self._annotator._on_preview_toggle(s, v))

            # Assemble Layers/Tables/Clip/Rings/Preview tabs
            tab_widget = QTabWidget()
            tab_widget.addTab(layers_tab, "Layers")
            tab_widget.addTab(tables_tab, "Tables")
            tab_widget.addTab(clip_tab, "Clip")
            tab_widget.addTab(edit_tab, "Rings")
            tab_widget.addTab(preview_tab, "Preview")
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
        # napari >=0.5 emits status as a dict; older versions emit a string.
        # showMessage() requires a string, so flatten dict → "name coords: value".
        def _fmt_status(v):
            if isinstance(v, dict):
                return f"{v.get('layer_name', '')}{v.get('coordinates', '')}{v.get('value', '')}"
            return str(v) if v is not None else ""
        if hasattr(viewer_right, 'status'):
            viewer_right.events.status.connect(
                lambda e: self._host.statusBar().showMessage(_fmt_status(e.value))
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

    def _update_mode_buttons(self):
        """Sync mode button checked state with annotator's current modes.

        Covers Annotation/Lattice plus Wireframe/Mesh visibility — keyboard
        shortcuts (L / W / Shift+W) toggle the state directly, so the buttons
        need to mirror whatever the annotator currently holds.
        """
        a = self._annotator
        is_lattice = a.lattice_mode
        for ann_btn, lat_btn in zip(self._mode_ann_btns, self._mode_lat_btns):
            if ann_btn:
                ann_btn.setChecked(not is_lattice)
            if lat_btn:
                lat_btn.setChecked(is_lattice)
        for wf_btn in getattr(self, '_mode_wf_btns', []):
            if wf_btn and wf_btn.isChecked() != a.wireframe_visible:
                wf_btn.blockSignals(True)
                wf_btn.setChecked(a.wireframe_visible)
                wf_btn.blockSignals(False)
        for mesh_btn in getattr(self, '_mode_mesh_btns', []):
            if mesh_btn and mesh_btn.isChecked() != a.surface_visible:
                mesh_btn.blockSignals(True)
                mesh_btn.setChecked(a.surface_visible)
                mesh_btn.blockSignals(False)

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
