import numpy as np
import pyqtgraph as pg
from qtpy.QtWidgets import (
    QCheckBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget,
)


class HistogramLUTWidget(QWidget):
    """MIPAV-style 2D histogram + transfer function widget.

    Mirrors MIPAV's ViewJComponentHistoLUT.java:
    - 2D plot: X = image intensity, Y = output brightness (0–255)
    - Histogram bars drawn as background (log scale, bottom-up like MIPAV)
    - Yellow transfer function line with 4 draggable control points:
      * Endpoints (first/last): move Y only (like MIPAV setLinearIndex)
      * Middle points: move freely in X and Y (gamma/windowing)
    - Right-side count labels on histogram (log scale, like MIPAV)
    - Dragging endpoints updates napari contrast_limits; middle points
      adjust the transfer curve shape (gamma-like)
    - One widget per viewer — embedded in dockLayerList, switches with side

    MIPAV reference: ViewJComponentHistoLUT.java, TransferFunction.java
    Default 4 points match MIPAV's linearMode():
      (min, 0), (min+range/3, 85), (min+2*range/3, 170), (max, 255)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layer = None
        self._updating = False
        self._hist_counts = None
        self._hist_edges = None
        self._data_min = 0.0
        self._data_max = 65535.0
        self._bar_item = None
        self._count_labels = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # --- Layout: vertical color bar (left) + 2D plot (right) ---
        plot_row = QHBoxLayout()
        plot_row.setSpacing(2)

        # Vertical LUT color bar (MIPAV's ViewJComponentLUT — 20px wide).
        # Uses an ImageItem for a smooth gradient that reflects the layer's
        # colormap (gray for gray, red for red channel, etc.).
        self._lut_bar = pg.GraphicsLayoutWidget()
        self._lut_bar.setFixedWidth(30)
        self._lut_bar.setMinimumHeight(180)
        self._lut_bar.setMaximumHeight(250)
        self._lut_bar.setBackground((40, 40, 40))
        self._lut_bar_plot = self._lut_bar.addPlot()
        self._lut_bar_plot.hideAxis('bottom')
        self._lut_bar_plot.hideAxis('left')
        self._lut_bar_plot.setMouseEnabled(x=False, y=False)
        self._lut_bar_plot.hideButtons()
        self._lut_bar_plot.setYRange(0, 255)
        self._lut_bar_plot.setXRange(0, 1)
        self._lut_bar_plot.getViewBox().setDefaultPadding(0)
        # Gradient image: 1 pixel wide × 256 tall, will be colored per-layer
        self._lut_gradient = np.linspace(0, 255, 256).reshape(256, 1).astype(np.uint8)
        self._lut_bar_img = pg.ImageItem(self._lut_gradient)
        self._lut_bar_img.setRect(0, 0, 1, 255)
        self._lut_bar_plot.addItem(self._lut_bar_img)
        plot_row.addWidget(self._lut_bar)

        # Main 2D plot
        self._graphics = pg.GraphicsLayoutWidget()
        self._graphics.setMinimumHeight(180)
        self._graphics.setMaximumHeight(250)
        self._graphics.setBackground((40, 40, 40))
        self._plot = self._graphics.addPlot()
        self._plot.setLabel('bottom', 'Image Intensities',
                            color='#CCCCCC', **{'font-size': '9pt'})
        self._plot.setLabel('left', 'Output Brightness',
                            color='#CCCCCC', **{'font-size': '9pt'})
        self._plot.setMouseEnabled(x=False, y=False)
        self._plot.hideButtons()
        self._plot.setYRange(0, 255)
        self._plot.getViewBox().setBackgroundColor((50, 50, 60, 255))
        # Extra bottom margin so axis labels don't get cut off
        self._plot.setContentsMargins(0, 0, 0, 8)
        for axis_name in ('bottom', 'left'):
            axis = self._plot.getAxis(axis_name)
            axis.setStyle(tickFont=pg.QtGui.QFont('', 8))
            axis.setTextPen(pg.mkPen('#CCCCCC'))
            axis.setPen(pg.mkPen('#999999'))

        # Transfer function line (yellow, like MIPAV)
        self._tf_line = pg.PlotCurveItem(pen=pg.mkPen('y', width=2))
        self._tf_line.setZValue(10)
        self._plot.addItem(self._tf_line)

        # 4 draggable control points (MIPAV linearMode default)
        # Points 0,3 = endpoints (Y only); Points 1,2 = middle (free X+Y)
        self._points = []
        for _ in range(4):
            pt = pg.TargetItem(
                pos=(0, 0), size=10, symbol='s',
                pen=pg.mkPen('y', width=2),
                brush=pg.mkBrush(0, 0, 0, 180),
                movable=True)
            pt.setZValue(20)
            self._plot.addItem(pt)
            self._points.append(pt)
        for i, pt in enumerate(self._points):
            pt.sigPositionChanged.connect(
                lambda _p, idx=i: self._on_point_moved(idx))

        plot_row.addWidget(self._graphics, stretch=1)
        layout.addLayout(plot_row)

        # --- Controls row ---
        ctrl_row = QHBoxLayout()
        self._log_cb = QCheckBox("Log scale")
        self._log_cb.setChecked(True)
        self._log_cb.stateChanged.connect(self._redraw_histogram)
        ctrl_row.addWidget(self._log_cb)
        ctrl_row.addStretch()
        self._range_label = QLabel("")
        self._range_label.setStyleSheet("font-size: 9px;")
        ctrl_row.addWidget(self._range_label)
        layout.addLayout(ctrl_row)

    def set_layer(self, layer):
        """Bind to a single napari Image layer."""
        if self._layer is not None:
            try:
                self._layer.events.contrast_limits.disconnect(
                    self._on_layer_contrast_changed)
            except (TypeError, RuntimeError):
                pass
            try:
                self._layer.events.colormap.disconnect(
                    self._on_layer_colormap_changed)
            except (TypeError, RuntimeError):
                pass
        self._layer = layer
        if layer is not None:
            layer.events.contrast_limits.connect(self._on_layer_contrast_changed)
            layer.events.colormap.connect(self._on_layer_colormap_changed)
            self._update_lut_bar_colormap()
            self._compute_histogram()
            self._redraw_histogram()
            self._sync_points_from_layer()

    def _update_lut_bar_colormap(self):
        """Update the vertical LUT color bar to match the layer's colormap."""
        if self._layer is None:
            return
        # Build a 256×1 RGBA image using the layer's colormap
        cmap = self._layer.colormap
        # Sample the colormap at 256 positions
        positions = np.linspace(0, 1, 256)
        colors = cmap.map(positions)  # (256, 4) RGBA float 0–1
        # Convert to uint8 RGBA image (256 rows × 1 col × 4 channels)
        rgba = (colors * 255).astype(np.uint8).reshape(256, 1, 4)
        self._lut_bar_img.setImage(rgba)

    def _compute_histogram(self):
        """Compute histogram bins from layer data."""
        if self._layer is None:
            return
        data = np.asarray(self._layer.data)
        self._data_min = float(data.min())
        self._data_max = float(data.max())
        if self._data_min == self._data_max:
            self._data_max = self._data_min + 1
        self._hist_counts, self._hist_edges = np.histogram(
            data.ravel(), bins=256, range=(self._data_min, self._data_max))

    def _redraw_histogram(self):
        """Redraw histogram bars and count labels."""
        if self._hist_counts is None:
            return

        if hasattr(self, '_hist_bar_items'):
            for item in self._hist_bar_items:
                self._plot.removeItem(item)
            self._hist_bar_items = []
        for lbl in self._count_labels:
            self._plot.removeItem(lbl)
        self._count_labels.clear()

        counts = self._hist_counts.astype(float)
        raw_max = float(self._hist_counts.max()) if len(counts) > 0 else 1.0
        use_log = self._log_cb.isChecked()
        if use_log:
            counts = np.log1p(counts)

        c_max = counts.max() if counts.max() > 0 else 1.0
        norm_counts = counts / c_max * 255.0

        bin_centers = (self._hist_edges[:-1] + self._hist_edges[1:]) / 2.0
        bin_width = self._hist_edges[1] - self._hist_edges[0]

        # Draw individual bars colored by LUT position (MIPAV: each bar uses
        # its transfer function mapped color — dark red for low intensity,
        # bright red for high intensity).
        if hasattr(self, '_hist_bar_items'):
            for item in self._hist_bar_items:
                self._plot.removeItem(item)
        self._hist_bar_items = []
        # Batch bars into groups for performance (every 4 bins)
        step = max(1, len(bin_centers) // 64)
        for i in range(0, len(bin_centers), step):
            end_i = min(i + step, len(bin_centers))
            h = norm_counts[i:end_i].max()
            cx = bin_centers[i:end_i].mean()
            w = bin_width * step
            # Color: map bin intensity position through channel colormap
            t = (cx - self._data_min) / (self._data_max - self._data_min)
            cmap = self._layer.colormap if self._layer is not None else None
            if cmap is not None:
                rgba = cmap.map(np.array([max(0.15, t)]))[0]  # floor at 0.15 so dark end stays visible
                r, g, b = int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)
            else:
                r, g, b = int(40 + t * 200), 10, 10
            bar = pg.BarGraphItem(
                x=[cx], height=[h], width=w,
                brush=pg.mkBrush(r, g, b, 180),
                pen=pg.mkPen(None))
            self._plot.addItem(bar)
            self._hist_bar_items.append(bar)
        self._bar_item = None  # no longer a single item

        # Right-side count labels (like MIPAV's log/linear tick marks)
        x_pos = self._data_max
        if use_log and raw_max > 1:
            log_max = np.log(raw_max)
            for frac in [1.0, 0.5, 0.25, 0.125]:
                count_val = int(np.exp(log_max * frac) + 0.5)
                y_pos = frac * 255.0
                lbl = pg.TextItem(str(count_val), color=(100, 150, 255),
                                  anchor=(0, 0.5))
                lbl.setPos(x_pos, y_pos)
                lbl.setZValue(5)
                font = lbl.textItem.font()
                font.setPointSize(7)
                lbl.setFont(font)
                self._plot.addItem(lbl)
                self._count_labels.append(lbl)
        elif raw_max > 0:
            for frac in [1.0, 0.75, 0.5, 0.25]:
                count_val = int(raw_max * frac + 0.5)
                y_pos = frac * 255.0
                lbl = pg.TextItem(str(count_val), color=(100, 150, 255),
                                  anchor=(0, 0.5))
                lbl.setPos(x_pos, y_pos)
                lbl.setZValue(5)
                font = lbl.textItem.font()
                font.setPointSize(7)
                lbl.setFont(font)
                self._plot.addItem(lbl)
                self._count_labels.append(lbl)

        self._tf_line.setZValue(10)
        for pt in self._points:
            pt.setZValue(20)

        self._plot.setXRange(self._data_min, self._data_max, padding=0.02)
        self._plot.setYRange(0, 255, padding=0.02)

    def _sync_points_from_layer(self):
        """Set 4 control points from layer's contrast_limits (MIPAV linearMode)."""
        if self._layer is None or self._updating:
            return
        self._updating = True
        try:
            lo, hi = self._layer.contrast_limits
            rng = hi - lo if hi > lo else 1.0
            # MIPAV default: 4 points evenly spaced along the ramp
            self._points[0].setPos(lo, 0)
            self._points[1].setPos(lo + rng / 3, 85)
            self._points[2].setPos(lo + 2 * rng / 3, 170)
            self._points[3].setPos(hi, 255)
            self._update_tf_line()
            self._range_label.setText(f"min: {lo:.0f}  max: {hi:.0f}")
        finally:
            self._updating = False

    def _update_tf_line(self):
        """Redraw the yellow transfer function line through all 4 points."""
        xs = [self._points[i].pos().x() for i in range(4)]
        ys = [self._points[i].pos().y() for i in range(4)]
        # Flat extensions beyond endpoints (like MIPAV)
        self._tf_line.setData(
            x=[self._data_min] + xs + [self._data_max],
            y=[ys[0]] + ys + [ys[3]])

    def _on_point_moved(self, idx):
        """Called when user drags a control point."""
        if self._updating or self._layer is None:
            return
        self._updating = True
        try:
            pt = self._points[idx]
            px, py = pt.pos().x(), pt.pos().y()
            py = max(0.0, min(255.0, py))

            if idx == 0 or idx == 3:
                # Endpoints: Y moves freely, X fixed at contrast limit
                # (MIPAV: endpoints only move in Y direction)
                fixed_x = self._points[idx].pos().x()
                pt.setPos(fixed_x, py)
            else:
                # Middle points: free X+Y, but X stays ordered
                # (MIPAV setLinearIndex constraint)
                x_lo = self._points[idx - 1].pos().x()
                x_hi = self._points[idx + 1].pos().x()
                px = max(x_lo + 1, min(x_hi - 1, px))
                pt.setPos(px, py)

            self._update_tf_line()

            # Update contrast_limits from endpoint X positions
            lo_x = self._points[0].pos().x()
            hi_x = self._points[3].pos().x()
            if lo_x < hi_x:
                self._layer.contrast_limits = (lo_x, hi_x)

            # Approximate gamma from BOTH middle points' curvature.
            # MIPAV's piecewise-linear transfer function controls each
            # segment independently, but napari only has a single gamma.
            # We average both middle points' deviation from the linear ramp.
            lo_y = self._points[0].pos().y()
            hi_y = self._points[3].pos().y()
            if hi_x > lo_x and hi_y > lo_y:
                gammas = []
                for ref in (1, 2):
                    mid_x = self._points[ref].pos().x()
                    mid_y = self._points[ref].pos().y()
                    t = (mid_x - lo_x) / (hi_x - lo_x)
                    t = max(0.01, min(0.99, t))
                    norm_y = (mid_y - lo_y) / (hi_y - lo_y)
                    norm_y = max(0.01, min(0.99, norm_y))
                    g = np.log(norm_y) / np.log(t)
                    gammas.append(max(0.1, min(5.0, g)))
                self._layer.gamma = (gammas[0] + gammas[1]) / 2.0

            self._range_label.setText(
                f"min: {lo_x:.0f}  max: {hi_x:.0f}")
        finally:
            self._updating = False

    def _on_layer_contrast_changed(self, event=None):
        """Called when napari's built-in slider changes contrast_limits."""
        self._sync_points_from_layer()

    def _on_layer_colormap_changed(self, event=None):
        """Called when the layer's colormap changes — redraw bars and LUT bar."""
        self._update_lut_bar_colormap()
        self._redraw_histogram()
