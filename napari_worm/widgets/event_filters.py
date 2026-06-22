from qtpy.QtCore import QEvent, QObject, Qt, QTimer


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
            step = 5.0 if (event.modifiers() & Qt.ShiftModifier) else 1.0
            if ann.lattice_mode and ann.lattice_last_placed is not None:
                ann._nudge_last_point(direction, step=step)
                return True  # consume event — don't let napari rotate
            elif (not ann.lattice_mode
                  and ann.annotation_last_placed is not None):
                ann._nudge_annotation_point(direction, step=step)
                return True
            else:
                # Navigate timepoints
                if direction == 'right':
                    ann._grid_next(ann.viewer_left)
                elif direction == 'left':
                    ann._grid_prev(ann.viewer_left)
                return True  # consume event
        return False  # pass other events through


class _TableDeleteFilter(QObject):
    """Intercept Delete/Backspace key on table widgets to remove selected row."""
    def __init__(self, annotator, side: int, table_type: str):
        super().__init__()
        self._ann = annotator
        self._side = side
        self._type = table_type  # 'annotation' or 'lattice'

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress and event.key() in (
                Qt.Key_Delete, Qt.Key_Backspace):
            if self._type == 'annotation':
                self._ann._delete_annotation_row(self._side)
            else:
                self._ann._delete_lattice_row(self._side)
            return True
        return False


class _ClipDragFilter(QObject):
    """Intercept Shift+MouseDrag on the canvas to rotate the arbitrary clip plane.

    Installed on the Qt canvas native widget BEFORE vispy processes events,
    so the camera rotation is suppressed while the clip plane rotates.
    Matches MIPAV's VolumeTriPlanarRenderBase.java:1921-1926 which swaps the
    rotation target from the scene to ArbRotate() during the drag.
    Only active when the current timepoint's frame_visible is true (the
    equivalent of MIPAV's DisplayArb() gate).
    """
    def __init__(self, annotator, side: int):
        super().__init__()
        self._ann = annotator
        self._side = side
        self._dragging = False
        self._last = None

    def eventFilter(self, obj, event):
        t = event.type()
        if t == QEvent.MouseButtonPress:
            if event.modifiers() & Qt.ShiftModifier:
                state = self._ann._get_clip_state(self._side)
                if state['frame_visible']:
                    self._dragging = True
                    self._last = event.pos()
                    # Drop volume ray-cast resolution (MIPAV trick) —
                    # restored on release.
                    self._ann._set_volume_quality(self._side, hi=False)
                    return True  # consume — vispy never sees this
        elif t == QEvent.MouseMove and self._dragging:
            cur = event.pos()
            dx = cur.x() - self._last.x()
            dy = cur.y() - self._last.y()
            self._ann._rotate_clip_by_mouse(self._side, dx, dy)
            self._last = cur
            return True
        elif t == QEvent.MouseButtonRelease and self._dragging:
            self._dragging = False
            self._last = None
            # Restore crisp volume sampling and force a final (non-throttled)
            # refresh so the frame settles exactly on the last rotation state
            # instead of whatever the throttle happened to catch.
            self._ann._set_volume_quality(self._side, hi=True)
            self._ann._flush_clip_refresh(self._side)
            return True
        return False


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
            # Clear table highlights when clicking on the canvas
            QTimer.singleShot(0, lambda: self._dual_window.clear_all_highlights())
        return False  # pass event through
