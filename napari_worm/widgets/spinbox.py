from qtpy.QtWidgets import QSpinBox


class _CommitOnEnterSpinBox(QSpinBox):
    """SpinBox that fires immediately on arrow clicks but waits for Enter when typing.

    - Arrow buttons / keyboard arrows: ``committed`` emitted on every step
    - Typing digits: ``committed`` emitted only on Enter or focus-out
    """
    from qtpy.QtCore import Signal
    committed = Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._typing = False
        self.lineEdit().textEdited.connect(self._on_text_edited)
        self.valueChanged.connect(self._on_value_changed)
        self.editingFinished.connect(self._on_editing_finished)

    def _on_text_edited(self, _text):
        self._typing = True

    def _on_value_changed(self, _val):
        if not self._typing:
            self.committed.emit()

    def _on_editing_finished(self):
        if self._typing:
            self._typing = False
            self.committed.emit()
