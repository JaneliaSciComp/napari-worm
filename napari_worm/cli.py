import argparse
from pathlib import Path

from qtpy.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLabel, QSpinBox, QVBoxLayout,
)

from napari_worm.app.annotator import WormAnnotator


def _parse_downsample(s: str | None) -> tuple | None:
    """Parse 'Z,Y,X' string to a 3-tuple of positive ints, or None if blank."""
    if not s or not s.strip():
        return None
    try:
        parts = [int(p.strip()) for p in s.strip().split(',')]
        if len(parts) != 3 or any(p < 1 for p in parts):
            raise ValueError
        return tuple(parts)
    except ValueError:
        raise ValueError(
            f"--downsample must be three positive integers in Z,Y,X order "
            f"(e.g. 1,2,2 to halve Y and X). Got: {s!r}")


def main():
    parser = argparse.ArgumentParser(
        description="napari_worm: 3D Cell Annotation Tool for C. elegans")
    parser.add_argument("volume", nargs="?", default=None,
                        help="TIFF file or directory of TIFFs "
                             "(opens file dialog if omitted)")
    parser.add_argument("--annotations", "-a", default=None,
                        help="Existing annotations CSV (optional)")
    parser.add_argument("--no-grid", action="store_true",
                        help="Use dask 4D slider instead of dual window mode")
    parser.add_argument("--start", "-s", type=int, default=0,
                        help="Starting timepoint index (default: 0)")
    parser.add_argument("--channels", "-c", default=None,
                        help="Comma-separated channel dir names (e.g. RegA,RegB). "
                             "Auto-discovered if omitted.")
    parser.add_argument("--downsample", "-d", default=None,
                        help="Block-mean downsample factors as Z,Y,X integers "
                             "(e.g. --downsample 1,2,2 halves Y and X). "
                             "Axis order is detected from TIFF metadata.")
    args = parser.parse_args()

    volume = args.volume
    if volume is None:
        # No path on command line — show a GUI dialog for path selection
        import json
        app = QApplication.instance() or QApplication([])

        # Load last-used path from settings file
        settings_file = Path.home() / ".napari_worm_settings.json"
        last_dir = ""
        last_start = 100
        last_downsample = ""
        if settings_file.exists():
            try:
                settings = json.loads(settings_file.read_text())
                last_dir = settings.get("last_dir", "")
                last_start = settings.get("last_start", 100)
                last_downsample = settings.get("last_downsample", "")
            except Exception:
                pass

        # Build a small dialog with path text field + Browse button + start timepoint
        from qtpy.QtWidgets import (QDialog, QDialogButtonBox, QLineEdit,
                                     QPushButton)
        dlg = QDialog()
        dlg.setWindowTitle("napari-worm — Select Volume Directory")
        dlg.setMinimumWidth(600)
        layout = QVBoxLayout(dlg)

        layout.addWidget(QLabel("Enter or browse to a volume directory (folder of TIFFs):"))
        path_row = QHBoxLayout()
        path_edit = QLineEdit(last_dir)
        path_edit.setPlaceholderText("/path/to/RegB  or  /Volumes/shroff/.../For_Tracking/RegB")
        browse_btn = QPushButton("Browse...")
        path_row.addWidget(path_edit)
        path_row.addWidget(browse_btn)
        layout.addLayout(path_row)

        start_row = QHBoxLayout()
        start_row.addWidget(QLabel("Start timepoint:"))
        start_spin = QSpinBox()
        start_spin.setRange(0, 9999)
        start_spin.setValue(last_start)
        start_spin.setFixedWidth(80)
        start_row.addWidget(start_spin)
        start_row.addStretch()
        layout.addLayout(start_row)

        ds_row = QHBoxLayout()
        ds_row.addWidget(QLabel("Downsample Z,Y,X:"))
        ds_edit = QLineEdit(last_downsample)
        ds_edit.setPlaceholderText("e.g. 1,2,2  (blank = no downsampling)")
        ds_edit.setFixedWidth(160)
        ds_row.addWidget(ds_edit)
        ds_row.addStretch()
        layout.addLayout(ds_row)

        def _browse():
            start = path_edit.text() or last_dir or "/Volumes"
            d = QFileDialog.getExistingDirectory(dlg, "Select directory", start,
                                                  QFileDialog.ShowDirsOnly)
            if d:
                path_edit.setText(d)

        browse_btn.clicked.connect(_browse)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        # Accept on Enter in the text field
        path_edit.returnPressed.connect(dlg.accept)

        if not dlg.exec():
            print("No directory selected — exiting.")
            return
        volume = path_edit.text().strip()
        if not volume:
            print("No directory selected — exiting.")
            return
        args.start = start_spin.value()

        args.downsample = args.downsample or ds_edit.text().strip() or None

        # Save the selected path, start timepoint, and downsample for next time
        try:
            settings_file.write_text(json.dumps({
                "last_dir": volume,
                "last_start": args.start,
                "last_downsample": args.downsample or "",
            }))
        except Exception:
            pass

    # If the selected directory has no TIFFs but contains Reg* subdirectories,
    # auto-resolve to the first channel dir (e.g. For_Tracking/ → RegB/)
    vp = Path(volume)
    if vp.is_dir() and not list(vp.glob("*.tif"))[:1]:
        reg_dirs = sorted([
            d for d in vp.iterdir()
            if d.is_dir() and d.name.startswith('Reg') and list(d.glob("*.tif"))[:1]
        ], key=lambda d: d.name)
        if reg_dirs:
            volume = str(reg_dirs[-1])  # prefer RegB (last alphabetically)
            print(f"No TIFFs in selected directory — using {reg_dirs[-1].name}/")
            print(f"  (sibling channels will be auto-discovered)")

    try:
        downsample = _parse_downsample(args.downsample)
    except ValueError as e:
        print(f"Error: {e}")
        return

    WormAnnotator(volume, args.annotations,
                  grid_mode=not args.no_grid, start_t=args.start,
                  channels=args.channels, downsample=downsample).run()


if __name__ == "__main__":
    main()
