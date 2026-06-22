from pathlib import Path

import dask.array as da
import numpy as np
import tifffile
from dask import delayed


def _block_mean(data: np.ndarray, factors: tuple) -> np.ndarray:
    """Downsample by block-averaging. factors[i] is the factor for axis i."""
    result = data
    for ax, f in enumerate(factors):
        if f <= 1:
            continue
        n = result.shape[ax]
        n2 = (n // f) * f
        idx = [slice(None)] * result.ndim
        idx[ax] = slice(0, n2)
        result = result[tuple(idx)]
        shape = list(result.shape)
        shape[ax] = n2 // f
        shape.insert(ax + 1, f)
        result = result.reshape(shape).mean(axis=ax + 1)
    return result.astype(data.dtype)


def _tiff_axes(path: Path) -> str:
    """Return axis-order string from TIFF metadata; default 'ZYX' for 3-D."""
    try:
        with tifffile.TiffFile(str(path)) as tf:
            if tf.series:
                axes = tf.series[0].axes
                if axes:
                    return axes.upper()
    except Exception:
        pass
    return 'ZYX'


def load_volume(path: str | Path, downsample: tuple | None = None) -> np.ndarray:
    """Load a 3D TIFF volume.

    downsample=(fZ, fY, fX): block-average each spatial axis by that factor.
    Axis order is detected from TIFF metadata so the factors apply to the
    correct physical dimensions regardless of on-disk storage order.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Volume not found: {path}")
    axes = _tiff_axes(path)
    data = tifffile.imread(str(path))
    print(f"Loaded volume: {path.name}  shape={data.shape}  axes={axes}  dtype={data.dtype}")
    if downsample is not None and any(f > 1 for f in downsample):
        fz, fy, fx = downsample
        # Normalize common Z aliases (ImageJ uses 'I', some writers use 'Q'/'S')
        _Z_ALIASES = {'I', 'Q', 'S'}
        axis_factor = {c: fz for c in _Z_ALIASES}
        axis_factor.update({'Z': fz, 'Y': fy, 'X': fx})
        factors = tuple(axis_factor.get(c, 1) for c in axes)
        data = _block_mean(data, factors)
        print(f"  Downsampled Z×{fz} Y×{fy} X×{fx} → shape={data.shape}")
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


def _channel_colormap(name: str) -> str:
    """Assign colormap by channel directory name (matching MIPAV conventions)."""
    n = name.lower()
    if '405' in n or 'blue' in n:
        return 'blue'
    if '488' in n or 'green' in n:
        return 'green'
    if '561' in n or 'red' in n:
        return 'red'
    if n == 'rega':
        return 'red'
    if n == 'regb':
        return 'green'
    return 'gray'


def discover_channels(volume_path: Path, explicit: str | None = None) -> list[tuple[Path, str, str]]:
    """Discover channel directories.

    Returns list of (directory, name, colormap) tuples.
    Single channel returns one entry with 'gray' colormap.
    """
    if explicit:
        parent = volume_path.parent
        channels = []
        for ch_name in explicit.split(','):
            ch_name = ch_name.strip()
            ch_dir = parent / ch_name
            if ch_dir.is_dir() and list(ch_dir.glob("*.tif"))[:1]:
                channels.append((ch_dir, ch_name, _channel_colormap(ch_name)))
            else:
                print(f"Warning: channel dir not found or empty: {ch_dir}")
        if channels:
            return channels

    # Auto-discover sibling directories that contain .tif files
    parent = volume_path.parent
    siblings = sorted([
        d for d in parent.iterdir()
        if d.is_dir() and d.name.startswith('Reg') and list(d.glob("*.tif"))[:1]
    ], key=lambda d: d.name)

    if len(siblings) >= 2:
        channels = [(d, d.name, _channel_colormap(d.name)) for d in siblings]
        print(f"Discovered {len(channels)} channels: {[c[1] for c in channels]}")
        return channels

    # Single channel — gray
    return [(volume_path, volume_path.name, 'gray')]


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
