"""
napari_worm: 3D Cell Annotation Tool for C. elegans

Usage:
    python napari_worm.py                                         # opens directory picker dialog
    python napari_worm.py /path/to/volume.tif                    # single file
    python napari_worm.py /path/to/directory/                     # dual view (MIPAV-style)
    python napari_worm.py /path/to/directory/ --start 100         # start at timepoint 100
    python napari_worm.py /path/to/directory/ --no-grid           # dask 4D slider (fallback)
    python napari_worm.py /path/to/volume.tif -a existing.csv     # load existing annotations
    python napari_worm.py /path/to/RegB/ --channels RegA,RegB    # explicit multi-channel
    python napari_worm.py /path/to/RegB/ --start 100             # auto-discovers RegA if present
"""
