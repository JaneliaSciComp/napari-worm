# napari-worm

3D cell annotation and lattice building tool for *C. elegans* embryo analysis using [Napari](https://napari.org). Designed as a modern replacement for MIPAV's worm untwisting plugin.

Developed at the **Shroff Lab**, Janelia Research Campus.

**Author**: Diyi Chen (SciComp, Janelia Research Campus)

## Features

- **3D click-to-annotate**: Cmd+Click places markers at peak intensity along the camera ray (gradient ascent + intensity-weighted centroid for sub-voxel accuracy)
- **MIPAV-style dual view**: Two consecutive timepoints side-by-side with independent 3D canvases and linked navigation
- **Lattice mode**: Build left/right lattice pairs (nose-to-tail), with smooth natural cubic spline curves matching MIPAV's `NaturalSpline3(BT_FREE)`
- **Seam cells**: Cmd+Shift+Click to mark seam cells (H0, H1, H2, V1-V6, T) with automatic MIPAV-compatible naming
- **Insert / drag / nudge**: Click on curves to insert pairs, drag points to reposition, arrow keys to nudge by 1 voxel
- **Wireframe mesh**: 32 longitudinal splines around elliptical cross-sections (toggle with `W` key), matching MIPAV's `generateCurves()` / `generateEllipses()` algorithm
- **MIPAV-compatible output**: Saves `annotations_test.csv` and `lattice_test.csv` per timepoint in the expected directory structure
- **Toast notifications**: In-app notifications for save confirmations, mode changes, and seam cell placement

## Requirements

- macOS (for Napari GUI)
- [Pixi](https://pixi.sh) package manager

## Installation

```bash
git clone git@github.com:JaneliaSciComp/napari-worm.git
cd napari-worm
pixi install
```

## Usage

```bash
# Dual-view mode (two consecutive timepoints side-by-side):
pixi run python napari_worm.py /path/to/RegB/ --start 100

# Single file mode:
pixi run python napari_worm.py /path/to/volume.tif

# Load existing annotations:
pixi run python napari_worm.py /path/to/volume.tif -a annotations.csv

# Dask 4D slider mode (fallback for large datasets):
pixi run python napari_worm.py /path/to/RegB/ --no-grid
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Cmd+Click` | Annotate at peak intensity (annotation mode) / place lattice point (lattice mode) |
| `Cmd+Shift+Click` | Place seam cell (lattice mode) |
| `L` | Toggle lattice mode |
| `W` | Toggle wireframe mesh |
| `D` | Done with lattice (save + exit lattice mode) |
| `S` / `Cmd+S` | Save annotations + lattice |
| `Right` / `]` | Next timepoint pair |
| `Left` / `[` | Previous timepoint pair |
| `Cmd+Z` | Undo last annotation or lattice point |
| Arrow keys | Nudge selected lattice point by 1 voxel |

## Output Format

Annotations and lattice data are saved per-timepoint in MIPAV-compatible CSV format (`name,x_voxels,y_voxels,z_voxels,R,G,B`):

```
RegB/Decon_reg_100/Decon_reg_100_results/integrated_annotation/annotations_test.csv
RegB/Decon_reg_100/Decon_reg_100_results/lattice_final/lattice_test.csv
```

## Architecture

Single-file tool (`napari_worm.py`) built on Napari's viewer API:

- `DualViewWindow` — Two independent `QtViewer` canvases in a `QSplitter`, with tabified dock widgets for panel switching
- `WormAnnotator` — Main class handling annotation, lattice, and wireframe state
- `generate_wireframe_mesh()` — Builds 32 longitudinal splines from lattice L/R pairs using orthogonal frame construction (tangent from center spline derivative, right vector from L-to-R direction, up from cross product)

