# napari-worm

3D cell annotation and lattice building tool for *C. elegans* embryo analysis using [Napari](https://napari.org). Designed as a modern replacement for MIPAV's worm untwisting plugin.

Developed at the **Shroff Lab**, Janelia Research Campus.

**Author**: Diyi Chen (SciComp, Janelia Research Campus)

## Features

- **3D click-to-annotate**: Cmd+Click places markers at peak intensity along the camera ray (gradient ascent + intensity-weighted centroid for sub-voxel accuracy)
- **MIPAV-style dual view**: Two consecutive timepoints side-by-side with independent 3D canvases and linked navigation
- **Multi-channel support**: Auto-discovers RegA/RegB (or wavelength 405/488/561/637) channels and overlays them with additive blending (red + green). Per-channel histogram/contrast controls. Peak detection blends all visible channels (normalized by contrast limits) for accurate cross-channel placement, matching MIPAV's accurate mode.
- **Trilinear ray sampling**: Ray-volume intersection uses trilinear interpolation (`scipy.ndimage.map_coordinates`) instead of nearest-voxel, matching MIPAV's `getFloatTriLinearBounds` accurate mode.
- **Histogram + contrast controls**: MIPAV-style transfer function widget with 4 draggable control points, log scale, and per-channel colormap-aware display
- **Annotation tables**: Layers/Tables tab widget with live annotation table (Name, X, Y, Z, Intensity, Seg) and lattice table (Pair, L/R coords). Inline coordinate editing updates 3D view live. Click row to highlight point. DELETE key removes points. Lattice segment column for associating annotations with lattice pairs. All operations undoable.
- **Lattice mode**: Build left/right lattice pairs (nose-to-tail), with smooth natural cubic spline curves matching MIPAV's `NaturalSpline3(BT_FREE)`
- **Seam cells**: Cmd+Shift+Click to mark seam cells (H0, H1, H2, V1-V6, T) with automatic MIPAV-compatible naming
- **Insert / drag / nudge**: Click on curves to insert pairs, drag points to reposition, arrow keys to nudge by 1 voxel
- **Wireframe mesh**: 32 longitudinal splines around elliptical cross-sections (toggle with `W` key), matching MIPAV's `generateCurves()` / `generateEllipses()` algorithm
- **Surface mesh**: Solid triangle mesh rendered with smooth Phong shading and turbo colormap (toggle with `Shift+W`), matching MIPAV's `generateTriMesh()` — head/tail caps + body quads
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

# Multi-channel (auto-discovers sibling Reg* directories):
pixi run python napari_worm.py /path/to/For_Tracking/RegB/ --start 100
# ^ auto-detects RegA alongside RegB → red/green overlay

# Explicit channel selection:
pixi run python napari_worm.py /path/to/RegB/ --channels RegA,RegB --start 100

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
| `Shift+W` | Toggle surface mesh |
| `D` | Done with lattice (save + exit lattice mode) |
| `S` / `Cmd+S` | Save all (annotations + lattice) |
| `Delete` / `Backspace` | Remove selected row in table |
| `Right` / `]` | Next timepoint pair |
| `Left` / `[` | Previous timepoint pair |
| `Cmd+Z` | Undo last annotation or lattice point |
| Arrow keys | Nudge selected lattice point by 1 voxel |

## Output Format

Annotations and lattice data are saved per-timepoint in MIPAV-compatible CSV format (`name,x_voxels,y_voxels,z_voxels,R,G,B[,lattice_segment]`):

```
RegB/Decon_reg_100/Decon_reg_100_results/integrated_annotation/annotations_test.csv
RegB/Decon_reg_100/Decon_reg_100_results/lattice_final/lattice_test.csv
```

## Architecture

Single-file tool (`napari_worm.py`) built on Napari's viewer API:

- `DualViewWindow` — Two independent `QtViewer` canvases in a `QSplitter`, with tabified dock widgets for panel switching
- `WormAnnotator` — Main class handling annotation, lattice, wireframe, and surface state
- `generate_wireframe_mesh()` — Builds 32 longitudinal splines from lattice L/R pairs using orthogonal frame construction (tangent from center spline derivative, right vector from L-to-R direction, up from cross product)
- `generate_surface_mesh()` — Converts ellipse cross-sections into a triangle mesh `(vertices, faces, values)` for napari's `add_surface()`, matching MIPAV's `generateTriMesh()` vertex layout and face indexing

