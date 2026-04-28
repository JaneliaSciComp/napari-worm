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
- **Wireframe mesh**: Lattice-aligned rings (one per L/R pair) + 32 longitudinal splines (toggle with `W`), matching MIPAV's `displayContours[]` / `generateEllipses()` algorithm
- **Surface mesh**: Solid triangle mesh rendered with smooth Phong shading and turbo colormap (toggle with `Shift+W`), matching MIPAV's `generateTriMesh()` — head/tail caps + body quads
- **Cross-section ring editing**: Rings tab → "Enable ring editing" (auto-enables Lattice + Wireframe) + Cmd+Click+Drag a wireframe ring vertex to reshape that cross-section radially. Five falloff modes: Single / Narrow / Medium / Wide use MIPAV's Fourier kernel (`precalculateCrossSectionBasis`, `LatticeModel.java:8352`); Gaussian is a 5-vertex weighted bulge (clicked + ±1 at 0.37× + ±2 at 0.14×, MIPAV's `quickGaussian` branch, `LatticeModel.java:8511`) — smoother than narrow Fourier kernels. Overrides persist to `<results>/model_crossSections/latticeCrossSection_<i>.csv` (one file per edited ring, MIPAV-compatible) and reload automatically on next launch. Reset button wipes current-timepoint overrides.
- **Arbitrary clip plane**: Per-timepoint arbitrary-orientation clipping with position and slab-thickness sliders. Shift+Drag on the canvas rotates the plane (MIPAV-style); red frame outline shows the clip slab. Volume ray-cast step size drops during drag for smooth interaction, then snaps back to full quality on release. Clips Image/Surface/Shapes/Points layers together.
- **Dual-view navigation**: NEXT/BACK buttons (and `]`/`[`) advance the sliding-window pair with auto-save of annotations for currently displayed timepoints before moving — matches MIPAV's `PlugInDialogVolumeRenderDual` workflow.
- **Threshold slider**: Global lower-contrast-limit slider in the layer-controls area, synced across both viewers and all channels. Non-destructive (adjusts contrast_limits, not data).
- **Preview mode (straightening)**: Preview tab → "Enable straightened view" resamples the twisted volume into a straightened tube using the lattice. Output axes are `Z=AP` (head→tail), `Y=DV`, `X=ML`. Cmd+Click in this view places annotations; positions are mapped back to twisted-pixel space via retwist so they persist when preview toggles off and save in the original coordinate system. When cross-section ring overrides exist for the timepoint, each AP slice is sampled out to its own per-ring outer-bound radius (max distance of the 32 cross-section vertices from the midline center) instead of a single global extent — narrower regions stay tight, wider regions get more room. Lattice/Wireframe/Ring-edit controls are greyed out while preview is on (read-mostly). Auto-exits on timepoint change. All worm-space math (splines, basis vectors, `straighten_volume`, `retwist`) runs in [Caroline Malin-Mayor's `celegans_model` package](https://github.com/ShroffLab/pyShroffCelegansModels) — napari-worm is a UI shell on top.
- **MIPAV-compatible output**: Saves per-timepoint with 1-indexed naming matching MIPAV exactly: annotations as `A1, A2, ...` in `annotations_test.csv`, lattice in `lattice_test.csv`, cross-section overrides as `latticeCrossSection_1.csv` through `latticeCrossSection_<n>.csv` (no `_0`).
- **Toast notifications**: In-app notifications for save confirmations, mode changes, and seam cell placement

## Quick Install (one command)

**macOS** (paste in Terminal):
```bash
curl -fsSL https://raw.githubusercontent.com/JaneliaSciComp/napari-worm/main/install.sh | bash
```

**Windows** (paste in PowerShell):
```powershell
irm https://raw.githubusercontent.com/JaneliaSciComp/napari-worm/main/install.ps1 | iex
```

This installs [Pixi](https://pixi.sh) (if needed), clones the repo, and sets up all dependencies. Run the same command again to update.

**Prerequisites**: Git ([Windows download](https://git-scm.com/download/win), pre-installed on macOS) and NRS drive access.

## Manual Installation

```bash
git clone https://github.com/JaneliaSciComp/napari-worm.git
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
| `Cmd+Click` (macOS) / `Ctrl+Click` (Windows) | Annotate at peak intensity (annotation mode) / place lattice point (lattice mode) |
| `Cmd+Shift+Click` (macOS) / `Ctrl+Shift+Click` (Windows) | Place seam cell (lattice mode) |
| `L` | Toggle lattice mode |
| `W` | Toggle wireframe mesh |
| `Shift+W` | Toggle surface mesh |
| `D` | Done with lattice (save + exit lattice mode) |
| `S` / `Cmd+S` | Save all (annotations + lattice) |
| `Delete` / `Backspace` | Remove selected row in table |
| `Right` / `]` / NEXT button | Next timepoint pair (auto-saves current annotations first) |
| `Left` / `[` / BACK button | Previous timepoint pair (auto-saves current annotations first) |
| `Shift+Drag` on canvas | Rotate arbitrary clip plane (when clip is enabled) |
| `Cmd+Click+Drag` on wireframe vertex | Reshape cross-section ring (requires Lattice + Wireframe + "Enable ring editing") |
| `Cmd+Z` | Undo last annotation or lattice point |
| Arrow keys | Nudge selected lattice point by 1 voxel |

## Output Format

Annotations and lattice data are saved per-timepoint in MIPAV-compatible CSV format (`name,x_voxels,y_voxels,z_voxels,R,G,B[,lattice_segment]`):

```
RegB/Decon_reg_100/Decon_reg_100_results/integrated_annotation/annotations_test.csv
RegB/Decon_reg_100/Decon_reg_100_results/lattice_final/lattice_test.csv
RegB/Decon_reg_100/Decon_reg_100_results/model_crossSections/latticeCrossSection_<i>.csv
```

## Architecture

Single-file tool (`napari_worm.py`) built on Napari's viewer API:

- `DualViewWindow` — Two independent `QtViewer` canvases in a `QSplitter`, with tabified dock widgets for panel switching
- `WormAnnotator` — Main class handling annotation, lattice, wireframe, surface, and preview state
- `generate_wireframe_mesh()` — Builds 32 longitudinal splines from lattice L/R pairs using orthogonal frame construction (tangent from center spline derivative, right vector from L-to-R direction, up from cross product)
- `generate_surface_mesh()` — Converts ellipse cross-sections into a triangle mesh `(vertices, faces, values)` for napari's `add_surface()`, matching MIPAV's `generateTriMesh()` vertex layout and face indexing
- `_make_celegans_model()` — Adapter that converts napari's `(N, 3)` (z,y,x) lattice arrays into Caroline's `(N, 2, 3)` `PythonCelegansModel`. Preview mode delegates `straighten_volume` and `retwist` to that model.

## Dependencies

- `napari`, `pyqt`, `numpy`, `pandas`, `tifffile`, `zarr`, `dask`, `scipy`, `pyqtgraph`, `superqt` (via `pixi.toml`)
- [`celegans-model`](https://github.com/ShroffLab/pyShroffCelegansModels) (Caroline Malin-Mayor / Funke Lab, BSD-3) — worm-space coordinate transforms used by Preview mode. Installed as a Python dep.

