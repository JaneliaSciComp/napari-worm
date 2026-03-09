# napari_worm: 3D Cell Annotation Tool

**Project Lead**: Diyi Chen (SciComp)
**Lab**: Shroff Lab, Janelia Research Campus
**Created**: February 12, 2026
**Status**: Phase 5 - Dual View + Lattice Working
**Last Updated**: March 4, 2026
**Repository**: https://github.com/JaneliaSciComp/napari-worm (private)

---

## Execution Environments

This project can be run in two environments:

1.  **Local Mac Environment (Recommended for GUI)**: Running on your local machine allows you to use the Napari graphical user interface. The project should be cloned to your local machine, and the data accessed via a mounted network drive.

2.  **Cluster Environment (for scripting and headless processing)**: The project can also be run on a cluster for batch processing or scripting, but the Napari GUI will not be available. If running on a new cluster environment, ensure to navigate to the project directory and run `pixi install` to set up the environment.

---

## Running Locally on Mac (Successful Launch on Feb 13, 2026)

This section documents the successful launch of Napari on a local Mac.

### Summary of Steps Taken

1.  **Cloned Project Locally:** The project was copied from the network drive to `/Users/chend/julia/napari_worm`.
2.  **Pinned Python Version:** The `pixi.toml` file was modified to pin the Python version to `3.11` to resolve a `pydantic` compatibility issue with Python `3.14`.
3.  **Installed Dependencies:** `pixi install` was run in the local project directory.
4.  **Launched Napari:** The application was successfully launched with the following command.

### Launch Commands

From the project's root directory (`/Users/chend/julia/napari_worm`):

```bash
# Single file mode:
pixi run python napari_worm.py "/Volumes/shroff/.../RegB/Decon_reg_100.tif"

# Dual-view mode (directory of TIFFs):
pixi run python napari_worm.py "/Volumes/shroff/.../RegB/"

# Dual-view starting at timepoint 100:
pixi run python napari_worm.py "/Volumes/shroff/.../RegB/" --start 100

# Dask 4D slider mode (fallback):
pixi run python napari_worm.py "/Volumes/shroff/.../RegB/" --no-grid
```

---

## Progress Log

### Feb 13, 2026 — Phase 1 Complete
1.  [x] **Run napari on Mac and load MIPAV TIFF files**
2.  [x] Single-file loading, Ctrl+Click annotation, save/load CSV

### Feb 17, 2026 — Phase 5 (Time Series / Dual View)
3.  [x] **Dask lazy loading** — Pass a directory to load all TIFFs as a 4D dask array with time slider. Works but slow over network (~82 MB per timepoint read). Available via `--no-grid` flag.
4.  [x] **Grid mode (dual view)** — Show two consecutive timepoints side by side using `viewer.grid`. Grid layout, navigation (Right/Left keys), and linked rotation all work correctly.
5.  [x] **MIPAV dual-view code review** — Analyzed `PlugInDialogVolumeRenderDual.java` ([PR #9](https://github.com/JaneliaSciComp/mipav/pull/9)). MIPAV uses two completely separate renderer panels (`JSplitPane` with independent OpenGL canvases). Each panel has its own mouse handling via `setActiveRenderer(this)` callback. Rotations are NOT synced.
6.  [x] **Translate-based dual view** — Tried replacing `viewer.grid` with spatial offset. Reverted because: (a) rotation moves entire scene so left/right models swap positions, (b) image quality was blurry. Grid mode is better.
7.  [x] **Numeric file sorting** — Fixed `sorted(glob("*.tif"))` from lexicographic (`0, 1, 10, 100`) to numeric (`0, 1, 2, ..., 9, 10, 11`).
8.  [x] **Global undo stack** — Ctrl+Z undoes the most recent annotation regardless of which side. Works across navigation (undo from cached annotations if you navigated away). Can press multiple times to walk back.
9.  [x] **Per-timepoint save** — Annotations save to MIPAV-compatible path: `RegB/Decon_reg_X/Decon_reg_X_results/integrated_annotation/annotations_test.csv`
10. [x] **Navigation spinboxes** — Docked widget at bottom with two spinboxes ("Left t=" and "Right t=") for independent timepoint selection. Can jump directly to any timepoint (e.g., t=100) without pressing Right repeatedly. Arrow keys still step both by 1.
11. [x] **`--start` flag** — Launch directly at a specific timepoint: `--start 100` opens t=100 and t=101.
12. [x] **Global undo fix** — Undo from cache now also updates the currently displayed layer if that timepoint is visible on screen.
13. [x] **Early timepoints are noisy** — Confirmed that Decon_reg_0 through ~Decon_reg_10 have diffuse/noisy signal. Clear nuclei visible from ~Decon_reg_100 onward. Annotations exist for timepoints 100-140.

### Current working workflow (Mar 4, 2026)

**Annotation mode** (default):
1. Launch → two canvases side by side, left panel shows t=100 controls
2. Click right canvas → left panel switches to t=101 controls
3. `Cmd+Click` on either canvas → peak-intensity annotation added (gradient ascent + centroid)
4. `Cmd+Z` → undo last annotation
5. `Right`/`]` or `Left`/`[` → navigate both timepoints; spinboxes for independent jump
6. `S` → saves `annotations_test.csv` per timepoint

**Lattice mode** (press `L`):
1. `Cmd+Click` → Left lattice point (cyan square, named a0L, a1L...)
2. `Cmd+Shift+Click` → Right lattice point (magenta square, named a0R, a1R...)
3. Yellow lines appear between each complete L/R pair; white centerline after ≥2 pairs
4. `Cmd+Z` → undo last lattice point
5. Press `L` again → back to annotation mode
6. `S` → saves `lattice_test.csv` per timepoint (interleaved a0L,a0R,a1L,a1R...)

### TIFF File Structure (from `tifffile` inspection)
- **Not tiled**, uncompressed, strip-based ImageJ TIFFs
- 308 Z-slices per file, each 425x325 uint16 (~82 MB per volume)
- No compression (`Compression: 1`)
- Mark Kittisopikul noted: Ben uses NetCDF/Zarr downstream; zarr conversion could help with chunked access and multiscale pyramids but not needed for MVP.
- **Multiscale only works in 2D** (Mark K.): napari only renders the lowest resolution scale in 3D. Zarr pyramids would NOT help our 3D MIP annotation workflow. See: [napari.Viewer.add_image docs](https://napari.org/stable/api/napari.Viewer.html#napari.Viewer.add_image).

---

## Current TODO

1.  [x] **Fix click routing without sidebar**
2.  [x] **Peak accuracy** — gradient ascent + intensity-weighted centroid (sub-voxel)
3.  [x] **MIPAV-style dual view** — tabified dock panels, independent cameras, no freeze
4.  [x] **MIPAV-compatible save format** — `name,x_voxels,y_voxels,z_voxels,R,G,B`
5.  [x] **Lattice mode** — L/R pairs, midline, save lattice_test.csv
6.  [ ] **Validate against MIPAV** — annotate same timepoints in both tools, compare coordinates
7.  [ ] **Trilinear interpolation in `sample_ray`** — replace `np.round()` with trilinear blend (MIPAV's `getFloatTriLinearBounds`). Improves initial ray-peak seed quality.
8.  [ ] **"Next Peak" / N key** — secant slope peak cycling along the ray. Algorithm: slope = `values[i+2] - values[i]`, find +→- transition above threshold, interpolate sub-voxel. Solves MIP occlusion: foreground nucleus is bright, press N to jump to dim nucleus behind it.
9.  [ ] **Intensity profile chart** — matplotlib widget showing ray intensity after each click.
10. [ ] **Adjustable threshold** — expose `threshold_fraction=0.5` as UI slider.
9.  [x] **MIPAV-style dual view — COMPLETE (Mar 4, 2026)**

    **Architecture**: `DualViewWindow` wraps `viewer_left.window._qt_window` (NapariQtMainWindow) as host. Replaces central widget with `QSplitter` of two `QtViewer` widgets + nav bar at bottom. Right viewer's dock widgets tabified behind left viewer's via `tabifyDockWidget` — `raise_()` on click switches tabs (no layout recalculation, no GL freeze).

    **Key implementation lessons**:
    - Use `viewer_left.window._qt_window` as host (not a custom QMainWindow) → styling, menus, console, all work for free
    - Reparent `_qt_viewer` into canvas splitter BEFORE `setCentralWidget` to prevent Qt from destroying it
    - Show window + `processEvents()` BEFORE loading layers — GL context must exist when vispy visuals are created
    - Use `tabifyDockWidget` + `raise_()` for panel switching — `setVisible()` on dock widgets triggers GL canvas resize → macOS freeze
    - Defer dock switch with `QTimer.singleShot(0, ...)` in `_CanvasClickFilter`
    - All Points layers always in `pan_zoom` mode — our Cmd+Click handler adds points programmatically
    - `QTimer` at 100ms forces canvas redraws (mirrors MIPAV's `startAnimator`)

    **Keyboard controls** (Mac):
    - `Cmd+Click`: annotate at peak intensity
    - `L`: toggle lattice mode
    - `Cmd+Z`: undo last annotation or lattice point
    - `S` / `Cmd+S`: save annotations + lattice
    - `Right` / `]`: next timepoint pair; `Left` / `[`: previous

10. [x] **MIPAV-compatible annotation save format (Mar 4, 2026)**

    Saves to `name,x_voxels,y_voxels,z_voxels,R,G,B` with float sub-voxel coordinates.
    - Annotations: `A0, A1, A2...` → `integrated_annotation/annotations_test.csv`
    - Lattice: `a0L, a0R, a1L, a1R...` (interleaved pairs) → `lattice_final/lattice_test.csv`
    - Uses `_test` suffix to avoid overwriting real MIPAV output while validating

11. [x] **Lattice mode — COMPLETE (Mar 4, 2026)**

    Mirrors MIPAV's `addLeftRightMarker` workflow:
    - Press `L` to enter lattice mode; `L` again to return to annotation mode
    - `Cmd+Click` → Left lattice point (cyan square); `Cmd+Shift+Click` → Right lattice point (magenta square)
    - Yellow lines connect each L/R pair; white path shows centerline through midpoints
    - `Cmd+Z` undoes last lattice point with line/midline update
    - Auto-names pairs a0, a1, a2... (matching real MIPAV lattice.csv format)
    - 5 layers per viewer: Volume, Annotations, Lattice Left, Lattice Right, Lattice Lines, Lattice Mid
    - Saves per-timepoint to `lattice_final/lattice_test.csv`

12. [x] **Smooth lattice curves — COMPLETE (Mar 9, 2026)**

    Replaced straight-line midline with smooth Natural Cubic Splines matching MIPAV's
    `NaturalSpline3(BT_FREE, ...)` from WildMagic. Uses `scipy.interpolate.CubicSpline(bc_type='natural')`
    with arc-length parametrization. Three smooth curves now drawn:
    - **Red** center midline (through L/R midpoints) — matches MIPAV `Color.red`
    - **Magenta** left curve (through left lattice points) — matches MIPAV `Color.magenta`
    - **Green** right curve (through right lattice points) — matches MIPAV `Color.green`
    - Yellow cross-rungs still connect each L/R pair
    - 7 layers per viewer: Volume, Annotations, Lattice Left, Lattice Right, Lattice Lines, Lattice Mid, Lattice Left Curve, Lattice Right Curve

13. [x] **Lattice workflow overhaul — PARTIAL (Mar 9, 2026)**

    Redesigned lattice mode to match MIPAV's full workflow (based on meeting with Ryan):

    **Completed:**
    - `Cmd+Click` alternates L→R automatically (nose to tail, left to right)
    - `Cmd+Shift+Click` = seam cell (H0, H1, H2, V1-V6, T) — also L/R pairs
    - All points (lattice + seam cells) saved in single `lattice_test.csv` matching MIPAV format:
      `a0L, a0R, a1L, a1R, H0L, H0R, ...` (interleaved, nose to tail order)
    - Positional renumbering via `_renumber_lattice_pairs()` (matches MIPAV `updateSeamCount()`)
    - Thinner lattice lines (`edge_width=1`)
    - `D` key = done (save + exit lattice mode)
    - Arrow key nudging via `_ArrowKeyFilter` event filter (intercepts before napari canvas)
    - Mouse drag for moving existing points (generator-based `mouse_drag_callbacks`)
    - Callbacks registered on all layers so Cmd+Click works regardless of sidebar selection
    - Network drive save retry (handles `BlockingIOError`)

    **Known issues / TODO:**
    - [ ] Click routing still requires selecting a lattice layer in sidebar first — need to verify
          the all-layers callback registration fix works
    - [ ] Insert point between existing pairs not working reliably — `_find_insertion_index()`
          threshold too restrictive; needs curve-parameter projection instead of distance threshold
    - [ ] Arrow key nudge needs testing — may still be intercepted by napari in some cases
    - [ ] Mouse drag needs testing — drag detection threshold (8 voxels) may be too small

    **MIPAV references for remaining work:**
    - `addInsertionPoint()` — LatticeModel.java:6248 — inserts both L+R, R interpolated
    - `modifyLattice()` — LatticeModel.java:2353 — CTRL+drag moves picked point
    - `moveSelectedPoint()` — LatticeModel.java:2441 — arrow key nudge (camera-relative)
    - `updateSeamCount()` — LatticeModel.java:8859 — positional renaming after insert

14. [ ] **Wireframe mesh**

    Build a 3D wireframe surface around the worm body from the lattice curves.
    Per Mark Kittisopikul: "Essentially we are drawing a circle in 3D space centered around
    the midpoint between each pair of seam cells. To draw that circle you need to establish
    the plane in which to draw it." This requires vector geometry.

    **Algorithm** (from MIPAV `LatticeModel.generateCurves()` + `generateEllipses()`):
    1. At each uniformly-sampled point along the center spline, compute an orthogonal frame:
       - **Tangent**: 1st derivative of center spline (`centerSpline.GetTangent(t)`)
       - **Right vector**: direction from left curve to right curve (interpolated), normalized
       - **Up vector**: cross product of tangent × right vector
    2. Compute the worm diameter at that point (distance between left and right curves)
    3. Draw a circle (or ellipse) in the plane defined by (right, up) centered at the midline point
       - Major axis = half the L/R distance; minor axis ≈ half the major axis
    4. Connect circles at adjacent slices with longitudinal lines → wireframe tube
    5. Display as a napari `Surface` or `Shapes` layer

    **Key MIPAV references**:
    - `generateEllipses(int extent)` — LatticeModel.java:6561
    - Orthogonal frame construction — LatticeModel.java:4354-4398
    - Sampling plane corners — `center ± rightV ± upV`

    **Open questions**:
    - How many points per cross-section circle? (MIPAV uses `numEllipsePts`)
    - Render as wireframe lines or semi-transparent surface mesh?
    - Interactive editing of the mesh? (relates to "Surface editing" and "How to round mesh" TODOs)

---

## Test Data

| Dataset | Path | Files | Description |
|---------|------|-------|-------------|
| efn-2/Pos1 | `/nrs/shroff/data_internal/celegans_mipav_data/2025_12_16/efn-2/120222_efn-2/Pos1/SPIMB/Reg_Sample/For_Tracking/RegB/` | 141 TIFFs | Full time-series, 16-bit 3D volumes (~82 MB each) |

**Single timepoint with existing annotations (for comparison):**
```python
tiff_path = "/nrs/shroff/data_internal/celegans_mipav_data/2025_12_16/efn-2/120222_efn-2/Pos1/SPIMB/Reg_Sample/For_Tracking/RegB/Decon_reg_100.tif"
```
Note: Annotations exist for timepoints 100-140. Use `Decon_reg_100.tif` if you want to compare with existing MIPAV annotations.

**Annotations available at:**
```
RegB/Decon_reg_X/Decon_reg_X_results/integrated_annotation/annotations.csv
```

---

## Objective

Build a simple, straightforward 3D cell annotation tool using **Napari** for C. elegans embryo analysis. The tool should:

1. **Load and visualize 3D multi-channel microscopy volumes**
2. **Allow clicking in 3D to place annotation markers at nuclei centers**
3. **Automatically find peak intensity along the viewing ray**
4. **Save/load annotations for the untwisting pipeline**

This tool is inspired by MIPAV's worm untwisting plugin, but designed to be simpler and leverage Napari's built-in capabilities directly.

---

## Why Napari?

| MIPAV | Napari |
|-------|--------|
| Java, complex setup | Python, easy `pip install` |
| Large monolithic codebase | Simple scripts |
| Hard to modify | Easy to customize |
| Limited integration | Works with Python ecosystem |

**Key Napari features we use:**
- `viewer.add_image()` - 3D volume rendering
- `viewer.add_points()` - Annotation markers
- `layer.get_ray_intersections()` - 3D picking (converts click to ray)
- Mouse callbacks - Custom click handling

---

## Core Concept: 3D Clicking

When you click in Napari's 3D view:

```
Click on screen
      ↓
Napari calculates a RAY through the volume
      ↓
layer.get_ray_intersections() gives us:
  - near_point (where ray enters volume)
  - far_point (where ray exits volume)
      ↓
We sample intensity along this ray
      ↓
Find the PEAK (brightest point = nucleus center)
      ↓
Place annotation marker at peak position
```

This is exactly what MIPAV does, but Napari gives us the ray intersection for free!

---

## Implementation Plan

### Phase 1: Basic 3D Clicking (MVP)

**Goal**: Click in 3D → place marker at peak intensity

```python
import napari
import numpy as np

# Load data
viewer = napari.Viewer(ndisplay=3)
volume_layer = viewer.add_image(data, name='Volume')
points_layer = viewer.add_points(ndim=3, name='Annotations')

@volume_layer.mouse_drag_callbacks.append
def on_click(layer, event):
    if 'Control' not in event.modifiers:
        return

    # Get ray through volume
    near, far = layer.get_ray_intersections(
        event.position, event.view_direction, event.dims_displayed
    )

    if near is None:
        return

    # Sample along ray and find peak
    peak_pos = find_peak_along_ray(layer.data, near, far)

    # Add annotation
    points_layer.add(peak_pos)

napari.run()
```

**Deliverables:**
- [x] Load TIFF/zarr volumes
- [x] Ctrl+Click to place annotations
- [x] Simple peak detection (max intensity along ray)
- [x] Save/load annotations as CSV (supports MIPAV format)

### Phase 2: Better Peak Detection

**Goal**: Sub-voxel accuracy using secant method (like MIPAV)

The secant method finds where the intensity derivative crosses zero:

```python
def find_peaks_secant(values):
    """Find peaks using MIPAV's secant slope method."""
    # Calculate slopes over 2-sample span (smoother than point-to-point)
    slopes = (values[2:] - values[:-2]) / 2.0

    # Find where slope changes from positive to negative
    peaks = []
    for i in range(len(slopes) - 1):
        if slopes[i] > 0 and slopes[i+1] < 0:
            # Interpolate exact position
            t = -slopes[i] / (slopes[i+1] - slopes[i])
            peak_idx = i + t + 1
            peaks.append(peak_idx)

    return peaks
```

**Deliverables:**
- [ ] Secant-based peak detection
- [ ] Sub-voxel interpolation
- [ ] Threshold filtering (ignore dim peaks)
- [ ] "Next Peak" navigation

### Phase 3: UI Enhancements

**Goal**: Intensity profile display, better controls

**Deliverables:**
- [ ] Show intensity profile along ray (matplotlib widget)
- [ ] Click on profile to adjust marker position
- [ ] Keyboard shortcuts (N = next peak, Delete = remove)
- [ ] Multi-channel support

### Phase 4: Integration

**Goal**: Work with existing pipeline

**Deliverables:**
- [ ] Load existing MIPAV annotations
- [ ] Export for T4D pipeline
- [ ] Time series support (4D)

### Phase 5: Dual-View Time Series (In Progress)

**Goal**: View two consecutive timepoints side by side for annotation, inspired by MIPAV's dual panel.

**Approaches tried**:
1.  ~~Dask 4D slider~~ — Works but slow over network. Available via `--no-grid` flag.
2.  **Napari grid mode (current)** — Shows two volumes side by side with linked rotation. Click routing requires selecting the Volume layer in sidebar first. This is the working approach.
3.  ~~Translate-based offset~~ — Volumes in same scene with spatial offset. Reverted: rotation moves entire scene (left/right swap), image quality degraded.

**How MIPAV does it** (for reference):
- Two separate `VolumeTriPlanarRender` instances in a `JSplitPane`
- Each has its own OpenGL canvas and independent mouse handling
- Active panel indicated by red border
- `setActiveRenderer(this)` callback routes clicks
- Per-timepoint state: annotations, rotation matrix, clip planes
- See: `PlugInDialogVolumeRenderDual.java` in [mipav repo](https://github.com/JaneliaSciComp/mipav)

**Keyboard controls**:
- `Right` / `]`: advance both timepoints by 1
- `Left` / `[`: go back both by 1
- `Ctrl+Z`: undo last annotation (global, works across sides and navigation)
- `S`: save annotations per-timepoint (MIPAV-compatible path)
- **Navigation widget**: spinboxes at bottom for independent left/right timepoint selection

**Annotations saved per-timepoint** (MIPAV-compatible):
```
RegB/Decon_reg_X/Decon_reg_X_results/integrated_annotation/annotations_test.csv
```

---

## File Structure

```
napari_worm/
├── PROJECT_PLAN.md       # This document
├── COPY_TEST_DATA.md     # Instructions for copying test data
├── pixi.toml             # Environment config (napari, tifffile, etc.)
├── pixi.lock             # Locked dependencies
├── napari_worm.py        # Main tool (single file!)
├── .gitignore
└── Shroff_Lab_migration/ # Data migration scripts (reference)
```

**Note**: We keep it simple - one main file, not a complex package structure.
**Repository**: https://github.com/JaneliaSciComp/napari-worm (private)

**Important**: Napari requires a display. Run on Mac or via VNC/X11 forwarding.

---

## Key Functions to Implement

### 1. `find_peak_along_ray(data, start, end)`
Sample intensity along ray, return position of maximum.

### 2. `find_peaks_secant(values, threshold)`
Find all peaks using derivative analysis.

### 3. `interpolate_position(positions, fractional_idx)`
Get 3D position from fractional index.

### 4. `save_annotations(points, filepath)`
Export to CSV (z, y, x format).

### 5. `load_annotations(filepath)`
Import from CSV.

---

## Technical Notes

### Napari's Ray Intersection API

```python
near_point, far_point = layer.get_ray_intersections(
    position=event.position,        # Click position in world coords
    view_direction=event.view_direction,  # Camera direction
    dims_displayed=event.dims_displayed,  # Which dims shown (e.g., [0,1,2])
    world=True                      # Return world coordinates
)
```

### Sampling Along Ray

```python
def sample_ray(data, start, end, n_samples=None):
    """Sample intensity values along a ray."""
    if n_samples is None:
        n_samples = int(np.linalg.norm(end - start)) + 1

    positions = np.linspace(start, end, n_samples)
    values = []

    for pos in positions:
        idx = np.round(pos).astype(int)
        if np.all(idx >= 0) and np.all(idx < data.shape):
            values.append(data[tuple(idx)])
        else:
            values.append(0)

    return positions, np.array(values)
```

---

## Development Timeline

| Week | Goal |
|------|------|
| 1 | Basic clicking + peak detection working |
| 2 | Secant method + save/load |
| 3 | UI enhancements (profile display) |
| 4 | Test with real data, get feedback |

---

## References

### Napari Docs
- [3D Interactivity](https://napari.org/stable/guides/3D_interactivity.html)
- [Cursor Ray Example](https://napari.org/dev/gallery/cursor_ray.html)
- [Custom Mouse Functions](https://napari.org/dev/gallery/custom_mouse_functions.html)

### MIPAV Reference
- Key algorithm: `PickVolume3D()` in `VolumeTriPlanarRender.java`
- Peak detection: `nextPeak()` in `SelectionChartPanel.java`
- Dual view: `PlugInDialogVolumeRenderDual.java` — two separate renderers in `JSplitPane`
- Source: https://github.com/JaneliaSciComp/mipav (fork: https://github.com/dchen116/mipav)
- PR with dual-view changes: https://github.com/JaneliaSciComp/mipav/pull/9

### Napari Dask Tutorial
- [Using dask with napari](https://napari.org/stable/tutorials/processing/dask.html)
- [Grid mode](https://napari.org/dev/gallery/grid_mode.html)
- [OME-Zarr visualization](https://imaging.epfl.ch/field-guide/sections/image_data_visualization/notebooks/visualization_zarr.html)

---

## Contact

Diyi Chen - chend@janelia.hhmi.org
