import numpy as np


def _precalculate_cross_section_basis(n_max: int, n_samples: int) -> np.ndarray:
    """Replicates MIPAV's precalculateCrossSectionBasis (LatticeModel.java:8352-8399).

    Length-``n_samples`` impulse ``[1, 0, ..., 0]``, forward FFT, zero-padded into
    a length-``n_max`` complex array at ``(n_max - n_samples)//2``, inverse FFT.
    Returns the real part (length ``n_max``) — a bandlimited Dirichlet kernel.
    Narrower ``n_samples`` → wider falloff; ``n_samples == n_max`` → impulse
    (only the clicked vertex moves).
    """
    fft_shift_offset = (n_max - n_samples) // 2
    impulse = np.zeros(n_samples)
    impulse[0] = 1.0
    fft = np.fft.fft(impulse)
    padded = np.zeros(n_max, dtype=complex)
    padded[fft_shift_offset:fft_shift_offset + n_samples] = fft
    return np.real(np.fft.ifft(padded))


def _precalculate_cross_section_bases(num_ellipse_pts: int = 32) -> dict[int, np.ndarray]:
    """Precompute Fourier falloff kernels for all supported widths.

    Matches MIPAV's ``precalculateCrossSectionBases`` (LatticeModel.java:8402-8412)
    which stores kernels for ``nSamples ∈ {1, 2, 4, 8, 16, 32}`` indexed by
    ``log2(nSamples)``. Returns a dict keyed by ``nSamples`` for direct lookup.
    """
    return {ns: _precalculate_cross_section_basis(num_ellipse_pts, ns)
            for ns in (1, 2, 4, 8, 16, 32)}


_CROSS_SECTION_BASES = _precalculate_cross_section_bases(32)


def _apply_fourier_falloff(ring: np.ndarray, center: np.ndarray,
                           selected_vertex: int, delta_length: float,
                           n_samples: int, n_ellipse_pts: int = 32) -> np.ndarray:
    """Apply MIPAV's Fourier-kernel radial falloff to one cross-section ring.

    Mirrors the ``fourierPrecalculated`` branch of LatticeModel.updateLattice
    (LatticeModel.java:8606-8661): each of the 32 ring vertices displaces
    radially (toward/away from ``center``) by
    ``kernel[i] * (n_ellipse_pts / n_samples) * delta_length`` where ``i`` is
    the index offset from ``selected_vertex``.

    - ``n_samples = 32`` → impulse kernel → only the clicked vertex moves
      (the "Single" edit mode).
    - ``n_samples ∈ {4, 8, 16}`` → bandlimited kernel → bulge of decreasing width
      around the clicked vertex (Narrow / Medium / Wide).

    Parameters
    ----------
    ring : (n_ellipse_pts, 3) absolute ring vertex positions
    center : (3,) ring center point
    selected_vertex : index of the vertex the user grabbed
    delta_length : signed change in radius (positive = outward)
    n_samples : one of the keys of ``_CROSS_SECTION_BASES``

    Returns
    -------
    (n_ellipse_pts, 3) new ring positions. Input is not mutated.
    """
    if n_samples not in _CROSS_SECTION_BASES:
        return ring.copy()
    kernel = _CROSS_SECTION_BASES[n_samples]
    ratio = n_ellipse_pts // n_samples
    out = ring.copy()
    for i in range(n_ellipse_pts):
        idx = (selected_vertex + i) % n_ellipse_pts
        vertex = out[idx]
        radial = center - vertex
        r_norm = np.linalg.norm(radial)
        if r_norm < 1e-12:
            continue
        radial_unit = radial / r_norm
        displacement = kernel[i] * ratio * delta_length
        # MIPAV: current.sub(delta) — move vertex AWAY from center by positive displacement
        out[idx] = vertex - radial_unit * displacement
    return out


# Sentinel n_samples value for the Gaussian (5-vertex weighted) edit mode.
# Picked so the existing Fourier dispatch (n_samples ∈ {1,2,4,8,16,32}) is
# untouched and so the value can still be stored in `cross_section_n_samples: int`.
_GAUSSIAN_MODE = 0

# Hardcoded weights from MIPAV's `quickGaussian` branch (LatticeModel.java:8511-8544).
# Index = signed offset from the clicked vertex; weight ≈ exp(-offset²/2).
_GAUSSIAN_WEIGHTS = {0: 1.0, -1: 0.37, 1: 0.37, -2: 0.14, 2: 0.14}


def _apply_gaussian_falloff(ring: np.ndarray, center: np.ndarray,
                            selected_vertex: int, delta_length: float,
                            n_ellipse_pts: int = 32) -> np.ndarray:
    """Apply MIPAV's `quickGaussian` 5-vertex weighted radial falloff.

    Mirrors the dormant ``quickGaussian`` branch of LatticeModel.updateLattice
    (LatticeModel.java:8511-8544; ``final boolean quickGaussian = false`` keeps
    it disabled in MIPAV today). Only the clicked vertex and its ±1, ±2
    neighbors move radially toward/away from ``center``:

    - clicked vertex: ``1.0 × delta_length``
    - vertex ± 1:     ``0.37 × delta_length`` (≈ 1/e)
    - vertex ± 2:     ``0.14 × delta_length`` (≈ 1/e²)

    All other vertices unchanged. Local 5-vertex bulge → smoother than the
    Fourier-precalculated kernels at narrow widths (which can be spiky from
    bandlimit ringing).

    Parameters mirror :func:`_apply_fourier_falloff`. Returns a new array;
    ``ring`` is not mutated.
    """
    out = ring.copy()
    for offset, weight in _GAUSSIAN_WEIGHTS.items():
        idx = (selected_vertex + offset) % n_ellipse_pts
        vertex = out[idx]
        radial = center - vertex
        r_norm = np.linalg.norm(radial)
        if r_norm < 1e-12:
            continue
        radial_unit = radial / r_norm
        out[idx] = vertex - radial_unit * (weight * delta_length)
    return out
