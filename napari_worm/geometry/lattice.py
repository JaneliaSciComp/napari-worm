def _lattice_pair_name(pair_idx: int) -> str:
    """Return name prefix for the i-th L/R lattice pair (matches MIPAV a0/a1/... convention)."""
    return f"a{pair_idx}"


# Seam cell naming: MIPAV uses uppercase H0, H1, H2, V1-V6, T
# Each seam cell is also an L/R pair (H0L/H0R, V1L/V1R, TL/TR)
_SEAM_CELL_SEQUENCE = ['H0', 'H1', 'H2', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'T']


def _renumber_lattice_pairs(pair_infos: list[dict]) -> list[dict]:
    """Re-label all lattice pair names based on positional order.

    Matches MIPAV's updateSeamCount() (LatticeModel.java:8859).
    Walks through pairs nose→tail, numbering seam and non-seam independently:
      - Non-seam:  a0, a1, a2, ...
      - Seam (≤10 total): H0, H1, H2, V1, V2, V3, V4, V5, V6, T

    Pairs with ``custom_name=True`` (user-renamed) are skipped — their
    name is preserved across structural edits. Auto-numbered slots advance
    only for non-custom pairs, so the visible sequence stays consecutive
    among auto names.
    """
    seam_count = 0
    lattice_count = 0
    for info in pair_infos:
        if info.get('custom_name'):
            continue
        if info['type'] == 'seam':
            if seam_count < len(_SEAM_CELL_SEQUENCE):
                info['name'] = _SEAM_CELL_SEQUENCE[seam_count]
            else:
                info['name'] = f'S{seam_count}'
            seam_count += 1
        else:
            info['name'] = f'a{lattice_count}'
            lattice_count += 1
    return pair_infos
