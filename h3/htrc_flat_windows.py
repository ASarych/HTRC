"""Exact variable-geometry query windows for the H3 Flat operator.

Only the query launch grid is reduced. K/V leaves and selection geometry are
unchanged. No CUDA imports and no hardcoded short-window tile count.
"""
from __future__ import annotations

from typing import NamedTuple


class QueryWindow(NamedTuple):
    start: int
    end: int
    valid_rows: int
    tile_count: int
    padded_rows: int


def iter_query_windows(tokens: int, window_tiles: int, *, tile_rows: int = 64,
                       center_group: int = 1):
    """Yield minimal group-aligned tile counts, never fewer valid Q rows."""
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 1
           for v in (tokens, window_tiles, tile_rows, center_group)):
        raise ValueError("Query dimensions must be positive integers")
    if window_tiles % center_group:
        raise ValueError("Configured query tiles must divide center group")
    window_rows = window_tiles * tile_rows
    for start in range(0, tokens, window_rows):
        end = min(start + window_rows, tokens)
        valid = end - start
        required = (valid + tile_rows - 1) // tile_rows
        active = ((required + center_group - 1) // center_group) * center_group
        if active > window_tiles:
            raise AssertionError("Query geometry exceeds configured window")
        yield QueryWindow(start, end, valid, active, active * tile_rows)


def compact_workspace_view(workspace, heads: int, active_tiles: int, config):
    """Reinterpret prefix of existing contiguous GPU workspace, no allocation.

    A shorter tail has fewer centers and tile partials per head. Reusing just
    :h,:tail on a multi-head buffer would create non-contiguous per-head gaps.
    Flattening the SAME storage and viewing its prefix avoids that problem.
    Sequential windows on one CUDA stream must not overlap writes.
    """
    if workspace is None:
        return None
    if not isinstance(workspace, dict) or heads < 1 or active_tiles < 1:
        raise ValueError("Invalid workspace or query dimensions")
    group = int(config.get("kern", {}).get("group", 1))
    if group < 1 or active_tiles % group:
        raise ValueError("Tail tile count must align to center group")
    max_tiles = int(workspace["_out"].shape[1])
    if (workspace["_out"].shape[0] != heads or
            active_tiles > max_tiles):
        raise ValueError("Workspace head count or tile capacity mismatch")
    if active_tiles == max_tiles:
        return workspace
    centers = active_tiles // group
    nb = int(workspace["G"].shape[2])
    dim = int(workspace["G"].shape[3])
    kx = int(workspace["IDX"].shape[-1])
    splits = int(workspace["_partials"]["pm"].shape[2])
    tile_rows = int(config["tile"])

    def prefix(buffer, shape):
        size = 1
        for extent in shape:
            size *= extent
        if not buffer.is_contiguous() or buffer.numel() < size:
            raise ValueError("Incompatible workspace prefix capacity/layout")
        return buffer.view(-1)[:size].view(shape)

    result = dict(workspace)
    shapes = {
        "G": (heads, centers, nb, dim),
        "VB": (heads, centers, nb, dim),
        "LZ": (heads, centers, nb),
        "SG": (heads, centers, nb),
        "sel": (heads * centers, nb),
        "IDX": (heads, centers, kx),
        "EXCL": (heads * centers, nb),
    }
    for key, shape in shapes.items():
        result[key] = prefix(workspace[key], shape)
    pshape = (heads, active_tiles, splits, tile_rows)
    result["_partials"] = {
        "pm": prefix(workspace["_partials"]["pm"], pshape),
        "pl": prefix(workspace["_partials"]["pl"], pshape),
        "acc": prefix(workspace["_partials"]["acc"], pshape + (dim,)),
    }
    result["_out"] = prefix(workspace["_out"],
                            (heads, active_tiles, tile_rows, dim))
    return result
