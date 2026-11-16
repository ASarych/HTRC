"""Research-only two-pass wrapper around the UNMODIFIED HTRC Triton Tile.

Pass1 writes original far-field FP32 PM/PL/PACC. Pass2 reads exactly
those buffers and continues near-field math, on the same CUDA stream.
Opt-in proxy is patched only for one launch; no production import.
"""
from __future__ import annotations

from unittest.mock import patch

PINNED_GRID = (4 * 32, 4, 1)


class TwoPassTileProxy:
    def __init__(self, original, *, grid=PINNED_GRID):
        self.original = original
        self.grid = tuple(grid)
        self.compiled = None
        self.calls = 0

    def __getitem__(self, grid):
        if tuple(grid) != self.grid:
            raise AssertionError('Unexpected Tile grid for two-pass research')
        launch = self.original[grid]

        def two_pass(*args, **kwargs):
            if (kwargs.get('PASS') != 0 or kwargs.get('TAIL') != 0
                or kwargs.get('DOT') != 1 or kwargs.get('I8') != 1
                or kwargs.get('RB') != 1 or kwargs.get('GR') != 1
                or kwargs.get('FUSED_VB', 0) != 0
                or kwargs.get('VB_SPLIT_STREAM', 0) != 0
                or kwargs.get('FAR_PV_ACC') != 0
                or kwargs.get('NEAR_PV_ACC') != 0
                or kwargs.get('LATE_NEAR_Q') != 0
                or kwargs.get('BN') != 64
                or kwargs.get('BN_NEAR') != 64
                or kwargs.get('BATCH_TILES_PER_HEAD') != 32
                or kwargs.get('T') != 64
                or kwargs.get('D') != 128):
                raise AssertionError('Research Tile launch diverged from original PASS0 geometry')
            # Pass2 consumes the SAME PM/PL/PACC buffers in positional args
            # 9/10/11. No extra scratch tensor or intermediate conversion.
            if len(args) <= 16:
                raise AssertionError('Tile positional operands incomplete')
            far = launch(*args, **dict(kwargs, PASS=1))
            near = launch(*args, **dict(kwargs, PASS=2))
            self.compiled = (far, near)
            self.calls += 1
            return near

        return two_pass


def launch_two_pass_tile(batch, states, cfg, tr, *, buffers):
    """One stock batch launch with a temporarily substituted Triton launcher."""
    if buffers is None:
        raise ValueError('Preallocated FP32 partials are mandatory')
    q = states['Q']
    if (tuple(q.shape) != (4, 32, 64, 128)
        or states.get('_vb_mode') != 'materialized'
        or not states.get('_direct_q')
        or cfg['splits'] != 4 or cfg['exact_leaves'] != 512
        or cfg['kern'].get('tile_bn') != 64
        or cfg['kern'].get('tile_warps') != 4
        or cfg['kern'].get('tile_stages') != 2):
        raise AssertionError('Unpinned four-head Tile32 Balanced/512 two-pass input')
    proxy = TwoPassTileProxy(tr._tile_kernel)
    with patch.object(tr, '_tile_kernel', proxy):
        returned, compiled = batch.launch_prepared_tile_batch(
            states, cfg, tr, buffers=buffers)
    if returned is not buffers or proxy.calls != 1 or proxy.compiled is None:
        raise AssertionError('Expected exactly one two-pass Tile launch')
    if compiled is not proxy.compiled[1]:
        raise AssertionError('Last compiled kernel must be near pass')
    return returned, proxy.compiled


def validate_original_pass_source(source):
    """Refuse if source control flow no longer guarantees ordered exact resumes."""
    required = (
        'if PASS == 2:',
        'm_i = tl.load(PM + prior * T + r)',
        'l_i = tl.load(PL + prior * T + r)',
        'acc = tl.load(PACC + (prior * T + r)[:, None] * D + d[None, :])',
        'if PASS != 2:',
        'if PASS != 1:',
        'tl.store(PM + base * T + r, m_i)',
        'tl.store(PL + base * T + r, l_i)',
        'tl.store(PACC + (base * T + r)[:, None] * D + d[None, :], acc)',
    )
    for phrase in required:
        if source.count(phrase) != 1:
            raise RuntimeError('Canonical Tile PASS1/PASS2 source drift: ' + phrase)
    far_start = source.index('if PASS != 2:')
    near_start = source.index('if PASS != 1:')
    if not (source.index('if PASS == 2:') < far_start < near_start
            < source.index('tl.store(PM + base * T + r, m_i)')):
        raise RuntimeError('Canonical Tile pass ordering changed')
    return True
