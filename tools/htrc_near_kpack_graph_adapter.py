"""Research-only opt-in shared near-K8 scratch for 56-head Graph captures.

One 8-head x 16-centre K8+scale buffer is shared by both full/partial
window CUDA Graphs. The intercepted Python dispatcher is ONLY
flow._one_window; the original eager reference is never patched.
No default HTRC code is modified.
"""
from __future__ import annotations
from contextlib import contextmanager
from unittest.mock import patch
import math

GROUP_HEADS, MAX_TILES, ROWS, DIM = 8, 16, 64, 128
NEAR_LEAVES, LEAF, SPLITS, BN = 512, 16, 4, 64
NEAR_TOKENS = NEAR_LEAVES * LEAF
SCRATCH_BYTES = GROUP_HEADS * MAX_TILES * NEAR_TOKENS * (DIM + 4)
assert SCRATCH_BYTES == 138412032


class NearKGraphAdapter:
    def __init__(self, torch, flow, batch, pack_kernel, tile_kernel):
        self.torch = torch
        self.flow = flow
        self.batch = batch
        self.pack_kernel = pack_kernel
        self.tile_kernel = tile_kernel
        self.original_window = flow._one_window
        self._packed_k = None
        self._packed_ks = None
        self._calls = 0
        self._windows_seen = set()

    @contextmanager
    def capturing(self):
        """Patch only _one_window; Python eager parity outside it is original."""
        with patch.object(self.flow, '_one_window', self._research_window):
            yield self

    def _research_window(self, *args, **kwargs):
        if kwargs.get('vb_mode', 'materialized') != 'materialized':
            raise ValueError('K8 near pack requires original materialized VB')
        # The eager Flat reference does not call flow._one_window and
        # cannot inherit this temporary batch launcher patch.
        with patch.object(self.batch, 'launch_prepared_tile_batch', self.launch):
            return self.original_window(*args, **kwargs)

    def _allocate_once(self, device):
        if self._packed_k is not None:
            return
        if self.torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Near K8 scratch must be allocated before CUDA Graph capture')
        free, _ = self.torch.cuda.mem_get_info(device)
        if free < SCRATCH_BYTES + 256 * (1 << 20):
            raise MemoryError('Insufficient free memory for 132 MiB shared K8 scratch + 256 MiB floor')
        self._packed_k = self.torch.empty(
            (GROUP_HEADS * MAX_TILES, NEAR_TOKENS, DIM),
            device=device, dtype=self.torch.int8)
        self._packed_ks = self.torch.empty(
            (GROUP_HEADS * MAX_TILES, NEAR_TOKENS),
            device=device, dtype=self.torch.float32)

    def launch(self, states, cfg, tr, *, buffers=None,
               fused_vb=False, research_resource_variant=None):
        if fused_vb or research_resource_variant is not None:
            raise ValueError('Near K8 variant rejects unrelated research Tile modes')
        q = states['Q']
        h, tiles, rows, d = map(int, q.shape)
        if (h != GROUP_HEADS or tiles not in (15, 16)
            or rows != ROWS or d != DIM
            or states.get('_vb_mode') != 'materialized'
            or not bool(states.get('_direct_q', False))
            or not bool(states.get('_direct_v', False))):
            raise AssertionError('Unpinned group8/Tile16 near K8 Graph geometry')
        kern = cfg['kern']
        if (cfg['order'] != 'seq' or cfg['store'] != 'int8'
            or cfg['dot'] != 'bf16' or int(cfg['tile']) != ROWS
            or int(cfg['exact_leaves']) != NEAR_LEAVES
            or int(cfg['splits']) != SPLITS
            or int(kern.get('group', 1)) != 1
            or int(kern.get('tile_bn', -1)) != BN
            or int(kern.get('tile_near_bn', BN) or BN) != BN
            or int(kern.get('tile_warps', -1)) != 4
            or int(kern.get('tile_stages', -1)) != 2):
            raise AssertionError('K8 Graph variant requires original Balanced/512 Tile16 BT16')
        keys = int(states['K'].shape[1])
        nb = keys // LEAF
        if (keys != 15264 or tuple(states['K'].shape) != (h, keys, d)
            or tuple(states['KS'].shape) != (h, keys)
            or tuple(states['V'].shape) != (h, keys, d)
            or tuple(states['IDX'].shape) != (h, tiles, NEAR_LEAVES)
            or tuple(states['MU'].shape) != (h, tiles, d)
            or tuple(states['G'].shape) != (h, tiles, nb, d)
            or tuple(states['VB'].shape) != (h, tiles, nb, d)
            or tuple(states['LZ'].shape) != (h, tiles, nb)
            or tuple(states['SG'].shape) != (h, tiles, nb)
            or tuple(states['EXCL'].shape) != (h, tiles, nb)):
            raise AssertionError('Packed near K8 source/state dims unexpected')
        if (states['K'].dtype != self.torch.int8
            or states['KS'].dtype != self.torch.float32
            or states['V'].dtype != self.torch.bfloat16
            or buffers is None):
            raise AssertionError('K8/KS/native V dtype or output buffers incorrect')
        for name, shape in (
            ('pm', (h, tiles, SPLITS, ROWS)),
            ('pl', (h, tiles, SPLITS, ROWS)),
            ('acc', (h, tiles, SPLITS, ROWS, d)),
        ):
            if (name not in buffers
                or tuple(buffers[name].shape) != shape
                or buffers[name].dtype != self.torch.float32):
                raise AssertionError('Graph scratch partial buffer mismatch: ' + name)
        if not q.is_contiguous() or not states['IDX'].is_contiguous():
            raise AssertionError('K8 pseudo-head Q and IDX must be contiguous')
        self._allocate_once(q.device)
        self.pack_kernel[(h * tiles, NEAR_TOKENS // 32)](
            states['K'], states['KS'], states['IDX'],
            self._packed_k, self._packed_ks,
            SRC_KEYS=keys, KX=NEAR_LEAVES,
            NEAR_TOKENS=NEAR_TOKENS, CENTRES_PER_HEAD=tiles,
            D=d, BLOCK=32, num_warps=4)

        pseudo = q.view(h * tiles, 1, rows, d)
        qs = tuple(map(int, pseudo.stride()))
        v = states['V']
        vs = tuple(map(int, v.stride()))
        z = states['LZ'].view(h * tiles, nb)
        compiled = self.tile_kernel[(h * tiles, SPLITS, 1)](
            pseudo, states['MU'].view(h * tiles, d),
            states['G'].view(h * tiles, nb, d), z,
            states['VB'].view(h * tiles, nb, d),
            states['EXCL'].view(h * tiles, nb),
            states['IDX'].view(h * tiles, NEAR_LEAVES),
            self._packed_k.view(GROUP_HEADS * MAX_TILES * NEAR_TOKENS, d), v,
            buffers['pm'].view(h * tiles, SPLITS, ROWS),
            buffers['pl'].view(h * tiles, SPLITS, ROWS),
            buffers['acc'].view(h * tiles, SPLITS, ROWS, d),
            states['SG'].view(h * tiles, nb), z, z, z,
            self._packed_ks.view(GROUP_HEADS * MAX_TILES * NEAR_TOKENS),
            nb, NEAR_LEAVES, 256, 2048, 0, 1.0 / math.sqrt(d),
            T=rows, D=d, BN=BN, BN_NEAR=BN, LEAF=LEAF,
            PREC='tf32', DOT=1, GR=1, FP8=0, TAIL=0, RB=1, I8=1,
            PASS=0, LATE_NEAR_Q=0, FAR_PREFETCH_LZ=0,
            FAR_PV_ACC=0, NEAR_PV_ACC=0,
            BATCH_TILES_PER_HEAD=1, BATCH_KEY_ROWS=NEAR_TOKENS,
            Q_DIRECT=1, Q_SH=qs[0], Q_ST=qs[1],
            Q_SR=qs[2], Q_SD=qs[3],
            V_DIRECT=1, V_SH=vs[0], V_SS=vs[1], V_SD=vs[2],
            REAL_TILES_PER_HEAD=tiles,
            num_warps=4, num_stages=2,
        )
        self._calls += 1
        self._windows_seen.add(tiles)
        return buffers, compiled

    def retention(self):
        """Retain these buffers until the captured session has been cleared."""
        if self._packed_k is None or self._packed_ks is None:
            raise AssertionError('K8 scratch was never initialized')
        if not self._windows_seen.issuperset({15, 16}):
            raise AssertionError('Both Graph geometries must use the same scratch')
        return (self._packed_k, self._packed_ks)

    def recorded_scratch(self):
        return {'bytes': SCRATCH_BYTES,
                'allocated_once_for_both_geometries': True,
                'tile_counts': sorted(self._windows_seen),
                'captured_tile_launches': self._calls}
