"""Production entry point of the flat (tree-free) HTRC operator: CUDA-graph head runner.

Same call shape as htrc_executor_heads.HeadRunner:

    runner = FlatRunner(queries, keys, free_bytes=free_vram, mode="balanced")
    outputs = runner.run([(head, q, k, v), ...])        # q [Q,128], k/v [K,128] on CUDA
    runner.run(jobs, sink=lambda i, out: block_out[i].copy_(out))   # nothing retained

The operator has no host synchronisation, so one CUDA graph of a whole head is captured once
(static q/k/v/out buffers) and replayed per head: ~1.1 ms/head less than eager launching
(measured 6.6 -> 5.5 ms). Heads run one after another: four CUDA streams were slower
(9.8 vs 6.6 ms/head, the GPU is already full); graph + streams gained only ~3 %.

modes (error: 72 full-Q maps, RTX 5070 Ti; ms: eager. With the graph, measured ms/head:
fast 5.5, balanced 6.7, accurate 9.6, max 12.4 at chunk 256; chunk 64 costs +0.4..+1.9 ms):
    fast      256 exact leaves, tiles 64            error 9.9/26.5/36.7 % (median/p90/max)  ~7.1 ms
    balanced  512 exact leaves, tiles 64            5.7/22.2/24.0 %                         ~7.8 ms
    accurate  1024 exact leaves, k-means, tiles 64  2.7/6.6/8.0 %                           ~14.6 ms
    max       1024 exact leaves, k-means, tiles 32  2.2/6.1/6.8 %                           ~17.4 ms
(error = relative L2 of the attention output against exact softmax attention; errors are per
head/block/call, block 0 is the easiest.)

Memory model (measured over the q/k/v inputs, eager = graph pool): 80 MiB + per-tile state x
tile_chunk; tile_chunk 16/32/64/128/256 -> 109/138/196/312/544 MiB (tiles 64).
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

try:                                   # the memory model below must stay importable without torch
    import torch
except ImportError:                    # pragma: no cover
    torch = None


def _load(name, filename):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


MODES = {
    "fast": dict(exact_leaves=256, order="seq", tile=64),
    "balanced": dict(exact_leaves=512, order="seq", tile=64),
    "accurate": dict(exact_leaves=1024, order="kmeans", tile=64),
    "max": dict(exact_leaves=1024, order="kmeans", tile=32),
}
CHUNKS = (1024, 512, 256, 128, 64, 32, 16)


STATIC_MIB = 40.0      # graph runner's static q(fp32)/k/v(bf16) copies (measured: +39.8 MiB)
KMEANS_MIB = 65.0      # k-means ordering extra (measured +57..+105 MiB over the model; conservative)


def predicted_peak_mib(keys: int, tile: int, tile_chunk: int, splits: int = 8, leaf: int = 16,
                       dim: int = 128, order: str = "seq", graph: bool = True) -> float:
    """Peak above the caller's q/k/v: 80 MiB base + bf16 g/vbar + split partials per tile in the
    chunk (x1.2), + static graph copies, + k-means scratch. Checked on the -FlatRunner report."""
    nb = keys // leaf
    per_tile = (2 * nb * dim * 2 + splits * tile * dim * 4 + nb * 9) / 2**20 * 1.2
    return (80.0 + per_tile * tile_chunk + (STATIC_MIB if graph else 0.0)
            + (KMEANS_MIB if order == "kmeans" else 0.0))


def plan_chunk(free_bytes: int, keys: int, tile: int, splits: int = 8, order: str = "seq",
               graph: bool = True, queries: int = 0) -> int:
    """Largest chunk that fits (fewer launches per head). A chunk beyond the tile count is the
    whole sequence: predicted at the real tile count, not at the nominal chunk."""
    nt = -(-int(queries) // int(tile)) if queries else 0
    for c in CHUNKS:
        eff = min(c, nt) if nt else c
        if predicted_peak_mib(keys, tile, eff, splits, order=order, graph=graph) * 2**20 <= free_bytes:
            return c
    raise MemoryError(
        f"flat HTRC needs ~{predicted_peak_mib(keys, tile, CHUNKS[-1], splits, order=order, graph=graph):.0f} MiB at the smallest chunk, "
        f"{int(free_bytes) >> 20} MiB free")


class FlatRunner:
    def __init__(self, queries: int, keys: int, *, free_bytes: int | None = None,
                 mode: str = "balanced", tile_chunk: int | None = None, use_graph: bool = True,
                 device="cuda", dim: int = 128, sink_leaves: int = 0, tail_rows: int = 0,
                 **overrides):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {list(MODES)}")
        self.flat = _load("_apl_htrc_flat", "htrc_flat.py")
        self.mode = mode
        self.device = torch.device(device)
        base = dict(self.flat.PROD)
        base.update(MODES[mode])
        overrides = dict(overrides)
        if "kern" in overrides:                         # merge, do not replace, the kernel knobs
            base["kern"] = {**base.get("kern", {}), **overrides.pop("kern")}
        base.update(overrides)
        base["sink_leaves"] = int(sink_leaves)           # leading leaves kept exact (conditioning KV)
        self.tail_rows = int(tail_rows)                  # keys past the last full leaf, exact
        self._tk = self._tv = None
        self.ws = {}                                     # reusable workspace (eager path only)
        if tile_chunk is None:
            if free_bytes is not None:
                tile_chunk = plan_chunk(free_bytes, keys, base["tile"], base.get("splits", 8),
                                        base["order"], use_graph, queries=int(queries))
            else:
                tile_chunk = base["tile_chunk"]
        base["tile_chunk"] = int(tile_chunk)
        self.cfg = base
        self.predicted_mib = predicted_peak_mib(keys, base["tile"], base["tile_chunk"], base.get("splits", 8),
                                                order=base["order"], graph=use_graph)
        self.queries, self.keys, self.dim = int(queries), int(keys), int(dim)
        self.graph = None
        self._capturing = False
        self.graph_error = None
        self._sq = self._sk = self._sv = self._so = None
        if use_graph:
            self._capture()

    def _call(self, q, k, v, tail=None):
        if tail is None and self._tk is not None:
            tail = (self._tk, self._tv)
        ws = self.ws if self.graph is None and not self._capturing else None
        return self.flat.flat_attention(q, k, v, tail=tail, ws=ws, **self.cfg)

    def _capture(self):
        dev = self.device
        with torch.cuda.device(dev):
            self._sq = torch.zeros((self.queries, self.dim), device=dev, dtype=torch.float32)
            self._sk = torch.zeros((self.keys, self.dim), device=dev, dtype=torch.bfloat16)
            self._sv = torch.zeros_like(self._sk)
            # realistic warm-up data so k-means / softmax see finite, non-degenerate values
            self._sq.normal_(); self._sk.normal_(); self._sv.normal_()
            if self.tail_rows:
                self._tk = torch.randn((self.tail_rows, self.dim), device=dev, dtype=torch.float32)
                self._tv = torch.randn_like(self._tk)
            self._capturing = True
            try:
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side), torch.inference_mode():
                    for _ in range(2):
                        self._call(self._sq, self._sk, self._sv)
                torch.cuda.current_stream().wait_stream(side)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.inference_mode(), torch.cuda.graph(g, capture_error_mode="thread_local"):
                    self._so = self._call(self._sq, self._sk, self._sv)
                torch.cuda.synchronize()
                self.graph = g
                self._capturing = False
            except Exception as exc:  # noqa: BLE001  (fall back to eager launching)
                self.graph = None
                self._capturing = False
                self.graph_error = f"{type(exc).__name__}: {exc}"[:300]
                self._sq = self._sk = self._sv = self._so = None
                torch.cuda.empty_cache()

    def _set_tail(self, tail):
        if self.tail_rows and self.graph is not None:
            self._tk.copy_(tail[0])
            self._tv.copy_(tail[1])

    def head(self, q, k, v, tail=None):
        """One head; returns a fresh fp32 [Q, D] tensor. tail=(tk, tv) [tail_rows, D]."""
        if self.graph is None:
            with torch.inference_mode():
                return self._call(q, k, v, tail=tail)
        self._set_tail(tail)
        self._sq.copy_(q)
        self._sk.copy_(k)
        self._sv.copy_(v)
        self.graph.replay()
        return self._so.clone()

    def run(self, jobs, sink=None):
        """jobs: list of (head_index, q, k, v[, (tail_k, tail_v)]). Outputs in job order, or via
        sink(i, output)."""
        results = [None] * len(jobs)
        for i, job in enumerate(jobs):
            q, k, v = job[1], job[2], job[3]
            tail = job[4] if len(job) > 4 else None
            if self.graph is None or sink is None:
                out = self.head(q, k, v, tail=tail)
                if sink is None:
                    results[i] = out
                else:
                    sink(i, out)
            else:                                   # graph + sink: hand over the static output, no clone
                self._set_tail(tail)
                self._sq.copy_(q)
                self._sk.copy_(k)
                self._sv.copy_(v)
                self.graph.replay()
                sink(i, self._so)
        return results if sink is None else None

    def close(self):
        self.graph = None
        self._sq = self._sk = self._sv = self._so = None
