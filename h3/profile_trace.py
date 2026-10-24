"""Optional one-shot torch.profiler capture of a few transformer blocks, summarised into one
telemetry event (`block_profile`) in the run's H3_run_*.jsonl.  Off by default.

  $env:APL_H3_PROFILE_TRACE="2:10:2"     # cycle 2 (1-based), first block 10, 2 blocks
  (optional) $env:APL_H3_PROFILE_TOP="12"; $env:APL_H3_PROFILE_STACK="0" (default 0: Python stacks made the chrome-trace export fail/empty and slowed the cycle ~2x; 1 = capture them)

The raw chrome trace is written to a temp file, parsed and deleted, so the run still produces one
report.  Summary: GPU busy/idle time, per-stream busy time, multi-stream overlap, kernel categories,
top kernels, largest idle gaps (with the kernels before/after and the CPU sync call inside the
gap), CPU-side synchronisation time and the live Python thread list.  Any failure is swallowed.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from typing import Any

_CATS = (
    ("gemm", ("gemm", "cutlass", "nvjet", "cublas", "scaled_mm", "fp4", "fp8", "matmul", "sm120", "sm100", "sm90")),
    ("attention", ("attn", "attention", "flash", "sol_", "_tile_kernel", "_stats_kernel", "_rowmax", "_merge_kernel", "softmax")),
    ("memcpy", ("memcpy", "memset")),
    ("norm_elementwise", ("elementwise", "vectorized", "layer_norm", "rms", "norm", "silu", "gelu", "mul", "add", "copy", "cat", "rope", "index")),
)


def _category(name: str, cat: str) -> str:
    low = name.lower()
    if cat in ("gpu_memcpy", "gpu_memset"):
        return "memcpy"
    for label, keys in _CATS:
        if any(k in low for k in keys):
            return label
    return "other"


def _union(intervals):
    total, last_end, merged = 0.0, None, []
    for s, e in sorted(intervals):
        if last_end is None or s > last_end:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
        last_end = merged[-1][1]
    for s, e in merged:
        total += e - s
    return total, merged


def analyze_trace(trace: dict, wall_s: float | None = None, top: int = 12, threads: list[str] | None = None) -> dict:
    ev = trace.get("traceEvents", [])
    gpu = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    rt = [e for e in ev if e.get("ph") == "X" and e.get("cat") == "cuda_runtime"]
    cpu = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("cpu_op", "python_function", "user_annotation")
           and e.get("dur", 0) >= 200]            # aten ops are present without stacks, python_function needs APL_H3_PROFILE_STACK=1
    out: dict[str, Any] = {"gpu_events": len(gpu)}
    if not gpu:
        out["error"] = "no GPU events in trace"
        return out
    t0 = min(e["ts"] for e in gpu)
    t1 = max(e["ts"] + e["dur"] for e in gpu)
    span = t1 - t0
    busy, merged = _union([(e["ts"], e["ts"] + e["dur"]) for e in gpu])
    out.update(window_ms=round(span / 1000, 2), gpu_busy_ms=round(busy / 1000, 2),
               gpu_idle_ms=round((span - busy) / 1000, 2), gpu_busy_pct=round(100 * busy / span, 1))
    if wall_s:
        out["block_wall_ms"] = round(wall_s * 1000, 1)
    per_stream: dict[Any, list] = {}
    for e in gpu:
        per_stream.setdefault((e.get("args") or {}).get("stream", e.get("tid")), []).append(e)
    out["streams"] = {}
    for sid, lst in per_stream.items():
        b, _ = _union([(e["ts"], e["ts"] + e["dur"]) for e in lst])
        out["streams"][str(sid)] = {"events": len(lst), "busy_ms": round(b / 1000, 2),
                                    "memcpy_ms": round(sum(e["dur"] for e in lst if e["cat"] != "kernel") / 1000, 2)}
    # time with >=2 streams active at once
    points = []
    for sid, lst in per_stream.items():
        _, m = _union([(e["ts"], e["ts"] + e["dur"]) for e in lst])
        for s, e_ in m:
            points += [(s, 1), (e_, -1)]
    points.sort()
    active, last, multi = 0, None, 0.0
    for t, d in points:
        if last is not None and active >= 2:
            multi += t - last
        active += d
        last = t
    out["multi_stream_overlap_ms"] = round(multi / 1000, 2)
    cats: dict[str, float] = {}
    names: dict[str, list] = {}
    for e in gpu:
        c = _category(e.get("name", ""), e["cat"])
        cats[c] = cats.get(c, 0.0) + e["dur"]
        n = names.setdefault(e.get("name", "")[:90], [0, 0.0])
        n[0] += 1
        n[1] += e["dur"]
    out["category_ms"] = {k: round(v / 1000, 2) for k, v in sorted(cats.items(), key=lambda kv: -kv[1])}
    out["top_kernels"] = [{"name": k, "n": v[0], "ms": round(v[1] / 1000, 2)}
                          for k, v in sorted(names.items(), key=lambda kv: -kv[1][1])[:top]]
    out["memcpy"] = {}
    for e in gpu:
        if e["cat"] == "gpu_memcpy":
            kind = (e.get("name") or "memcpy")[:30]
            slot = out["memcpy"].setdefault(kind, {"n": 0, "ms": 0.0, "MiB": 0.0})
            slot["n"] += 1
            slot["ms"] += e["dur"] / 1000
            slot["MiB"] += float((e.get("args") or {}).get("bytes", 0)) / (1 << 20)
    for slot in out["memcpy"].values():
        slot["ms"], slot["MiB"] = round(slot["ms"], 2), round(slot["MiB"], 1)
    # biggest idle gaps on the merged GPU timeline
    gaps = []
    sorted_gpu = sorted(gpu, key=lambda e: e["ts"])
    for i in range(1, len(merged)):
        gs, ge = merged[i - 1][1], merged[i][0]
        if ge - gs >= 200:
            before = [e for e in sorted_gpu if e["ts"] + e["dur"] <= gs + 1][-1:] or [{"name": "?"}]
            after = [e for e in sorted_gpu if e["ts"] >= ge - 1][:1] or [{"name": "?"}]
            syncs = {}
            for r in rt:
                if r["ts"] < ge and r["ts"] + r["dur"] > gs and r["dur"] >= 50:
                    syncs[r.get("name", "")[:40]] = syncs.get(r.get("name", "")[:40], 0.0) + r["dur"]
            over = []
            if (ge - gs) >= 2000:                      # for big gaps: what was the CPU doing (needs with_stack)
                for c in cpu:
                    ov = min(c["ts"] + c["dur"], ge) - max(c["ts"], gs)
                    if ov >= 0.3 * (ge - gs) or ov >= 20000:
                        over.append((ov, c.get("name", "")[:110], c.get("cat")))
                over.sort(reverse=True)
            gaps.append({"ms": round((ge - gs) / 1000, 2), "last_kernel_before": before[0].get("name", "")[:60],
                         "first_kernel_after": after[0].get("name", "")[:60],
                         "cpu_in_gap": {k: round(v / 1000, 2) for k, v in sorted(syncs.items(), key=lambda kv: -kv[1])[:3]},
                         "cpu_ops_overlapping": [{"ms": round(o[0] / 1000, 1), "name": o[1], "cat": o[2]} for o in over[:8]]})
    gaps.sort(key=lambda g: -g["ms"])
    out["idle_gaps_ge_0p2ms"] = {"count": len(gaps), "total_ms": round(sum(g["ms"] for g in gaps), 2), "top": gaps[:top]}
    sync = {}
    for r in rt:
        nm = r.get("name", "")
        if "Synchronize" in nm or "Memcpy" in nm or "StreamWaitEvent" in nm:
            slot = sync.setdefault(nm, [0, 0.0])
            slot[0] += 1
            slot[1] += r["dur"]
    out["cpu_sync_calls"] = {k: {"n": v[0], "ms": round(v[1] / 1000, 2)} for k, v in sorted(sync.items(), key=lambda kv: -kv[1][1])[:8]}
    out["cpu_threads_launching"] = len({r.get("tid") for r in rt})
    out["python_threads"] = threads or []
    return out


class _Sampler(threading.Thread):
    """Poor-man's py-spy: samples the profiled thread's Python stack every few ms (no trace export needed)."""

    def __init__(self, target_ident: int, interval_s: float = 0.002):
        super().__init__(name="APL-H3-profile-sampler", daemon=True)
        self.target = target_ident
        self.interval = interval_s
        self.samples: list = []
        self._stop_evt = threading.Event()

    def run(self):
        import sys
        while not self._stop_evt.is_set():
            fr = sys._current_frames().get(self.target)
            if fr is not None:
                stack, f = [], fr
                while f is not None and len(stack) < 14:
                    co = f.f_code
                    stack.append((co.co_filename.replace("\\", "/").rsplit("/", 1)[-1], co.co_name, f.f_lineno))
                    f = f.f_back
                self.samples.append((time.perf_counter(), tuple(stack)))
            self._stop_evt.wait(self.interval)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=2.0)

    def summary(self, top: int = 12) -> dict:
        n = len(self.samples)
        out: dict[str, Any] = {"samples": n, "interval_ms": self.interval * 1000}
        if not n:
            return out
        leaf: dict[str, int] = {}
        incl: dict[str, int] = {}
        for _t, st in self.samples:
            k = "%s:%s:%d" % st[0]
            leaf[k] = leaf.get(k, 0) + 1
            for fn in {"%s:%s" % (a, b) for a, b, _l in st}:
                incl[fn] = incl.get(fn, 0) + 1
        out["leaf_top"] = [{"at": k, "pct": round(100 * v / n, 1)} for k, v in sorted(leaf.items(), key=lambda kv: -kv[1])[:top]]
        out["inclusive_top"] = [{"fn": k, "pct": round(100 * v / n, 1)} for k, v in sorted(incl.items(), key=lambda kv: -kv[1])[:top]]
        # longest stalls: consecutive samples with the same leaf location
        runs, i = [], 0
        while i < n:
            j = i
            key = self.samples[i][1][0]
            while j + 1 < n and self.samples[j + 1][1][0] == key:
                j += 1
            dur = (self.samples[j][0] - self.samples[i][0]) * 1000 + self.interval * 1000
            if dur >= 30:
                runs.append((dur, ["%s:%s:%d" % fr for fr in self.samples[i][1][:8]]))
            i = j + 1
        runs.sort(key=lambda r: -r[0])
        out["stalls_ge_30ms"] = [{"ms": round(d, 0), "stack_leaf_first": st} for d, st in runs[:6]]
        return out


class BlockProfiler:
    def __init__(self):
        spec = os.environ.get("APL_H3_PROFILE_TRACE", "").strip()
        self.cfg = None
        self.prof = None
        self.done = False
        self.t_start = 0.0
        self.sampler = None
        if spec:
            try:
                p = [int(x) for x in spec.split(":")]
                self.cfg = (p[0], p[1], p[2] if len(p) > 2 else 1)
            except Exception:
                self.cfg = None

    def enabled(self) -> bool:
        return self.cfg is not None and not self.done

    def before_block(self, cycle_number: int, idx: int) -> None:
        if not self.enabled() or self.prof is not None:
            return
        cyc, first, _n = self.cfg
        if cycle_number != cyc or idx != first:
            return
        try:
            import torch
            from torch.profiler import ProfilerActivity, profile
            torch.cuda.synchronize()
            self.prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                                with_stack=os.environ.get("APL_H3_PROFILE_STACK", "0") == "1")
            self.prof.__enter__()
            self.sampler = _Sampler(threading.get_ident())
            self.sampler.start()
            self.t_start = time.perf_counter()
        except Exception:
            self.prof = None
            self.done = True

    def after_block(self, idx: int, emit) -> None:
        if self.prof is None or self.done:
            return
        _cyc, first, n = self.cfg
        if idx != first + n - 1:
            return
        self.done = True
        prof, self.prof = self.prof, None
        try:
            import torch
            torch.cuda.synchronize()
            wall = time.perf_counter() - self.t_start
            sampler, self.sampler = self.sampler, None
            if sampler is not None:
                sampler.stop()
            prof.__exit__(None, None, None)
            tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
            tmp.close()
            try:
                prof.export_chrome_trace(tmp.name)
                size = os.path.getsize(tmp.name)
                try:
                    with open(tmp.name, "r", encoding="utf-8") as fh:
                        trace = json.load(fh)
                except Exception as exc:
                    raise RuntimeError(f"chrome trace unreadable (size={size} bytes): {exc}") from exc
            finally:
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass
            threads = [f"{t.name}{'' if t.daemon else ' (non-daemon)'}" for t in threading.enumerate()]
            top = int(os.environ.get("APL_H3_PROFILE_TOP", "12") or 12)
            summary = analyze_trace(trace, wall, top, threads)
            if sampler is not None:
                summary["python_sampler"] = sampler.summary(top)
            emit("H3", "block_profile",
                 f"block profile cycle={self.cfg[0]} blocks={first}..{first + n - 1} wall={wall * 1000:.0f}ms "
                 f"gpu_busy={summary.get('gpu_busy_pct', '?')}%", blocks=n, summary=summary)
        except Exception as exc:
            emit("H3", "block_profile", f"block profile failed: {type(exc).__name__}: {exc}"[:300])
