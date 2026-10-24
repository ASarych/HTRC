"""Shape and retained-buffer planning for the opt-in HTRC E1 executor.

No CUDA allocation or ComfyUI import occurs in this module. Transient PyTorch /
vendor allocations are covered by a declared reserve, not a hard allocator cap.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ExecutorConfig:
    open_fraction: float = 0.02
    target_batch: int = 12288
    root_chunk: int = 80
    leaf_work_chunk: int = 32
    leaf_producer: str = "reference"
    state_queue: str = "unique"
    frontier_capacity: int = 1048576
    state_capacity: int = 32768
    transient_reserve_bytes: int = 512 * 1024 * 1024

    def __post_init__(self):
        if self.state_queue not in ("unique", "mark_scan"):
            raise ValueError("state_queue must be unique or mark_scan")
        if self.leaf_producer not in ("reference", "reuse"):
            raise ValueError("leaf_producer must be reference or reuse")
        if not math.isfinite(self.open_fraction) or not 0 < self.open_fraction < 1:
            raise ValueError("open_fraction must be finite and strictly between 0 and 1")
        for name in ("target_batch", "root_chunk", "leaf_work_chunk",
                     "frontier_capacity", "state_capacity"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.transient_reserve_bytes) is not int or self.transient_reserve_bytes < 0:
            raise ValueError("transient_reserve_bytes must be a nonnegative integer")


@dataclass(frozen=True)
class BufferSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str

    @property
    def nbytes(self):
        sizes = {"float32": 4, "bfloat16": 2, "int32": 4, "int64": 8, "bool": 1}
        return math.prod(self.shape) * sizes[self.dtype]


@dataclass(frozen=True)
class ExecutorPlan:
    query_count: int
    source_count: int
    leaf_count: int
    centroid_capacity: int
    root_capacity: int
    node_capacity: int
    state_capacity: int
    config: ExecutorConfig
    buffers: tuple[BufferSpec, ...]
    workspace_budget_bytes: int

    @property
    def retained_bytes(self):
        return sum(b.nbytes for b in self.buffers)

    @property
    def admission_bytes(self):
        return self.retained_bytes + self.config.transient_reserve_bytes

    @property
    def source_prefix_count(self):
        return self.leaf_count * 16

    @property
    def source_tail_count(self):
        return self.source_count - self.source_prefix_count

    def manifest(self):
        return {
            "boundary": "htrc_geometry_and_prefix_core",
            "query_count": self.query_count,
            "source_count": self.source_count,
            "source_prefix_count": self.source_prefix_count,
            "source_tail_count": self.source_tail_count,
            "retained_bytes": self.retained_bytes,
            "transient_reserve_bytes": self.config.transient_reserve_bytes,
            "admission_bytes": self.admission_bytes,
            "workspace_budget_bytes": self.workspace_budget_bytes,
            "budget_is_allocator_cap": False,
            "buffers": [
                {"name": b.name, "shape": list(b.shape), "dtype": b.dtype,
                 "bytes": b.nbytes} for b in self.buffers
            ],
        }


def make_plan(query_count: int, source_count: int, *,
              workspace_budget_bytes: int = 2 * 1024**3,
              config: ExecutorConfig | None = None) -> ExecutorPlan:
    """Plan one D128 head. Source-tail attention is outside this core API."""
    for name, value, minimum in (("query_count", query_count, 1),
                                  ("source_count", source_count, 16),
                                  ("workspace_budget_bytes", workspace_budget_bytes, 1)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    cfg = config or ExecutorConfig()
    q, b = query_count, source_count // 16
    c = min(32, (min(q, 128) + 1) // 2)
    r, n = min(80, b), 2 * b
    # Tree internal nodes have >=2 children, hence total nodes < 2*B.
    # Prefix scans temporarily carry fanout*frontier_capacity in int64;
    # existing authority kernels consume int32 totals.
    for name, bound in (("key space", c*n), ("query count", q),
                        ("root expansion", q*r*4),
                        ("frontier expansion", cfg.frontier_capacity*4)):
        if bound >= 2**31:
            raise ValueError(f"{name} exceeds the existing int32 kernel ABI")
    s = min(cfg.state_capacity, c*n)
    if s < c*r:
        raise ValueError(f"state_capacity must fit the root batch ({c*r})")
    m = min(cfg.target_batch, s)
    specs = []

    def add(name, shape, dtype="float32"):
        specs.append(BufferSpec(name, tuple(shape), dtype))

    if cfg.leaf_producer == "reuse":
        add("leaf_logz", (c, b))
        add("leaf_grad", (c, b, 128))
        add("leaf_value", (c, b, 128))
        add("leaf_small", (c, b, 8, 8))
        add("leaf_trace", (c, b))
        add("leaf_q8", (b, 128, 8))
        # Ten finite-check stages per group of four centroids.
        add("leaf_nonfinite", ((c + 3) // 4, 10), "bool")
    if cfg.state_queue == "mark_scan":
        add("queue_marks", (c * n,), "int32")
        add("queue_prefix", (c * n,), "int32")
        add("queue_slot_map", (c * n,), "int32")
        add("queue_keys", (s,), "int32")
        add("queue_row_slots", (cfg.frontier_capacity,), "int32")
        add("queue_metadata", (2,), "int32")  # unique count, invalid key rows
    add("solver_a", (m, 128, 16))
    add("solver_b", (m, 128, 16))
    add("parent_grad", (m, 128))
    add("parent_value", (m, 128))
    add("state_logz", (s,))
    add("state_grad", (s, 128), "bfloat16")
    add("state_value", (s, 128), "bfloat16")
    add("state_lambda", (s, 8))
    add("state_u", (s, 128, 8), "bfloat16")
    add("state_trace", (s,))
    add("state_captured_trace", (s,))
    for side in ("a", "b"):
        add("frontier_q_" + side, (cfg.frontier_capacity,), "int32")
        add("frontier_n_" + side, (cfg.frontier_capacity,), "int32")
    add("root_score", (q, r))
    add("root_mass", (q, r))
    add("log_total", (q,))
    add("acc_m", (q,))
    add("acc_z", (q,))
    add("acc_n", (q, 128))
    add("status", (4,), "int64")  # code, resource, required, available
    plan = ExecutorPlan(q, source_count, b, c, r, n, s, cfg,
                        tuple(specs), workspace_budget_bytes)
    if plan.admission_bytes > workspace_budget_bytes:
        raise MemoryError(
            f"HTRC plan requires {plan.admission_bytes} bytes "
            f"including declared transient reserve; budget={workspace_budget_bytes}"
        )
    return plan
