"""CPU/NumPy teaching reference for the local HTRC tree identities.

This deliberately uses *exact softmax leaves*, NOT H3's RRC3 antipodal
B16 leaf law. It is a runnable reference for tree aggregation, Taylor-2
log mass, local value dipoles and heuristic mass-weighted opening, not an
optimized replica of the experimental CUDA/Triton implementation.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np

Array = np.ndarray

@dataclass(frozen=True)
class Stats:
    log_mass: float
    gradient: Array
    hessian: Array
    value: Array
    value_jacobian: Array

@dataclass(frozen=True)
class TreeNode:
    stats: Stats
    children: tuple["TreeNode", ...]
    key_indices: Array

def _softmax_weights(logits: Array) -> tuple[float, Array]:
    maximum = float(np.max(logits))
    shifted = np.exp(logits - maximum)
    total = float(np.sum(shifted))
    return maximum + float(np.log(total)), shifted / total

def exact_stats(keys: Array, values: Array, center: Array) -> Stats:
    """Compute derivatives of a real attention block at scaled query center."""
    keys, values, center = np.asarray(keys), np.asarray(values), np.asarray(center)
    if keys.ndim != 2 or values.ndim != 2 or len(keys) != len(values) or len(keys) == 0:
        raise ValueError("keys/values must be matching nonempty matrices")
    if center.shape != (keys.shape[1],):
        raise ValueError("scaled query dimension mismatch")
    log_mass, weights = _softmax_weights(keys @ center)
    mu = weights @ keys
    val = weights @ values
    ck = keys - mu
    cv = values - val
    hess = (ck * weights[:, None]).T @ ck
    jac = (cv * weights[:, None]).T @ ck
    return Stats(log_mass, mu, hess, val, jac)

def merge_stats(children: Sequence[Stats]) -> Stats:
    """Exact log-sum-exp, Hessian and conditional-value dipole merge."""
    if not children:
        raise ValueError("cannot merge an empty set")
    logs = np.asarray([s.log_mass for s in children], dtype=np.float64)
    log_mass, weights = _softmax_weights(logs)
    mu = sum((w * s.gradient for w, s in zip(weights, children)))
    val = sum((w * s.value for w, s in zip(weights, children)))
    hess = np.zeros_like(children[0].hessian)
    jac = np.zeros_like(children[0].value_jacobian)
    for w, s in zip(weights, children):
        dg = s.gradient - mu
        dv = s.value - val
        hess += w * (s.hessian + np.outer(dg, dg))
        jac += w * (s.value_jacobian + np.outer(dv, dg))
    return Stats(log_mass, mu, hess, val, jac)

def build_forest(keys: Array, values: Array, center: Array, *,
                 leaf_size: int = 16, fanout: int = 4) -> tuple[TreeNode, ...]:
    """Deterministic contiguous grouping; original HTRC uses learned geometry."""
    if leaf_size < 1 or fanout < 2:
        raise ValueError("leaf_size must be positive and fanout >= 2")
    keys = np.asarray(keys, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    leaves = [TreeNode(exact_stats(keys[i:i+leaf_size], values[i:i+leaf_size], center),
                       (), np.arange(i, min(i+leaf_size, len(keys))))
              for i in range(0, len(keys), leaf_size)]
    level = leaves
    while len(level) > fanout:
        level = [TreeNode(merge_stats([n.stats for n in level[i:i+fanout]]),
                          tuple(level[i:i+fanout]), np.array([], dtype=int))
                 for i in range(0, len(level), fanout)]
    return tuple(level)

def exact_attention(keys: Array, values: Array, query: Array) -> Array:
    """Attention for query already scaled by 1 / sqrt(d_k)."""
    _, w = _softmax_weights(np.asarray(keys) @ np.asarray(query))
    return w @ np.asarray(values)

def predict(node: TreeNode, delta: Array) -> tuple[float, Array, float]:
    s = node.stats
    quad = float(delta @ s.hessian @ delta)
    log_mass = s.log_mass + float(s.gradient @ delta) + 0.5 * quad
    value = s.value + s.value_jacobian @ delta
    return log_mass, value, quad

def evaluate(keys: Array, values: Array, query: Array, center: Array,
             forest: Sequence[TreeNode], *, opening_threshold: float = 0.01
             ) -> tuple[Array, dict]:
    """Evaluate approximate attention; exact fallback at opened leaves.

    The curvature-based score is a heuristic, never a certified error bound.
    """
    if opening_threshold < 0:
        raise ValueError("opening_threshold must be nonnegative")
    query = np.asarray(query, dtype=np.float64)
    delta = query - np.asarray(center, dtype=np.float64)
    root_logits = [predict(n, delta)[0] for n in forest]
    norm = np.logaddexp.reduce(root_logits)
    accepted: list[tuple[float, Array]] = []
    visited = 0
    exact_leaves = 0

    def traverse(node: TreeNode) -> None:
        nonlocal visited, exact_leaves
        visited += 1
        log_mass, val, quad = predict(node, delta)
        frac = float(np.exp(min(0.0, log_mass - norm)))
        score = frac * float(np.expm1(min(0.5 * abs(quad), 80.0)))
        if node.children and score > opening_threshold:
            for child in node.children:
                traverse(child)
        elif not node.children and score > opening_threshold:
            ix = node.key_indices
            log_mass, weights = _softmax_weights(keys[ix] @ query)
            accepted.append((log_mass, weights @ values[ix]))
            exact_leaves += 1
        else:
            accepted.append((log_mass, val))
    for root in forest:
        traverse(root)
    logs = np.asarray([x[0] for x in accepted])
    _, weights = _softmax_weights(logs)
    output = sum(w * x[1] for w, x in zip(weights, accepted))
    return output, {"visited_nodes": visited, "accepted_nodes": len(accepted),
                    "exact_leaves": exact_leaves}
