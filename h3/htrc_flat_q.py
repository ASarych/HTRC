"""APL_H3_FLAT_Q: one speed/quality coefficient for flat HTRC.

Measured steps only (captures from the 2026-09-26 math probe, 40,759-token video shape, RTX 5070 Ti).
Not a validated production setting: the image and video quality checks for these steps are in the
evidence folder of this branch.
"""
import os


_FLAT_Q_STEPS = (
    (0.1, 16, 2, 0.1005),   # 3.17 ms   8.1 / 23.4   (fast mode itself: 3.94 ms, 7.2 / 20.9)
    (0.2, 16, 4, 0.1005),   # 2.88 ms  10.1 / 26.8
    (0.3, 16, 4, 0.0754),   # 2.59 ms  12.5 / 29.7
    (0.4, 32, 2, 0.0754),   # 2.50 ms  13.6 / 30.4
    (0.5, 32, 2, 0.0377),   # 2.24 ms  16.3 / 35.7
    (0.6, 32, 4, 0.0503),   # 2.15 ms  17.3 / 37.1
    (0.7, 32, 4, 0.0377),   # 2.08 ms  21.4 / 39.4
    (0.8, 32, 4, 0.0251),   # 1.97 ms  26.1 / 40.8
    (0.9, 32, 4, 0.0126),   # 1.84 ms  30.1 / 42.1
    (1.0, 32, 4, 0.0063),   # 1.81 ms  33.0 / 44.6
)


def _flat_q_plan(tokens: int, exact_kv_tokens: int):
    """APL_H3_FLAT_Q -> {q, leaf, group, exact_leaves} for this token count, or None when unset / 0.

    The coefficient snaps down to the nearest measured step. The exact near-field count scales with
    the key count, so the same coefficient means the same share on a 2k-token image and a 40k-token
    video. Leaf 32 needs a 32-aligned conditioning sink; otherwise the step keeps leaf 16.
    """
    raw = os.environ.get("APL_H3_FLAT_Q", "").strip()
    if not raw:
        return None
    try:
        q = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"APL_H3_FLAT_Q must be a number in [0, 1], got {raw!r}") from exc
    if not 0.0 <= q <= 1.0:
        raise RuntimeError(f"APL_H3_FLAT_Q must be in [0, 1], got {raw!r}")
    step = None
    for cand in _FLAT_Q_STEPS:
        if cand[0] <= q + 1.0e-9:
            step = cand
    if step is None:                                   # below the first step: unchanged operator
        return None
    _q, leaf, group, frac = step
    if exact_kv_tokens % leaf != 0 or tokens // leaf < 1:
        leaf = 16
    free_leaves = max(0, tokens // leaf - exact_kv_tokens // leaf)
    exact = max(1, min(free_leaves, int(round(frac * free_leaves)))) if free_leaves else 0
    return {"q": float(step[0]), "leaf": int(leaf), "group": int(group), "exact_leaves": int(exact)}

