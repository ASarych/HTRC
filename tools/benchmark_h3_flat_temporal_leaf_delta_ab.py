"""Research-only H3 Flat temporal leaf-state delta feasibility A/B.

The fixed 16-token leaf partition is NOT rebuilt in Flat. The expensive
per-query-centre leaf summaries G/VB/LZ/SG are refreshed every call. This
probe compares an observed-delta, mean-leaf update of those summaries to
native HTRC Triton full leaf-statistics on consecutive QKV banks, and checks
real Tile+Merge output. No production change, predictive runtime, quality
coefficient, or cross-denoise cache is enabled.

The update is generally APPROXIMATE. It is a diagnostic, not an accepted
speed optimization or a future-timestep predictor. One persistent JSON.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

D = 128
LEAF = 16
TILE = 64
SCALE = 1.0 / math.sqrt(D)


def head_leaf_mean_delta(old: torch.Tensor, new: torch.Tensor, nkeys: int) -> torch.Tensor:
    """[H,S,D] -> [H,leaves,D] float32 observed per-leaf mean delta."""
    if old.shape != new.shape or old.ndim != 3 or old.shape[-1] != D:
        raise ValueError("Delta inputs must have matching [H,S,128] shape")
    if nkeys < LEAF or nkeys % LEAF or nkeys > old.shape[1]:
        raise ValueError("nkeys must be valid 16-row-aligned prefix")
    h = int(old.shape[0])
    a = old[:, :nkeys, :].float()
    b = new[:, :nkeys, :].float()
    return (b - a).reshape(h, nkeys // LEAF, LEAF, D).mean(dim=2)


def temporal_delta_into(previous: dict, mu_next: torch.Tensor,
                        delta_k: torch.Tensor, delta_v: torch.Tensor,
                        target: dict) -> None:
    """Cheap first-order *hypothesis* on prior quantized HTRC leaf summaries.

    Old g = G8 * SG approximates scaled E[K] in old leaf softmax.
    Update g by scaled mean dK; update vbar by mean dV; update logZ by
    dmu dot old g plus scaled mu_next dot mean dK. Omitted covariance
    terms mean it is NOT exact for arbitrary changed Q/K/V.
    """
    g8 = previous["G"]
    h, c, leaves, d = g8.shape
    if d != D or delta_k.shape != (h, leaves, D) or delta_v.shape != (h, leaves, D):
        raise ValueError("Leaf delta shape mismatch")
    if previous["MU"].shape != (h, c, D) or mu_next.shape != (h, c, D):
        raise ValueError("MU shape mismatch")
    for name in ("G", "VB", "LZ", "SG"):
        if target[name].shape != previous[name].shape:
            raise ValueError(f"Target {name} shape mismatch")
    g0 = g8.float() * previous["SG"].unsqueeze(-1)
    dmu = mu_next.float() - previous["MU"].float()
    corrected_g = g0 + SCALE * delta_k[:, None, :, :]
    amplitude = corrected_g.abs().amax(dim=-1).clamp_min(1.0e-30)
    scale_g = amplitude * (1.0 / 127.0)
    x = corrected_g * (127.0 / amplitude).unsqueeze(-1)
    new_g8 = torch.where(x >= 0, x + 0.5, x - 0.5).to(torch.int8)
    corrected_lz = (
        previous["LZ"]
        + (dmu[:, :, None, :] * g0).sum(dim=-1)
        + SCALE * torch.einsum("hcd,hld->hcl", mu_next.float(), delta_k)
    )
    corrected_vb = previous["VB"].float() + delta_v[:, None, :, :]
    target["G"].copy_(new_g8)
    target["SG"].copy_(scale_g)
    target["VB"].copy_(corrected_vb.to(previous["VB"].dtype))
    target["LZ"].copy_(corrected_lz)


def dirty_stats_coverage(mu_old: torch.Tensor, mu_new: torch.Tensor,
                         k_old: torch.Tensor, k_new: torch.Tensor,
                         v_old: torch.Tensor, v_new: torch.Tensor,
                         nkeys: int) -> dict:
    """A leaf summary is *exactly* reusable only if MU and every K/V row match.

    Q rows themselves may change while their MU stays equal, so this gates
    stats only, never rowmax, selection, Tile, or final output reuse.
    """
    h, centers, dim = mu_old.shape
    if dim != D or mu_new.shape != mu_old.shape:
        raise ValueError("MU pair shape mismatch")
    if any(x.shape != k_old.shape for x in (k_new, v_old, v_new)):
        raise ValueError("K/V shapes differ")
    leaves = nkeys // LEAF
    ck = (k_old[:, :nkeys, :] != k_new[:, :nkeys, :]).reshape(
        h, leaves, LEAF, D).any(dim=(2, 3))
    cv = (v_old[:, :nkeys, :] != v_new[:, :nkeys, :]).reshape(
        h, leaves, LEAF, D).any(dim=(2, 3))
    cm = (mu_old != mu_new).any(dim=-1)
    dleaf = ck | cv
    dirty = cm[:, :, None] | dleaf[:, None, :]
    total = h * centers * leaves
    return {
        "total_head_center_leaf_pairs": total,
        "changed_mu_centers": int(cm.sum().item()),
        "total_mu_centers": int(cm.numel()),
        "changed_k_leaves": int(ck.sum().item()),
        "changed_v_leaves": int(cv.sum().item()),
        "changed_k_or_v_leaves": int(dleaf.sum().item()),
        "total_head_leaves": int(dleaf.numel()),
        "dirty_head_center_leaf_pairs": int(dirty.sum().item()),
        "exact_reusable_pairs": int((~dirty).sum().item()),
        "exact_reusable_fraction": float((~dirty).sum().item() / total),
        "selection_must_still_be_recomputed": True,
    }


def selected_leaf_patch(previous_idx: torch.Tensor, next_idx: torch.Tensor,
                        leaves: int) -> dict:
    """Measure compact IDX membership *changes*, not a reusable G/VB cache.

    This is an ORACLE patch size: next_idx is freshly computed. Discovering
    missing incoming leaves efficiently is a separate, unsolved problem.
    """
    if previous_idx.shape != next_idx.shape or previous_idx.ndim != 3:
        raise ValueError("IDX pair shape mismatch")
    h, centers, kx = previous_idx.shape
    if leaves < kx:
        raise ValueError("Leaf capacity smaller than selected count")
    old = torch.zeros((h * centers, leaves), device=previous_idx.device, dtype=torch.bool)
    new = torch.zeros_like(old)
    old.scatter_(1, previous_idx.reshape(h * centers, kx).long(), True)
    new.scatter_(1, next_idx.reshape(h * centers, kx).long(), True)
    drops = old & ~new
    adds = new & ~old
    changed = (old != new).any(dim=1)
    num_add = int(adds.sum().item())
    num_remove = int(drops.sum().item())
    unchanged = int((~changed).sum().item())
    total_centers = h * centers
    replacement = num_add  # both sides have equal kx
    return {
        "head_centers": total_centers,
        "selected_per_center": kx,
        "unchanged_selected_set_centers": unchanged,
        "unchanged_selected_set_fraction": unchanged / total_centers,
        "removed_memberships": num_remove,
        "added_memberships": num_add,
        "replaced_fraction_of_selected": replacement / (total_centers * kx),
        "full_idx_int32_bytes": total_centers * kx * 4,
        "oracle_minimum_pair_patch_bytes": (num_remove + num_add) * 4,
        "patch_discovery_is_free": False,
        "selection_unchanged_does_not_imply_stats_or_output_unchanged": True,
    }


def selected_idx_position_patch(previous_idx: torch.Tensor,
                                next_idx: torch.Tensor) -> dict:
    """Demonstrate exact ordered IDX reconstruction from changed positions.

    Stable set membership is NOT enough to preserve floating accumulation
    order in Tile. This measures actual positional edits and validates that
    applying the patch reproduces next_idx byte-for-byte. Patch discovery
    still uses a freshly built next_idx and is NOT accelerated here.
    """
    if previous_idx.shape != next_idx.shape or previous_idx.ndim != 3:
        raise ValueError("Ordered IDX pair shape mismatch")
    mask = previous_idx != next_idx
    patch = next_idx[mask]
    restored = previous_idx.clone()
    restored[mask] = patch
    if not torch.equal(restored, next_idx):
        raise AssertionError("Position patch did not reconstruct exact IDX")
    count = int(mask.sum().item())
    full = int(next_idx.numel() * next_idx.element_size())
    # Conservative 4B linear position plus 4B int32 ID per replacement.
    return {
        "replaced_index_positions": count,
        "total_index_positions": int(next_idx.numel()),
        "position_change_fraction": count / next_idx.numel(),
        "patch_bytes_position_u32_plus_id_i32": count * 8,
        "full_idx_bytes": full,
        "round_trip_ordered_idx_bit_exact": True,
        "requires_fresh_idx_for_patch_discovery": True,
    }


def cache_footprint_projection(heads: int, tokens: int, exact_leaves: int) -> dict:
    """Full-attention all-centre state size vs compact selected IDX only."""
    leaves = tokens // LEAF
    centers = math.ceil(tokens / TILE)
    pairs = heads * centers * leaves
    full_stats = pairs * (D + 2 * D + 4 + 4)  # G INT8, VB BF16, LZ/SG FP32
    selected = heads * centers * min(leaves, exact_leaves) * 4
    return {
        "centers_per_head": centers,
        "leaves_per_head": leaves,
        "estimated_G_VB_LZ_SG_bytes_one_block": full_stats,
        "estimated_G_VB_LZ_SG_gib_one_block": full_stats / (1024 ** 3),
        "estimated_selected_IDX_int32_bytes_one_block": selected,
        "estimated_selected_IDX_int32_mib_one_block": selected / (1024 ** 2),
        "note": "All full-Q centres, one H3 block, one step; does not include keys, values, masks, projection weights, concurrency or 49-block persistence."
    }


def tensor_difference(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    if reference.shape != candidate.shape:
        raise ValueError("Comparison tensor shape mismatch")
    a = reference.float()
    b = candidate.float()
    diff = b - a
    return {
        "relative_l2": float((diff.norm() / a.norm().clamp_min(1.0e-20)).item()),
        "max_abs": float(diff.abs().amax().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "bit_exact": bool(torch.equal(reference, candidate)),
    }


def load_trusted_capture(path: Path, heads: int) -> tuple[tuple[torch.Tensor, ...], tuple[int, ...]]:
    """Caller-owned local capture only; never load arbitrary downloaded pickle."""
    if not path.is_file():
        raise FileNotFoundError(path)
    data = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if not isinstance(data, dict) or any(k not in data for k in
                                             ("q_qhd", "k_khd", "v_khd")):
        raise ValueError("Capture lacks q_qhd/k_khd/v_khd")
    raw = [data[k] for k in ("q_qhd", "k_khd", "v_khd")]
    if any(x.ndim != 3 or x.shape != raw[0].shape or x.shape[-1] != D
           for x in raw):
        raise ValueError("Capture must contain common [S,H,128] QKV")
    if int(raw[0].shape[1]) < heads:
        raise ValueError("Capture contains fewer distinct heads than requested")
    ids = list(range(int(raw[0].shape[1])))
    if "raw_head_index" in data:
        ids = data["raw_head_index"]
        ids = ids.tolist() if isinstance(ids, torch.Tensor) else list(ids)
        if len(ids) != int(raw[0].shape[1]):
            raise ValueError("Captured raw_head_index length disagrees with QKV")
    chosen = tuple(map(int, ids[:heads]))
    if len(set(chosen)) != heads:
        raise ValueError("Captured heads must be distinct")
    tensors = tuple(x[:, :heads, :].permute(1, 0, 2).contiguous().to(
        device="cuda", dtype=torch.bfloat16) for x in raw)
    return tensors, chosen


def synthetic_next(banks: tuple[torch.Tensor, ...], mode: str,
                   std: float, sparse_fraction: float, seed: int) -> tuple[torch.Tensor, ...]:
    q, k, v = banks
    h, tokens, _ = q.shape
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    next_q, next_k, next_v = (x.clone() for x in banks)

    def shift(x, noise):
        x.copy_((x.float() + noise).to(x.dtype))

    if mode == "dense_drift":
        for x in (next_q, next_k, next_v):
            shift(x, torch.randn(x.shape, generator=generator, device=x.device) * std)
    elif mode == "sparse_drift":
        cg = torch.Generator(device="cpu").manual_seed(seed)
        centers = math.ceil(tokens / TILE)
        leaves = tokens // LEAF
        ncenters = max(1, math.ceil(centers * sparse_fraction))
        nleaves = max(1, math.ceil(leaves * sparse_fraction))
        for center in torch.randperm(centers, generator=cg)[:ncenters].tolist():
            lo, hi = center * TILE, min(tokens, (center + 1) * TILE)
            x = next_q[:, lo:hi, :]
            shift(x, torch.randn(x.shape, generator=generator, device=x.device) * std)
        for leaf in torch.randperm(leaves, generator=cg)[:nleaves].tolist():
            lo, hi = leaf * LEAF, (leaf + 1) * LEAF
            for all_keys in (next_k, next_v):
                x = all_keys[:, lo:hi, :]
                shift(x, torch.randn(x.shape, generator=generator, device=x.device) * std)
    elif mode == "uniform_leaf_drift":
        leaves = tokens // LEAF
        for x in (next_k, next_v):
            noise = torch.randn((h, leaves, 1, D), generator=generator,
                                device=x.device) * std
            base = x[:, :leaves * LEAF, :].float().reshape(h, leaves, LEAF, D)
            x[:, :leaves * LEAF, :].copy_((base + noise).reshape(h, leaves * LEAF, D).to(x.dtype))
    else:
        raise ValueError("Unsupported synthetic delta mode")
    return next_q, next_k, next_v


def make_q_window(q: torch.Tensor, start: int, max_centers: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    h, tokens, d = q.shape
    if start < 0 or start >= tokens or start % TILE:
        raise ValueError("query-start must be in-range and tile-aligned")
    valid = min(max_centers * TILE, tokens - start)
    centers = math.ceil(valid / TILE)
    src = q[:, start:start + valid, :]
    if centers * TILE != valid:
        src = torch.cat((src, src[:, -1:, :].expand(h, centers * TILE - valid, d)), dim=1)
    tiled = src.contiguous().view(h, centers, TILE, d)
    mu = tiled.mean(dim=2, dtype=torch.float32)
    if valid % TILE:
        mu[:, -1, :] = tiled[:, -1, :valid % TILE, :].mean(dim=1, dtype=torch.float32)
    return tiled, mu.contiguous(), valid


def allocate_stage(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   start: int, max_centers: int, exact_leaves: int,
                   tri) -> tuple[dict, tuple[torch.Tensor, torch.Tensor] | None, int]:
    h, tokens, _ = q.shape
    nkeys = tokens // LEAF * LEAF
    if nkeys < LEAF:
        raise ValueError("At least 16 K/V tokens are required")
    leaves = nkeys // LEAF
    Q, MU, valid = make_q_window(q, start, max_centers)
    centers = int(Q.shape[1])
    keys = k[:, :nkeys, :].contiguous()
    values = v[:, :nkeys, :].contiguous()
    k8 = torch.empty_like(keys, dtype=torch.int8)
    ks = torch.empty((h, nkeys), device=q.device, dtype=torch.float32)
    tri._quant_key_int8_rows[(tri.triton.cdiv(h * nkeys, 8),)](
        keys.view(h * nkeys, D), k8.view(h * nkeys, D),
        ks.view(h * nkeys), h * nkeys, D, 8, num_warps=4)
    state = {
        "Q": Q, "MU": MU, "K": k8, "V": values, "KS": ks,
        "_KBF": keys,
        "G": torch.empty((h, centers, leaves, D), dtype=torch.int8, device=q.device),
        "VB": torch.empty((h, centers, leaves, D), dtype=torch.bfloat16, device=q.device),
        "LZ": torch.empty((h, centers, leaves), dtype=torch.float32, device=q.device),
        "SG": torch.empty((h, centers, leaves), dtype=torch.float32, device=q.device),
        "EXCL": torch.empty((h, centers, leaves), dtype=torch.int8, device=q.device),
        "IDX": torch.empty((h, centers, min(exact_leaves, leaves)),
                           dtype=torch.int32, device=q.device),
        "_direct_q": False, "_direct_v": False,
    }
    tails = ((k[:, nkeys:, :].contiguous(), v[:, nkeys:, :].contiguous())
             if tokens != nkeys else None)
    return state, tails, valid


def fill_native_leaf_stats(state: dict, keybf: torch.Tensor,
                           valbf: torch.Tensor, tri) -> None:
    h, centers, leaves, d = state["G"].shape
    nkeys = leaves * LEAF
    bt, bl = 16, 4
    blocks = math.ceil(centers / bt)
    tri._stats_kernel[(h * blocks, tri.triton.cdiv(leaves, bl))](
        state["MU"].view(h * centers, d),
        keybf.view(h * nkeys, d),
        valbf.view(h * nkeys, d),
        state["G"].view(h * centers, leaves, d),
        state["LZ"].view(h * centers, leaves),
        state["VB"].view(h * centers, leaves, d),
        state["SG"].view(h * centers, leaves),
        state["LZ"].view(h * centers, leaves),
        h * centers, leaves, SCALE,
        D=d, BT=bt, LEAF=LEAF, BL=bl, PREC="tf32", FP8=0, I8=1,
        BATCH_TILES_PER_HEAD=centers,
        BATCH_KEY_ROWS=nkeys, BATCH_TILE_BLOCKS_PER_HEAD=blocks,
        DIRECT_KV=0, K_SH=0, K_SS=0, K_SD=0,
        V_SH=0, V_SS=0, V_SD=0,
        num_warps=4,
    )


def refresh_selection(state: dict, cfg: dict, tri) -> None:
    h, centers, leaves, d = state["G"].shape
    q = state["Q"].view(h * centers, TILE, d)
    mu = state["MU"].view(h * centers, d)
    sel = tri.rowmax_select(
        q, mu, state["G"].view(h * centers, leaves, d),
        state["LZ"].view(h * centers, leaves),
        prec="tf32", bn=32, warps=4, splits=int(cfg["splits"]),
        stride=1, dot=1, ws=None,
        sg=state["SG"].view(h * centers, leaves)
    )
    ids = torch.topk(sel, state["IDX"].shape[-1], dim=-1, sorted=False).indices
    state["IDX"].copy_(ids.to(torch.int32).view_as(state["IDX"]))
    excluded = state["EXCL"].view(h * centers, leaves)
    excluded.zero_()
    excluded.scatter_(1, ids, 1)


def output_buffers(state: dict, cfg: dict) -> tuple[dict, torch.Tensor]:
    h, centers = state["Q"].shape[:2]
    splits = int(cfg["splits"])
    base = (h, centers, splits, TILE)
    dev = state["Q"].device
    partials = {
        "pm": torch.empty(base, dtype=torch.float32, device=dev),
        "pl": torch.empty(base, dtype=torch.float32, device=dev),
        "acc": torch.empty(base + (D,), dtype=torch.float32, device=dev),
    }
    out = torch.empty((h, centers, TILE, D), dtype=torch.float32, device=dev)
    return partials, out


def attention_output(state, cfg, tri, batch, tail, buffers):
    partials, out = buffers
    batch.launch_prepared_tile_batch(state, cfg, tri, buffers=partials)
    merged, _ = batch.launch_prepared_merge_batch(
        state, cfg, tri, partials, tails=tail, output=out, merge_rows=32
    )
    return merged


def timed_cuda(fn, warmups: int, rounds: int) -> dict:
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    samples, wall = [], []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
        wall.append((time.perf_counter() - t0) * 1000.0)
    return {"median_gpu_ms": float(statistics.median(samples)),
            "median_sync_wall_ms": float(statistics.median(wall)),
            "samples_gpu_ms": samples}


def run_case(case_name: str, old_banks, new_banks, args, tri, batch, cfg) -> dict:
    q0, k0, v0 = old_banks
    q1, k1, v1 = new_banks
    if any(x.shape != q0.shape for x in (k0, v0, q1, k1, v1)):
        raise ValueError("All two-state QKV tensors must have identical shapes")
    h, tokens, d = q0.shape
    nkeys = tokens // LEAF * LEAF
    previous, _, valid = allocate_stage(q0, k0, v0, args.query_start,
                                         args.centers, cfg["exact_leaves"], tri)
    fresh, tail1, _ = allocate_stage(q1, k1, v1, args.query_start,
                                     args.centers, cfg["exact_leaves"], tri)
    estimated = dict(fresh)
    for name in ("G", "VB", "LZ", "SG", "EXCL", "IDX"):
        estimated[name] = torch.empty_like(fresh[name])
    fill_native_leaf_stats(previous, previous["_KBF"], previous["V"], tri)
    refresh_selection(previous, cfg, tri)
    fill_native_leaf_stats(fresh, fresh["_KBF"], fresh["V"], tri)
    refresh_selection(fresh, cfg, tri)

    dk = head_leaf_mean_delta(k0, k1, nkeys)
    dv = head_leaf_mean_delta(v0, v1, nkeys)
    temporal_delta_into(previous, fresh["MU"], dk, dv, estimated)
    refresh_selection(estimated, cfg, tri)

    shared_buffers = output_buffers(fresh, cfg)
    original_output = attention_output(fresh, cfg, tri, batch, tail1, shared_buffers).clone()
    updated_output = attention_output(estimated, cfg, tri, batch, tail1, shared_buffers).clone()
    torch.cuda.synchronize()
    gold_ids = torch.sort(fresh["IDX"].view(-1, fresh["IDX"].shape[-1]), dim=-1).values
    estimated_ids = torch.sort(estimated["IDX"].view(-1, estimated["IDX"].shape[-1]), dim=-1).values
    changed_selected_centers = (gold_ids != estimated_ids).any(dim=-1)
    actual_selected_patch = selected_leaf_patch(previous["IDX"], fresh["IDX"], nkeys // LEAF)
    actual_ordered_patch = selected_idx_position_patch(previous["IDX"], fresh["IDX"])
    predicted_selected_patch = selected_leaf_patch(fresh["IDX"], estimated["IDX"], nkeys // LEAF)
    dirty = dirty_stats_coverage(previous["MU"], fresh["MU"],
                                 k0, k1, v0, v1, nkeys)
    summary = {name: tensor_difference(fresh[name], estimated[name])
               for name in ("G", "VB", "LZ", "SG")}
    # Quantized G error must also be evaluated in its FP32 scale domain.
    gold_g = fresh["G"].float() * fresh["SG"].unsqueeze(-1)
    est_g = estimated["G"].float() * estimated["SG"].unsqueeze(-1)
    summary["G_dequantized"] = tensor_difference(gold_g, est_g)
    quality = tensor_difference(original_output, updated_output)
    exact_all = (quality["bit_exact"] and
                 all(summary[k]["bit_exact"] for k in ("G", "VB", "LZ", "SG")) and
                 bool(torch.equal(fresh["IDX"], estimated["IDX"])) and
                 bool(torch.equal(fresh["EXCL"], estimated["EXCL"])))

    def fresh_path():
        fill_native_leaf_stats(fresh, fresh["_KBF"], fresh["V"], tri)
        refresh_selection(fresh, cfg, tri)
        attention_output(fresh, cfg, tri, batch, tail1, shared_buffers)

    def delta_path():
        dk_live = head_leaf_mean_delta(k0, k1, nkeys)
        dv_live = head_leaf_mean_delta(v0, v1, nkeys)
        temporal_delta_into(previous, fresh["MU"], dk_live, dv_live, estimated)
        refresh_selection(estimated, cfg, tri)
        attention_output(estimated, cfg, tri, batch, tail1, shared_buffers)

    timings = {
        "fresh_stats_selection_tile_merge": timed_cuda(fresh_path, args.warmups, args.rounds),
        "observed_delta_selection_tile_merge": timed_cuda(delta_path, args.warmups, args.rounds),
        "boundary": "Post-MU/post-K8: includes full leaf-stats or both K/V mean deltas, selection/topk/EXCL, Tile and Merge; EXCLUDES new QKV projection, Q MU, K8, tensor input allocation and GPU model weights.",
    }
    base_ms = timings["fresh_stats_selection_tile_merge"]["median_sync_wall_ms"]
    change_ms = timings["observed_delta_selection_tile_merge"]["median_sync_wall_ms"]
    timings["wall_speedup_fresh_over_delta"] = base_ms / change_ms if change_ms else None
    return {
        "name": case_name,
        "input_kind": "paired_real_QKV_captures" if case_name == "real_pair" else "synthetic_temporal_delta",
        "heads": h, "tokens": tokens, "valid_query_rows": valid,
        "query_start": args.query_start,
        "query_centers": int(previous["MU"].shape[1]),
        "leaves": nkeys // LEAF, "exact_leaves_selected": int(fresh["IDX"].shape[-1]),
        "dirty_coverage": dirty,
        "actual_selected_leaf_change_between_steps": actual_selected_patch,
        "exact_ordered_idx_position_patch_oracle": actual_ordered_patch,
        "estimated_selected_leaf_error_vs_fresh": predicted_selected_patch,
        "cache_footprint_if_all_query_windows_persist": cache_footprint_projection(h, tokens, int(fresh["IDX"].shape[-1])),
        "summary_error_vs_fresh": summary,
        "selected_leaf_set_different_centers": int(changed_selected_centers.sum().item()),
        "selected_leaf_set_total_centers": int(changed_selected_centers.numel()),
        "selected_leaf_set_different_fraction": float(changed_selected_centers.float().mean().item()),
        "output_error_vs_fresh_flat": quality,
        "strict_bitwise_identity_pass": exact_all,
        "eligible_for_quality_preserving_integration": exact_all,
        "timings": timings,
        "note": "Delta candidate uses observed NEW K/V (not a forecast) and approximated previous INT8 G; no quality regression accepted; numerical FAIL is a valid feasibility result.",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=18723)
    ap.add_argument("--centers", type=int, default=16)
    ap.add_argument("--query-start", type=int, default=0)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--delta-std", type=float, default=0.02)
    ap.add_argument("--sparse-fraction", type=float, default=0.05)
    ap.add_argument("--scenarios", default="dense_drift,sparse_drift,uniform_leaf_drift")
    ap.add_argument("--capture-a", type=Path, default=None)
    ap.add_argument("--capture-b", type=Path, default=None)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--warmups", type=int, default=2)
    ap.add_argument("--out", type=Path, default=Path(
        r"D:\ComfyOutput\H3\math_probe\flat_temporal_leaf_delta_ab.json"))
    args = ap.parse_args()
    if args.heads < 1 or args.tokens < LEAF or args.centers < 1 or min(args.rounds, args.warmups) < 1:
        ap.error("heads, tokens, centers, rounds and warmups must be positive (tokens >=16)")
    if not (0.0 < args.sparse_fraction <= 1.0) or args.delta_std <= 0:
        ap.error("sparse-fraction must be (0,1], delta-std positive")
    if args.capture_b is not None and args.capture_a is None:
        ap.error("--capture-b requires --capture-a")
    scenarios = [m.strip() for m in args.scenarios.split(",") if m.strip()]
    if not scenarios or any(m not in ("dense_drift", "sparse_drift", "uniform_leaf_drift")
                            for m in scenarios):
        ap.error("Unknown --scenarios")
    report = {
        "experiment": "h3_flat_temporal_leaf_summary_delta_feasibility_v1",
        "completed": False,
        "accepted_for_runtime": False,
        "real_model_weights_loaded": False,
        "predicts_future_timestep_before_new_qkv": False,
        "quality_coefficient_used": False,
        "leaf_rows": LEAF,
        "fixed_exact_leaves": 256,
        "fixed_center_group": 1,
        "inputs": {}, "cases": [], "errors": [],
        "notes": [
            "The 16-token leaf partition itself does NOT need rebuilding in Flat.",
            "Primary compact-structure signal: count changed selected leaf IDs (IDX membership) and oracle patch bytes between states, without treating the full G/VB cache as portable across steps.",
            "Secondary diagnostic: approximate derived per-centre per-leaf G8/SG/VB/LZ from previous summaries and observed K/V mean deltas + changed MU, only on one group/window.",
            "A full cross-step G/VB/LZ/SG cache for ALL query windows is prohibitively large at H3 shape; selected IDX is compact but computing its correct next members is still unsolved.",
            "Covariance effects omitted: approximation NOT safe for production unless parity/quality is proven.",
            "Native HTRC Triton exact full leaf-stats and unchanged Tile/Merge are the control.",
            "Exact dirty gate counts reusable stats only when BOTH MU and all K/V tokens are unchanged; selection depends on full Q.",
            "Synthetic drift tests cannot establish temporal similarity in real successive H3 denoising steps.",
            "One JSON, no H3 production hook or CUDA Graph, no video generation."
        ],
    }
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for native HTRC Triton proof")
        from h3 import htrc_flat_head_batch as batch
        from h3 import htrc_flat_live_backend as live
        from h3 import htrc_flat_model_shadow as shared
        tri = live._triton()
        tri._dll()
        cfg = shared.flat_config("fast", args.centers)
        if cfg["exact_leaves"] != 256 or cfg["kern"].get("group", 1) != 1:
            raise RuntimeError("Fixed Flat maths changed")
        torch.cuda.reset_peak_memory_stats()
        paired_next = None
        if args.capture_a is not None:
            old, source_head_ids = load_trusted_capture(args.capture_a, args.heads)
            report["inputs"]["capture_a"] = str(args.capture_a)
            report["inputs"]["raw_head_ids"] = source_head_ids
            if args.capture_b is not None:
                scenarios = ["real_pair"]
                paired_next, next_head_ids = load_trusted_capture(args.capture_b, args.heads)
                report["inputs"]["capture_b"] = str(args.capture_b)
                if next_head_ids != source_head_ids:
                    raise ValueError("Capture pair uses different ordered raw head IDs")
                if any(x.shape != old[0].shape for x in paired_next):
                    raise ValueError("QKV capture pair has different head/token dimensions")
        else:
            gen = torch.Generator(device="cuda")
            gen.manual_seed(args.seed)
            old = tuple(torch.randn((args.heads, args.tokens, D), device="cuda",
                                    generator=gen, dtype=torch.bfloat16) for _ in range(3))
        report["gpu"] = torch.cuda.get_device_name()
        report["inputs"].update({"heads": int(old[0].shape[0]),
                                  "tokens": int(old[0].shape[1]),
                                  "delta_std": args.delta_std,
                                  "sparse_fraction": args.sparse_fraction,
                                  "scenarios": scenarios})
        for i, mode in enumerate(scenarios):
            new = (paired_next if mode == "real_pair"
                   else synthetic_next(old, mode, args.delta_std,
                                       args.sparse_fraction, args.seed + 101 + i))
            print(f"TEMPORAL_LEAF_DELTA_CASE {mode} start", flush=True)
            one = run_case(mode, old, new, args, tri, batch, cfg)
            report["cases"].append(one)
            print(f"TEMPORAL_LEAF_DELTA_CASE {mode} done; exact={one['strict_bitwise_identity_pass']}; "
                  f"output_rel_l2={one['output_error_vs_fresh_flat']['relative_l2']:.7g}; "
                  f"reusable={one['dirty_coverage']['exact_reusable_fraction']:.4f}", flush=True)
            del new
        report["completed"] = True
        report["peak_torch_allocated_mib"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
        report["all_cases_bit_exact"] = all(x["strict_bitwise_identity_pass"]
                                              for x in report["cases"])
        # This exploratory predictor is NOT authorized for live H3 even if a
        # synthetic sample happens to match. Separate real-step proof needed.
        return_code = 0
    except BaseException as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}"[:1500])
        report["traceback_tail"] = traceback.format_exc()[-4000:]
        return_code = 1
    finally:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("REPORT:", args.out, flush=True)
        print("COMPLETED:", report["completed"], flush=True)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
