"""Research-only model-resident H3 Direct-Flow Graph, disabled by default.

Opt-in graph path uses real, fresh post-RoPE BHSD QKV. All math is the
already-validated HTRC Flat fast/balanced 256/512-leaf / group1 paths. Graph objects
own only STAGED GPU buffers, never source/model QKV or model weights.
Same shape group sessions may be reused across model blocks/timesteps;
every replay requantizes K, refreshes K/V tail and stages Q. The first
full call compares bitwise with the native eager HTRC Flat on true QKV.
No implicit fallback or replacement of the default Flat path.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import threading
import time

import torch

from . import htrc_flat_head_batch as batch
from . import htrc_flat_graph_group_plan as group_plan
from . import htrc_flat_model_shadow as shared
from . import htrc_flat_windows as windows

_LOCK = threading.RLock()
_SESSIONS = {}
_STATE = {"geometry": None, "verified": False, "reports": [], "last_event": None}
_MIB = 1 << 20


def _memory_probe(device, baseline=None):
    """Research allocator snapshot; NOT an exact private Graph-pool breakdown."""
    allocated = int(torch.cuda.memory_allocated(device))
    reserved = int(torch.cuda.memory_reserved(device))
    free, _ = torch.cuda.mem_get_info(device)
    data = {
        "allocated_bytes": allocated,
        "reserved_bytes": reserved,
        "allocated_mib": round(allocated/_MIB, 3),
        "reserved_mib": round(reserved/_MIB, 3),
        "reserved_minus_allocated_mib": round((reserved-allocated)/_MIB, 3),
        "free_mib": round(free/_MIB, 3),
    }
    if baseline is not None:
        data["allocated_delta_mib"] = round(
            (allocated-baseline["allocated_bytes"])/_MIB, 3)
        data["reserved_delta_mib"] = round(
            (reserved-baseline["reserved_bytes"])/_MIB, 3)
    return data


def _env_int(name, default, lo, hi):
    value = int(os.environ.get(name, str(default)))
    if not lo <= value <= hi:
        raise ValueError(f"{name} outside {lo}..{hi}")
    return value


def _geometry_key(win):
    return win.tile_count, win.valid_rows


def _copy_q(q_bhsd, geom, qbuf):
    real = int(geom.valid_rows)
    if qbuf.shape != (int(q_bhsd.shape[1]), int(geom.padded_rows), 128):
        raise ValueError("Direct-Flow Q stage shape mismatch")
    src = q_bhsd[0, :, geom.start:geom.end, :]
    qbuf[:, :real, :].copy_(src)
    if real < geom.padded_rows:
        qbuf[:, real:, :].copy_(
            src[:, -1:, :].expand(-1, geom.padded_rows-real, -1)
        )


def _one_window(window_capture, ids, cfg, tr, cache, ws, tail, geom,
                vb_mode="materialized"):
    states, _, _, _, _ = batch.prepare_batched_head_states(
        window_capture, ids, cfg, tr, tile_count=geom.tile_count,
        quant_backend="triton", input_cast="gpu", kv_cache=cache,
        query_workspace=ws, valid_query_rows=geom.valid_rows,
        query_staging="native_torch_mu", vb_mode=vb_mode,
    )
    if vb_mode == "split_stream":
        partial, _ = batch.launch_prepared_tile_batch_split_stream(
            states, cfg, tr, buffers=ws["_partials"], vb_split=ws["VB_SPLIT"]
        )
    elif vb_mode == "fused_vb":
        partial, _ = batch.launch_prepared_tile_batch_fused_vb(
            states, cfg, tr, buffers=ws["_partials"]
        )
    else:
        partial, _ = batch.launch_prepared_tile_batch(
            states, cfg, tr, buffers=ws["_partials"]
        )
    merged, _ = batch.launch_prepared_merge_batch(
        states, cfg, tr, partial, tails=tail,
        output=ws["_out"], merge_rows=32,
    )
    return states, merged


def _new_session(bank, geometries, tiles, cfg, tr, warmups,
                 vb_mode="materialized", memory_trace=None,
                 baseline_memory=None):
    h, seq = int(bank[0].shape[1]), int(bank[0].shape[2])
    def note(phase, geom=None):
        if memory_trace is None:
            return
        entry = {"phase": phase, "session_heads": h}
        if geom is not None:
            entry["geometry"] = [geom.tile_count, geom.valid_rows]
        entry.update(_memory_probe(bank[0].device, baseline_memory))
        memory_trace.append(entry)

    ids = list(range(h))
    sk = torch.empty_like(bank[1], memory_format=torch.contiguous_format)
    sv = torch.empty_like(bank[2], memory_format=torch.contiguous_format)
    sk.copy_(bank[1])
    sv.copy_(bank[2])
    cap = {
        "k_khd": sk[0].permute(1, 0, 2),
        "v_khd": sv[0].permute(1, 0, 2),
        "raw_head_index": ids,
    }
    cache = batch.prepare_gpu_head_kv_cache(cap, ids, tr, native_direct=True)
    tail = shared._tail(cap, ids)
    stage = {
        "k": sk, "v": sv, "cap": cap, "cache": cache,
        "tail": tail, "ids": ids, "graphs": {},
    }
    note("after_stage_kv_cache")
    unique = {}
    for geom in geometries:
        unique[_geometry_key(geom)] = geom
    current = torch.cuda.current_stream(bank[0].device)
    started = time.perf_counter()
    for key, geom in unique.items():
        qbuf = torch.empty((h, geom.padded_rows, 128),
                           device=bank[0].device, dtype=torch.bfloat16)
        _copy_q(bank[0], geom, qbuf)
        win = dict(cap)
        win["q_qhd"] = qbuf.permute(1, 0, 2)
        ws = shared.alloc_workspace(win, h, geom.tile_count, cfg,
                                    vb_mode=vb_mode)
        note("after_workspace_alloc", geom)
        def perform():
            return _one_window(win, ids, cfg, tr, cache, ws, tail, geom,
                               vb_mode=vb_mode)
        side = torch.cuda.Stream(device=bank[0].device)
        side.wait_stream(current)
        with torch.cuda.stream(side):
            for _ in range(warmups):
                st, merged = perform()
                del st, merged
        current.wait_stream(side)
        torch.cuda.synchronize(bank[0].device)
        note("after_warmup", geom)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            states, merged = perform()
        torch.cuda.synchronize(bank[0].device)
        note("after_capture", geom)
        stage["graphs"][key] = {
            "graph": graph, "q": qbuf, "win": win, "ws": ws,
            "states": states, "output": merged,
        }
    stage["capture_wall_ms"] = (time.perf_counter()-started)*1000
    stage["geometry"] = (seq, h, tiles)
    return stage


def _refresh_kv(stage, bank, tr):
    stage["k"].copy_(bank[1])
    stage["v"].copy_(bank[2])
    cache = stage["cache"]
    kbf = cache["KBF"]
    h, n, d = map(int, kbf.shape)
    block_rows = 8
    tr._quant_key_int8_native[(tr.triton.cdiv(h*n, block_rows),)](
        kbf, cache["K"].view(h*n, d), cache["KS"].view(h*n),
        n, d, h, block_rows, *map(int, kbf.stride()),
        num_warps=4,
    )
    if stage["tail"] is not None:
        tk, tv = stage["tail"]
        count = int(tk.shape[1])
        tk.copy_(stage["k"][0, :, -count:, :])
        tv.copy_(stage["v"][0, :, -count:, :])


def _replay(stage, bank, geometries, tr, output, query_provider=None):
    _refresh_kv(stage, bank, tr)
    for geom in geometries:
        item = stage["graphs"][_geometry_key(geom)]
        if query_provider is None:
            _copy_q(bank[0], geom, item["q"])
        else:
            # Research-only producer fills every Q row before Graph replay.
            query_provider(geom.start, geom.end, item["q"])
        item["graph"].replay()
        merged = item["output"]
        nhead = int(output.shape[1])
        selected = merged.reshape(nhead, geom.padded_rows, 128)
        selected = selected[:, :geom.valid_rows, :]
        output[geom.start:geom.end, :, :].copy_(
            selected.permute(1, 0, 2)
        )


def _measurement(call, device):
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    wall = time.perf_counter()
    call()
    cpu_submit = (time.perf_counter()-wall)*1000
    stop.record()
    stop.synchronize()
    return {
        "gpu_event_ms": start.elapsed_time(stop),
        "cpu_submit_ms": cpu_submit,
        "sync_wall_ms": (time.perf_counter()-wall)*1000,
    }


def _write_report(payload):
    path = Path(os.environ.get(
        "APL_HTRC_DIRECTFLOW_REPORT",
        r"D:\ComfyOutput\H3\math_probe\flat_directflow_real_h3_ab.json",
    ))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def clear_session_cache():
    """Explicit best-effort release when model/geometry is retired."""
    with _LOCK:
        # Graph pools must never be freed while GPU replay is still running
        # on a previously used CUDA stream.
        pending = _STATE.get("last_event")
        if pending is not None:
            pending.synchronize()
        _SESSIONS.clear()
        _STATE.update(geometry=None, verified=False, reports=[], last_event=None)


def graph_attention_bthd(q, k, v, *, heads, mode, tiles,
                         budget_mib, reserve_mib, workspace, observer=None,
                         compare_only=False, query_provider=None):
    """Actual opt-in Flat output [1,S,H,128]. Fail closed on parity/VRAM."""
    if torch.is_grad_enabled():
        raise RuntimeError("H3 Direct-Flow is inference-only")
    if type(compare_only) is not bool:
        raise TypeError("compare_only must be a bool (research API, off by default)")
    if query_provider is not None and (
        not compare_only or not callable(query_provider)
    ):
        raise ValueError(
            "Pre-QKV query provider requires compare_only=True and a callable")
    if (mode not in ("fast", "balanced") or not workspace
        or not isinstance(tiles, int) or tiles not in (16, 32, 64)
        or any(x.dtype != torch.bfloat16 for x in (q,k,v))):
        raise ValueError(
            "H3 Direct-Flow supports only fast/balanced / tiles16,32,64 / workspace / BF16"
        )
    cap = shared.capture_from_bhsd(q, k, v, heads)
    s, total_heads, _ = map(int, cap["q_qhd"].shape)
    cfg = shared.flat_config(mode, tiles)
    expected_leaves = {"fast": 256, "balanced": 512}[mode]
    if (int(cfg["exact_leaves"]) != expected_leaves
        or int(cfg.get("kern", {}).get("group", 1)) != 1):
        raise RuntimeError(
            f"Fixed Flat {expected_leaves} exact leaves / centre group1 changed"
        )
    from . import htrc_flat_live_backend as live
    tr = live._triton()
    tr._dll()
    dev = q.device
    # Reserve the actual H3 output BEFORE planning the temporary head-group
    # memory. The eager live backend uses the same accounting order.
    output = torch.empty((s, total_heads, 128), device=dev, dtype=q.dtype)
    # Keep the existing 11-head planner unchanged by default. The
    # separately opted-in reuse-aware planner counts the graph-owned BF16
    # K/V stages ONCE (native cache views alias them), includes distinct
    # full/tail graph pools, and prefers fewer replayed head groups.
    policy=os.environ.get(
        "APL_HTRC_DIRECTFLOW_GROUP_POLICY","legacy"
    ).strip().lower()
    if policy not in ("legacy","reuse_aware"):
        raise ValueError("APL_HTRC_DIRECTFLOW_GROUP_POLICY must be legacy or reuse_aware")
    head_limit = _env_int(
        "APL_HTRC_DIRECTFLOW_GROUP", 32 if policy=="reuse_aware" else 11, 1, 32
    )
    warmups = _env_int("APL_HTRC_DIRECTFLOW_WARMUPS", 4, 3, 20)
    max_extra = _env_int("APL_HTRC_DIRECTFLOW_MAX_EXTRA_MIB", 768, 32, 4096)
    min_free = _env_int(
        "APL_HTRC_DIRECTFLOW_MIN_FREE_MIB", max(256, int(reserve_mib)), 64, 4096
    )
    vb_mode=os.environ.get(
        "APL_HTRC_DIRECTFLOW_VB_MODE","materialized"
    ).strip().lower()
    if vb_mode not in ("materialized","split_stream","fused_vb"):
        raise ValueError(
            "APL_HTRC_DIRECTFLOW_VB_MODE must be materialized, split_stream or fused_vb"
        )
    if policy=="reuse_aware":
        actual_free,_ = torch.cuda.mem_get_info(dev)
        plan=group_plan.choose_group(
            tokens=s,heads=total_heads,tiles=tiles,mode=mode,vb_mode=vb_mode,
            max_group=head_limit,budget_mib=int(budget_mib),
            reserve_mib=int(reserve_mib),max_extra_mib=max_extra,
            free_mib=actual_free/_MIB,min_free_mib=min_free
        )
        group_cap=int(plan["group_size"])
    else:
        legacy_cap,plan=shared._head_plan(
            cap, total_heads, tiles, cfg, budget_mib=int(budget_mib),
            reserve_mib=int(reserve_mib), workspace=True,
        )
        group_cap=min(legacy_cap,head_limit)
    forced=group_plan.research_forced_group_size(
        os.environ.get("APL_HTRC_DIRECTFLOW_RESEARCH_FORCE_GROUP"),
        compare_only=compare_only,policy=policy,
        max_group=head_limit,heads=total_heads,
    )
    if forced is not None:
        plan=dict(plan)
        plan["research_forced_group_size"]=forced
        plan["planner_recommended_group_size"]=group_cap
        plan["research_force_bypasses_capacity_estimate"]=True
        plan["research_force_runtime_guard"]="UNCHANGED"
        # This research-only override can exceed the conservative
        # 1.5x staged model, but the *actual* post-capture hard memory
        # and free-VRAM gates below still fail closed.
        group_cap=forced
    geom = (dev.index, s, tiles, str(q.dtype), mode, group_cap, vb_mode,
            compare_only, policy, query_provider is not None)
    geometries = list(windows.iter_query_windows(
        s, tiles, tile_rows=int(cfg["tile"]), center_group=1
    ))
    call_wall_started = time.perf_counter()
    with _LOCK:
        # Replacing a geometry releases all previous staging and graph pools;
        # never grow a process-global unbounded set of H3 graph instances.
        if _STATE["geometry"] != geom:
            clear_session_cache()
            _STATE["geometry"] = geom
        previous = _STATE.get("last_event")
        if previous is not None:
            # No CPU sync in steady-state: GPU waits for the last writer
            # before a different stream can overwrite graph-owned staging.
            torch.cuda.current_stream(dev).wait_event(previous)
        baseline_reserved = int(torch.cuda.memory_reserved(dev))
        verify = not bool(_STATE["verified"])
        # Only the opt-in research first call takes detailed snapshots.
        # They are absent from warmed whole-attention timings.
        memory_trace = [] if (compare_only and verify) else None
        baseline_memory = (_memory_probe(dev)
                           if memory_trace is not None else None)
        if memory_trace is not None:
            memory_trace.append({"phase": "baseline", **baseline_memory})
        rows = []
        try:
            for begin, end in batch.iter_head_groups(total_heads, group_cap):
                h = end-begin
                bank = (q[:,begin:end,:,:],k[:,begin:end,:,:],v[:,begin:end,:,:])
                key = (h,)
                stage = _SESSIONS.get(key)
                first_group = stage is None
                if first_group:
                    stage = _new_session(
                        bank, geometries, tiles, cfg, tr, warmups,
                        vb_mode=vb_mode, memory_trace=memory_trace,
                        baseline_memory=baseline_memory,
                    )
                    _SESSIONS[key] = stage
                if memory_trace is not None:
                    memory_trace.append({
                        "phase": "after_session", "session_heads": h,
                        "new_session": first_group,
                        **_memory_probe(dev, baseline_memory),
                    })
                # Absolute cache cap includes stage buffers and graph pools
                # for all distinct head-group sizes, not a per-group limit.
                extra_reserved = max(0,int(torch.cuda.memory_reserved(dev))
                                     -baseline_reserved)
                free, _ = torch.cuda.mem_get_info(dev)
                if extra_reserved > max_extra*_MIB or free < min_free*_MIB:
                    raise MemoryError(
                        f"Direct-Flow graph GPU memory gate: extra_reserved="
                        f"{extra_reserved/_MIB:.1f} MiB cap={max_extra} MiB, "
                        f"free={free/_MIB:.1f} MiB minimum={min_free} MiB"
                    )
                group_out = output[:,begin:end,:]
                def replay():
                    _replay(stage, bank, geometries, tr, group_out,
                            query_provider=query_provider)
                # Actual real H3 QKV, after construction warmup; the GPU
                # timing excludes first capture but includes fresh staging,
                # K8, all windows, merge and output writes.
                if verify:
                    graph_t = _measurement(replay, dev)
                    if memory_trace is not None:
                        memory_trace.append({
                            "phase": "after_group_graph_replay",
                            "session_heads": h, "head_range": [begin, end],
                            **_memory_probe(dev, baseline_memory),
                        })
                    if compare_only:
                        # Research already compares against sampled all-key
                        # FP32 dense. An eager Flat call per group has a large
                        # temporary allocator footprint, contaminating the
                        # Graph-memory gate and other arms' measurements.
                        # Never claim bit parity when that call was skipped.
                        rows.append({
                            "head_range": [begin, end],
                            "eager_flat_diagnostic_status": "SKIPPED_COMPARE_ONLY",
                            "exact_flat_parity": None,
                            "relative_l2_to_eager_flat_percent": None,
                            "max_abs_to_eager_flat": None,
                            "graph_gpu_event_ms": graph_t["gpu_event_ms"],
                            "eager_gpu_event_ms": None,
                            "graph_sync_wall_ms": graph_t["sync_wall_ms"],
                            "eager_sync_wall_ms": None,
                            "capture_wall_ms": (
                                stage["capture_wall_ms"] if first_group else 0.0
                            ),
                        })
                    else:
                        # Production remains fail-closed with mandatory
                        # bit-exact native eager Flat first-call parity.
                        eager_out = None
                        def eager():
                            nonlocal eager_out
                            eager_out = live.flat_attention_bthd(
                                *bank, heads=h, mode=mode, tiles=tiles,
                                budget_mib=budget_mib, reserve_mib=reserve_mib,
                                workspace=True, observer=None,
                                query_staging="native_torch_mu",
                                kv_staging="native_direct",
                                dispatch="eager",
                            )
                        eager_t = _measurement(eager, dev)
                        exact = bool(torch.equal(group_out, eager_out[0]))
                        diff = group_out.float()-eager_out[0].float()
                        maximum = float(diff.abs().amax().item())
                        rel_l2_pct = float(
                            diff.norm().item() /
                            max(eager_out[0].float().norm().item(),1.0e-30)*100.0
                        )
                        if not exact and not compare_only:
                            raise AssertionError(
                                f"Direct-Flow real H3 mismatch at heads "
                                f"[{begin},{end}): max_abs={maximum:.8g}"
                            )
                        rows.append({
                            "head_range":[begin,end],
                            "exact_flat_parity":exact,
                            "relative_l2_to_eager_flat_percent":rel_l2_pct,
                            "max_abs_to_eager_flat":maximum,
                            "graph_gpu_event_ms":graph_t["gpu_event_ms"],
                            "eager_gpu_event_ms":eager_t["gpu_event_ms"],
                            "graph_sync_wall_ms":graph_t["sync_wall_ms"],
                            "eager_sync_wall_ms":eager_t["sync_wall_ms"],
                            "capture_wall_ms":stage["capture_wall_ms"] if first_group else 0.0,
                        })
                        del eager_out
                else:
                    replay()
            if verify:
                _STATE["verified"] = True
                result = {
                    "experiment":"h3_directflow_real_model_live_first_call_v1",
                    "status":("RECORDED_FIRST_CALL" if compare_only
                              else "PASS_FIRST_REAL_CALL"),
                    "strict_parity_required":not compare_only,
                    "reference_for_quality":"NOT_EAGER_FLAT__USE_EXACT_SOFTMAX",
                    "mode":mode,"dispatch":"graph","vb_mode":vb_mode,
                    "group_policy":policy,"head_group_cap":group_cap,
                    "group_plan":plan,
                    "capture_and_parity_first_call_wall_ms":(time.perf_counter()-call_wall_started)*1000,
                    "real_model_qkv":not compare_only,
                    "qkv_source":("pre_qkv_window_producer_synthetic"
                                  if query_provider is not None else
                                  "caller_supplied_may_be_synthetic" if compare_only
                                  else "real_model_caller_contract"),
                    "query_provider_research_only":query_provider is not None,
                    "full_model_denoise_or_video_complete":False,
                    "completed":True,
                    "source":("synthetic pre-QKV dense BF16 projection, no RoPE"
                              if query_provider is not None else
                              "actual H3 post-RoPE BHSD QKV before out_proj"),
                    "gpu":torch.cuda.get_device_name(dev),
                    "heads":total_heads, "tokens":s, "head_groups":rows,
                    "window_tiles":tiles,
                    "query_windows":len(geometries),
                    "graph_replays_per_attention":len(geometries)*math.ceil(total_heads/group_cap),
                    "eager_first_call_diagnostic":(
                        "SKIPPED_COMPARE_ONLY" if compare_only else "EXECUTED"
                    ),
                    "all_groups_bit_exact":(
                        None if compare_only else
                        all(x["exact_flat_parity"] for x in rows)
                    ),
                    "speedup_eager_over_graph_gpu_by_group_sums":(
                        None if compare_only else
                        sum(x["eager_gpu_event_ms"] for x in rows)/
                        sum(x["graph_gpu_event_ms"] for x in rows)
                    ),
                    "extra_reserved_mib":max(
                        0, int(torch.cuda.memory_reserved(dev))-baseline_reserved
                    )/_MIB,
                    "model_resident_free_after_mib":torch.cuda.mem_get_info(dev)[0]/_MIB,
                    "graph_cache_group_sizes":sorted(x[0] for x in _SESSIONS),
                    "quality_coefficient_used":False,
                    "fixed_exact_leaves":int(cfg["exact_leaves"]),
                    "center_group":1,
                    "notes":[
                        ("Research compare_only: eager Flat diagnostic SKIPPED "
                         "to prevent its temporary allocator reservations from "
                         "contaminating Graph head-group A/B; sampled all-key "
                         "FP32 dense reference is external."
                         if compare_only else
                         "First real model attention call only, group-by-group exact parity."),
                        "One or two reused graph geometries per size, no stale model QKV pointers.",
                        "Graph capture time excluded from GPU sample, recorded separately.",
                        "Wall times include independent per-group synchronizations; "
                        "do not interpret group sum as whole-denoise gain.",
                        "Model/video completion not established by this one-call gate.",
                    ]
                }
                if memory_trace is not None:
                    result["memory_trace"] = memory_trace
                    result["memory_trace_semantics"] = (
                        "Allocator allocated/reserved/free at first-call stages; "
                        "reserved-minus-allocated is NOT a measured private "
                        "CUDA Graph pool. No profiling on timed replays."
                    )
                _STATE["reports"].append(result)
                _write_report(result)
            if observer is not None:
                observer("htrc_directflow_graph", {
                    "backend":"htrc_flat_gpu_directflow_graph",
                    "vb_mode":vb_mode,"group_policy":policy,
                    "head_group_cap":group_cap,
                    "heads":total_heads,"tokens":s,
                    "graph_cache_group_sizes":sorted(x[0] for x in _SESSIONS),
                    "first_call_parity_verified":bool(
                        _STATE["verified"] and
                        (not _STATE["reports"] or
                         _STATE["reports"][-1].get("all_groups_bit_exact",False))
                    ),
                    "quality_coefficient_used":False,
                    "free_bytes":torch.cuda.mem_get_info(dev)[0],
                    "ts_perf":time.perf_counter(),
                })
            completed = torch.cuda.Event()
            completed.record(torch.cuda.current_stream(dev))
            _STATE["last_event"] = completed
            return output.unsqueeze(0)
        except BaseException as exc:
            # One JSON even on a failed gate; keep the original exception.
            try:
                failure = {
                    "experiment":"h3_directflow_real_model_live_first_call_v1",
                    "completed":False,"mode":mode,"dispatch":"graph",
                    "vb_mode":vb_mode,"group_policy":policy,
                    "head_group_cap":group_cap,
                    "status":"FAIL_CLOSED",
                    "strict_parity_required":not compare_only,
                    "real_model_qkv":not compare_only,
                    "heads":total_heads, "tokens":s,
                    "head_groups_completed":rows,
                    "error":f"{type(exc).__name__}: {exc}"[:1200],
                    "model_video_complete":False,
                    "quality_coefficient_used":False,
                }
                if memory_trace is not None:
                    failure["memory_trace"] = memory_trace
                    failure["memory_at_failure"] = _memory_probe(
                        dev, baseline_memory)
                    failure["memory_trace_semantics"] = (
                        "Allocator snapshots, NOT isolated private Graph pools."
                    )
                _write_report(failure)
            except Exception:
                pass
            # Failed or partly launched graph kernels must finish before
            # freeing persistent staging. Never silently fall back.
            torch.cuda.synchronize(dev)
            clear_session_cache()
            del output
            raise
