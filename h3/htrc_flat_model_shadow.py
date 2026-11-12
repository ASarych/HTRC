"""Read-only MiniMax H3 model-resident Flat shadow, never a production replacement.

Inputs: native post-QKV/RMS/RoPE BHSD tensors (B=1) from
comfy.ldm.minimax.model.optimized_attention, plus original backend output.
The original output is ALWAYS returned by the caller. All Q/K/V views
stay on GPU; no PT capture dump, synthetic repeated heads or weight reload.
"""
from __future__ import annotations

import gc
import math
import time

import torch

from . import htrc_flat, htrc_flat_head_batch as batch
from .htrc_flat_runner import MODES


def capture_from_bhsd(q, k, v, heads):
    """Expose native Q/K/V as [S,H,D] views without a full QKV copy."""
    if not all(isinstance(x, torch.Tensor) for x in (q,k,v)):
        raise TypeError("H3 shadow requires plain GPU tensors (unwrapped from container)")
    if not all(x.device.type == "cuda" for x in (q,k,v)):
        raise ValueError("H3 shadow requires CUDA-resident post-RoPE QKV")
    if q.device != k.device or q.device != v.device:
        raise ValueError("Native Q/K/V must share a CUDA device")
    if not all(x.ndim == 4 and int(x.shape[0]) == 1 for x in (q,k,v)):
        raise ValueError("Only native B=1 BHSD QKV is supported")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("H3 Flat shadow requires same-token self-attention Q/K/V")
    _,h,s,d = map(int,q.shape)
    if h != int(heads) or d != 128 or s < 16:
        raise ValueError("Expected post-RoPE MiniMax H3 [1,heads,S,128]")
    if q.dtype not in (torch.float16,torch.bfloat16,torch.float32):
        raise TypeError("Unsupported native query dtype")
    if k.dtype not in (torch.float16,torch.bfloat16,torch.float32):
        raise TypeError("Unsupported native key dtype")
    if v.dtype not in (torch.float16,torch.bfloat16,torch.float32):
        raise TypeError("Unsupported native value dtype")
    return {
        "q_qhd":q.detach()[0].permute(1,0,2),
        "k_khd":k.detach()[0].permute(1,0,2),
        "v_khd":v.detach()[0].permute(1,0,2),
        "raw_head_index":list(range(h)),
    }


def reference_bthd(output, seq, heads):
    """Make a [S,H,128] view of the original optimized_attention return."""
    if not isinstance(output,torch.Tensor):
        raise TypeError("Original attention must return a Tensor")
    if output.device.type!="cuda":
        raise ValueError("Original attention output must remain CUDA-resident")
    if output.ndim == 3 and tuple(output.shape)==(1,seq,heads*128):
        return output.detach()[0].view(seq,heads,128)
    if output.ndim == 4 and tuple(output.shape)==(1,seq,heads,128):
        return output.detach()[0]
    raise ValueError(f"Unexpected original attention output shape {tuple(output.shape)}")


def flat_config(mode, tiles):
    if mode not in ("fast","balanced") or tiles < 1:
        raise ValueError("Only 64-row seq fast/balanced Flat is supported")
    config=dict(htrc_flat.PROD)
    config.update(MODES[mode])
    config["tile_chunk"]=int(tiles)
    config["sink_leaves"]=0
    config["kern"]=dict(config["kern"])
    if (config["tile"]!=64 or config["order"]!="seq"
        or config["dot"]!="bf16" or config["store"]!="int8"):
        raise ValueError("Unvalidated Flat configuration")
    return config


def alloc_workspace(capture, head_count, tiles, cfg):
    """Preallocate query workspace, but NEVER share results between windows."""
    h,n=int(head_count),int(capture["k_khd"].shape[0])//16
    kx=min(int(cfg["exact_leaves"]),n)
    centers=tiles//int(cfg.get("kern",{}).get("group",1))
    ns=int(cfg["splits"])
    dev=capture["q_qhd"].device
    def make(shape,dtype):
        return torch.empty(shape,dtype=dtype,device=dev)
    ps=(h,tiles,ns,64)
    return {
        "G":make((h,centers,n,128),torch.int8),
        "VB":make((h,centers,n,128),torch.bfloat16),
        "LZ":make((h,centers,n),torch.float32),
        "SG":make((h,centers,n),torch.float32),
        "sel":make((h*centers,n),torch.float32),
        "IDX":make((h,centers,kx),torch.int32),
        "EXCL":make((h*centers,n),torch.int8),
        "_partials":{"pm":make(ps,torch.float32),
                     "pl":make(ps,torch.float32),
                     "acc":make(ps+(128,),torch.float32)},
        "_out":make((h,tiles,64,128),torch.float32),
    }


def _head_plan(capture, cases, tiles, cfg, *, budget_mib, reserve_mib, workspace):
    """Conservative per-group planning against current *model-resident* free VRAM."""
    device=capture["q_qhd"].device
    free,_=torch.cuda.mem_get_info(device)
    mib=1024*1024
    budget=min(int(free),int(budget_mib*mib))
    nkeys=int(capture["k_khd"].shape[0])//16*16
    plan=batch.plan_tile_head_batches(
        cases,free_bytes=budget,keys_per_head=nkeys,
        tile_count=tiles,exact_leaves=int(cfg["exact_leaves"]),
        reserve_bytes=int(reserve_mib*mib),
        fixed_temp_bytes=128*mib,safety_factor=1.5,
    )
    group=int(plan.heads_per_group)
    # Resident KBF/VBF/K8/KS and query buffers are additional to old
    # temporary-state plan; clamp group count rather than evict model weights.
    nb=nkeys//16
    ns=int(cfg["splits"])
    kx=min(int(cfg["exact_leaves"]),nb)
    per_cache=nkeys*(128*2+128*2+128+4)
    centers=tiles//int(cfg.get("kern",{}).get("group",1))
    per_workspace=(centers*nb*(128+256+4+4+4+1)
                  +centers*kx*4+tiles*ns*64*(4+4+128*4)
                  +tiles*64*128*4) if workspace else 0
    usable=budget-int(reserve_mib*mib)-128*mib
    per_head=(plan.estimated_bytes_per_head
              +math.ceil(per_cache*1.3)
              +math.ceil(per_workspace*1.3))
    if usable < per_head:
        raise MemoryError(
            f"Model-resident HTRC: {usable/mib:.1f} MiB usable, "
            f"{per_head/mib:.1f} MiB required for one head")
    group=min(group,int(usable//per_head),cases)
    return group,{
        "free_at_plan_mib":free/mib, "budget_mib":budget/mib,
        "reserve_mib":reserve_mib, "head_group_cap":group,
        "estimated_mib_per_head":per_head/mib,
    }


def _tail(capture, positions):
    keys=capture["k_khd"]
    values=capture["v_khd"]
    tail=(int(keys.shape[0])%16)
    if not tail:
        return None
    pos=torch.tensor(positions,dtype=torch.long,device=keys.device)
    def pick(src):
        return src[-tail:].index_select(1,pos).permute(1,0,2).contiguous()
    return pick(keys),pick(values)


@torch.inference_mode()
def run_model_shadow(
    q, k, v, original_output, *, heads, head_cases=(4,56),
    mode="fast", tiles=16, budget_mib=1024, reserve_mib=256,
    use_workspace=True, max_rel_l2=None,
):
    """Run full-Q Flat on TRUE native model heads, keeping original output.

    Returns compact metrics only. Never reorders/modifies q/k/v/output
    and never touches Comfy's global attention dispatch.
    """
    capture=capture_from_bhsd(q,k,v,heads)
    seq=int(capture["q_qhd"].shape[0])
    native=reference_bthd(original_output,seq,int(heads))
    cfg=flat_config(mode,tiles)
    tri=htrc_flat._load_triton()
    tri._dll()
    out={
        "kind":"model_resident_post_rope_self_attention_shadow",
        "native_head_count":int(heads),
        "native_head_ids":list(range(int(heads))),
        "true_distinct_heads":True,
        "query_rows_per_head":seq,
        "aligned_key_rows":seq//16*16,
        "exact_tail_key_rows":seq%16,
        "query_window_rows":tiles*64,
        "query_window_count":math.ceil(seq/(tiles*64)),
        "native_input_dtype":str(q.dtype),
        "native_input_stride":list(q.stride()),
        "mode":mode,"tiles":tiles,"workspace":bool(use_workspace),
        "cases":[],"completed":False,
    }
    for requested in head_cases:
        case={"head_count":int(requested),"groups":[],
              "status":"STARTED","error":None,
              "quality":None}
        out["cases"].append(case)
        if requested<1 or requested>heads:
            case.update(status="UNSUPPORTED_HEAD_COUNT",
                        error=f"Requested {requested} distinct heads, native has {heads}")
            break
        try:
            group_size,plan=_head_plan(
                capture,int(requested),tiles,cfg,
                budget_mib=budget_mib,reserve_mib=reserve_mib,
                workspace=use_workspace
            )
            case["planner"]=plan
            case["groups_initial"]=math.ceil(requested/group_size)
            total_err,total_ref,total_abs,largest_abs=0.,0.,0.,0.
            n_total=0
            pos=0
            while pos<requested:
                end=min(requested,pos+group_size)
                ids=list(range(pos,end))   # REAL distinct native heads
                cached=None
                ws=None
                try:
                    torch.cuda.synchronize()
                    group_before=torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    start=time.perf_counter()
                    cached=batch.prepare_gpu_head_kv_cache(capture,ids,tri)
                    torch.cuda.synchronize()
                    kv_ms=(time.perf_counter()-start)*1000
                    if use_workspace:
                        ws=alloc_workspace(capture,len(ids),tiles,cfg)
                    setup_peak=torch.cuda.max_memory_allocated()
                    setup_live=torch.cuda.memory_allocated()
                    tails=_tail(capture,ids)
                    group_row={
                        "head_range":[pos,end], "unique_native_head_ids":ids,
                        "kv_build_wall_ms":kv_ms,
                        "kv_cache_mib":batch.gpu_head_kv_cache_bytes(cached)/batch.MIB,
                        "workspace_mib":batch.gpu_query_workspace_bytes(ws)/batch.MIB if ws else 0.,
                        "build_peak_allocated_mib":setup_peak/batch.MIB,
                        "resident_after_setup_mib":setup_live/batch.MIB,
                        "query_window_count":0,
                        "operator_wall_ms":0.,
                        "operator_prep_wall_ms":0.,
                        "operator_tile_merge_wall_ms":0.,
                        "max_peak_allocated_mib":setup_peak/batch.MIB,
                    }
                    windows=math.ceil(seq/(tiles*64))
                    for wi in range(windows):
                        beg=wi*tiles*64
                        stop=min(seq,(wi+1)*tiles*64)
                        valid=stop-beg
                        src=capture["q_qhd"][beg:stop]
                        if valid<tiles*64:
                            # Repeat last REAL query instead of zero-padding.
                            # valid_query_rows adjusts final partial-tile MU.
                            src=torch.cat((
                                src,src[-1:].expand(tiles*64-valid,-1,-1)
                            ),dim=0)
                        c=dict(capture)
                        c["q_qhd"]=src
                        torch.cuda.synchronize()
                        start=time.perf_counter()
                        states,_,nkeys,nb,kx=batch.prepare_batched_head_states(
                            c,ids,cfg,tri,tile_count=tiles,
                            quant_backend="triton",input_cast="gpu",
                            kv_cache=cached,query_workspace=ws,
                            valid_query_rows=valid,
                        )
                        torch.cuda.synchronize()
                        prep_ms=(time.perf_counter()-start)*1000
                        parts,_=batch.launch_prepared_tile_batch(
                            states,cfg,tri,
                            buffers=(ws.get("_partials") if ws else None)
                        )
                        candidate,_=batch.launch_prepared_merge_batch(
                            states,cfg,tri,parts,tails=tails,
                            output=(ws["_out"] if ws else None),
                            merge_rows=32
                        )
                        torch.cuda.synchronize()
                        wall_ms=(time.perf_counter()-start)*1000
                        # Compare ONLY valid Q rows with the original
                        # model backend. No full output tensor is retained.
                        pred=candidate.reshape(len(ids),tiles*64,128)[:,:valid]
                        baseline=native[beg:stop,ids,:].permute(1,0,2)
                        delta=pred.float()-baseline.float()
                        ref=baseline.float()
                        total_err+=float(delta.square().sum(dtype=torch.float64).item())
                        total_ref+=float(ref.square().sum(dtype=torch.float64).item())
                        total_abs+=float(delta.abs().sum(dtype=torch.float64).item())
                        largest_abs=max(largest_abs,float(delta.abs().amax().item()))
                        n_total+=int(delta.numel())
                        if not bool(torch.isfinite(pred).all().item()):
                            raise ArithmeticError("Non-finite grouped Flat output")
                        group_row["query_window_count"]+=1
                        group_row["operator_wall_ms"]+=wall_ms
                        group_row["operator_prep_wall_ms"]+=prep_ms
                        group_row["operator_tile_merge_wall_ms"]+=wall_ms-prep_ms
                        group_row["max_peak_allocated_mib"]=max(
                            group_row["max_peak_allocated_mib"],
                            torch.cuda.max_memory_allocated()/batch.MIB
                        )
                        del states,parts,candidate,pred,baseline,delta,ref,c
                    case["groups"].append(group_row)
                    pos=end
                    print(
                        "MODEL_FLAT_GROUP",pos,"/",requested,
                        "windows",windows,
                        "wall_ms",round(group_row["operator_wall_ms"],2),
                        "kv_ms",round(kv_ms,2),flush=True
                    )
                    del cached,ws,tails
                    gc.collect()
                except torch.cuda.OutOfMemoryError as exc:
                    # A CUDA OOM during a live model block can leave partially
                    # initialized graph/kernel state. DO NOT retry same model
                    # invocation or accidentally count a partial group.
                    gc.collect()
                    case["oom_at_head"]=pos
                    case["head_group_size_at_oom"]=group_size
                    raise MemoryError(
                        "Model-resident shadow OOM; original model attention "
                        "output remains authoritative; lower --budget-mib "
                        "or reduce --head-cases for a separate test"
                    ) from exc
            case["quality"]={
                "rel_l2_vs_original_attention":math.sqrt(total_err/max(total_ref,1e-30)),
                "max_abs_vs_original_attention":largest_abs,
                "mean_abs_vs_original_attention":total_abs/max(n_total,1),
                "checked_output_values":n_total,
                "finite":True,
                "target_reference":"native_comfy_optimized_attention",
                "pass_threshold":max_rel_l2,
            }
            case["timing"]={
                "sum_group_kv_build_wall_ms":sum(x["kv_build_wall_ms"] for x in case["groups"]),
                "sum_group_full_Q_operator_wall_ms":sum(x["operator_wall_ms"] for x in case["groups"]),
                "sum_group_build_plus_operator_wall_ms":sum(
                    x["kv_build_wall_ms"]+x["operator_wall_ms"] for x in case["groups"]
                ),
            }
            quality=case["quality"]["rel_l2_vs_original_attention"]
            case["status"]=(
                "PASS_SHADOW_MEASURED" if max_rel_l2 is None or quality<=max_rel_l2
                else "QUALITY_THRESHOLD_EXCEEDED"
            )
        except BaseException as exc:
            case["status"]="FAILED_NO_REPLACEMENT"
            case["error"]=f"{type(exc).__name__}: {exc}"[:1000]
            break
    out["completed"]=all(x["status"]=="PASS_SHADOW_MEASURED" for x in out["cases"])
    return out
