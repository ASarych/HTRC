"""Experimental universal H3 Flat Tile launcher plus VRAM head-group planner.

The head COUNT affects the launch grid only, not Triton constexpr.
Prepared INT8/BF16 state inputs are required. This helper does not
include leaf stats, selection, topk, merge or full H3 execution.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Iterator

def _source_tensor_version(tensor):
    # ComfyUI creates inference tensors without version counters. Cache users
    # still require the identical source objects and a single-call lifetime;
    # no cache-owning code mutates native K/V during that lifetime.
    return None if tensor.is_inference() else tensor._version


MIB = 1 << 20
DEFAULT_FIXED_TEMP_BYTES = 128 * MIB


@dataclass(frozen=True)
class TileHeadBatchPlan:
    total_heads: int
    heads_per_group: int
    groups: int
    free_bytes_at_plan: int
    reserve_bytes: int
    fixed_temp_bytes: int
    estimated_bytes_per_head: int
    estimated_max_additional_bytes: int
    tile_count_per_head: int
    keys_per_head: int
    safety_factor: float


def estimate_tile_head_bytes(
    *, keys_per_head: int, tile_count: int, tile_rows: int = 64,
    dim: int = 128, splits: int = 4, exact_leaves: int = 256,
    partial_copies: int = 2
) -> int:
    """Estimate prepared inputs and split partials for one head, in bytes."""
    fields = (keys_per_head, tile_count, tile_rows, dim, splits)
    if any(not isinstance(x, int) or x <= 0 for x in fields):
        raise ValueError("keys, tiles, rows, dim, splits must be positive ints")
    if keys_per_head % 16:
        raise ValueError("keys must be 16-token-leaf aligned")
    if exact_leaves < 0 or partial_copies < 1:
        raise ValueError("invalid near leaves or partial copies")
    leaves = keys_per_head // 16
    near_leaves = min(int(exact_leaves), leaves)
    qr = tile_count * tile_rows
    nl = tile_count * leaves
    # Q float32 and MU float32.
    q_mu = qr * dim * 4 + tile_count * dim * 4
    # K int8, V bf16, KS float32.
    near = keys_per_head * (dim + dim * 2 + 4)
    # G int8, VB bf16, LZ float32, SG float32, EXCL int8.
    far = nl * (dim + dim * 2 + 4 + 4 + 1)
    # Selection idx int32.
    index = tile_count * near_leaves * 4
    # PM, PL, PACC in float32, usually with both A/B copies.
    partial = partial_copies * tile_count * splits * tile_rows * (dim + 2) * 4
    return q_mu + near + far + index + partial


def plan_tile_head_batches(
    total_heads: int, *, free_bytes: int, keys_per_head: int,
    tile_count: int, tile_rows: int = 64, dim: int = 128,
    splits: int = 4, exact_leaves: int = 256, partial_copies: int = 2,
    reserve_bytes: int = 256 * MIB,
    fixed_temp_bytes: int = DEFAULT_FIXED_TEMP_BYTES,
    safety_factor: float = 1.5,
    max_heads_per_group: int | None = None
) -> TileHeadBatchPlan:
    """Choose maximum head group from CURRENT free bytes, with safety reserve.

    An estimate is not proof of allocator/graph-pool safety; test the
    actual peak GPU allocation and leave enough reserve for the model.
    """
    if not isinstance(total_heads, int) or total_heads < 1:
        raise ValueError("total_heads must be a positive int")
    if not isinstance(free_bytes, int) or free_bytes < 0:
        raise ValueError("free_bytes must be a nonnegative int")
    if any(not isinstance(v, int) or v < 0 for v in
           (reserve_bytes, fixed_temp_bytes)):
        raise ValueError("reserve and fixed temporary memory must be nonnegative")
    if not math.isfinite(safety_factor) or safety_factor < 1:
        raise ValueError("safety_factor must be finite and >= 1")
    if max_heads_per_group is not None and (
        not isinstance(max_heads_per_group, int) or max_heads_per_group < 1
    ):
        raise ValueError("max_heads_per_group must be a positive int or None")
    raw = estimate_tile_head_bytes(
        keys_per_head=keys_per_head, tile_count=tile_count,
        tile_rows=tile_rows, dim=dim, splits=splits,
        exact_leaves=exact_leaves, partial_copies=partial_copies
    )
    per_head = math.ceil(raw * safety_factor)
    allowance = free_bytes - reserve_bytes - fixed_temp_bytes
    group = min(total_heads, max_heads_per_group or total_heads,
                max(0, allowance // per_head))
    if group < 1:
        raise MemoryError(
            "Even one head exceeds the estimated free GPU budget: "
            f"free={free_bytes/MIB:.1f} MiB, reserve={reserve_bytes/MIB:.1f} MiB, "
            f"fixed={fixed_temp_bytes/MIB:.1f} MiB, per_head={per_head/MIB:.1f} MiB"
        )
    return TileHeadBatchPlan(
        total_heads=total_heads, heads_per_group=int(group),
        groups=math.ceil(total_heads/group),
        free_bytes_at_plan=free_bytes, reserve_bytes=reserve_bytes,
        fixed_temp_bytes=fixed_temp_bytes,
        estimated_bytes_per_head=per_head,
        estimated_max_additional_bytes=fixed_temp_bytes+group*per_head,
        tile_count_per_head=tile_count, keys_per_head=keys_per_head,
        safety_factor=float(safety_factor)
    )


def iter_head_groups(total_heads: int, heads_per_group: int) -> Iterator[tuple[int,int]]:
    """Head-major contiguous ranges, including any short final group."""
    if total_heads < 1 or heads_per_group < 1:
        raise ValueError("head count and group size must be >=1")
    for start in range(0,total_heads,heads_per_group):
        yield start, min(total_heads,start+heads_per_group)


def launch_prepared_tile_batch(states, config, triton_module, *, buffers=None,
                               fused_vb=False):
    """One _tile_kernel call for arbitrary H prepared, same-shape native heads.

    The call uses existing BEST math with TAIL=0. Tail-in-Merge must be
    executed separately; PM/PL/PACC are not normalized final attention.
    """
    import torch
    req = ("Q","MU","K","V","KS","G","VB","LZ","SG","EXCL","IDX")
    missing=[name for name in req if name not in states]
    if missing:
        raise ValueError(f"Missing prepared state: {missing}")
    direct=bool(states.get("_direct_q",False))
    direct_v=bool(states.get("_direct_v",False))
    if fused_vb:
        if (states.get("_vb_mode")!="fused_vb" or not direct or not direct_v):
            raise ValueError("Fused VB requires explicit opt-in and direct BF16 Q/K/V")
        kbf=states.get("_stream_kbf")
        if kbf is None or kbf.dtype!=torch.bfloat16 or not kbf.is_cuda:
            raise ValueError("Fused VB requires native BF16 K")
    if any(not states[name].is_contiguous()
           for name in req if name not in (("Q","V") if direct_v else ("Q",))):
        raise ValueError("Prepared packed state tensors must be contiguous")
    if not direct_v and not states["V"].is_contiguous():
        raise ValueError("Packed V must be contiguous")
    if not direct and not states["Q"].is_contiguous():
        raise ValueError("Staged Q must be contiguous")
    q=states["Q"]
    if q.ndim != 4:
        raise ValueError("Q must be [heads,tiles,rows,dim]")
    heads,tiles,tile,dim=map(int,q.shape)
    if min(heads,tiles,tile,dim)<1:
        raise ValueError("Empty prepared Q dimensions")
    keys=int(states["K"].shape[1])
    leaves=keys//16
    if keys%16:
        raise ValueError("K rows must be 16-token leaf aligned")
    if (config.get("order")!="seq" or config.get("store")!="int8"
        or config.get("dot")!="bf16" or config.get("sink_leaves",0)!=0
        or int(config["tile"])!=tile or dim!=128):
        raise ValueError("Only seq/int8/bf16/sink0/dim128 Tile is supported")
    group=int(config.get("kern",{}).get("group",1))
    if group not in (1,2,4) or tiles % group:
        raise ValueError("Query tile count must be divisible by centre group 1/2/4")
    centers=tiles//group
    if states["MU"].shape!=(heads,centers,dim):
        raise ValueError("MU dimensions do not match Q")
    if states["K"].shape!=(heads,keys,dim) or states["V"].shape!=states["K"].shape:
        raise ValueError("K/V must be [heads,keys,dim]")
    if states["KS"].shape!=(heads,keys):
        raise ValueError("KS scales mismatch")
    if states["G"].shape!=(heads,centers,leaves,dim):
        raise ValueError("G dims mismatch")
    if (not fused_vb and states["VB"].shape!=states["G"].shape
        or fused_vb and states["VB"].numel()!=1):
        raise ValueError("VB state layout mismatch for selected mode")
    if fused_vb and (kbf.shape!=(heads,keys,dim)
        or states["V"].dtype!=torch.bfloat16
        or states["K"].dtype!=torch.int8):
        raise ValueError("Fused VB source layout/dtype mismatch")
    if any(states[name].shape!=(heads,centers,leaves) for name in ("LZ","SG","EXCL")):
        raise ValueError("LZ/SG/EXCL dims mismatch")
    idx=states["IDX"]
    if idx.ndim!=3 or idx.shape[:2]!=(heads,centers) or idx.shape[2]<1:
        raise ValueError("IDX must be [heads,tiles,exact_leaves]")
    kx=int(idx.shape[2])
    splits=int(config["splits"])
    kern=config.get("kern",{})
    bn=int(kern.get("tile_bn",32))
    near_bn=int(kern.get("tile_near_bn") or bn)
    warps=int(kern.get("tile_warps",4))
    stages=int(kern.get("tile_stages",1))
    if splits<1:
        raise ValueError("Split count must be positive")
    partial_shape=(heads,tiles,splits,tile)
    acc_shape=partial_shape+(dim,)
    if buffers is None:
        pm=torch.empty(partial_shape,device=q.device,dtype=torch.float32)
        pl=torch.empty_like(pm)
        acc=torch.empty(acc_shape,device=q.device,dtype=torch.float32)
        buffers={"pm":pm,"pl":pl,"acc":acc}
    else:
        for name,shape in (("pm",partial_shape),("pl",partial_shape),("acc",acc_shape)):
            if name not in buffers or tuple(buffers[name].shape)!=shape:
                raise ValueError(f"Missing/bad output partial: {name}")
            if (buffers[name].dtype!=torch.float32
                or buffers[name].device!=q.device
                or not buffers[name].is_contiguous()):
                raise ValueError("Partial buffers must be contiguous CUDA float32 on Q device")
        pm,pl,acc=buffers["pm"],buffers["pl"],buffers["acc"]
    flat=lambda name:states[name].view((-1,)+states[name].shape[2:])
    qptr=q if direct else flat("Q")
    qstrides=tuple(map(int,q.stride())) if direct else (0,0,0,0)
    vptr=states["V"] if direct_v else states["V"].view(heads*keys,dim)
    vstrides=tuple(map(int,states["V"].stride())) if direct_v else (0,0,0)
    lz=flat("LZ")
    vb_kernel_kwargs={}
    if fused_vb:
        vb_kernel_kwargs={
            "FUSED_VB":1,"KBF":kbf,
            "KB_SH":int(kbf.stride(0)),
            "KB_SS":int(kbf.stride(1)),
            "KB_SD":int(kbf.stride(2)),
        }
    compiled=triton_module._tile_kernel[(heads*tiles,splits,1)](
        qptr,flat("MU"),flat("G"),lz,flat("VB"),
        flat("EXCL"),flat("IDX"),
        states["K"].view(heads*keys,dim),
        vptr,
        pm.view(heads*tiles,splits,tile),
        pl.view(heads*tiles,splits,tile),
        acc.view(heads*tiles,splits,tile,dim),
        flat("SG"),lz,lz,lz,states["KS"].view(heads*keys),
        leaves,kx,
        triton_module._split(leaves,splits,bn),
        triton_module._split(kx*16,splits,bn),
        0,1.0/math.sqrt(dim),
        T=tile,D=dim,BN=bn,BN_NEAR=near_bn,LEAF=16,
        PREC="tf32",DOT=1,GR=group,FP8=0,TAIL=0,RB=1,I8=1,
        PASS=0,LATE_NEAR_Q=0,FAR_PREFETCH_LZ=0,
        FAR_PV_ACC=0,NEAR_PV_ACC=0,
        BATCH_TILES_PER_HEAD=tiles,BATCH_KEY_ROWS=keys,
        Q_DIRECT=int(direct),Q_SH=qstrides[0],Q_ST=qstrides[1],
        Q_SR=qstrides[2],Q_SD=qstrides[3],
        V_DIRECT=int(direct_v),V_SH=vstrides[0],
        V_SS=vstrides[1],V_SD=vstrides[2],
        num_warps=warps,num_stages=stages,**vb_kernel_kwargs
    )
    return buffers,compiled

def launch_prepared_tile_batch_fused_vb(states, config, triton_module, *,
                                         buffers=None):
    """Opt-in, no-global-VB Tile specialization; old Tile remains default."""
    if states.get("_vb_mode")!="fused_vb":
        raise ValueError("Fused Tile invoked on non-fused prepared state")
    return launch_prepared_tile_batch(states,config,triton_module,
                                      buffers=buffers,fused_vb=True)




def launch_prepared_tile_batch_split_stream(states, config, triton_module, *,
                                            buffers, vb_split):
    """Experimental 4-split BF16 VB producer/Tile consumer with bounded staging.

    Produce ONLY one split's VB and consume it immediately in the existing
    far/near Tile math; then overwrite the same VB scratch for the next
    split. G/LZ/SG/IDX/EXCL are unchanged and still full-size. Requires
    SCORE_ONLY stats preparation. Never select from the default Tile API.
    GPU parity is mandatory before any deployment.
    """
    import math
    import torch
    if states.get("_vb_mode")!="split_stream":
        raise ValueError("split-stream Tile requires SCORE_ONLY preparation")
    q=states["Q"]
    if q.ndim!=4:
        raise ValueError("Expected [H,tiles,64,128] native Q")
    heads,tiles,tile,dim=map(int,q.shape)
    group=int(config.get("kern",{}).get("group",1))
    if group!=1 or tile!=64 or dim!=128 or not bool(states.get("_direct_q")):
        raise ValueError("split-stream Tile supports native Q/group1/64rows/dim128")
    if config.get("dot")!="bf16" or config.get("store")!="int8":
        raise ValueError("split-stream Tile requires pinned INT8/BF16 math")
    kbf=states.get("_stream_kbf")
    if kbf is None or kbf.dtype!=torch.bfloat16:
        raise ValueError("Missing original BF16 K for VB recomputation")
    k8,v,ks=states["K"],states["V"],states["KS"]
    keys=int(k8.shape[1])
    leaves=keys//16
    if (keys%16 or k8.shape!=(heads,keys,dim)
        or v.shape!=(heads,keys,dim)
        or kbf.shape!=(heads,keys,dim)
        or ks.shape!=(heads,keys)):
        raise ValueError("Incompatible K/K8/V/KS split shapes")
    if v.dtype!=torch.bfloat16 or k8.dtype!=torch.int8:
        raise ValueError("Streaming far requires BF16 original V and INT8 K")
    if states["G"].shape!=(heads,tiles,leaves,dim):
        raise ValueError("Invalid full G state")
    if states["LZ"].shape!=(heads,tiles,leaves):
        raise ValueError("Invalid full LZ state")
    if states["MU"].shape!=(heads,tiles,dim):
        raise ValueError("Invalid MU state")
    if states["VB"].numel()!=1:
        raise ValueError("Full VB was unexpectedly allocated in split-stream mode")
    splits=int(config["splits"])
    kern=config.get("kern",{})
    bn=int(kern.get("tile_bn",32))
    near_bn=int(kern.get("tile_near_bn") or bn)
    warps=int(kern.get("tile_warps",4))
    stages=int(kern.get("tile_stages",1))
    if splits!=4 or bn!=64 or near_bn!=64 or warps!=4:
        raise ValueError("Unvalidated split-stream kernel geometry")
    span=triton_module._split(leaves,splits,bn)
    if (tuple(vb_split.shape)!=(heads,tiles,span,dim)
        or vb_split.dtype!=torch.bfloat16
        or not vb_split.is_contiguous()):
        raise ValueError("Invalid split VB scratch dimensions or layout")
    partial_shape=(heads,tiles,splits,tile)
    for name,shape in (("pm",partial_shape),
                       ("pl",partial_shape),
                       ("acc",partial_shape+(dim,))):
        if (name not in buffers or tuple(buffers[name].shape)!=shape
            or buffers[name].dtype!=torch.float32
            or not buffers[name].is_contiguous()):
            raise ValueError("Incompatible partial buffer "+name)
    if any(t.device!=q.device for t in
           (vb_split,kbf,v,k8,ks,buffers["pm"],buffers["pl"],buffers["acc"])):
        raise ValueError("All stream buffers must be on the same CUDA device")
    htc=heads*tiles
    qstride=tuple(map(int,q.stride()))
    direct_v=bool(states.get("_direct_v",False))
    kstride=tuple(map(int,kbf.stride()))
    vstride=tuple(map(int,v.stride()))
    vptr=v if direct_v else v.view(heads*keys,dim)
    triton_module._dll()
    last=None
    # Split loop order exactly matches the original sp=0..3 Tile partials.
    # Same-stream sequential launches prevent VB scratch overwrite races.
    for split in range(splits):
        first=split*span
        if first<leaves:
            triton_module._vbar_split_kernel[
                (heads*math.ceil(tiles/16),triton_module.triton.cdiv(span,4))
            ](
                states["MU"].view(htc,dim),
                kbf,v,vb_split,
                htc,leaves,1.0/math.sqrt(dim),
                D=dim,BT=16,LEAF=16,BL=4,PREC="tf32",
                BATCH_TILES_PER_HEAD=tiles,
                BATCH_TILE_BLOCKS_PER_HEAD=math.ceil(tiles/16),
                LEAF_START=first,LEAF_SPAN=span,
                DIRECT_KV=int(direct_v),
                K_SH=(kstride[0] if direct_v else 0),
                K_SS=(kstride[1] if direct_v else 0),
                K_SD=(kstride[2] if direct_v else 0),
                V_SH=(vstride[0] if direct_v else 0),
                V_SS=(vstride[1] if direct_v else 0),
                V_SD=(vstride[2] if direct_v else 0),
                num_warps=4,
            )
        last=triton_module._tile_kernel[(htc,1,1)](
            q,states["MU"].view(htc,dim),
            states["G"].view(htc,leaves,dim),
            states["LZ"].view(htc,leaves),
            vb_split,
            states["EXCL"].view(htc,leaves),
            states["IDX"].view(htc,-1),
            k8.view(heads*keys,dim),vptr,
            buffers["pm"].view(htc,splits,tile),
            buffers["pl"].view(htc,splits,tile),
            buffers["acc"].view(htc,splits,tile,dim),
            states["SG"].view(htc,leaves),
            states["LZ"].view(htc,leaves),
            states["LZ"].view(htc,leaves),
            states["LZ"].view(htc,leaves),
            ks.view(heads*keys),
            leaves,int(states["IDX"].shape[-1]),
            span,triton_module._split(int(states["IDX"].shape[-1])*16,splits,bn),
            0,1.0/math.sqrt(dim),
            T=tile,D=dim,BN=bn,BN_NEAR=near_bn,LEAF=16,
            PREC="tf32",DOT=1,GR=1,FP8=0,TAIL=0,RB=1,I8=1,
            PASS=0,LATE_NEAR_Q=0,FAR_PREFETCH_LZ=0,
            FAR_PV_ACC=0,NEAR_PV_ACC=0,
            BATCH_TILES_PER_HEAD=tiles,BATCH_KEY_ROWS=keys,
            Q_DIRECT=1,Q_SH=qstride[0],Q_ST=qstride[1],
            Q_SR=qstride[2],Q_SD=qstride[3],
            V_DIRECT=int(direct_v),V_SH=vstride[0] if direct_v else 0,
            V_SS=vstride[1] if direct_v else 0,
            V_SD=vstride[2] if direct_v else 0,
            VB_SPLIT_STREAM=1,VB_START=first,VB_SPAN=span,
            SPLIT_ID=split,SPLIT_COUNT=splits,
            num_warps=warps,num_stages=stages,
        )
    return buffers,last


def launch_prepared_merge_batch(
    states, config, triton_module, partials, *, tails=None,
    output=None, merge_rows: int = 32
):
    """Normalize grouped Tile partials, with each head's OWN exact tail.

    tails=(K_tail, V_tail), each [H,ntail,D], ntail in 1..16.
    None means no exact tail (ordinary grouped Merge).
    Input Tile must have been launched with TAIL=0. For the
    currently validated path, use merge_rows=32.
    """
    import torch
    q=states["Q"]
    direct=bool(states.get("_direct_q",False))
    if q.ndim!=4 or (not direct and not q.is_contiguous()):
        raise ValueError("Q must be [heads,tiles,rows,dim], staged or native strided")
    heads,tiles,tile,dim=map(int,q.shape)
    if min(heads,tiles,tile,dim)<1 or dim!=128:
        raise ValueError("Nonempty dim128 head-major Q required")
    if config.get("dot")!="bf16" or config.get("sink_leaves",0)!=0:
        raise ValueError("Only BF16 dot, sink0 experimental merge supported")
    splits=int(config["splits"])
    pshape=(heads,tiles,splits,tile)
    ashape=pshape+(dim,)
    for name,shape in (("pm",pshape),("pl",pshape),("acc",ashape)):
        if name not in partials or tuple(partials[name].shape)!=shape:
            raise ValueError(f"Incompatible partial {name}")
        if not partials[name].is_contiguous():
            raise ValueError(f"Partial {name} must be contiguous")
    if output is None:
        output=torch.empty((heads,tiles,tile,dim),
                           device=q.device,dtype=torch.float32)
    if (tuple(output.shape)!=(heads,tiles,tile,dim)
            or output.dtype!=torch.float32 or output.device!=q.device
            or not output.is_contiguous()):
        raise ValueError("Output must be contiguous float32 [H,tiles,T,D]")
    pm=partials["pm"].view(heads*tiles,splits,tile)
    pl=partials["pl"].view(heads*tiles,splits,tile)
    acc=partials["acc"].view(heads*tiles,splits,tile,dim)
    out=output.view(heads*tiles,tile,dim)
    if tails is None:
        compiled=triton_module._merge_kernel[(heads*tiles,)](
            pm,pl,acc,out,
            NS=splits,T=tile,D=dim,num_warps=4
        )
        return output,compiled
    if not isinstance(tails,(tuple,list)) or len(tails)!=2:
        raise ValueError("tails must be a K/V pair or None")
    tk,tv=tails
    if tk.ndim!=3 or not tk.is_contiguous() or not tv.is_contiguous():
        raise ValueError("Tail K/V must be contiguous [H,ntail,D]")
    ntail=int(tk.shape[1])
    if (ntail<1 or ntail>16 or tuple(tk.shape)!=(heads,ntail,dim)
            or tuple(tv.shape)!=tuple(tk.shape)):
        raise ValueError("Exactly 1..16 tail tokens per head required")
    if tk.device!=q.device or tv.device!=q.device:
        raise ValueError("Tail K/V must share Q device")
    if merge_rows<16 or merge_rows>tile or tile%merge_rows or merge_rows&(merge_rows-1):
        raise ValueError("merge_rows must be a power-of-two divisor of T, >=16")
    # Same round-to-BF16 semantics as validated pinned tail-in-merge.
    tkb=tk.to(torch.bfloat16).contiguous()
    tvb=tv.to(torch.bfloat16).contiguous()
    compiled=triton_module._merge_tail_kernel[
        (heads*tiles,tile//merge_rows)
    ](
        pm,pl,acc,(q if direct else q.view(heads*tiles,tile,dim)),
        tkb.view(heads*ntail,dim),
        tvb.view(heads*ntail,dim),
        out,ntail,1.0/math.sqrt(dim),
        NS=splits,T=tile,R=merge_rows,D=dim,
        DOT=1,PREC="tf32",
        BATCH_TILES_PER_HEAD=tiles,BATCH_TAIL_ROWS=ntail,
        Q_DIRECT=int(direct),
        Q_SH=(int(q.stride(0)) if direct else 0),
        Q_ST=(int(q.stride(1)) if direct else 0),
        Q_SR=(int(q.stride(2)) if direct else 0),
        Q_SD=(int(q.stride(3)) if direct else 0),
        num_warps=4
    )
    return output,compiled


def prepare_batched_head_states(capture, raw_heads, config, triton_module,
                                *, tile_count: int, quant_backend="torch",
                                input_cast="gpu", stage_timings_ms=None,
                                kv_cache=None, query_workspace=None,
                                valid_query_rows=None, query_staging="gather",
                                stage_gpu_events=None, vb_mode="materialized"):
    """One head-group preparation: one stats launch, one rowmax, one topk.

    Reads trusted native same-block captured Q/K/V; repeated raw head
    indices require the caller to mark the measurement as a proxy.
    Outputs contiguous tensors for the batch Tile + Merge helpers.
    The arithmetic is the same as the pinned INT8/BF16 seq/sink0 path.
    """
    import torch
    triton_module._dll()
    if vb_mode not in ("materialized", "split_stream", "fused_vb"):
        raise ValueError("vb_mode must be materialized, split_stream or fused_vb")
    if quant_backend not in ("torch", "triton"):
        raise ValueError("quant_backend must be torch or triton")
    if input_cast not in ("gpu", "cpu"):
        raise ValueError("input_cast must be gpu or cpu")
    if query_staging not in ("gather", "view_copy", "native_direct", "native_torch_mu"):
        raise ValueError("query_staging must be gather, view_copy, native_direct or native_torch_mu")
    if query_staging in ("view_copy","native_direct","native_torch_mu") and input_cast != "gpu":
        raise ValueError("Direct query staging needs GPU-resident Q input")
    if query_workspace is not None and (
        not isinstance(query_workspace, dict) or kv_cache is None
        or quant_backend != "triton" or input_cast != "gpu"
    ):
        raise ValueError("Query workspace requires GPU K/V cache and Triton quantizer")
    # Optional GPU event stage intervals: no CPU synchronization and no
    # change to the default path. They include GPU stream enqueue gaps.
    # They are diagnostics, NOT kernel-only execution measurements.
    stage_gpu_prev=None
    if stage_gpu_events is not None:
        if not isinstance(stage_gpu_events,list) or capture["q_qhd"].device.type!="cuda":
            raise ValueError("stage_gpu_events requires a CUDA capture and list")
        stage_gpu_prev=torch.cuda.Event(enable_timing=True)
        stage_gpu_prev.record()
    stage_start=time.perf_counter()
    def mark_stage(name):
        nonlocal stage_start,stage_gpu_prev
        if stage_gpu_prev is not None:
            stage_finish=torch.cuda.Event(enable_timing=True)
            stage_finish.record()
            stage_gpu_events.append((name,stage_gpu_prev,stage_finish))
            stage_gpu_prev=stage_finish
        if stage_timings_ms is not None:
            torch.cuda.synchronize()
            now=time.perf_counter()
            stage_timings_ms[name]=(now-stage_start)*1000.0
            stage_start=now
    if not isinstance(raw_heads, (list, tuple)) or not raw_heads:
        raise ValueError("raw_heads must be a nonempty head ID list")
    if not isinstance(tile_count,int) or tile_count<1:
        raise ValueError("tile_count must be positive")
    if (config.get("order")!="seq" or config.get("store")!="int8"
            or config.get("dot")!="bf16" or config.get("sink_leaves",0)!=0
            or int(config["tile"])!=64):
        raise ValueError("Only seq/int8/bf16/sink0, tile64 is supported")
    if config.get("select") not in (None,"max"):
        raise ValueError("Batched head preparation supports max leaf selection only")
    if int(config.get("sel_stride",1))!=1:
        raise ValueError("Batched preparation requires select stride1")
    if any(k not in capture for k in ("q_qhd","k_khd","v_khd","raw_head_index")):
        raise ValueError("Saved capture lacks Q/K/V/raw_head_index")
    idxmap=capture["raw_head_index"]
    available=[int(x) for x in (idxmap.tolist()
                                if isinstance(idxmap,torch.Tensor) else idxmap)]
    if len(set(available))!=len(available):
        raise ValueError("Duplicate captured native raw_head_index")
    lookup={raw:i for i,raw in enumerate(available)}
    try:
        positions=[lookup[int(raw)] for raw in raw_heads]
    except KeyError as exc:
        raise ValueError(f"Head ID missing in capture: {exc}") from exc
    h=len(positions)
    q_cap,k_cap,v_cap=(capture[x] for x in ("q_qhd","k_khd","v_khd"))
    if q_cap.device != k_cap.device or q_cap.device != v_cap.device:
        raise ValueError("All Q/K/V capture tensors must share one device")
    if q_cap.device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU or CUDA Q/K/V capture tensors supported")
    if q_cap.device.type == "cuda" and input_cast == "cpu":
        raise ValueError("CPU BF16 staging is invalid for GPU-resident Q/K/V")
    tile=64
    group=int(config.get("kern",{}).get("group",1))
    if group not in (1,2,4) or tile_count % group:
        raise ValueError("Query tile count must divide centre group 1/2/4")
    centers=tile_count//group
    if vb_mode in ("split_stream","fused_vb") and (
        group != 1 or kv_cache is None or query_workspace is None
        or query_staging not in ("native_direct","native_torch_mu")
    ):
        raise ValueError("Streaming VB requires group1, resident KV, native Q and workspace")
    n_query=tile_count*tile
    if n_query>int(q_cap.shape[0]):
        raise ValueError("Not enough query rows in native capture")
    if valid_query_rows is not None and (
        not isinstance(valid_query_rows,int)
        or not 1 <= valid_query_rows <= n_query
    ):
        raise ValueError("valid_query_rows must be in 1..tile_count*64")
    if k_cap.shape!=v_cap.shape or int(k_cap.shape[0])<16:
        raise ValueError("Capture K/V must have aligned full-leaf key rows")
    dim=int(q_cap.shape[-1])
    if dim!=128:
        raise ValueError("Only dim128 is supported")
    nb=int(k_cap.shape[0])//16
    nkeys=nb*16
    kx=min(int(config["exact_leaves"]),nb)
    if kx<1:
        raise ValueError("Exact leaf count must be nonzero")
    dev=torch.device("cuda")
    # Native strided Q with a prebuilt K/V cache never gathers a head.
    # Defer this small CPU->CUDA positions upload until an actual gather;
    # unneeded H2D copies inside CUDA Graph capture would be invalid.
    pos=None
    def gather(key, rows):
        nonlocal pos
        if pos is None:
            pos=torch.tensor(positions,dtype=torch.long,device=q_cap.device)
        raw=capture[key][:rows]
        return raw.index_select(1,pos).permute(1,0,2).contiguous()
    if query_staging == "view_copy":
        # Research-only alternate Q packing: one BF16/F16->FP32 copy.
        # Only valid for consecutive *source positions* in native
        # post-RoPE QKV. Do not use for proxy repeated heads.
        if (q_cap.device.type != "cuda" or
            positions != list(range(positions[0],positions[0]+h))):
            raise ValueError("view_copy requires consecutive unique CUDA native head positions")
        native=q_cap[:n_query,positions[0]:positions[0]+h,:]
        native_hsd=native.permute(1,0,2)
        Q=torch.empty((h,tile_count,tile,dim),device=q_cap.device,dtype=torch.float32)
        Q.view(h,n_query,dim).copy_(native_hsd)
    elif query_staging in ("native_direct","native_torch_mu"):
        if (q_cap.device.type!="cuda" or
            positions != list(range(positions[0],positions[0]+h))):
            raise ValueError("Native Q requires consecutive unique CUDA heads")
        native=q_cap[:n_query,positions[0]:positions[0]+h,:]
        Q=native.permute(1,0,2).view(h,tile_count,tile,dim)
        if Q.dtype not in (torch.bfloat16,torch.float16,torch.float32):
            raise TypeError("Unsupported native Q dtype")
        if query_staging=="native_direct":
            # Existing independent Triton MU reference.
            # During CUDA Graph capture, a new MU allocation inside
            # perform() may have a capture-private pointer that Triton's
            # launcher cannot dereference. Prefer a persistent CUDA
            # workspace tensor allocated BEFORE graph capture.
            MU=(query_workspace.get("MU") if query_workspace is not None
                else None)
            if MU is None:
                if os.environ.get("APL_HTRC_GRAPH_CAPTURE_SAFE_MU") == "1":
                    raise RuntimeError(
                        "Capture-safe native MU requires a preallocated GPU buffer")
                MU=torch.empty((h,centers,dim),device=dev,dtype=torch.float32)
            elif (not isinstance(MU,torch.Tensor)
                  or tuple(MU.shape)!=(h,centers,dim)
                  or MU.dtype!=torch.float32 or MU.device!=q_cap.device
                  or not MU.is_contiguous()):
                raise ValueError("Invalid capture-safe preallocated MU buffer")
            triton_module._direct_q_mu_kernel[(h,centers)](
                Q,MU,centers,group,tile,dim,
                int(Q.stride(0)),int(Q.stride(1)),int(Q.stride(2)),int(Q.stride(3)),
                int(valid_query_rows if valid_query_rows is not None else n_query),
                num_warps=4
            )
        else:
            # Experimental ATen reduction directly from native Q strides;
            # accumulates FP32 without materializing all Q rows in FP32.
            # This does not change downstream leaf geometry or selection.
            grouped_q=Q.view(h,centers,group*tile,dim)
            MU=grouped_q.mean(dim=2,dtype=torch.float32).contiguous()
            if valid_query_rows is not None and valid_query_rows % (group*tile):
                final_center=(valid_query_rows-1)//(group*tile)
                real_rows=valid_query_rows-final_center*group*tile
                MU[:,final_center,:]=grouped_q[
                    :,final_center,:real_rows,:].mean(dim=1,dtype=torch.float32)
    else:
        # Pinned A/B control; exactly the same sequence as before.
        Q=gather("q_qhd",n_query).to(dev).float().view(h,tile_count,tile,dim).contiguous()
    if query_staging not in ("native_direct","native_torch_mu") and group==1:
        MU=Q.mean(dim=2).contiguous()
        if valid_query_rows is not None and valid_query_rows % tile:
            final_tile=(valid_query_rows-1)//tile
            real_rows=valid_query_rows-final_tile*tile
            MU[:,final_tile,:]=Q[:,final_tile,:real_rows,:].mean(dim=1)
    elif query_staging not in ("native_direct","native_torch_mu"):
        grouped_q=Q.view(h,centers,group*tile,dim)
        MU=grouped_q.mean(dim=2).contiguous()
        if valid_query_rows is not None and valid_query_rows % (group*tile):
            final_center=(valid_query_rows-1)//(group*tile)
            real_rows=valid_query_rows-final_center*group*tile
            MU[:,final_center,:]=grouped_q[:,final_center,:real_rows,:].mean(dim=1)
    mark_stage("query_gather_h2d_mu" if q_cap.device.type=="cpu"
               else "query_gather_gpu_mu")
    if kv_cache is None:
        # CPU BF16 conversion avoids a whole-group FP32 GPU staging copy.
        def gather_bf16(name):
            raw=gather(name,nkeys)
            if input_cast=="cpu":
                return raw.to(torch.bfloat16).to(dev).contiguous()
            return raw.to(dev).to(torch.bfloat16).contiguous()
        kbf=gather_bf16("k_khd")
        vbf=gather_bf16("v_khd")
        mark_stage("key_value_gather_h2d" if q_cap.device.type=="cpu"
                   else "key_value_gather_gpu")
        if quant_backend=="torch":
            # Retain the original numerical/allocator path as control.
            kf=kbf.float()
            KS=(kf.abs().amax(-1)*(1.0/127.0)).clamp_min_(1.0e-20).contiguous()
            K=(kf/KS[:,:,None]).round_().clamp_(-127,127).to(torch.int8).contiguous()
            del kf
        else:
            # One fused K int8 / scale kernel for all grouped key rows.
            K=torch.empty((h,nkeys,dim),device=dev,dtype=torch.int8)
            KS=torch.empty((h,nkeys),device=dev,dtype=torch.float32)
            br=8
            triton_module._quant_key_int8_rows[
                (triton_module.triton.cdiv(h*nkeys,br),)
            ](kbf.view(h*nkeys,dim),K.view(h*nkeys,dim),
              KS.view(h*nkeys),h*nkeys,dim,br,num_warps=4)
        mark_stage("key_int8_quantization")
    else:
        # KBF/VBF/K8/KS are invariant only within this attention call.
        # Stats/selection remain query-dependent and are recomputed below.
        if quant_backend!="triton" or input_cast!="gpu" or q_cap.device.type!="cuda":
            raise ValueError("KV cache supports only CUDA-resident Triton K8")
        if (not isinstance(kv_cache,dict)
            or kv_cache.get("raw_heads")!=tuple(map(int,raw_heads))
            or kv_cache.get("source_k") is not k_cap
            or kv_cache.get("source_v") is not v_cap
            or kv_cache.get("source_k_version")!=_source_tensor_version(k_cap)
            or kv_cache.get("source_v_version")!=_source_tensor_version(v_cap)
            or kv_cache.get("head_positions")!=tuple(positions)
            or kv_cache.get("nkeys")!=nkeys):
            raise ValueError("KV cache identity, head order or K length mismatch")
        kbf=kv_cache["KBF"]
        vbf=kv_cache["VBF"]
        K=kv_cache["K"]
        KS=kv_cache["KS"]
        direct_kv=bool(kv_cache.get("native_kv",False))
        if (kbf.shape!=(h,nkeys,dim) or vbf.shape!=kbf.shape
            or K.shape!=kbf.shape or KS.shape!=(h,nkeys)
            or kbf.dtype!=torch.bfloat16 or vbf.dtype!=torch.bfloat16
            or K.dtype!=torch.int8 or KS.dtype!=torch.float32
            or any(t.device!=q_cap.device for t in (kbf,vbf,K,KS))
            or not K.is_contiguous() or not KS.is_contiguous()
            or (not direct_kv and (not kbf.is_contiguous() or not vbf.is_contiguous()))):
            raise ValueError("KV cache invalid tensor shape/dtype/device/layout")
        mark_stage("kv_cache_hit")
    V=vbf
    direct_kv=bool(kv_cache is not None and kv_cache.get("native_kv",False))
    def state_buffer(name, shape, dtype):
        if query_workspace is None:
            return torch.empty(shape,device=q_cap.device,dtype=dtype)
        buf=query_workspace.get(name)
        if buf is None:
            buf=torch.empty(shape,device=q_cap.device,dtype=dtype)
            query_workspace[name]=buf
        elif (not isinstance(buf,torch.Tensor)
              or tuple(buf.shape)!=tuple(shape)
              or buf.dtype!=dtype or buf.device!=q_cap.device
              or not buf.is_contiguous()):
            raise ValueError(f"Incompatible query workspace buffer: {name}")
        return buf
    # Reuse storage only; the stats kernel overwrites every element for every Q tile.
    G=state_buffer("G",(h,centers,nb,dim),torch.int8)
    VB=state_buffer("VB",((h,centers,nb,dim) if vb_mode=="materialized" else (1,)),torch.bfloat16)
    LZ=state_buffer("LZ",(h,centers,nb),torch.float32)
    SG=state_buffer("SG",(h,centers,nb),torch.float32)
    BT,BL=16,4
    blocks=math.ceil(centers/BT)
    # Head count is a dynamic grid dimension, not a kernel specialization.
    triton_module._stats_kernel[(h*blocks,triton_module.triton.cdiv(nb,BL))](
        MU.view(h*centers,dim),
        (kbf if direct_kv else kbf.view(h*nkeys,dim)),
        (vbf if direct_kv else vbf.view(h*nkeys,dim)),
        G.view(h*centers,nb,dim),
        LZ.view(h*centers,nb),
        (VB.view(h*centers,nb,dim) if vb_mode=="materialized" else VB),
        SG.view(h*centers,nb),
        LZ.view(h*centers,nb),
        h*centers,nb,1.0/math.sqrt(dim),
        D=dim,BT=BT,LEAF=16,BL=BL,PREC="tf32",FP8=0,I8=1,
        BATCH_TILES_PER_HEAD=centers,
        BATCH_KEY_ROWS=nkeys,
        BATCH_TILE_BLOCKS_PER_HEAD=blocks,
        DIRECT_KV=int(direct_kv),
        K_SH=(int(kbf.stride(0)) if direct_kv else 0),
        K_SS=(int(kbf.stride(1)) if direct_kv else 0),
        K_SD=(int(kbf.stride(2)) if direct_kv else 0),
        V_SH=(int(vbf.stride(0)) if direct_kv else 0),
        V_SS=(int(vbf.stride(1)) if direct_kv else 0),
        V_SD=(int(vbf.stride(2)) if direct_kv else 0),
        SCORE_ONLY=int(vb_mode in ("split_stream","fused_vb")),
        num_warps=4
    )
    mark_stage("leaf_stats")
    # The streaming VB producer reads the genuine BF16 K (not K8) with
    # the same MU as the original stats kernel, after selection is fixed.
    stream_kbf=kbf if vb_mode in ("split_stream","fused_vb") else None
    del kbf
    muf=MU.view(h*centers,dim)
    gf=G.view(h*centers,nb,dim)
    lzf=LZ.view(h*centers,nb)
    sgf=SG.view(h*centers,nb)
    if query_staging in ("native_direct","native_torch_mu"):
        sel=state_buffer("sel",(h*centers,nb),torch.float32)
        lps=triton_module._split(nb,int(config["splits"]),32)
        triton_module._rowmax_kernel[
            (h*centers,triton_module.triton.cdiv(nb,lps))
        ](Q,muf,gf,lzf,sel,sgf,nb,lps,
          T=group*tile,TS=group*tile,STRIDE=1,D=dim,BN=32,
          PREC="tf32",DOT=1,I8=1,
          Q_DIRECT=1,BATCH_CENTERS_PER_HEAD=centers,
          Q_GROUP=group,Q_TILE=tile,
          Q_SH=int(Q.stride(0)),Q_ST=int(Q.stride(1)),
          Q_SR=int(Q.stride(2)),Q_SD=int(Q.stride(3)),
          num_warps=4,num_stages=1)
    else:
        qf=Q.view(h*centers,group*tile,dim)
        sel=triton_module.rowmax_select(
            qf,muf,gf,lzf,
            prec="tf32",bn=32,warps=4,
            splits=int(config["splits"]),stride=1,dot=1,
            ws=query_workspace,sg=sgf
        )
    if (query_workspace is not None
        and "_TOPK_VALUES" in query_workspace
        and "_TOPK_IDS" in query_workspace):
        # Keep top-k GPU outputs stable through Graph capture/replay.
        # The old eager/benchmark path remains unchanged by default.
        topv=query_workspace["_TOPK_VALUES"]
        ids=query_workspace["_TOPK_IDS"]
        if (not all(isinstance(t,torch.Tensor) for t in (topv,ids))
            or tuple(topv.shape)!=(h*centers,kx)
            or tuple(ids.shape)!=(h*centers,kx)
            or topv.dtype!=sel.dtype or ids.dtype!=torch.int64
            or topv.device!=q_cap.device or ids.device!=q_cap.device
            or not topv.is_contiguous() or not ids.is_contiguous()):
            raise ValueError("Invalid capture-safe topk output buffers")
        torch.topk(sel,kx,dim=-1,sorted=False,out=(topv,ids))
    else:
        if os.environ.get("APL_HTRC_GRAPH_CAPTURE_SAFE_MU") == "1":
            raise RuntimeError(
                "Capture-safe Graph requires preallocated topk buffers")
        ids=torch.topk(sel,kx,dim=-1,sorted=False).indices
    if query_workspace is None:
        idx=ids.to(torch.int32).view(h,centers,kx).contiguous()
        excl=torch.zeros((h*centers,nb),device=dev,dtype=torch.int8)
    else:
        idx=state_buffer("IDX",(h,centers,kx),torch.int32)
        # copy_ casts the persistent INT64 indices to INT32 into the
        # existing output buffer, avoiding a short-lived CUDA allocation.
        idx.copy_(ids.view(h,centers,kx))
        excl=state_buffer("EXCL",(h*centers,nb),torch.int8)
        # Mandatory reset: do NOT reuse prior window's selection/EXCL contents.
        excl.zero_()
    excl.scatter_(1,ids,1)
    mark_stage("rowmax_topk_exclusion")
    states={
        "Q":Q,"MU":MU,"K":K,"V":V,"KS":KS,
        "G":G,"VB":VB,"LZ":LZ,"SG":SG,
        "EXCL":excl.view(h,centers,nb),
        "IDX":idx,
        "_direct_q":query_staging in ("native_direct","native_torch_mu"),
        "_direct_v":direct_kv,
        "_vb_mode":vb_mode,
        "_stream_kbf":stream_kbf,
    }
    return states,config,nkeys,nb,kx


def prepare_gpu_head_kv_cache(capture, raw_heads, triton_module, *, native_direct=False):
    """Cache KBF, VBF, INT8 K and scale KS for one GPU attention call.

    The caller must NOT reuse this across a different attention call,
    K/V tensor, head order or denoising step. Leaf stats depend on
    window-specific query MU and must be recomputed.
    """
    import torch
    triton_module._dll()
    if not isinstance(raw_heads,(list,tuple)) or not raw_heads:
        raise ValueError("Cache requires nonempty head list")
    if any(k not in capture for k in ("raw_head_index","k_khd","v_khd")):
        raise ValueError("Capture missing native K/V or head IDs")
    ksrc,vsrc=capture["k_khd"],capture["v_khd"]
    if (ksrc.device.type!="cuda" or vsrc.device!=ksrc.device
        or ksrc.ndim!=3 or ksrc.shape!=vsrc.shape or ksrc.shape[-1]!=128):
        raise ValueError("Expected same-shape CUDA K/V [tokens,heads,128]")
    nkeys=(int(ksrc.shape[0])//16)*16
    if nkeys<16:
        raise ValueError("Missing aligned K rows")
    ids=capture["raw_head_index"]
    available=[int(x) for x in (ids.tolist()
                                if isinstance(ids,torch.Tensor) else ids)]
    if not available or len(set(available))!=len(available):
        raise ValueError("Invalid capture head IDs")
    lookup={head:i for i,head in enumerate(available)}
    try:
        indices=[lookup[int(head)] for head in raw_heads]
    except KeyError as exc:
        raise ValueError(f"Uncaptured head ID {exc}") from exc
    pos=torch.tensor(indices,device=ksrc.device,dtype=torch.long)
    h=len(indices)
    if native_direct:
        # This first Direct KV gate is BF16-only. No hidden conversion or copy.
        if (ksrc.dtype!=torch.bfloat16 or vsrc.dtype!=torch.bfloat16
            or indices!=list(range(indices[0],indices[0]+h))):
            raise ValueError("native_direct KV requires unique consecutive BF16 CUDA heads")
        kbf=ksrc[:nkeys,indices[0]:indices[0]+h,:].permute(1,0,2)
        vbf=vsrc[:nkeys,indices[0]:indices[0]+h,:].permute(1,0,2)
    else:
        def gather(name):
            return capture[name][:nkeys].index_select(
                1,pos
            ).permute(1,0,2).contiguous().to(torch.bfloat16).contiguous()
        kbf=gather("k_khd")
        vbf=gather("v_khd")
    k8=torch.empty((h,nkeys,128),device=ksrc.device,dtype=torch.int8)
    ks=torch.empty((h,nkeys),device=ksrc.device,dtype=torch.float32)
    br=8
    if native_direct:
        triton_module._quant_key_int8_native[
            (triton_module.triton.cdiv(h*nkeys,br),)
        ](kbf,k8.view(h*nkeys,128),ks.view(h*nkeys),
          nkeys,128,h,br,*map(int,kbf.stride()),num_warps=4)
    else:
        triton_module._quant_key_int8_rows[
            (triton_module.triton.cdiv(h*nkeys,br),)
        ](kbf.view(h*nkeys,128),k8.view(h*nkeys,128),
          ks.view(h*nkeys),h*nkeys,128,br,num_warps=4)
    return {
        "raw_heads":tuple(map(int,raw_heads)),
        "head_positions":tuple(indices),
        "source_k":ksrc,"source_v":vsrc,
        "source_k_version":_source_tensor_version(ksrc),"source_v_version":_source_tensor_version(vsrc),
        "nkeys":nkeys,"native_kv":bool(native_direct),
        "KBF":kbf,"VBF":vbf,"K":k8,"KS":ks,
    }


def gpu_head_kv_cache_bytes(cache):
    """Size of unique resident K-side buffers (not aliased source QKV)."""
    keys=("K","KS") if cache.get("native_kv",False) else ("KBF","VBF","K","KS")
    return sum(cache[k].numel()*cache[k].element_size() for k in keys)

def gpu_query_workspace_bytes(workspace):
    """Count unique LIVE preallocated query buffers, not cached K/V."""
    import torch
    if not isinstance(workspace,dict):
        raise ValueError("Expected query workspace dict")
    seen=set()
    total=0
    def walk(value):
        nonlocal total
        if isinstance(value,dict):
            for item in value.values():
                walk(item)
        elif isinstance(value,torch.Tensor):
            storage=value.untyped_storage()
            ptr=(value.device, storage.data_ptr())
            if ptr not in seen:
                seen.add(ptr)
                total+=storage.nbytes()
    walk(workspace)
    return total
