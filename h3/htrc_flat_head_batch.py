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


def launch_prepared_tile_batch(states, config, triton_module, *, buffers=None):
    """One _tile_kernel call for arbitrary H prepared, same-shape native heads.

    The call uses existing BEST math with TAIL=0. Tail-in-Merge must be
    executed separately; PM/PL/PACC are not normalized final attention.
    """
    import torch
    req = ("Q","MU","K","V","KS","G","VB","LZ","SG","EXCL","IDX")
    missing=[name for name in req if name not in states]
    if missing:
        raise ValueError(f"Missing prepared state: {missing}")
    if any(not states[name].is_contiguous() for name in req):
        raise ValueError("All prepared batch tensors must be contiguous")
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
    if states["MU"].shape!=(heads,tiles,dim):
        raise ValueError("MU dimensions do not match Q")
    if states["K"].shape!=(heads,keys,dim) or states["V"].shape!=states["K"].shape:
        raise ValueError("K/V must be [heads,keys,dim]")
    if states["KS"].shape!=(heads,keys):
        raise ValueError("KS scales mismatch")
    if states["G"].shape!=(heads,tiles,leaves,dim) or states["VB"].shape!=states["G"].shape:
        raise ValueError("G/VB dims mismatch")
    if any(states[name].shape!=(heads,tiles,leaves) for name in ("LZ","SG","EXCL")):
        raise ValueError("LZ/SG/EXCL dims mismatch")
    idx=states["IDX"]
    if idx.ndim!=3 or idx.shape[:2]!=(heads,tiles) or idx.shape[2]<1:
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
    flat=lambda name:states[name].view((heads*tiles,)+states[name].shape[2:])
    lz=flat("LZ")
    compiled=triton_module._tile_kernel[(heads*tiles,splits,1)](
        flat("Q"),flat("MU"),flat("G"),lz,flat("VB"),
        flat("EXCL"),flat("IDX"),
        states["K"].view(heads*keys,dim),
        states["V"].view(heads*keys,dim),
        pm.view(heads*tiles,splits,tile),
        pl.view(heads*tiles,splits,tile),
        acc.view(heads*tiles,splits,tile,dim),
        flat("SG"),lz,lz,lz,states["KS"].view(heads*keys),
        leaves,kx,
        triton_module._split(leaves,splits,bn),
        triton_module._split(kx*16,splits,bn),
        0,1.0/math.sqrt(dim),
        T=tile,D=dim,BN=bn,BN_NEAR=near_bn,LEAF=16,
        PREC="tf32",DOT=1,GR=1,FP8=0,TAIL=0,RB=1,I8=1,
        PASS=0,LATE_NEAR_Q=0,FAR_PREFETCH_LZ=0,
        FAR_PV_ACC=0,NEAR_PV_ACC=0,
        BATCH_TILES_PER_HEAD=tiles,BATCH_KEY_ROWS=keys,
        num_warps=warps,num_stages=stages
    )
    return buffers,compiled



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
    if q.ndim!=4 or not q.is_contiguous():
        raise ValueError("Q must be contiguous [heads,tiles,rows,dim]")
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
        pm,pl,acc,q.view(heads*tiles,tile,dim),
        tkb.view(heads*ntail,dim),
        tvb.view(heads*ntail,dim),
        out,ntail,1.0/math.sqrt(dim),
        NS=splits,T=tile,R=merge_rows,D=dim,
        DOT=1,PREC="tf32",
        BATCH_TILES_PER_HEAD=tiles,BATCH_TAIL_ROWS=ntail,
        num_warps=4
    )
    return output,compiled


def prepare_batched_head_states(capture, raw_heads, config, triton_module,
                                *, tile_count: int, quant_backend="torch",
                                input_cast="gpu", stage_timings_ms=None,
                                kv_cache=None, query_workspace=None):
    """One head-group preparation: one stats launch, one rowmax, one topk.

    Reads trusted native same-block captured Q/K/V; repeated raw head
    indices require the caller to mark the measurement as a proxy.
    Outputs contiguous tensors for the batch Tile + Merge helpers.
    The arithmetic is the same as the pinned INT8/BF16 seq/sink0 path.
    """
    import torch
    triton_module._dll()
    if quant_backend not in ("torch", "triton"):
        raise ValueError("quant_backend must be torch or triton")
    if input_cast not in ("gpu", "cpu"):
        raise ValueError("input_cast must be gpu or cpu")
    if query_workspace is not None and (
        not isinstance(query_workspace, dict) or kv_cache is None
        or quant_backend != "triton" or input_cast != "gpu"
    ):
        raise ValueError("Query workspace requires GPU K/V cache and Triton quantizer")
    # Synchronized stages are for an optional untimed diagnostic.
    stage_start=time.perf_counter()
    def mark_stage(name):
        nonlocal stage_start
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
    n_query=tile_count*tile
    if n_query>int(q_cap.shape[0]):
        raise ValueError("Not enough query rows in native capture")
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
    pos=torch.tensor(positions,dtype=torch.long,device=q_cap.device)
    # One source-device gather per Q/K/V rather than 3*H independent transfers.
    def gather(key, rows):
        raw=capture[key][:rows]
        return raw.index_select(1,pos).permute(1,0,2).contiguous()
    Q=gather("q_qhd",n_query).to(dev).float().view(h,tile_count,tile,dim).contiguous()
    MU=Q.mean(dim=2).contiguous()
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
            or kv_cache.get("source_k_version")!=k_cap._version
            or kv_cache.get("source_v_version")!=v_cap._version
            or kv_cache.get("head_positions")!=tuple(positions)
            or kv_cache.get("nkeys")!=nkeys):
            raise ValueError("KV cache identity, head order or K length mismatch")
        kbf=kv_cache["KBF"]
        vbf=kv_cache["VBF"]
        K=kv_cache["K"]
        KS=kv_cache["KS"]
        if (kbf.shape!=(h,nkeys,dim) or vbf.shape!=kbf.shape
            or K.shape!=kbf.shape or KS.shape!=(h,nkeys)
            or kbf.dtype!=torch.bfloat16 or vbf.dtype!=torch.bfloat16
            or K.dtype!=torch.int8 or KS.dtype!=torch.float32
            or any(not t.is_contiguous() or t.device!=q_cap.device
                   for t in (kbf,vbf,K,KS))):
            raise ValueError("KV cache invalid tensor shape/dtype/device/layout")
        mark_stage("kv_cache_hit")
    V=vbf
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
    G=state_buffer("G",(h,tile_count,nb,dim),torch.int8)
    VB=state_buffer("VB",(h,tile_count,nb,dim),torch.bfloat16)
    LZ=state_buffer("LZ",(h,tile_count,nb),torch.float32)
    SG=state_buffer("SG",(h,tile_count,nb),torch.float32)
    BT,BL=16,4
    blocks=math.ceil(tile_count/BT)
    # Head count is a dynamic grid dimension, not a kernel specialization.
    triton_module._stats_kernel[(h*blocks,triton_module.triton.cdiv(nb,BL))](
        MU.view(h*tile_count,dim),
        kbf.view(h*nkeys,dim),
        vbf.view(h*nkeys,dim),
        G.view(h*tile_count,nb,dim),
        LZ.view(h*tile_count,nb),
        VB.view(h*tile_count,nb,dim),
        SG.view(h*tile_count,nb),
        LZ.view(h*tile_count,nb),
        h*tile_count,nb,1.0/math.sqrt(dim),
        D=dim,BT=BT,LEAF=16,BL=BL,PREC="tf32",FP8=0,I8=1,
        BATCH_TILES_PER_HEAD=tile_count,
        BATCH_KEY_ROWS=nkeys,
        BATCH_TILE_BLOCKS_PER_HEAD=blocks,
        num_warps=4
    )
    mark_stage("leaf_stats")
    del kbf
    qf=Q.view(h*tile_count,tile,dim)
    muf=MU.view(h*tile_count,dim)
    gf=G.view(h*tile_count,nb,dim)
    lzf=LZ.view(h*tile_count,nb)
    sgf=SG.view(h*tile_count,nb)
    sel=triton_module.rowmax_select(
        qf,muf,gf,lzf,
        prec="tf32",bn=32,warps=4,
        splits=int(config["splits"]),stride=1,dot=1,
        ws=query_workspace,sg=sgf
    )
    ids=torch.topk(sel,kx,dim=-1,sorted=False).indices
    if query_workspace is None:
        idx=ids.to(torch.int32).view(h,tile_count,kx).contiguous()
        excl=torch.zeros((h*tile_count,nb),device=dev,dtype=torch.int8)
    else:
        idx=state_buffer("IDX",(h,tile_count,kx),torch.int32)
        idx.copy_(ids.to(torch.int32).view(h,tile_count,kx))
        excl=state_buffer("EXCL",(h*tile_count,nb),torch.int8)
        # Mandatory reset: do NOT reuse prior window's selection/EXCL contents.
        excl.zero_()
    excl.scatter_(1,ids,1)
    mark_stage("rowmax_topk_exclusion")
    states={
        "Q":Q,"MU":MU,"K":K,"V":V,"KS":KS,
        "G":G,"VB":VB,"LZ":LZ,"SG":SG,
        "EXCL":excl.view(h,tile_count,nb),
        "IDX":idx
    }
    return states,config,nkeys,nb,kx


def prepare_gpu_head_kv_cache(capture, raw_heads, triton_module):
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
    def gather(name):
        return capture[name][:nkeys].index_select(
            1,pos
        ).permute(1,0,2).contiguous().to(torch.bfloat16).contiguous()
    kbf=gather("k_khd")
    vbf=gather("v_khd")
    h=len(indices)
    k8=torch.empty((h,nkeys,128),device=ksrc.device,dtype=torch.int8)
    ks=torch.empty((h,nkeys),device=ksrc.device,dtype=torch.float32)
    br=8
    triton_module._quant_key_int8_rows[
        (triton_module.triton.cdiv(h*nkeys,br),)
    ](kbf.view(h*nkeys,128),k8.view(h*nkeys,128),
      ks.view(h*nkeys),h*nkeys,128,br,num_warps=4)
    return {
        "raw_heads":tuple(map(int,raw_heads)),
        "head_positions":tuple(indices),
        "source_k":ksrc,"source_v":vsrc,
        "source_k_version":ksrc._version,"source_v_version":vsrc._version,
        "nkeys":nkeys,
        "KBF":kbf,"VBF":vbf,"K":k8,"KS":ks,
    }


def gpu_head_kv_cache_bytes(cache):
    """Size of unique resident K-side buffers (not aliased source QKV)."""
    return sum(cache[k].numel()*cache[k].element_size()
               for k in ("KBF","VBF","K","KS"))

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
