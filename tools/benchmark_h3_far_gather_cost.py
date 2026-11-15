"""HTRC research: measure coalesced dense versus packed-indirect G/VB/LZ/SG/EXCL copies.

NOT a packed attention implementation and NOT an attention-speedup test.
Both arms write separate scratch data with different numbers of valid leaf rows.
All correctness checks occur outside CUDA Graph event timing. Exactly ONE JSON.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))
from analyze_h3_far_pack import reference_pack
from analyze_h3_far_gather import gather_geometry
from benchmark_h3_far_pack_cost import _alloc as alloc_pack
from benchmark_h3_far_pack_cost import _launch as launch_pack
from benchmark_h3_far_pack_cost import _check as check_pack
from profile_h3_far_exclusion import ENV_KEYS

HEADS, TILES, NB, SPLITS, LPS, D = 4, 32, 954, 4, 256, 128
BN, N_TOKENS = 64, 15275
FIELDS = ("G", "VB", "LZ", "SG", "EXCL")


def validate_request(args):
    if not 7 <= args.rounds <= 12 or not 5 <= args.warmups <= 12:
        raise ValueError("Requires 7..12 samples and 5..12 warmups")
    if args.replays_per_sample != 16:
        raise ValueError("Pinned one-window timing requires exactly 16 replays per sample")


def allocate_scratch(torch, device):
    """Same 4x32x4x256 output CAPACITY for both arms; suffix undefined."""
    dims = (HEADS, TILES, SPLITS, LPS)
    return {key: torch.empty(dims + ((D,) if key in ("G", "VB") else ()),
                             dtype=({"G": torch.int8, "VB": torch.bfloat16,
                                     "LZ": torch.float32, "SG": torch.float32,
                                     "EXCL": torch.int8})[key], device=device)
            for key in FIELDS}


def check_source(torch, state):
    shapes = {"G": (HEADS, TILES, NB, D),
              "VB": (HEADS, TILES, NB, D),
              "LZ": (HEADS, TILES, NB),
              "SG": (HEADS, TILES, NB),
              "EXCL": (HEADS, TILES, NB)}
    dtypes = {"G": torch.int8, "VB": torch.bfloat16, "LZ": torch.float32,
              "SG": torch.float32, "EXCL": torch.int8}
    for key in FIELDS:
        obj = state[key]
        if tuple(obj.shape) != shapes[key] or obj.dtype != dtypes[key] or not obj.is_contiguous():
            raise ValueError(f"Prepared Graph {key} shape, dtype or stride invalid")


def launch_gather(state, packed, counts, out, *, arm, dense_kernel, packed_kernel):
    args = tuple(state[k] for k in FIELDS)
    destinations = tuple(out[k] for k in FIELDS)
    kw = {"NB": NB, "SPLITS": SPLITS, "LPS": LPS,
          "D": D, "BN": BN, "num_warps": 4, "num_stages": 2}
    if arm == "control_dense":
        return dense_kernel[(HEADS * TILES, SPLITS)](*args, *destinations, **kw)
    if arm == "candidate_packed":
        return packed_kernel[(HEADS * TILES, SPLITS)](
            *args, packed, counts, *destinations, **kw)
    raise ValueError("Unrecognized experimental arm")


def check_gather(torch, state, packed, counts, out, arm, phase, reference):
    """Check all defined leaves bitwise on GPU without reading output suffixes."""
    check_source(torch, state)
    if tuple(packed.shape) != (HEADS, TILES, SPLITS, LPS) or packed.dtype != torch.int32:
        raise AssertionError("Packed leaf index shape/dtype drift")
    if tuple(counts.shape) != (HEADS, TILES, SPLITS) or counts.dtype != torch.int32:
        raise AssertionError("Packed count shape/dtype drift")
    centers = HEADS * TILES
    if arm == "control_dense":
        leaf = torch.arange(SPLITS * LPS, device=counts.device, dtype=torch.int64)
        mask = leaf < NB
        selected = leaf.clamp(max=NB - 1).expand(centers, -1)
        valid = mask.expand(centers, -1)
    elif arm == "candidate_packed":
        # CPU oracle already verified exact PACKED valid prefixes and COUNTS.
        # Build masked addresses from that trusted oracle: do NOT read any
        # uninitialized PACKED suffix, even via torch.where on the GPU.
        safe_rows, valid_rows = [], []
        for head in reference["indices"]:
            for tile in head:
                safe_row, valid_row = [], []
                for split in tile:
                    if len(split) > LPS:
                        raise AssertionError("Reference split overflow")
                    safe_row.extend(split + [0] * (LPS - len(split)))
                    valid_row.extend([True] * len(split) + [False] * (LPS - len(split)))
                safe_rows.append(safe_row)
                valid_rows.append(valid_row)
        selected = torch.tensor(safe_rows, device=counts.device, dtype=torch.int64)
        valid = torch.tensor(valid_rows, device=counts.device, dtype=torch.bool)
        if selected.shape != (centers, SPLITS*LPS):
            raise AssertionError("Reference layout invalid")
        if not torch.equal(counts.detach().cpu(),torch.tensor(
                [[[len(split) for split in tile] for tile in head]
                 for head in reference["indices"]],dtype=torch.int32)):
            raise AssertionError("Packed prefix counts changed")
    else:
        raise ValueError("Unrecognized experimental arm")
    observed_count = int(valid.sum().item())
    for key in FIELDS:
        src = state[key].view(centers, NB, *((D,) if key in ("G", "VB") else ()))
        dest = out[key].view(centers, SPLITS*LPS, *((D,) if key in ("G", "VB") else ()))
        if key in ("G", "VB"):
            ix = selected[:, :, None].expand(-1, -1, D)
        else:
            ix = selected
        expected = torch.gather(src, 1, ix)
        # uint8 view compares raw BF16/FP32 bits, including signed zeros or NaN payloads.
        got = dest[valid].contiguous().view(torch.uint8)
        want = expected[valid].contiguous().view(torch.uint8)
        if not torch.equal(got, want):
            raise AssertionError(f"{phase}: {arm} {key} data mismatch")
        del expected, got, want
    if arm == "candidate_packed":
        if not bool(torch.all(out["EXCL"].view(centers, SPLITS*LPS)[valid] == 0)):
            raise AssertionError("Packed far gather included an excluded leaf")
    return {"raw_values_bit_exact": True,
            "uninitialized_suffix_not_read": True,
            "valid_leaf_rows_verified": observed_count,
            "fields": list(FIELDS),
            "not_a_softmax_or_attention_parity_check": True}


def compiled_resources(kernel):
    """Compiler metadata only; do not interpret as measured occupancy."""
    return {"registers_per_thread": getattr(kernel, "n_regs", None),
            "shared_bytes_per_cta": getattr(getattr(kernel, "metadata", None), "shared", None),
            "compiler_spills": getattr(kernel, "n_spills", None)}


def run_gpu(args):
    import torch
    from htrc_far_pack_triton import pack_ordered_far_index
    from htrc_far_gather_triton import far_dense_copy, far_packed_copy
    from h3 import htrc_flat_directflow_live as flow
    from h3 import htrc_flat_live_backend as live

    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 0):
        raise RuntimeError("Exact CUDA SM120 GPU required")
    if "RTX 5070 Ti" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Pinned RTX 5070 Ti GPU required")
    torch.set_grad_enabled(False)
    device = torch.device("cuda:0")
    torch.manual_seed(42026)
    shape = (1, HEADS, N_TOKENS, D)
    banks = [tuple(torch.randn(shape,device=device,dtype=torch.bfloat16)
                   for _ in range(3)) for _ in range(2)]
    saved_env = {k: os.environ.get(k) for k in ENV_KEYS}
    report = {"banks": [], "capture_blocks": [], "compiled": {}}
    outputs = []
    try:
        with tempfile.TemporaryDirectory(prefix="htrc-far-gather-") as tmp:
            overrides = {
                "APL_HTRC_TILE32_STATS_BT": "16",
                "APL_HTRC_DIRECTFLOW_GROUP": "11",
                "APL_HTRC_DIRECTFLOW_WARMUPS": "4",
                "APL_HTRC_DIRECTFLOW_VB_MODE": "materialized",
                "APL_HTRC_DIRECTFLOW_GROUP_POLICY": "legacy",
                "APL_HTRC_DIRECTFLOW_REPORT": str(Path(tmp)/"first_call.json"),
                "APL_HTRC_DIRECTFLOW_RESEARCH_FORCE_GROUP": None,
                "APL_HTRC_GRAPH_CAPTURE_SAFE_MU": "0",
                "APL_HTRC_DIRECTFLOW_MAX_EXTRA_MIB": "768",
                "APL_HTRC_DIRECTFLOW_MIN_FREE_MIB": "256",
                "APL_HTRC_DIRECTFLOW_TILES32_ADMISSION": "0",
            }
            for key,value in overrides.items():
                if value is None: os.environ.pop(key, None)
                else: os.environ[key] = value
            flow.clear_session_cache()
            bank_b_state = bank_b_packed = bank_b_counts = None
            refs = []
            for bi,bank in enumerate(banks):
                result = flow.graph_attention_bthd(
                    *bank, heads=HEADS, mode="balanced", tiles=TILES,
                    budget_mib=1024, reserve_mib=256, workspace=True)
                torch.cuda.synchronize(device)
                if bi == 0:
                    gate = json.loads((Path(tmp)/"first_call.json").read_text(encoding="utf-8"))
                    if gate.get("status") != "PASS_FIRST_REAL_CALL" or gate.get("all_groups_bit_exact") is not True:
                        raise AssertionError("Full first-call Graph/eager parity failed")
                if tuple(result.shape) != (1,N_TOKENS,HEADS,D):
                    raise AssertionError("Unexpected full attention output shape")
                outputs.append(result.detach().cpu().clone())
                session = flow._SESSIONS.get((HEADS,))
                if session is None or (TILES,2048) not in session["graphs"]:
                    raise AssertionError("Graph-prepared Tile32 window missing")
                state = session["graphs"][(TILES,2048)]["states"]
                check_source(torch, state)
                if tuple(state["IDX"].shape) != (HEADS,TILES,512):
                    raise AssertionError("IDX shape mismatch")
                reference = reference_pack(state["EXCL"].detach().cpu().tolist(),
                                           state["IDX"].detach().cpu().tolist(),
                                           splits=SPLITS,bn=BN,expected_near=512)
                geometry = gather_geometry(reference)
                if (geometry["existing_far_dot_blocks"] != 1920 or
                    geometry["packed_valid_source_leaves"] != 56576):
                    raise AssertionError("Unexpected packed far geometry")
                packed, counts = alloc_pack(torch, device)
                launch_pack(state["EXCL"],packed,counts,pack_ordered_far_index)
                pack_audit = check_pack(torch,packed,counts,reference,f"bank{bi}_pack")
                arm_checks = {}
                for arm in ("control_dense","candidate_packed"):
                    out = allocate_scratch(torch,device)
                    compiled = launch_gather(state,packed,counts,out,arm=arm,
                                              dense_kernel=far_dense_copy,
                                              packed_kernel=far_packed_copy)
                    torch.cuda.synchronize(device)
                    arm_checks[arm] = check_gather(torch,state,packed,counts,out,arm,f"bank{bi}",reference)
                    if arm not in report["compiled"]:
                        report["compiled"][arm] = compiled_resources(compiled)
                    del out
                report["banks"].append({
                    "bank":bi,"selection_mask_sha256":reference["selection"]["selection_mask_sha256"],
                    "pack_validation":pack_audit,"gather_validation":arm_checks,
                    "geometry":geometry,
                })
                refs.append(reference)
                if bi == 1:
                    bank_b_state = state
                    bank_b_packed = packed
                    bank_b_counts = counts
                else:
                    del packed, counts
                del result
            if torch.equal(outputs[0],outputs[1]):
                raise AssertionError("Independent QKV banks unexpectedly identical")
            if report["banks"][0]["selection_mask_sha256"] == report["banks"][1]["selection_mask_sha256"]:
                raise AssertionError("Selection masks were unexpectedly identical")
            eager = live.flat_attention_bthd(
                *banks[1],heads=HEADS,mode="balanced",tiles=TILES,
                budget_mib=1024,reserve_mib=256,workspace=True,
                query_staging="native_torch_mu",kv_staging="native_direct",dispatch="eager")
            if not torch.equal(eager.cpu(),outputs[1]):
                raise AssertionError("Bank B complete eager/Graph parity failed")
            del eager,outputs

            for capture_index,arm in enumerate(("control_dense","candidate_packed",
                                                "candidate_packed","control_dense"), 1):
                out = allocate_scratch(torch,device)
                launch_gather(bank_b_state,bank_b_packed,bank_b_counts,out,
                              arm=arm,dense_kernel=far_dense_copy,packed_kernel=far_packed_copy)
                pre = check_gather(torch,bank_b_state,bank_b_packed,bank_b_counts,out,arm,
                                   f"capture{capture_index}_before",refs[1])
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    launch_gather(bank_b_state,bank_b_packed,bank_b_counts,out,
                                  arm=arm,dense_kernel=far_dense_copy,packed_kernel=far_packed_copy)
                torch.cuda.synchronize(device)
                for _ in range(args.warmups):graph.replay()
                check_gather(torch,bank_b_state,bank_b_packed,bank_b_counts,out,arm,
                             f"capture{capture_index}_after_warmup",refs[1])
                samples = []
                for _ in range(args.rounds):
                    start,stop=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(args.replays_per_sample): graph.replay()
                    stop.record()
                    stop.synchronize()
                    samples.append(float(start.elapsed_time(stop))/args.replays_per_sample)
                post = check_gather(torch,bank_b_state,bank_b_packed,bank_b_counts,out,arm,
                                    f"capture{capture_index}_after_timing",refs[1])
                report["capture_blocks"].append({
                    "capture_index":capture_index,"arm":arm,"independent_capture":True,
                    "samples_per_capture":args.rounds,"replays_per_sample":args.replays_per_sample,
                    "gpu_event_ms_per_kernel_samples":samples,
                    "median_gpu_event_ms_per_kernel":statistics.median(samples),
                    "pre_timing_valid":pre,"post_timing_valid":post,
                })
                del graph,out
                gc.collect()
            a=[x["median_gpu_event_ms_per_kernel"] for x in report["capture_blocks"] if x["arm"]=="control_dense"]
            b=[x["median_gpu_event_ms_per_kernel"] for x in report["capture_blocks"] if x["arm"]=="candidate_packed"]
            if len(a)!=2 or len(b)!=2:
                raise AssertionError("Capture order incomplete")
            report.update({
                "status":"PASS_GATHER_COST_ONLY","completed":True,
                "first_complete_graph_eager_bit_exact":True,
                "bank_b_complete_graph_eager_bit_exact":True,
                "two_distinct_bf16_qkv_banks":True,
                "different_selection_masks_per_bank":True,
                "timings":{
                    "control_dense_capture_medians_ms":a,
                    "candidate_packed_capture_medians_ms":b,
                    "control_dense_median_ms":statistics.median(a),
                    "candidate_packed_median_ms":statistics.median(b),
                    "direct_dense_over_packed_ratio":statistics.median(a)/statistics.median(b),
                    "this_is_not_equal_work":True,
                    "timing_scope":"FIVE_FIELD_DATA_COPY_ONLY_ONE_2048_QUERY_ROW_WINDOW",
                    "packing_cost_excluded":True,
                    "no_tile_qk_pv_or_online_softmax":True,
                },
                "memory":{
                    "output_scratch_bytes_per_arm": (HEADS*TILES*SPLITS*LPS)*(D*(1+2)+4+4+1),
                    "pack_indices_and_counts_bytes":HEADS*TILES*SPLITS*(LPS*4+4),
                    "gpu_output_suffix_undefined":True,
                },
                "no_packed_attention_kernel":True,
                "no_far_softmax_parity_claim":True,
                "production_modified":False,
            })
            return report
    finally:
        try:flow.clear_session_cache()
        finally:
            for key,value in saved_env.items():
                if value is None:os.environ.pop(key,None)
                else:os.environ[key]=value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--expect-head",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--rounds",type=int,default=7)
    p.add_argument("--warmups",type=int,default=5)
    p.add_argument("--replays-per-sample",type=int,default=16)
    args=p.parse_args()
    report={
        "schema":"htrc-far-gather-cost-v1","status":"FAIL_CLOSED","completed":False,
        "probe":"GPU_DENSE_VS_INDIRECT_FAR_FIELD_COPY_ONLY",
        "tokens":N_TOKENS,"heads":HEADS,"tiles":TILES,"splits":SPLITS,
        "far_bn":BN,"near_budget":512,
        "source":"prepared native materialized full Graph EXCL/IDX and G/VB/LZ/SG",
        "timing_scope":"ONE_WINDOW_COPY_NOT_ATTENTION",
        "production_promotion":"PROHIBITED_RESEARCH_ONLY",
        "attention_candidate":"UNRUN","h3_model_or_video":"UNRUN",
    }
    try:
        validate_request(args)
        actual=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()
        if actual!=args.expect_head:
            raise RuntimeError(f"Expected exact HEAD {args.expect_head}; got {actual}")
        report["source_head"]=actual
        report.update(run_gpu(args))
    except BaseException as exc:
        report["status"]="FAIL_CLOSED"
        report["completed"]=False
        report["error"]=f"{type(exc).__name__}: {exc}"[:1800]
        report["traceback_tail"]=traceback.format_exc()[-4600:]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(f"one_json={args.output} status={report['status']}")
    return 0 if report["status"]=="PASS_GATHER_COST_ONLY" else 2


if __name__=="__main__":
    raise SystemExit(main())
