"""Inspect local CUDA/Triton without compiling or launching a GPU kernel.

This is a CAPABILITY INVENTORY only, not a test of TMA multicast,
cluster operations, physical L2 reuse, or HTRC attention performance.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

EVIDENCE = {
    "compute_capability": "https://developer.nvidia.com/cuda/gpus",
    "vendor_geforce_multicast": (
        "https://github.com/NVIDIA/cutlass/blob/main/examples/"
        "79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu"
    ),
    "triton_sm120_lowering": "https://github.com/triton-lang/triton/issues/10973",
    "cuda_l2": (
        "https://docs.nvidia.com/cuda/cuda-programming-guide/"
        "04-special-topics/l2-cache-control.html"
    ),
}


def classify_device(name: str, major: int, minor: int) -> dict:
    """Conservative vendor-source assessment; never infer GPU test PASS."""
    n = name.casefold()
    geforce_50 = "geforce" in n and "rtx 50" in n
    sm120_geforce_50 = geforce_50 and (major, minor) == (12, 0)
    return {
        "sm120_geforce_50": sm120_geforce_50,
        "vendor_tma_multicast": (
            "UNSUPPORTED_BY_NVIDIA_CUTLASS"
            if sm120_geforce_50 else "UNKNOWN_FAIL_CLOSED"
        ),
        "ptx_cluster_ops": (
            "SM90_PLUS_PTX_SUPPORT_NOT_A_RUNTIME_TEST"
            if major >= 9 else "UNKNOWN_FAIL_CLOSED"
        ),
        "triton_multi_cta_sm120": (
            "NOT_VALIDATED_ISSUE_10973_OPEN_ON_2026_11_14"
            if (major, minor) == (12, 0) else "UNKNOWN_FAIL_CLOSED"
        ),
        "tma_unicast": "NOT_EXECUTED_NOT_PROVEN",
        "l2_reuse": "NOT_MEASURED_NO_PHYSICAL_HBM_CLAIM",
        "decision": (
            "NO_GO_HARDWARE_TMA_MULTICAST_FOR_GEFORCE50"
            if sm120_geforce_50 else "UNKNOWN_DEVICE_NO_PRODUCTION_ADMISSION"
        ),
        "production_promotion": "PROHIBITED_CAPABILITY_ONLY",
    }


def inspect_local() -> dict:
    report = {
        "schema": "htrc-sm120-transport-inventory-v1",
        "status": "FAIL_CLOSED",
        "gpu_kernel_executed": False,
        "benchmark_executed": False,
        "h3_checkpoint_loaded": False,
        "cuda_graph_captured": False,
        "evidence_snapshot_date": "2026-11-14",
        "evidence": EVIDENCE,
        "ncu_executable": shutil.which("ncu"),
    }
    try:
        import torch
    except Exception as exc:
        report["error"] = f"torch_unavailable: {type(exc).__name__}: {exc}"[:360]
        return report
    report["torch"] = str(torch.__version__)
    report["torch_cuda_runtime"] = str(torch.version.cuda)
    if not torch.cuda.is_available():
        report["error"] = "CUDA is unavailable; no hardware verdict"
        return report

    prop = torch.cuda.get_device_properties(0)
    major, minor = map(int, torch.cuda.get_device_capability(0))
    name = str(prop.name)
    report["gpu"] = {
        "name": name,
        "capability_major": major,
        "capability_minor": minor,
        "total_memory_bytes": int(prop.total_memory),
        "sm_count": int(prop.multi_processor_count),
    }
    for name_attr in ("shared_memory_per_block", "shared_memory_per_block_optin",
                      "shared_memory_per_multiprocessor", "regs_per_multiprocessor",
                      "max_threads_per_multi_processor", "L2_cache_size"):
        value = getattr(prop, name_attr, None)
        if value is not None:
            report["gpu"][name_attr] = int(value)
    try:
        import triton
        report["triton_version"] = str(triton.__version__)
    except Exception as exc:
        report["triton_version"] = "UNAVAILABLE"
        report["triton_import_error"] = f"{type(exc).__name__}: {exc}"[:220]
    report["capability_assessment"] = classify_device(name, major, minor)
    report["status"] = "CAPABILITY_ONLY_NO_GPU_KERNEL"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="ONE JSON. No extra artifacts or GPU benchmark")
    args = parser.parse_args()
    result = inspect_local()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n",
                           encoding="utf-8")
    print(f"one_json={args.output} status={result['status']}")
    return 0 if result["status"] == "CAPABILITY_ONLY_NO_GPU_KERNEL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
