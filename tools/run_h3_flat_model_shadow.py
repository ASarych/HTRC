"""Real-checkpoint, one-native-H3-block attention shadow gate for Flat HTRC.

Loads ONE real H3 block through the existing ComfyNodes hybrid loader,
generates synthetic block INPUT ONLY, uses the real qkv_proj + RMS/RoPE
and intercepts the actual MiniMax optimized_attention call. Executes
Flat shadow on distinct model heads. Original attention output always
continues to out_proj. No full denoise, no persistent patch, one JSON.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback


HTRC_ROOT = Path(__file__).resolve().parents[1]
if str(HTRC_ROOT) not in sys.path:
    sys.path.insert(0,str(HTRC_ROOT))
from h3 import htrc_flat_model_shadow as shadow


def import_file(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load real H3 module: {path}")
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def find_comfynodes_root(explicit, comfy_root):
    tries=[]
    if explicit:
        tries.append(Path(explicit))
    if os.environ.get("APL_H3_COMFYNODES_ROOT"):
        tries.append(Path(os.environ["APL_H3_COMFYNODES_ROOT"]))
    if os.environ.get("APL_H3_REPO"):
        tries.append(Path(os.environ["APL_H3_REPO"]))
    tries.extend((
        comfy_root/"custom_nodes"/"ComfyNodes",
        comfy_root/"custom_nodes"/"ASarych-ComfyNodes",
        Path(r"D:\ComfyNodes"),
    ))
    for candidate in tries:
        if (candidate/"h3"/"hybrid_loader.py").is_file():
            return candidate.resolve(),[str(p) for p in tries]
    return None,[str(p) for p in tries]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--comfy-root",type=Path,default=Path(os.environ.get(
        "COMFYUI_ROOT",
        r"C:\Users\Misya\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Misyau\ComfyUI"
    )))
    p.add_argument("--comfynodes-root",type=Path,default=None,
                   help="Installed ComfyNodes checkout containing h3/hybrid_loader.py")
    p.add_argument("--unet",type=Path,default=Path(os.environ.get(
        "APL_H3_UNET",
        r"D:\ComfyModels\diffusion_models\minimax_h3_ref2va_pruned_nvfp4_lightx2v_8step_premerged.safetensors"
    )))
    p.add_argument("--sequence",type=int,default=37710)
    p.add_argument("--block",type=int,default=0)
    p.add_argument("--reserve-mb",type=int,default=3072)
    p.add_argument("--head-cases",default="4,56")
    p.add_argument("--mode",choices=("fast","balanced"),default="fast")
    p.add_argument("--tiles",type=int,default=16)
    p.add_argument("--budget-mib",type=int,default=1024)
    p.add_argument("--reserve-mib",type=int,default=256)
    p.add_argument("--workspace",choices=("on","off"),default="on")
    p.add_argument("--max-rel-l2",type=float,default=None,
                   help="Optional strict quality gate against original model attention")
    p.add_argument("--preflight-only",action="store_true",
                   help="Validate paths and CLI without loading model or CUDA")
    p.add_argument("--out",type=Path,default=Path(
        r"D:\ComfyOutput\H3\math_probe\flat_model_resident_shadow.json"
    ))
    args=p.parse_args()
    try:
        heads=[int(x.strip()) for x in args.head_cases.split(",")]
    except ValueError:
        p.error("--head-cases must be comma-separated positive integers")
    if not heads or any(x<1 for x in heads) or len(heads)!=len(set(heads)):
        p.error("--head-cases must contain unique positive integers")
    if any(x<1 for x in (args.sequence,args.tiles,args.budget_mib)):
        p.error("sequence, tiles and budget must be positive")
    if args.block<0 or args.reserve_mb<0 or args.reserve_mib<0:
        p.error("block/reserve must be nonnegative")
    if args.max_rel_l2 is not None and args.max_rel_l2<0:
        p.error("max-rel-l2 must not be negative")

    comfy=args.comfy_root.resolve()
    unet=args.unet.resolve()
    cn, tried=find_comfynodes_root(args.comfynodes_root,comfy)
    report={
        "format":"h3-flat-model-resident-shadow-v1",
        "status":"STARTED",
        "source":"HTRC experimental Flat GPU K/V reuse + query workspace",
        "model_source":"ComfyNodes APLH3HybridLoader",
        "checkpoint_path":str(unet),
        "checkpoint_present":unet.is_file(),
        "comfyui_root":str(comfy),
        "comfyui_present":(comfy/"comfy"/"ldm"/"minimax"/"model.py").is_file(),
        "comfynodes_root":str(cn) if cn else None,
        "comfynodes_search_paths":tried,
        "head_cases":heads,
        "block_index":args.block,
        "synthetic_block_inputs":True,
        "real_checkpoint_weights":False,
        "real_native_qkv_projection_and_norm_rope":False,
        "real_denoising_inputs":False,
        "attention_output_replaced":False,
        "quality_vs_model_attention_is_not_video_quality":True,
        "errors":[],
        "shadow":None,
    }
    result=1
    patcher=None
    attention_module=None
    original_attention=None
    comfy_mm=None
    previous_training=None
    try:
        if not report["comfyui_present"]:
            raise FileNotFoundError("Missing ComfyUI minimax module under --comfy-root")
        if cn is None:
            raise FileNotFoundError(
                "Missing ComfyNodes/h3/hybrid_loader.py; specify --comfynodes-root")
        if not args.preflight_only and not unet.is_file():
            raise FileNotFoundError(f"Model checkpoint missing: {unet}")
        if args.preflight_only:
            report["status"]="PASS_PREFLIGHT_ONLY"
            result=0
            return result

        for path in (comfy,cn,cn/"h3",cn/"tools"):
            name=str(path)
            if name not in sys.path:
                sys.path.insert(0,name)
        # The HTRC h3 package is pre-imported before setting ComfyNodes paths.
        os.environ["APL_H3_REPO"]=str(cn)

        import torch
        import folder_paths
        import comfy.model_management as mm
        import comfy.ldm.minimax.model as minimax
        comfy_mm=mm
        previous_training=bool(mm.in_training)
        attention_module=minimax
        original_attention=minimax.optimized_attention
        report["original_attention_name"]=getattr(
            original_attention,"__name__",type(original_attention).__name__
        )
        if callable(getattr(folder_paths,"add_model_folder_path",None)):
            folder_paths.add_model_folder_path("diffusion_models",str(unet.parent))
        loader=import_file(
            "htrc_flat_real_checkpoint_hybrid_loader",cn/"h3"/"hybrid_loader.py"
        )
        patcher=loader.APLH3HybridLoader().load(
            unet.name,reserve_mb=args.reserve_mb,profile_first_forward=False
        )[0]
        patcher.load(device_to=torch.device("cuda"),
                     lowvram_model_memory=0,full_load=False)
        patcher._remove_stream_hooks()
        patcher._drain_streaming()
        patcher._materialize_block(args.block)
        torch.cuda.synchronize()
        report["real_checkpoint_weights"]=True
        block=patcher.model.diffusion_model.blocks[args.block]
        seq=int(args.sequence)
        dim=int(block.adaln_proj.hidden)
        tdim=int(block.adaln_proj.linear.in_features)
        hdim=int(block.attn.head_dim)
        hcount=int(block.attn.heads)
        report["real_block"]={
            "index":args.block,"native_heads":hcount,
            "head_dim":hdim,"hidden_dim":dim,
            "sequence_rows":seq,
            "cuda_free_after_materialize_mib":
                torch.cuda.mem_get_info()[0]/(1024*1024),
            "torch_allocated_after_materialize_mib":
                torch.cuda.memory_allocated()/(1024*1024),
        }
        if hdim!=128:
            raise ValueError(f"Unexpected H3 head_dim={hdim}")
        # Match the existing ComfyNodes real-weight attention probe.
        # Synthetic activation/t_embed/identity RoPE are NOT true denoise data.
        x=torch.randn((seq,dim),device="cuda",dtype=torch.bfloat16)*0.02
        temb=torch.randn((1,tdim),device="cuda",dtype=torch.bfloat16)*0.02
        rope_pairs=min(48,max(1,hdim//2))
        eye=torch.eye(2,device="cuda",dtype=torch.bfloat16)
        rope=eye.view(1,1,1,1,2,2).expand(
            1,seq,1,rope_pairs,2,2
        ).contiguous()
        mod_segments=[(0,seq,0)]
        # Out-of-place fused RMS/RoPE path is stable for the source gate.
        mm.in_training=True
        intercepted=[False]

        def probe(qc,kc,vc,heads_arg,*pos,**kw):
            if intercepted[0]:
                return original_attention(qc,kc,vc,heads_arg,*pos,**kw)
            intercepted[0]=True
            torch.cuda.synchronize()
            start=time.perf_counter()
            original=original_attention(qc,kc,vc,heads_arg,*pos,**kw)
            torch.cuda.synchronize()
            report["native_attention_wall_ms"]=(
                time.perf_counter()-start
            )*1000.0
            report["native_output_shape"]=list(original.shape)
            q=qc.peek() if hasattr(qc,"peek") else qc
            k=kc.peek() if hasattr(kc,"peek") else kc
            v=vc.peek() if hasattr(vc,"peek") else vc
            report["native_qkv_shapes"]={
                "q":list(q.shape),"k":list(k.shape),"v":list(v.shape)
            }
            report["native_qkv_dtypes"]={
                "q":str(q.dtype),"k":str(k.dtype),"v":str(v.dtype)
            }
            if kw.get("mask") is not None or not bool(kw.get("skip_reshape",False)):
                report["errors"].append("Unexpected masked/non-BHSD H3 attention contract")
                return original
            report["real_native_qkv_projection_and_norm_rope"]=True
            try:
                report["shadow"]=shadow.run_model_shadow(
                    q,k,v,original,
                    heads=int(heads_arg),head_cases=heads,
                    mode=args.mode,tiles=args.tiles,
                    budget_mib=args.budget_mib,reserve_mib=args.reserve_mib,
                    use_workspace=args.workspace=="on",
                    max_rel_l2=args.max_rel_l2,
                )
            except BaseException as exc:
                report["errors"].append(
                    f"{type(exc).__name__}: {exc}"[:1100]
                )
                report["shadow_traceback_tail"]=traceback.format_exc()[-2300:]
            # Do not change H3 output, even if shadow measured a quality loss.
            return original

        minimax.optimized_attention=probe
        try:
            with torch.inference_mode():
                block_out=block(
                    x,temb,mod_segments,rope,transformer_options={}
                )
                torch.cuda.synchronize()
                report["real_block_output_shape"]=list(block_out.shape)
                report["real_block_output_finite"]=bool(
                    torch.isfinite(block_out).all().item()
                )
                del block_out
        finally:
            minimax.optimized_attention=original_attention

        if not intercepted[0]:
            raise RuntimeError("Native H3 optimized_attention was not intercepted")
        if report["errors"] or not report["shadow"] or not report["shadow"]["completed"]:
            report["status"]="BLOCK_FINISHED_SHADOW_FAILED"
        elif not report["real_block_output_finite"]:
            report["status"]="NONFINITE_REAL_BLOCK_OUTPUT"
        else:
            report["status"]="PASS_REAL_BLOCK_MODEL_RESIDENT_SHADOW"
            result=0
    except BaseException as exc:
        report["status"]="MODEL_SHADOW_GATE_FAILED"
        report["errors"].append(f"{type(exc).__name__}: {exc}"[:1100])
        report["traceback_tail"]=traceback.format_exc()[-2800:]
    finally:
        if attention_module is not None and original_attention is not None:
            attention_module.optimized_attention=original_attention
        if comfy_mm is not None and previous_training is not None:
            comfy_mm.in_training=previous_training
        if patcher is not None:
            try: patcher._drain_streaming()
            except BaseException: pass
            try: patcher.detach()
            except BaseException: pass
        gc.collect()
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(report,indent=2),encoding="utf-8")
        print("MODEL_SHADOW_REPORT:",args.out,flush=True)
        print("MODEL_SHADOW_STATUS:",report["status"],flush=True)
    return result


if __name__=="__main__":
    raise SystemExit(main())
