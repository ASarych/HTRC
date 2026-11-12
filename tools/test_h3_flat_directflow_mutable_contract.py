"""CPU-only source contracts for mutable-QKV CUDA Graph experiment.

These tests do not assert GPU numeric parity, capture success or runtime speed.
"""
from __future__ import annotations
import ast
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
BENCH=ROOT/"tools"/"benchmark_h3_flat_directflow_mutable_qkv_ab.py"
LIVE=ROOT/"h3"/"htrc_flat_live_backend.py"
PREP=ROOT/"h3"/"htrc_flat_head_batch.py"


class MutableGraphContract(unittest.TestCase):
    def test_python_ast(self):
        for path in (BENCH,LIVE,PREP):
            with self.subTest(path=path.name):
                ast.parse(path.read_text(encoding="utf-8"),filename=str(path))

    def test_qkv_refresh_and_quant_same_stream(self):
        src=BENCH.read_text(encoding="utf-8")
        self.assertIn('stage["k"].copy_(bank[1])',src)
        self.assertIn('stage["v"].copy_(bank[2])',src)
        self.assertIn('tr._quant_key_int8_native[',src)
        self.assertIn('tail_k.copy_(stage["k"][0,:,-ntail:,:])',src)
        self.assertIn('tail_v.copy_(stage["v"][0,:,-ntail:,:])',src)
        self.assertIn('target[:,:rows,:].copy_(src)',src)
        self.assertIn('reg["graph"].replay()',src)
        self.assertLess(src.index('update_kv_stage(stage,bank,tr)'),
                        src.index('reg["graph"].replay()'))

    def test_reuse_geometry_not_fixed_last_five(self):
        src=BENCH.read_text(encoding="utf-8")
        self.assertIn('def geometry_key(window):',src)
        self.assertIn('return (window.tile_count,window.valid_rows)',src)
        self.assertIn('windows.iter_query_windows(',src)
        self.assertIn('keys[geometry_key(geometry)]=geometry',src)
        self.assertIn('stage["registry"][geometry_key(geom)]',src)
        self.assertNotIn('tile_count=5',src)

    def test_true_input_change_and_exact_full_output(self):
        src=BENCH.read_text(encoding="utf-8")
        self.assertIn('sources=create_input_banks(args)',src)
        self.assertIn('torch.equal(eager_out,graph_out)',src)
        self.assertIn('if torch.equal(saved_output,graph_out):',src)
        self.assertIn('tuple(torch.randn(shape',src)
        self.assertIn('checks["OUTPUT_FULL"]',src)
        self.assertIn('checks[name]=bool(torch.equal(eager[name],graph[name]))',src)
        self.assertIn('snapshot_state(st)',src)
        self.assertIn('K8/KS refreshed once per attention call',src)
        self.assertIn('"eager_vs_replay_bit_exact":True',src)
        self.assertIn('report["cases"].append({',src)
        self.assertIn('"source_bank_total_bytes":sum(',src)

    def test_eager_workspace_needs_complete_qkv_capture(self):
        src=BENCH.read_text(encoding="utf-8")
        self.assertIn('eager_source_capture=shared.capture_from_bhsd(',src)
        self.assertIn('*sources[0],args.group_heads',src)
        self.assertIn('eager_ws=shared.alloc_workspace(',src)
        self.assertIn('eager_source_capture,args.group_heads,args.tiles,cfg',src)
        self.assertNotIn('stage["capture"],args.group_heads,args.tiles,cfg',src)


    def test_true_query_window_copy_on_cpu(self):
        # Run the EXACT non-CUDA query staging helper extracted from the
        # benchmark. This validates all real/padded Q rows for varied
        # window capacities and lengths without loading Triton.
        import torch
        from h3.htrc_flat_windows import iter_query_windows
        tree=ast.parse(BENCH.read_text(encoding="utf-8"))
        funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef)
               and n.name in ("copy_query_window","geometry_key")]
        self.assertEqual(len(funcs),2)
        module=ast.Module(body=funcs,type_ignores=[])
        ast.fix_missing_locations(module)
        scope={}
        exec(compile(module,str(BENCH),"exec"),scope)
        copy=scope["copy_query_window"]
        key=scope["geometry_key"]
        for h in (1,2,4):
            for tiles in (4,8,16):
                for seq in (17,63,64,65,257,1025):
                    src=torch.randn((1,h,seq,128),dtype=torch.bfloat16)
                    for win in iter_query_windows(seq,tiles):
                        with self.subTest(h=h,tiles=tiles,seq=seq,
                                          start=win.start):
                            qbuf=torch.empty(
                                (h,win.padded_rows,128),dtype=torch.bfloat16
                            )
                            copy(src,win,qbuf)
                            self.assertTrue(torch.equal(
                                qbuf[:,:win.valid_rows,:],
                                src[0,:,win.start:win.end,:]
                            ))
                            if win.valid_rows<win.padded_rows:
                                expected=src[0,:,win.end-1:win.end,:].expand(
                                    -1,win.padded_rows-win.valid_rows,-1
                                )
                                self.assertTrue(torch.equal(
                                    qbuf[:,win.valid_rows:,:],expected
                                ))
                            self.assertEqual(key(win),
                                             (win.tile_count,win.valid_rows))

    def test_cache_refresh_calls_requant_and_updates_tail_on_cpu(self):
        # Lightweight CPU quantizer stand-in: test that new input K/V
        # propagates into stage storage, K8/KS and the exact key tail.
        # This is a dataflow gate, NOT CUDA-kernel numerical parity.
        import torch
        from types import SimpleNamespace
        tree=ast.parse(BENCH.read_text(encoding="utf-8"))
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)
                and n.name=="update_kv_stage")
        mod=ast.Module(body=[fn],type_ignores=[])
        ast.fix_missing_locations(mod)
        scope={}
        exec(compile(mod,str(BENCH),"exec"),scope)
        refresh=scope["update_kv_stage"]
        class FakeQuant:
            def __getitem__(self,grid):
                def run(kbf,k8,ks,n,d,h,br,sh,ss,sd,*,num_warps):
                    src=kbf.float().reshape(h,n,d)
                    k8.copy_(torch.where(src>=0,1,-1).to(torch.int8))
                    ks.copy_(src.abs().mean(dim=2))
                return run
        tri=SimpleNamespace(
            _quant_key_int8_native=FakeQuant(),
            triton=SimpleNamespace(cdiv=lambda a,b:(a+b-1)//b)
        )
        h,seq,n,d=2,35,32,128
        stage_k=torch.zeros((1,h,seq,d),dtype=torch.bfloat16)
        stage_v=torch.zeros_like(stage_k)
        kbf=stage_k[0,:,:n,:]
        tail=(torch.empty((h,seq-n,d),dtype=torch.bfloat16),
              torch.empty((h,seq-n,d),dtype=torch.bfloat16))
        stage={
            "k":stage_k,"v":stage_v,"tail":tail,
            "cache":{
                "KBF":kbf,
                "K":torch.empty((h,n,d),dtype=torch.int8),
                "KS":torch.empty((h,n),dtype=torch.float32)
            }
        }
        source_a=(torch.zeros_like(stage_k),
                  torch.ones_like(stage_k),torch.full_like(stage_v,2))
        source_b=(torch.zeros_like(stage_k),
                  torch.full_like(stage_k,-3),torch.full_like(stage_v,5))
        refresh(stage,source_a,tri)
        self.assertTrue(torch.equal(stage["cache"]["K"],
                                    torch.ones((h,n,d),dtype=torch.int8)))
        self.assertTrue(torch.equal(stage["cache"]["KS"],
                                    torch.ones((h,n))))
        self.assertTrue(torch.equal(tail[0],source_a[1][0,:,-3:,:]))
        self.assertTrue(torch.equal(tail[1],source_a[2][0,:,-3:,:]))
        refresh(stage,source_b,tri)
        self.assertTrue(torch.equal(stage["cache"]["K"],
                                    -torch.ones((h,n,d),dtype=torch.int8)))
        self.assertTrue(torch.equal(stage["cache"]["KS"],
                                    torch.full((h,n),3.)))
        self.assertTrue(torch.equal(tail[0],source_b[1][0,:,-3:,:]))
        self.assertTrue(torch.equal(tail[1],source_b[2][0,:,-3:,:]))
        self.assertTrue(torch.equal(stage_k,source_b[1]))
        self.assertTrue(torch.equal(stage_v,source_b[2]))

    def test_memory_and_production_boundaries(self):
        src=BENCH.read_text(encoding="utf-8")
        live=LIVE.read_text(encoding="utf-8")
        self.assertIn('max_memory_reserved()',src)
        self.assertIn('signed_reserved_delta_mib',src)
        self.assertIn('peak_above_precapture_reserved_mib',src)
        self.assertIn('flat_directflow_mutable_qkv_ab.json',src)
        self.assertIn('capture_break_even_full_attention_calls_by_sync_wall',src)
        self.assertIn('"quality_coefficient_used":False',src)
        self.assertNotIn('torch.cuda.graph(',live)


if __name__=="__main__":
    unittest.main()
