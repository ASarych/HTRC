"""CPU-only contract and mathematical tests for the research delta probe.

GPU HTRC parity and performance remain UNRUN; this suite never invokes CUDA.
"""
from __future__ import annotations
import ast
from pathlib import Path
import sys
import unittest

import torch

ROOT=Path(__file__).resolve().parents[1]
BENCH=ROOT / "tools" / "benchmark_h3_flat_temporal_leaf_delta_ab.py"


class TemporalLeafDeltaContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import importlib.util
        spec=importlib.util.spec_from_file_location("_h3_temporal_leaf_delta_probe",BENCH)
        mod=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=mod
        spec.loader.exec_module(mod)
        cls.probe=mod

    def test_python_syntax_and_no_prod_change(self):
        src=BENCH.read_text(encoding="utf-8")
        tree=ast.parse(src,filename=str(BENCH))
        names={n.name for n in tree.body if isinstance(n,ast.FunctionDef)}
        self.assertTrue({"head_leaf_mean_delta","temporal_delta_into",
                         "dirty_stats_coverage","run_case","main"}<=names)
        self.assertIn('"accepted_for_runtime": False',src)
        self.assertIn('"predicts_future_timestep_before_new_qkv": False',src)
        self.assertIn('"quality_coefficient_used": False',src)
        self.assertIn('"fixed_exact_leaves": 256',src)
        self.assertIn('"fixed_center_group": 1',src)
        self.assertIn('flat_temporal_leaf_delta_ab.json',src)
        self.assertNotIn('torch.cuda.graph(',src)

    def test_observed_leaf_delta_is_mean_over_all_16_keys(self):
        p=self.probe
        old=torch.zeros((2,34,128),dtype=torch.bfloat16)
        new=old.clone()
        new[0,0:16,:]=3
        new[1,16:32,:]=-2
        new[0,0,0]=19
        delta=p.head_leaf_mean_delta(old,new,32)
        self.assertEqual(tuple(delta.shape),(2,2,128))
        self.assertEqual(delta[0,0,0].item(),4.0)
        self.assertEqual(delta[0,0,1].item(),3.0)
        self.assertEqual(delta[0,1,0].item(),0.0)
        self.assertEqual(delta[1,1,0].item(),-2.0)
        self.assertTrue(torch.equal(delta[1,0],torch.zeros(128)))

    def test_unchanged_stats_are_reused_algebraically(self):
        p=self.probe
        g=torch.zeros((1,2,3,128),dtype=torch.int8)
        g[:,:,:,0]=127
        g[:,:,:,1]=-38
        sg=torch.full((1,2,3),0.001,dtype=torch.float32)
        mu=torch.randn((1,2,128),dtype=torch.float32)
        vb=torch.randn((1,2,3,128),dtype=torch.bfloat16)
        lz=torch.randn((1,2,3),dtype=torch.float32)
        prev={"G":g,"SG":sg,"MU":mu,"VB":vb,"LZ":lz}
        target={k:torch.empty_like(prev[k]) for k in ("G","SG","VB","LZ")}
        zero=torch.zeros((1,3,128),dtype=torch.float32)
        p.temporal_delta_into(prev,mu,zero,zero,target)
        self.assertTrue(torch.equal(target["G"],g))
        self.assertTrue(torch.equal(target["VB"],vb))
        self.assertTrue(torch.equal(target["LZ"],lz))
        self.assertTrue(torch.allclose(target["SG"],sg,rtol=2e-7,atol=1e-10))

    def test_uniform_leaf_kv_shift_and_fixed_center(self):
        p=self.probe
        g=torch.zeros((1,1,2,128),dtype=torch.int8)
        g[...,0]=127
        sg=torch.full((1,1,2),0.003,dtype=torch.float32)
        mu=torch.zeros((1,1,128),dtype=torch.float32)
        mu[...,0]=2.0
        vb=torch.full((1,1,2,128),0.5,dtype=torch.bfloat16)
        lz=torch.zeros((1,1,2),dtype=torch.float32)
        prev={"G":g,"SG":sg,"MU":mu,"VB":vb,"LZ":lz}
        target={k:torch.empty_like(prev[k]) for k in ("G","SG","VB","LZ")}
        dk=torch.zeros((1,2,128),dtype=torch.float32)
        dk[0,0,0]=1.0
        dv=torch.zeros_like(dk)
        dv[0,0,1]=0.5
        p.temporal_delta_into(prev,mu,dk,dv,target)
        self.assertTrue(torch.isfinite(target["LZ"]).all())
        self.assertAlmostEqual(target["LZ"][0,0,0].item(),2.0*p.SCALE,places=6)
        self.assertAlmostEqual(target["LZ"][0,0,1].item(),0,places=7)
        self.assertEqual(target["VB"][0,0,0,1].item(),1.0)
        self.assertEqual(target["VB"][0,0,1,1].item(),0.5)

    def test_dirty_fraction_is_exact_pairwise_dependency(self):
        p=self.probe
        k0=torch.zeros((1,32,128),dtype=torch.bfloat16)
        v0=k0.clone()
        k1=k0.clone();v1=v0.clone()
        mu0=torch.zeros((1,2,128))
        mu1=mu0.clone()
        baseline=p.dirty_stats_coverage(mu0,mu1,k0,k1,v0,v1,32)
        self.assertEqual(baseline["exact_reusable_fraction"],1.0)
        mu1[0,0,0]=1.0
        k1[0,17,0]=1.0
        changed=p.dirty_stats_coverage(mu0,mu1,k0,k1,v0,v1,32)
        # Two centres and two leaves: centre0 invalidates both, leaf1
        # invalidates centre1; only centre1/leaf0 may be reused exactly.
        self.assertEqual(changed["total_head_center_leaf_pairs"],4)
        self.assertEqual(changed["dirty_head_center_leaf_pairs"],3)
        self.assertEqual(changed["exact_reusable_fraction"],0.25)

    def test_selected_index_patching_and_memory_scope(self):
        p=self.probe
        prev=torch.tensor([[[0,1,2],[1,2,3]]],dtype=torch.int32)
        curr=torch.tensor([[[0,1,2],[1,3,4]]],dtype=torch.int32)
        patch=p.selected_leaf_patch(prev,curr,5)
        self.assertEqual(patch["head_centers"],2)
        self.assertEqual(patch["unchanged_selected_set_centers"],1)
        self.assertEqual(patch["removed_memberships"],1)
        self.assertEqual(patch["added_memberships"],1)
        self.assertEqual(patch["full_idx_int32_bytes"],24)
        self.assertEqual(patch["oracle_minimum_pair_patch_bytes"],8)
        self.assertFalse(patch["patch_discovery_is_free"])
        ordered=p.selected_idx_position_patch(prev,curr)
        self.assertEqual(ordered["replaced_index_positions"],2)
        self.assertEqual(ordered["patch_bytes_position_u32_plus_id_i32"],16)
        self.assertTrue(ordered["round_trip_ordered_idx_bit_exact"])
        foot=p.cache_footprint_projection(56,18723,256)
        self.assertGreater(foot["estimated_G_VB_LZ_SG_gib_one_block"],6)
        self.assertLess(foot["estimated_selected_IDX_int32_mib_one_block"],20)

    def test_cannot_implicitly_claim_quality_or_future_forecast(self):
        p=self.probe
        a=torch.tensor([1.,2.,3.])
        bad=p.tensor_difference(a,a+1)
        self.assertFalse(bad["bit_exact"])
        self.assertGreater(bad["relative_l2"],0)
        self.assertTrue(p.tensor_difference(a,a)["bit_exact"])
        self.assertRaises(ValueError,p.head_leaf_mean_delta,
                          torch.zeros((1,8,128)),torch.zeros((1,8,128)),8)


if __name__=="__main__":
    unittest.main()
