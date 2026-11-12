"""CPU-only variable-tail geometry and zero-allocation workspace alias gates."""
from __future__ import annotations

import unittest
import torch

from h3.htrc_flat_windows import iter_query_windows, compact_workspace_view


class WindowGeometryTest(unittest.TestCase):
    def test_reference_18723_is_derived(self):
        wins=list(iter_query_windows(18723,16))
        self.assertEqual(len(wins),19)
        self.assertEqual(wins[-1].valid_rows,291)
        self.assertEqual(wins[-1].tile_count,5)
        self.assertEqual(wins[-1].padded_rows,320)

    def test_variable_tokens_window_tiles_center_groups(self):
        for tiles in (4,8,16,32):
            for group in (1,2,4):
                if tiles%group:
                    continue
                for tokens in (1,17,64,65,1024,18723,
                               tiles*64-1,tiles*64,tiles*64+1,tiles*128+19):
                    with self.subTest(tiles=tiles,group=group,tokens=tokens):
                        wins=list(iter_query_windows(tokens,tiles,center_group=group))
                        self.assertEqual(wins[0].start,0)
                        self.assertEqual(wins[-1].end,tokens)
                        self.assertEqual(sum(w.valid_rows for w in wins),tokens)
                        for a,b in zip(wins,wins[1:]):
                            self.assertEqual(a.end,b.start)
                        for w in wins:
                            self.assertEqual(w.tile_count%group,0)
                            self.assertLessEqual(w.tile_count,tiles)
                            self.assertLessEqual(w.valid_rows,w.padded_rows)
                            self.assertLessEqual(w.padded_rows-w.valid_rows,group*64-1)

    def test_rejects_invalid_geometry(self):
        for args,kwargs in (
            ((0,16),{}),((20,0),{}),
            ((20,3),{"center_group":2}),
            ((20,16),{"center_group":0}),
        ):
            with self.assertRaises(ValueError):
                list(iter_query_windows(*args,**kwargs))


class WorkspaceAliasTest(unittest.TestCase):
    def test_same_storage_contiguous_multihead(self):
        for heads in (1,4,11):
            for tiles in (8,16,32):
                for group in (1,2,4):
                    if tiles%group:
                        continue
                    cfg={"tile":64,"kern":{"group":group}}
                    n,dim,kx,splits=11,128,8,2
                    centers=tiles//group
                    partial=(heads,tiles,splits,64)
                    ws={
                        "G":torch.empty(heads,centers,n,dim,dtype=torch.int8),
                        "VB":torch.empty(heads,centers,n,dim,dtype=torch.bfloat16),
                        "LZ":torch.empty(heads,centers,n),
                        "SG":torch.empty(heads,centers,n),
                        "sel":torch.empty(heads*centers,n),
                        "IDX":torch.empty(heads,centers,kx,dtype=torch.int32),
                        "EXCL":torch.empty(heads*centers,n,dtype=torch.int8),
                        "_partials":{
                            "pm":torch.empty(partial),
                            "pl":torch.empty(partial),
                            "acc":torch.empty(partial+(dim,))
                        },
                        "_out":torch.empty(heads,tiles,64,dim)
                    }
                    for active in range(group,tiles+1,group):
                        with self.subTest(heads=heads,tiles=tiles,group=group,active=active):
                            sub=compact_workspace_view(ws,heads,active,cfg)
                            if active==tiles:
                                self.assertIs(sub,ws)
                                continue
                            for key in ("G","VB","LZ","SG","sel","IDX","EXCL","_out"):
                                self.assertEqual(sub[key].untyped_storage().data_ptr(),
                                                 ws[key].untyped_storage().data_ptr())
                                self.assertTrue(sub[key].is_contiguous())
                            for key in ("pm","pl","acc"):
                                self.assertEqual(sub["_partials"][key].untyped_storage().data_ptr(),
                                                 ws["_partials"][key].untyped_storage().data_ptr())
                                self.assertTrue(sub["_partials"][key].is_contiguous())

    def test_bound_and_head_mismatch(self):
        cfg={"tile":64,"kern":{"group":1}}
        ws={"_out":torch.empty(2,16,64,128)}
        with self.assertRaises(ValueError):
            compact_workspace_view(ws,2,17,cfg)
        with self.assertRaises(ValueError):
            compact_workspace_view(ws,1,5,cfg)


if __name__=="__main__":
    unittest.main()
