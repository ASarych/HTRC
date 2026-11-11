"""CPU-only validation of generic H3 Tile head group planner."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location(
    "_flat_head_plan_unit_test",ROOT/"h3"/"htrc_flat_head_batch.py"
)
mod=importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name]=mod
SPEC.loader.exec_module(mod)


class PlannerTests(unittest.TestCase):
    def test_arbitrary_head_counts(self):
        for count in (1,2,4,7,8,13,28,55,56,63):
            plan=mod.plan_tile_head_batches(
                count,free_bytes=16384*mod.MIB,
                keys_per_head=40752,tile_count=16,
                max_heads_per_group=9
            )
            self.assertEqual(plan.heads_per_group,min(count,9))
            groups=list(mod.iter_head_groups(count,plan.heads_per_group))
            self.assertEqual(groups[-1][1],count)
            self.assertEqual(sum(e-s for s,e in groups),count)

    def test_last_short_group(self):
        self.assertEqual(list(mod.iter_head_groups(56,11)),
            [(0,11),(11,22),(22,33),(33,44),(44,55),(55,56)])

    def test_more_memory_cannot_reduce_plan(self):
        def plan(memory):
            return mod.plan_tile_head_batches(
                56,free_bytes=memory*mod.MIB,
                keys_per_head=40752,tile_count=16,
                reserve_bytes=128*mod.MIB,
                fixed_temp_bytes=64*mod.MIB)
        self.assertLessEqual(plan(768).heads_per_group,plan(2048).heads_per_group)
        self.assertLessEqual(plan(2048).heads_per_group,plan(8192).heads_per_group)

    def test_budget_conservative(self):
        free=1024*mod.MIB
        reserve=256*mod.MIB
        p=mod.plan_tile_head_batches(
            56,free_bytes=free,keys_per_head=40752,
            tile_count=16,reserve_bytes=reserve
        )
        self.assertLessEqual(p.estimated_max_additional_bytes,free-reserve)

    def test_rejected_limits(self):
        with self.assertRaises(MemoryError):
            mod.plan_tile_head_batches(
                56,free_bytes=100*mod.MIB,
                keys_per_head=40752,tile_count=16)
        with self.assertRaises(ValueError):
            mod.estimate_tile_head_bytes(keys_per_head=40753,tile_count=16)
        with self.assertRaises(ValueError):
            list(mod.iter_head_groups(56,0))
        with self.assertRaises(ValueError):
            mod.plan_tile_head_batches(
                56,free_bytes=1024*mod.MIB,keys_per_head=40752,
                tile_count=16,safety_factor=0.9)


if __name__=="__main__":
    unittest.main()
