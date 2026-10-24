import unittest
import numpy as np
from htrc_reference import exact_stats, merge_stats, build_forest, evaluate, exact_attention

class TestHTRCMath(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1107)
        self.k = rng.normal(size=(128, 8))
        self.v = rng.normal(size=(128, 5))
        self.c = rng.normal(size=8) * 0.2
        self.d = rng.normal(size=8) * 0.1

    def test_parent_identities_are_exact(self):
        full = exact_stats(self.k, self.v, self.c)
        parts = [exact_stats(self.k[i:i+16], self.v[i:i+16], self.c) for i in range(0, 128, 16)]
        merged = merge_stats(parts)
        self.assertAlmostEqual(full.log_mass, merged.log_mass, places=12)
        for field in ("gradient", "hessian", "value", "value_jacobian"):
            np.testing.assert_allclose(getattr(full, field), getattr(merged, field), atol=1e-12)

    def test_value_jacobian_finite_difference(self):
        ref = exact_stats(self.k, self.v, self.c)
        direction = self.d / np.linalg.norm(self.d)
        eps = 1e-5
        plus = exact_attention(self.k, self.v, self.c + eps * direction)
        minus = exact_attention(self.k, self.v, self.c - eps * direction)
        np.testing.assert_allclose((plus-minus)/(2*eps), ref.value_jacobian @ direction,
                                   atol=1e-9, rtol=1e-8)

    def test_at_centroid_root_collapse_is_exact(self):
        forest = build_forest(self.k, self.v, self.c)
        approx, meta = evaluate(self.k, self.v, self.c, self.c, forest)
        dense = exact_attention(self.k, self.v, self.c)
        np.testing.assert_allclose(approx, dense, atol=1e-12)
        self.assertEqual(meta["exact_leaves"], 0)

    def test_open_all_exact_leaves_is_exact(self):
        forest = build_forest(self.k, self.v, self.c)
        approx, meta = evaluate(self.k, self.v, self.c + self.d,
                                self.c, forest, opening_threshold=0)
        dense = exact_attention(self.k, self.v, self.c + self.d)
        np.testing.assert_allclose(approx, dense, atol=1e-12)
        self.assertEqual(meta["exact_leaves"], 8)

    def test_shapes_and_threshold_validation(self):
        forest = build_forest(self.k, self.v, self.c)
        result, info = evaluate(self.k, self.v, self.c+self.d, self.c, forest,
                                opening_threshold=0.01)
        self.assertEqual(result.shape, (5,))
        self.assertGreaterEqual(info["visited_nodes"], 1)
        with self.assertRaises(ValueError):
            evaluate(self.k, self.v, self.c, self.c, forest, opening_threshold=-1)

if __name__ == "__main__":
    unittest.main()
