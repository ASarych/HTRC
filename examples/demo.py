import numpy as np
from htrc_reference import build_forest, evaluate, exact_attention

rng = np.random.default_rng(20260925)
keys = rng.normal(size=(512, 16))
values = rng.normal(size=(512, 12))
centroid = rng.normal(size=16) * 0.15
query = centroid + rng.normal(size=16) * 0.12
forest = build_forest(keys, values, centroid, leaf_size=16, fanout=4)
reference = exact_attention(keys, values, query)
for threshold in (0.1, 0.01, 0.001, 0.0):
    value, meta = evaluate(keys, values, query, centroid, forest, opening_threshold=threshold)
    rel_error = np.linalg.norm(value-reference)/max(np.linalg.norm(reference), 1e-12)
    print(f"threshold={threshold:<6} relative_error={rel_error:.6g} stats={meta}")
