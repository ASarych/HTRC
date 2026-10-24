"""Minimal standalone HTRC mathematical-reference implementation."""
from .model import Stats, TreeNode, exact_stats, merge_stats, build_forest, exact_attention, predict, evaluate
__all__ = ["Stats", "TreeNode", "exact_stats", "merge_stats", "build_forest", "exact_attention", "predict", "evaluate"]
