"""GloBE: Global-Basis Experts for Mixture-of-Experts FFNs.

A drop-in module that uses a single global basis bank across all MoE layers,
learns sparse token-independent mixtures via entmax/sparsemax with temperature annealing,
and precomposes & caches expert weights for dense-like inference.
"""

__version__ = "0.1.0"
