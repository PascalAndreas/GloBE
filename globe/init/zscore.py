"""Per-family Z-score normalization utilities for GloBE.

This module standardizes expert weight matrices on a per-family basis
(up and gate) across the expert dimension. The recorded mean and
scale can later be folded back into the learned adapters.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from torch import Tensor


@dataclass
class ZScoreStats:
    """Container for mean/std statistics of a weight family."""

    mean: Tensor
    std: Tensor


class ZScoreNormalizer:
    """Compute and apply Z-score normalization for expert families."""

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.stats: Dict[str, ZScoreStats] = {}

    # ------------------------------------------------------------------
    def fit(self, expert_weights: Dict[str, List[Tensor]]) -> None:
        """Collect layerwise statistics across all experts for each family.
        
        Following MoBE paper: compute mean/std across all expert weights in the family,
        not per-position. This gives scalar mean and std per family.
        """

        self.stats = {}
        for family, weights in expert_weights.items():
            if not weights:
                continue
            stacked = torch.stack(weights, dim=0)  # E × p × d
            
            # MoBE paper approach: compute scalar statistics across all weights
            # This treats the entire weight matrix collection as a single distribution
            all_weights = stacked.view(-1)  # Flatten all expert weights
            mean = all_weights.mean()  # Scalar mean
            std = all_weights.std().clamp_min(self.eps)  # Scalar std
            
            # Convert to same shape as original for broadcasting compatibility
            mean_expanded = torch.full_like(stacked[0], mean)
            std_expanded = torch.full_like(stacked[0], std)
            
            self.stats[family] = ZScoreStats(mean=mean_expanded, std=std_expanded)

    # ------------------------------------------------------------------
    def transform(self, expert_weights: Dict[str, List[Tensor]]) -> Dict[str, List[Tensor]]:
        """Normalize each family's experts using stored statistics."""

        if not self.stats:
            raise ValueError("fit() must be called before transform().")

        out: Dict[str, List[Tensor]] = {}
        for family, weights in expert_weights.items():
            if family not in self.stats:
                out[family] = weights
                continue
            stats = self.stats[family]
            out[family] = [(w - stats.mean) / stats.std for w in weights]
        return out

    # ------------------------------------------------------------------
    def inverse_transform(self, normalized_weights: Dict[str, List[Tensor]]) -> Dict[str, List[Tensor]]:
        """Restore weights to their original scale."""

        if not self.stats:
            raise ValueError("fit() must be called before inverse_transform().")

        out: Dict[str, List[Tensor]] = {}
        for family, weights in normalized_weights.items():
            if family not in self.stats:
                out[family] = weights
                continue
            stats = self.stats[family]
            out[family] = [w * stats.std + stats.mean for w in weights]
        return out

    # ------------------------------------------------------------------
    def save(self, filepath: Path) -> None:
        """Persist collected statistics to disk using torch.save."""

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: {"mean": v.mean, "std": v.std} for k, v in self.stats.items()}
        torch.save(payload, filepath)

    # ------------------------------------------------------------------
    def load(self, filepath: Path) -> None:
        """Load previously saved statistics."""

        payload = torch.load(Path(filepath), map_location="cpu")
        self.stats = {k: ZScoreStats(mean=v["mean"], std=v["std"]) for k, v in payload.items()}


# ----------------------------------------------------------------------
def normalize_expert_families(
    expert_weights: Dict[str, List[Tensor]],
    save_stats_path: Optional[Path] = None,
) -> Tuple[Dict[str, List[Tensor]], ZScoreNormalizer]:
    """Normalize expert weight families and optionally save statistics."""

    normalizer = ZScoreNormalizer()
    normalizer.fit(expert_weights)
    if save_stats_path is not None:
        normalizer.save(save_stats_path)
    normalized = normalizer.transform(expert_weights)
    return normalized, normalizer


# ----------------------------------------------------------------------
def load_and_denormalize(
    normalized_weights: Dict[str, List[Tensor]],
    stats_path: Path,
) -> Dict[str, List[Tensor]]:
    """Convenience function to load saved stats and denormalize weights."""

    normalizer = ZScoreNormalizer()
    normalizer.load(stats_path)
    return normalizer.inverse_transform(normalized_weights)
