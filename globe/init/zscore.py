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
    """Container for scaling statistics of a weight family."""

    # Row-wise scale (σ) used for z-score normalisation.  We keep a ``mean``
    # field for backward compatibility but it is always zero in the current
    # workflow which operates purely on rescaled weights without centring.
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

        Following the refined workflow, we compute a *row-wise* scale for each
        family without centring.  For a given row ``k`` we use

        ``sigma_k = sqrt(1/(E*d) * sum_{i,j} W_i[k, j]^2)``

        where ``E`` is the number of experts and ``d`` the hidden dimension.
        This produces a diagonal scaling matrix ``D`` used throughout
        training.  Means are set to zero which keeps the implementation
        compatible with earlier versions while avoiding per-expert offsets.
        """

        self.stats = {}
        for family, weights in expert_weights.items():
            if not weights:
                continue
            stacked = torch.stack(weights, dim=0)  # E × p × d

            # Compute row-wise root-mean-square (no centring)
            # Resulting shape: p
            row_rms = torch.sqrt((stacked ** 2).mean(dim=(0, 2))).clamp_min(self.eps)

            # Expand to ``p × d`` for broadcasting with original weights
            std_expanded = row_rms.view(-1, 1).expand_as(stacked[0])
            mean_expanded = torch.zeros_like(std_expanded)

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
            # Mean is zero so we simply divide by the row-wise scale
            out[family] = [w / stats.std for w in weights]
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
            # Means are zero so inverse is simply multiplication by the scale
            out[family] = [w * stats.std for w in weights]
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
