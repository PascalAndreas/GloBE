"""Global basis bank initialization following the documented workflow.

This module implements the training recipe outlined in
``globe_bank_initialization_training_workflow.md``.  The main entry
point is :func:`initialize_banks` which performs:

1. Per-family Z-score normalization of expert weights.
2. Truncated SVD warm start producing per-expert factors ``A_i`` and
   ``B_i``.
3. Initial bank creation from ``B_i`` centroids with non-negative least
   squares for the first set of codes ``alpha_i``.
4. Alternating minimization consisting of a coding step (entmax),
   dictionary update (MOD), adapter refit (OLS) and temperature
   annealing to target a median support size.

The resulting bank atoms, expert codes and adapter matrices are
returned alongside normalization statistics and simple metrics.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from sklearn.cluster import KMeans
from scipy.optimize import nnls

try:  # entmax may not be available in minimal environments
    from entmax import entmax15
except Exception:  # pragma: no cover - fallback
    entmax15 = None

from .zscore import normalize_expert_families, ZScoreNormalizer


@dataclass
class InitConfig:
    """Configuration for bank initialization."""

    rank: int
    num_bases: int
    steps: int = 25
    temperature: float = 1.0
    target_support: int = 12
    eta: float = 0.05
    epsilon: float = 1e-6
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float32


class BankInitializer:
    """High level interface for learning global basis banks."""

    def __init__(self, cfg: InitConfig):
        self.cfg = cfg
        self.device = cfg.device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.dtype = cfg.dtype

    # ------------------------------------------------------------------
    def _warm_start(self, weights: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Truncated SVD per expert.

        Args:
            weights: ``E × p × d`` tensor of expert weights.

        Returns:
            ``A0`` (``E × p × r``), ``B0`` (``E × r × d``), ``energy`` (``E``)
            where ``r`` is ``cfg.rank``.
        """

        r = self.cfg.rank
        A_list: List[Tensor] = []
        B_list: List[Tensor] = []
        energy: List[float] = []
        for W in weights:
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            energy.append(float((S[:r] ** 2).sum() / (S ** 2).sum()))
            A_list.append(U[:, :r] * S[:r])  # p × r
            B_list.append(Vh[:r, :])  # r × d
        return (
            torch.stack(A_list, dim=0),
            torch.stack(B_list, dim=0),
            torch.tensor(energy, device=self.device, dtype=self.dtype),
        )

    # ------------------------------------------------------------------
    def _initial_bank(self, B0: Tensor) -> Tuple[Tensor, Tensor]:
        """Build initial bank from centroids and NNLS codes."""

        E, r, d = B0.shape
        flat = B0.view(E, -1).cpu().numpy()
        kmeans = KMeans(n_clusters=self.cfg.num_bases, n_init=10).fit(flat)
        bank = (
            torch.from_numpy(kmeans.cluster_centers_)
            .to(self.device, self.dtype)
            .view(self.cfg.num_bases, r, d)
        )
        bank_flat = bank.view(self.cfg.num_bases, -1).T.cpu().numpy()
        codes = []
        for i in range(E):
            coeffs, _ = nnls(bank_flat, flat[i])
            codes.append(coeffs)
        alpha0 = torch.tensor(codes, device=self.device, dtype=self.dtype)
        return bank, alpha0

    # ------------------------------------------------------------------
    def _alternating_minimization(
        self, W: Tensor, B_targets: Tensor, bank: Tensor, alpha0: Tensor, A0: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Dict[str, float]]:
        """Run alternating minimization loop."""

        cfg = self.cfg
        alpha_logits = alpha0.clamp_min(cfg.epsilon).log()
        bank = bank.clone()
        A = A0.clone()
        T = cfg.temperature
        metrics = {"loss": []}

        for _ in range(cfg.steps):
            # Coding step --------------------------------------------------
            if entmax15 is not None:
                alpha = entmax15(alpha_logits / T, dim=-1)
            else:  # pragma: no cover - fallback to softmax
                alpha = F.softmax(alpha_logits / T, dim=-1)
            alpha = torch.where(alpha > cfg.epsilon, alpha, torch.zeros_like(alpha))

            # Dictionary step (MOD) ---------------------------------------
            X = B_targets.view(B_targets.shape[0], -1).T  # RD × E
            A_mat = alpha.T  # m × E
            AtA = A_mat @ A_mat.T
            reg = cfg.epsilon * torch.eye(AtA.shape[0], device=self.device, dtype=self.dtype)
            B_flat = X @ A_mat.T @ torch.linalg.inv(AtA + reg)  # RD × m
            bank = B_flat.T.view(cfg.num_bases, cfg.rank, B_targets.shape[-1])
            norms = bank.view(cfg.num_bases, -1).norm(dim=-1, keepdim=True).clamp_min(cfg.epsilon)
            bank = bank / norms.view(cfg.num_bases, 1, 1)
            alpha = alpha * norms.squeeze(-1)

            # Adapter refit -----------------------------------------------
            mixed = torch.einsum("em,mrd->erd", alpha, bank)
            new_A = []
            for i in range(W.shape[0]):
                Bi = mixed[i]
                pinv = torch.linalg.pinv(Bi)
                new_A.append(W[i] @ pinv)
            A = torch.stack(new_A, dim=0)

            # Reconstruction loss ----------------------------------------
            recon = torch.einsum("epr,erd->epd", A, mixed)
            loss = F.mse_loss(recon, W)
            metrics["loss"].append(float(loss.detach()))

            # Temperature annealing --------------------------------------
            support = (alpha > cfg.epsilon).sum(dim=-1).float()
            med = support.median().item()
            T = T * math.exp(cfg.eta * (med / cfg.target_support - 1.0))

        metrics["median_support"] = med
        metrics["final_temperature"] = T
        return bank, alpha, A, metrics

    # ------------------------------------------------------------------
    def initialize(self, expert_weights: Dict[str, List[Tensor]]):
        """Execute the full initialization workflow."""

        normalized, normalizer = normalize_expert_families(expert_weights)
        results: Dict[str, Dict[str, Tensor]] = {}
        all_metrics: Dict[str, Dict[str, float]] = {}

        for family, weights in normalized.items():
            if not weights:
                continue
            W = torch.stack(weights).to(self.device, self.dtype)
            A0, B0, energy = self._warm_start(W)
            bank, alpha0 = self._initial_bank(B0)
            bank, alpha, A, metrics = self._alternating_minimization(W, B0, bank, alpha0, A0)
            metrics["energy_captured_mean"] = float(energy.mean())
            results[family] = {"bank": bank, "codes": alpha, "adapters": A}
            all_metrics[family] = metrics

        return {"results": results, "normalizer": normalizer, "metrics": all_metrics}


def initialize_banks(
    expert_weights: Dict[str, List[Tensor]], cfg: InitConfig
) -> Tuple[Dict[str, Dict[str, Tensor]], ZScoreNormalizer, Dict[str, Dict[str, float]]]:
    """Convenience function wrapping :class:`BankInitializer`."""

    initializer = BankInitializer(cfg)
    out = initializer.initialize(expert_weights)
    return out["results"], out["normalizer"], out["metrics"]
