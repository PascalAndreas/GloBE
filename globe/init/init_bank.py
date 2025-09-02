"""Bank initialization via alternating minimization.

This module implements the workflow described in
`globe_bank_initialization_training_workflow.md`.  It provides a
`BankInitializer` that performs the following high level steps for each
expert family (Up/Gate):

1. **Per-family z-score normalization** of expert weights with mean/std
   recording so the original scale can be restored later.
2. **Truncated SVD warm start** producing per-expert adapter matrices
   ``A_i`` and low-rank targets ``B_i``.
3. **Initial bank construction** from centroids of ``B_i`` followed by an
   NNLS solve to obtain initial mixture codes ``α_i``.
4. **Alternating minimization** consisting of a coding step (entmax based
   sparse mapping), a dictionary update using MOD, adapter refitting via
   ordinary least squares and temperature annealing to target a median
   support size.
5. **Metric collection** and folding the normalization constants back
   into the adapter matrices for export.

The implementation is intentionally lightweight – it aims to provide a
reference implementation that matches the training plan and can be
extended or optimized further in the future.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from entmax import entmax15
from scipy.optimize import nnls
from sklearn.cluster import KMeans

from .zscore import ZScoreNormalizer


@dataclass
class BankInitResult:
    """Container for results of bank initialization for one family."""

    bank: Tensor  # (m, r*d)
    codes: Tensor  # (E, m)
    adapters: Tensor  # (E, p, r)
    normalizer_stats: Dict[str, float]
    metrics: Dict[str, float]


class BankInitializer:
    """Alternating minimization based bank initializer."""

    def __init__(
        self,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        activation=F.silu,
        seed: int = 0,
        eps: float = 1e-8,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = dtype
        self.activation = activation
        self.seed = seed
        self.eps = eps

        torch.manual_seed(seed)

    # ------------------------------------------------------------------
    # SVD warm start utilities
    # ------------------------------------------------------------------
    def _svd_warm_start(
        self, weights: List[Tensor], rank: int
    ) -> (Tensor, Tensor, List[float]):
        """Compute truncated SVD for each expert weight.

        Args:
            weights: list of ``p × d`` weight matrices ``W_i``
            rank: target rank ``r``

        Returns:
            ``adapters``: stack of ``A_i`` with shape ``(E, p, r)``
            ``targets``: stack of ``B_i`` with shape ``(E, r, d)``
            ``energy``: percentage of Frobenius energy captured per expert
        """

        adapters: List[Tensor] = []
        targets: List[Tensor] = []
        energy: List[float] = []

        for W in weights:
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            U_r = U[:, :rank]
            S_r = S[:rank]
            V_r = Vh[:rank, :]
            adapters.append(U_r * S_r)  # A_i^(0) = U_r Σ_r
            targets.append(V_r)  # B_i^(0) = V_r^T
            energy.append(float((S_r**2).sum() / (S**2).sum()))

        return (
            torch.stack(adapters, dim=0),
            torch.stack(targets, dim=0),
            energy,
        )

    # ------------------------------------------------------------------
    # Initial bank and codes
    # ------------------------------------------------------------------
    def _init_bank_from_centroids(
        self, targets: Tensor, bank_size: int
    ) -> Tensor:
        """Create initial bank atoms from k-means centroids."""

        num_experts, r, d = targets.shape
        flat = targets.view(num_experts, -1).cpu().numpy()
        kmeans = KMeans(n_clusters=bank_size, n_init=10, random_state=self.seed)
        centers = kmeans.fit(flat).cluster_centers_  # (m, r*d)
        bank = torch.from_numpy(centers).to(self.device, self.dtype)
        return bank

    def _nnls_codes(self, targets: Tensor, bank: Tensor) -> Tensor:
        """Solve NNLS for initial mixture codes."""

        num_experts = targets.shape[0]
        flat_targets = targets.view(num_experts, -1)
        bank_np = bank.cpu().numpy()

        codes: List[Tensor] = []
        for i in range(num_experts):
            y = flat_targets[i].cpu().numpy()
            sol, _ = nnls(bank_np, y)
            alpha = torch.from_numpy(sol).to(self.device, self.dtype)
            alpha = alpha / (alpha.sum() + self.eps)
            codes.append(alpha)

        return torch.stack(codes, dim=0)

    # ------------------------------------------------------------------
    # Adapter refitting
    # ------------------------------------------------------------------
    def _ols_adapter(self, W: Tensor, mixture: Tensor) -> Tensor:
        """Refit adapter ``A`` given original weights and mixed basis."""

        # mixture: (r × d), W: (p × d)
        lstsq = torch.linalg.lstsq(mixture.T, W.T)
        return lstsq.solution.T  # (p × r)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def initialize(
        self,
        expert_weights: Dict[str, List[Tensor]],
        bank_size: int,
        rank: int,
        steps: int = 50,
        target_support: int = 12,
        temp_init: float = 1.0,
        temp_eta: float = 0.05,
        temp_update_every: int = 10,
    ) -> Dict[str, BankInitResult]:
        """Run the bank initialization workflow.

        Args:
            expert_weights: mapping from family name to expert weight list
            bank_size: number of dictionary atoms ``m``
            rank: inner dimension ``r`` used for the warm start
            steps: number of alternating minimization iterations
            target_support: desired median support size of ``α_i``
            temp_init: initial temperature ``T``
            temp_eta: update rate for temperature annealing
            temp_update_every: how often to update temperature

        Returns:
            Dictionary mapping family name to :class:`BankInitResult`
        """

        normalizer = ZScoreNormalizer()
        norm_weights = normalizer.fit_transform(expert_weights)

        results: Dict[str, BankInitResult] = {}

        for family, weights in norm_weights.items():
            if not weights:
                continue

            weights = [w.to(self.device, self.dtype) for w in weights]
            num_experts, p, d = len(weights), weights[0].shape[0], weights[0].shape[1]

            adapters, targets, energy = self._svd_warm_start(weights, rank)
            flat_targets = targets.view(num_experts, -1)

            bank = self._init_bank_from_centroids(targets, bank_size)
            codes = self._nnls_codes(targets, bank)

            logits = torch.log(codes + self.eps).to(self.device)
            logits.requires_grad_(True)
            optimizer = torch.optim.Adam([logits], lr=0.1)
            temperature = temp_init

            for step in range(steps):
                optimizer.zero_grad()
                coeffs = entmax15(logits / temperature, dim=-1)
                recon = coeffs @ bank  # E × (r*d)
                loss = F.mse_loss(recon, flat_targets)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    coeffs = entmax15(logits / temperature, dim=-1)

                    # Dictionary step (MOD)
                    A = coeffs  # E × m
                    X = flat_targets  # E × (r*d)
                    gram = A.T @ A + self.eps * torch.eye(bank_size, device=self.device)
                    bank = torch.linalg.solve(gram, A.T @ X)  # m × (r*d)

                    # Normalize atoms
                    norms = bank.norm(dim=1, keepdim=True).clamp_min(self.eps)
                    bank /= norms

                    # Adapter refit
                    for i in range(num_experts):
                        mixture = coeffs[i] @ bank  # (r*d)
                        mixture = mixture.view(rank, d)
                        mixture = self.activation(mixture)
                        adapters[i] = self._ols_adapter(weights[i], mixture)

                    # Temperature annealing
                    if (step + 1) % temp_update_every == 0:
                        supports = (coeffs > self.eps).sum(dim=1).float()
                        median = supports.median()
                        temperature = float(
                            temperature
                            * torch.exp(temp_eta * (median / target_support - 1))
                        )

            with torch.no_grad():
                coeffs = entmax15(logits / temperature, dim=-1)
                supports = (coeffs > self.eps).sum(dim=1).float()
                coherence = torch.norm(
                    bank @ bank.T - torch.eye(bank_size, device=self.device), p="fro"
                )

                # Fold back normalization: multiply adapters by recorded std
                std = normalizer.stats[family]["global_std"]
                adapters = adapters * std

            metrics = {
                "mean_energy": float(torch.tensor(energy).mean()),
                "median_support": float(supports.median()),
                "atom_coherence": float(coherence),
            }

            results[family] = BankInitResult(
                bank=bank,
                codes=coeffs,
                adapters=adapters,
                normalizer_stats=normalizer.stats[family],
                metrics=metrics,
            )

        return results


__all__ = ["BankInitializer", "BankInitResult"]

