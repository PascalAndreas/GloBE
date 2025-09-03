"""Low level building blocks for global basis bank initialization.

This module provides small, pure PyTorch utilities used during bank
initialization.  High level orchestration and Hydra/CLI handling lives in
``globe.train.fit_banks``.  The functions here implement the math from
``globe_bank_initialization_training_workflow.md`` without any external
dependencies.  They are written to be easily testable and run on any
``torch`` device (CPU/MPS/CUDA).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Optional
import math

import torch
from torch import Tensor
import torch.nn.functional as F

try:  # entmax is optional
    from entmax import entmax15
except Exception:  # pragma: no cover - fallback
    entmax15 = None

from globe.modules.globe_bank import GloBEBank
from .zscore import normalize_expert_families, ZScoreNormalizer


# ---------------------------------------------------------------------------
# Configuration -----------------------------------------------------------------


@dataclass
class InitConfig:
    """Configuration for bank initialization primitives."""

    # Core dimensions
    rank: int  # r - inner dimension
    num_bases: int  # m - number of dictionary atoms

    # Training hyper-parameters
    steps: int = 25
    temperature: float = 1.0
    target_support: int = 12
    eta: float = 0.05  # temperature annealing rate
    epsilon: float = 1e-6

    # Device and precision
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float32


def _resolve_device(cfg: InitConfig) -> torch.device:
    if cfg.device is not None:
        return cfg.device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Warm start utilities ------------------------------------------------------


def truncated_svd(weights: Tensor, rank: int) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute per-expert truncated SVD warm starts.

    Args:
        weights: ``E × p × d`` tensor of expert weights.
        rank: target rank ``r``.

    Returns:
        Tuple ``(A0, B0, energy)`` where ``A0`` is ``E × p × r`` and ``B0`` is
        ``E × r × d``.
    """

    device = weights.device
    A_list: List[Tensor] = []
    B_list: List[Tensor] = []
    energy: List[float] = []
    for W in weights:
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        energy.append(float((S[:rank] ** 2).sum() / (S ** 2).sum()))
        A_list.append(U[:, :rank] * S[:rank])
        B_list.append(Vh[:rank, :])
    return (
        torch.stack(A_list, dim=0).to(device),
        torch.stack(B_list, dim=0).to(device),
        torch.tensor(energy, device=device),
    )


# ---------------------------------------------------------------------------
# Initial bank --------------------------------------------------------------


def _kmeans_pytorch(
    data: Tensor, num_clusters: int, iters: int = 25
) -> Tensor:
    """Very small K-means implementation using PyTorch."""

    E, D = data.shape
    # Randomly pick initial centroids without replacement
    perm = torch.randperm(E)[:num_clusters]
    centroids = data[perm].clone()

    for _ in range(iters):
        dists = torch.cdist(data, centroids)
        assignments = dists.argmin(dim=1)
        for k in range(num_clusters):
            mask = assignments == k
            if mask.any():
                centroids[k] = data[mask].mean(dim=0)

    return centroids


def build_initial_bank(B0: Tensor, cfg: InitConfig) -> Tuple[Tensor, Tensor]:
    """Create an initial bank and non-negative codes.

    This replaces the earlier SciPy/Sklearn based implementation with a
    fully PyTorch approach.  NNLS is approximated by solving the least
    squares system and clamping to the non-negative orthant.
    """

    device = _resolve_device(cfg)
    E, r, d = B0.shape
    flat = B0.view(E, -1)

    centroids = _kmeans_pytorch(flat, cfg.num_bases, iters=10)
    bank = centroids.view(cfg.num_bases, r, d).to(device, cfg.dtype)

    # Solve NNLS approximately using least squares + clamp
    bank_flat = bank.view(cfg.num_bases, -1).T  # RD × m
    # (m, E)
    codes = torch.linalg.lstsq(bank_flat, flat.T).solution.T
    codes = codes.clamp_min(0.0)
    return bank, codes.to(device, cfg.dtype)


# ---------------------------------------------------------------------------
# Alternating minimization --------------------------------------------------


def am_step(
    W: Tensor,
    B_targets: Tensor,
    bank_module: GloBEBank,
    alpha_logits: Tensor,
    A: Tensor,
    T: float,
    cfg: InitConfig,
) -> Tuple[Tensor, Tensor, float, float, float]:
    """Perform a single alternating minimization iteration.

    Args:
        W: ``E × p × d`` expert weight tensor.
        B_targets: ``E × r × d`` target bases from SVD.
        bank_module: ``GloBEBank`` instance holding learnable bases.
        alpha_logits: Logits parameterising mixture weights.
        A: Current adapters ``E × p × r``.
        T: Current temperature.
        cfg: Initialisation configuration.

    Returns:
        Tuple ``(alpha_logits, A, loss, T, median_support)`` with updated
        mixture logits, adapters, reconstruction loss, temperature and median
        support size.
    """

    # Coding step ---------------------------------------------------------
    if entmax15 is not None:
        alpha = entmax15(alpha_logits / T, dim=-1)
    else:  # pragma: no cover - fallback
        alpha = F.softmax(alpha_logits / T, dim=-1)
    alpha = torch.where(alpha > cfg.epsilon, alpha, torch.zeros_like(alpha))

    # Dictionary step (MOD) ----------------------------------------------
    X = B_targets.view(B_targets.shape[0], -1).T  # RD × E
    A_mat = alpha.T  # m × E
    AtA = A_mat @ A_mat.T
    reg = cfg.epsilon * torch.eye(AtA.shape[0], device=bank_module.bases.device, dtype=bank_module.bases.dtype)
    B_flat = X @ A_mat.T @ torch.linalg.inv(AtA + reg)  # RD × m
    new_bank = B_flat.T.view(cfg.num_bases, cfg.rank, B_targets.shape[-1])
    norms = new_bank.view(cfg.num_bases, -1).norm(dim=-1, keepdim=True).clamp_min(cfg.epsilon)
    new_bank = new_bank / norms.view(cfg.num_bases, 1, 1)
    alpha = alpha * norms.squeeze(-1)
    bank_module.bases.data.copy_(new_bank)

    # Adapter refit ------------------------------------------------------
    mixed = bank_module(alpha)
    new_A = []
    for i in range(W.shape[0]):
        Bi = mixed[i]
        pinv = torch.linalg.pinv(Bi)
        new_A.append(W[i] @ pinv)
    A = torch.stack(new_A, dim=0)

    # Reconstruction loss -------------------------------------------------
    recon = torch.einsum("epr,erd->epd", A, mixed)
    loss = F.mse_loss(recon, W)

    # Temperature update --------------------------------------------------
    support = (alpha > cfg.epsilon).sum(dim=-1).float()
    med = support.median().item()
    T = T * math.exp(cfg.eta * (med / cfg.target_support - 1.0))

    return alpha.log().detach(), A, float(loss.detach()), T, med


# ---------------------------------------------------------------------------
# Convenience API -----------------------------------------------------------


def initialize_banks(
    expert_weights: Dict[str, List[Tensor]], cfg: InitConfig
) -> Tuple[Dict[str, Dict[str, Tensor]], ZScoreNormalizer, Dict[str, Dict[str, float]]]:
    """End-to-end initialization using the primitives above.

    This function is retained for backward compatibility and testing.  It
    performs the complete workflow without any logging or CLI logic.
    """

    device = _resolve_device(cfg)
    dtype = cfg.dtype

    normalized, normalizer = normalize_expert_families(expert_weights)
    results: Dict[str, Dict[str, Tensor]] = {}
    metrics: Dict[str, Dict[str, float]] = {}

    for family, weights in normalized.items():
        if not weights:
            continue
        W = torch.stack(weights).to(device, dtype)
        A0, B0, energy = truncated_svd(W, cfg.rank)
        bank, alpha0 = build_initial_bank(B0, cfg)
        bank_module = GloBEBank(cfg.num_bases, cfg.rank)
        bank_module.bases.data.copy_(bank)
        alpha_logits = alpha0.clamp_min(cfg.epsilon).log()
        A = A0.clone()
        T = cfg.temperature
        loss = 0.0
        for _ in range(cfg.steps):
            alpha_logits, A, loss, T, _ = am_step(W, B0, bank_module, alpha_logits, A, T, cfg)
        alpha = (
            entmax15(alpha_logits / T, dim=-1)
            if entmax15 is not None
            else F.softmax(alpha_logits / T, dim=-1)
        )
        results[family] = {"bank": bank_module.bases.detach(), "codes": alpha, "adapters": A}
        metrics[family] = {
            "loss": float(loss),
            "energy_captured_mean": float(energy.mean()),
            "final_temperature": float(T),
        }

    return results, normalizer, metrics


# End of file

