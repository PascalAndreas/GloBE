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
    temperature: float = 3.0  # Higher initial temperature for stability
    target_support: int = 12
    eta: float = 0.01  # Conservative temperature annealing rate
    epsilon: float = 1e-4  # Higher threshold for less aggressive pruning

    # Device and precision
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float32
    
    # Additional AM loop parameters
    ridge_lambda: float = 1e-6  # Ridge regularization for MOD step
    temp_adjust_freq: int = 1  # How often to adjust temperature
    min_temperature: float = 0.5  # Higher minimum temperature to prevent collapse


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
    original_dtype = weights.dtype
    A_list: List[Tensor] = []
    B_list: List[Tensor] = []
    energy: List[float] = []
    for W in weights:
        # Convert to float32 for SVD operations (required for CPU fallback on MPS)
        W_float32 = W.float()
        U, S, Vh = torch.linalg.svd(W_float32, full_matrices=False)
        energy.append(float((S[:rank] ** 2).sum() / (S ** 2).sum()))
        A_list.append((U[:, :rank] * S[:rank]).to(original_dtype))  # Convert back to original dtype
        B_list.append(Vh[:rank, :].to(original_dtype))  # Convert back to original dtype
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
    
    # Handle case where we want more clusters than data points
    if num_clusters > E:
        # Initialize with data + random perturbations
        centroids = []
        for i in range(num_clusters):
            if i < E:
                # Use actual data points
                centroids.append(data[i].clone())
            else:
                # Create perturbations of existing points
                base_idx = i % E
                noise = torch.randn_like(data[base_idx]) * 0.1
                centroids.append(data[base_idx] + noise)
        centroids = torch.stack(centroids)
    else:
        # Standard case: pick initial centroids without replacement
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

    centroids = _kmeans_pytorch(flat, cfg.num_bases, iters=10)  # num_bases × (r*d)
    bank = centroids.view(cfg.num_bases, r, d).to(device, cfg.dtype)  # num_bases × r × d

    # Solve NNLS approximately using least squares + clamp
    bank_flat = bank.view(cfg.num_bases, -1).T  # RD × m
    # (m, E)
    
    # Move to CPU for lstsq if on MPS (not supported), and ensure float32
    if bank_flat.device.type == "mps":
        bank_flat_cpu = bank_flat.cpu().float()  # Ensure float32
        flat_cpu = flat.cpu().float()  # Ensure float32
        codes = torch.linalg.lstsq(bank_flat_cpu, flat_cpu.T).solution.T
        codes = codes.to(device, cfg.dtype)  # Convert back to original dtype
    else:
        # Ensure float32 for lstsq even on other devices
        bank_flat_f32 = bank_flat.float()
        flat_f32 = flat.float()
        codes = torch.linalg.lstsq(bank_flat_f32, flat_f32.T).solution.T
        codes = codes.to(cfg.dtype)  # Convert back to original dtype
    
    codes = codes.clamp_min(cfg.epsilon)  # Use epsilon instead of 0 to avoid log(0)
    # Normalize codes to be proper probability distributions
    codes = codes / codes.sum(dim=-1, keepdim=True)
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
    step: int = 0,
    log_metrics: bool = False,
) -> Tuple[Tensor, Tensor, float, float, float, Dict[str, float]]:
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
    # Use temperature-scaled logits for sparse activation
    if entmax15 is not None:
        alpha = entmax15(alpha_logits / T, dim=-1)
    else:  # pragma: no cover - fallback
        alpha = F.softmax(alpha_logits / T, dim=-1)
    
    # Apply epsilon pruning more conservatively
    alpha = torch.where(alpha > cfg.epsilon, alpha, torch.zeros_like(alpha))
    
    # Renormalize after pruning to maintain valid probability distribution
    alpha_sum = alpha.sum(dim=-1, keepdim=True).clamp_min(cfg.epsilon)
    alpha = alpha / alpha_sum

    # Dictionary step (MOD) ----------------------------------------------
    X = B_targets.view(B_targets.shape[0], -1).T  # RD × E
    A_mat = alpha.T  # m × E
    AtA = A_mat @ A_mat.T
    reg = cfg.epsilon * torch.eye(AtA.shape[0], device=bank_module.bases.device, dtype=bank_module.bases.dtype)
    
    # Move to CPU for matrix inversion if on MPS (not supported), and ensure float32
    if bank_module.bases.device.type == "mps":
        X_cpu = X.cpu().float()  # Ensure float32
        A_mat_cpu = A_mat.cpu().float()  # Ensure float32
        AtA_cpu = AtA.cpu().float()  # Ensure float32
        reg_cpu = reg.cpu().float()  # Ensure float32
        B_flat_cpu = X_cpu @ A_mat_cpu.T @ torch.linalg.inv(AtA_cpu + reg_cpu)  # RD × m
        B_flat = B_flat_cpu.to(bank_module.bases.device, bank_module.bases.dtype)
    else:
        # Ensure float32 for matrix operations even on other devices
        X_f32 = X.float()
        A_mat_f32 = A_mat.float()
        AtA_f32 = AtA.float()
        reg_f32 = reg.float()
        B_flat_f32 = X_f32 @ A_mat_f32.T @ torch.linalg.inv(AtA_f32 + reg_f32)  # RD × m
        B_flat = B_flat_f32.to(bank_module.bases.dtype)
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
        Wi = W[i]
        
        # Move to CPU for pseudoinverse if on MPS (not supported), and ensure float32
        if Bi.device.type == "mps":
            Bi_cpu = Bi.cpu().float()  # Ensure float32
            Wi_cpu = Wi.cpu().float()  # Ensure float32
            pinv = torch.linalg.pinv(Bi_cpu)
            Ai = (Wi_cpu @ pinv).to(Bi.device, Bi.dtype)
        else:
            # Ensure float32 for pseudoinverse even on other devices
            Bi_f32 = Bi.float()
            Wi_f32 = Wi.float()
            pinv = torch.linalg.pinv(Bi_f32)
            Ai = (Wi_f32 @ pinv).to(Bi.dtype)
        
        new_A.append(Ai)
    A = torch.stack(new_A, dim=0)

    # Reconstruction loss -------------------------------------------------
    recon = torch.einsum("epr,erd->epd", A, mixed)
    loss = F.mse_loss(recon, W)

    # Temperature update --------------------------------------------------
    support = (alpha > cfg.epsilon).sum(dim=-1).float()
    med = support.median().item()
    
    # More conservative temperature annealing with minimum bound
    T_new = T * math.exp(cfg.eta * (med / cfg.target_support - 1.0))
    T = max(T_new, cfg.min_temperature)  # Prevent temperature from getting too low

    # Collect comprehensive metrics
    metrics = {}
    if log_metrics:
        # A. Global reconstruction (weight space)
        recon_mse = loss.item()
        W_norm_sq = (W ** 2).sum().item()
        recon_rmse = recon_mse / W_norm_sq if W_norm_sq > 0 else recon_mse
        metrics.update({
            'recon/rMSE': recon_rmse,
            'recon/MSE': recon_mse,
        })
        
        # B. Coding step (α: entmax/sparsemax)
        alpha_np = alpha.detach().cpu()
        support_sizes = (alpha_np > cfg.epsilon).sum(dim=-1).float()
        entropy_vals = -(alpha_np * (alpha_np + cfg.epsilon).log()).sum(dim=-1)
        pruned_frac = (alpha_np < cfg.epsilon).float().mean().item()
        
        metrics.update({
            'alpha/median_support': support_sizes.median().item(),
            'alpha/mean_support': support_sizes.mean().item(),
            'alpha/p95_support': support_sizes.quantile(0.95).item(),
            'alpha/entropy_mean': entropy_vals.mean().item(),
            'alpha/epsilon_prune_frac': pruned_frac,
            'alpha/temp_T': T,
            'alpha/target_support': cfg.target_support,
            'alpha/support_error': support_sizes.median().item() - cfg.target_support,
        })
        
        # C. Dictionary step (bank B: MOD update)
        bank_data = bank_module.bases.detach().cpu()
        bank_flat = bank_data.view(bank_data.shape[0], -1)
        atom_norms = bank_flat.norm(dim=-1)
        
        # Coherence matrix (normalized atoms)
        bank_normalized = bank_flat / (atom_norms.unsqueeze(-1) + cfg.epsilon)
        coherence_matrix = bank_normalized @ bank_normalized.T
        # Off-diagonal coherence
        mask = ~torch.eye(coherence_matrix.shape[0], dtype=torch.bool)
        offdiag_coherence = coherence_matrix[mask].abs()
        
        metrics.update({
            'bank/atom_norm_mean': atom_norms.mean().item(),
            'bank/atom_norm_max_dev': (atom_norms - 1.0).abs().max().item(),
            'bank/coherence_offdiag_mean': offdiag_coherence.mean().item(),
            'bank/coherence_offdiag_max': offdiag_coherence.max().item(),
        })
        
        # D. A refit stability
        A_cpu = A.detach().cpu()
        mixed_cpu = mixed.detach().cpu()
        W_cpu = W.detach().cpu()
        
        refit_residuals = []
        refit_conds = []
        
        for i in range(W.shape[0]):
            residual = torch.norm(W_cpu[i] - A_cpu[i] @ mixed_cpu[i]) / torch.norm(W_cpu[i])
            refit_residuals.append(residual.item())
            
            # Condition number of mixed basis for this expert
            Bi = mixed_cpu[i]  # r × d
            BBt = Bi @ Bi.T  # r × r
            try:
                cond = torch.linalg.cond(BBt).item()
                refit_conds.append(cond if not torch.isnan(torch.tensor(cond)) else 1e6)
            except:
                refit_conds.append(1e6)
        
        metrics.update({
            'A/refit_resid_rel_mean': sum(refit_residuals) / len(refit_residuals),
            'A/refit_cond_BBt_mean': sum(refit_conds) / len(refit_conds),
            'A/refit_cond_BBt_p95': torch.tensor(refit_conds).quantile(0.95).item(),
        })
        
        # E. Health checks
        nan_infs = 0
        for tensor in [alpha, bank_module.bases, A]:
            nan_infs += torch.isnan(tensor).sum().item()
            nan_infs += torch.isinf(tensor).sum().item()
        
        metrics.update({
            'health/nan_infs': nan_infs,
            'loss/total': recon_mse,
            'loss/recon': recon_mse,
        })
    
    # Ensure no zeros before taking log to avoid -inf
    alpha_safe = torch.clamp(alpha, min=cfg.epsilon)
    return alpha_safe.log().detach(), A, float(loss.detach()), T, med, metrics


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
        # Extract hidden_dim from the B0 tensor shape
        _, _, hidden_dim = B0.shape
        bank_module = GloBEBank(cfg.num_bases, cfg.rank, hidden_dim)
        bank_module.bases.data.copy_(bank)
        alpha_logits = (alpha0 + cfg.epsilon).log()  # Add epsilon before log to avoid -inf
        A = A0.clone()
        T = cfg.temperature
        loss = 0.0
        for step in range(cfg.steps):
            alpha_logits, A, loss, T, _, _ = am_step(W, B0, bank_module, alpha_logits, A, T, cfg, step=step, log_metrics=False)
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

