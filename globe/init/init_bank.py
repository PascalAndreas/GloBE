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
from .warm_start import create_warm_start_proxies, SeedingConfig, SeedingMethod


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
    max_temperature: float = 10.0  # Maximum temperature to prevent support explosion
    
    # Regularization hyperparameters (separate from epsilon)
    lambda_B: float = 1e-4  # Bank regularization for MOD step
    lambda_A: float = 1e-6  # Adapter regularization for refit step
    
    # Warm start / seeding configuration
    seeding_config: Optional[SeedingConfig] = None


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


def create_warm_start(weights: Tensor, rank: int, seeding_config: Optional[SeedingConfig] = None) -> Tuple[Tensor, Tensor, Tensor, Dict[str, float]]:
    """Create warm start proxies using configurable seeding method.

    Args:
        weights: ``E × p × d`` tensor of expert weights.
        rank: target rank ``r``.
        seeding_config: Configuration for seeding method. If None, uses default TS-PCA.

    Returns:
        Tuple ``(A0, B0, energy, metrics)`` where ``A0`` is ``E × p × r``, ``B0`` is
        ``E × r × d``, energy is per-expert energy capture, and metrics contains
        seeding method statistics.
    """
    device = weights.device
    original_dtype = weights.dtype
    
    # Use default seeding config if none provided
    if seeding_config is None:
        seeding_config = SeedingConfig(method=SeedingMethod.TS_PCA)
    
    # Create proxies using the specified seeding method
    B0, seeding_metrics = create_warm_start_proxies(weights, rank, seeding_config)
    
    # For compatibility with existing code, we need to create A0 matrices
    # We'll use least squares to fit A_i such that W_i ≈ A_i @ B_i
    E, p, d = weights.shape
    A_list: List[Tensor] = []
    energy: List[float] = []
    
    for i in range(E):
        W_i = weights[i]  # p × d
        B_i = B0[i]      # r × d
        
        # Solve A_i such that W_i ≈ A_i @ B_i using more stable approach
        # W_i: p × d, B_i: r × d, we want A_i: p × r
        # Use normal equations: A_i = W_i @ B_i^T @ (B_i @ B_i^T + λI)^{-1}
        if device.type == "mps":
            # Move to CPU for matrix operations
            B_i_cpu = B_i.cpu().float()  # r × d
            W_i_cpu = W_i.cpu().float()  # p × d
            
            # Normal equations approach for better conditioning
            G = B_i_cpu @ B_i_cpu.T + 1e-6 * torch.eye(B_i_cpu.shape[0])  # r × r
            R = W_i_cpu @ B_i_cpu.T  # p × r
            try:
                L = torch.linalg.cholesky(G)
                A_i = torch.cholesky_solve(R.T, L).T.to(device, original_dtype)
            except RuntimeError:
                # Fallback to general solve if Cholesky fails
                A_i = torch.linalg.solve(G, R.T).T.to(device, original_dtype)
        else:
            B_i_f32 = B_i.float()  # r × d
            W_i_f32 = W_i.float()  # p × d
            
            # Normal equations approach for better conditioning
            G = B_i_f32 @ B_i_f32.T + 1e-6 * torch.eye(B_i_f32.shape[0], device=device)  # r × r
            R = W_i_f32 @ B_i_f32.T  # p × r
            try:
                L = torch.linalg.cholesky(G)
                A_i = torch.cholesky_solve(R.T, L).T.to(original_dtype)
            except RuntimeError:
                # Fallback to general solve if Cholesky fails
                A_i = torch.linalg.solve(G, R.T).T.to(original_dtype)
        
        A_list.append(A_i)
        
        # Compute energy captured (reconstruction quality)
        recon = A_i @ B_i
        mse = F.mse_loss(recon, W_i)
        w_norm_sq = (W_i ** 2).sum()
        energy_captured = 1.0 - (mse * W_i.numel()) / w_norm_sq.clamp_min(1e-8)
        energy.append(float(energy_captured))
    
    A0 = torch.stack(A_list, dim=0)  # E × p × r
    energy_tensor = torch.tensor(energy, device=device)
    
    return A0, B0, energy_tensor, seeding_metrics




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
    alpha: Tensor,
    A: Tensor,
    T: float,
    cfg: InitConfig,
    step: int = 0,
    log_metrics: bool = False,
) -> Tuple[Tensor, Tensor, float, float, float, Dict[str, float]]:
    """Perform a single alternating minimization iteration.

    Args:
        W: ``E × p × d`` expert weight tensor.
        B_targets: ``E × r × d`` target bases from warm start.
        bank_module: ``GloBEBank`` instance holding learnable bases.
        alpha: Current mixture weights (not logits).
        A: Current adapters ``E × p × r``.
        T: Current temperature.
        cfg: Initialisation configuration.

    Returns:
        Tuple ``(alpha, A, loss, T, median_support, metrics)`` with updated
        mixture weights, adapters, reconstruction loss, temperature and median
        support size.
    """

    # Coding step ---------------------------------------------------------
    # Apply temperature scaling and sparsity activation to current alpha
    if T != 1.0:
        # For temperature scaling, we need to work in log space temporarily
        alpha_safe = torch.clamp(alpha, min=cfg.epsilon)
        temp_logits = alpha_safe.log() / T
        if entmax15 is not None:
            alpha = entmax15(temp_logits, dim=-1)
        else:  # pragma: no cover - fallback
            alpha = F.softmax(temp_logits, dim=-1)
    # If T=1.0, alpha stays as-is (no temperature scaling needed)
    
    # Apply epsilon pruning more conservatively
    alpha = torch.where(alpha > cfg.epsilon, alpha, torch.zeros_like(alpha))
    
    # Renormalize after pruning to maintain valid probability distribution
    alpha_sum = alpha.sum(dim=-1, keepdim=True).clamp_min(cfg.epsilon)
    alpha = alpha / alpha_sum

    # Dictionary step (MOD) ----------------------------------------------
    device = bank_module.bases.device
    # Ensure all inputs are on the correct device
    alpha = alpha.to(device)
    X = B_targets.view(B_targets.shape[0], -1).T.to(device)  # RD × E
    A_mat = alpha.T  # m × E
    AtA = A_mat @ A_mat.T  # m × m
    reg = cfg.lambda_B * torch.eye(AtA.shape[0], device=device, dtype=bank_module.bases.dtype)
    
    # Use solve instead of inverse for better numerical stability
    # Solve (AtA + reg) @ B_flat.T = (A_mat @ X.T).T for B_flat
    if device.type == "mps":
        X_cpu = X.cpu().float()  # RD × E
        A_mat_cpu = A_mat.cpu().float()  # m × E
        AtA_cpu = AtA.cpu().float()  # m × m
        reg_cpu = reg.cpu().float()  # m × m
        rhs_cpu = (A_mat_cpu @ X_cpu.T).T  # RD × m
        B_flat_cpu = torch.linalg.solve(AtA_cpu + reg_cpu, rhs_cpu.T).T  # RD × m
        B_flat = B_flat_cpu.to(device, bank_module.bases.dtype)
    else:
        # Ensure all tensors are on the same device and float32 for matrix operations
        X_f32 = X.to(device).float()  # RD × E
        A_mat_f32 = A_mat.to(device).float()  # m × E
        AtA_f32 = AtA.to(device).float()  # m × m
        reg_f32 = reg.to(device).float()  # m × m
        rhs_f32 = (A_mat_f32 @ X_f32.T).T  # RD × m
        B_flat_f32 = torch.linalg.solve(AtA_f32 + reg_f32, rhs_f32.T).T  # RD × m
        B_flat = B_flat_f32.to(bank_module.bases.dtype)
    new_bank = B_flat.T.view(cfg.num_bases, cfg.rank, B_targets.shape[-1])
    norms = new_bank.view(cfg.num_bases, -1).norm(dim=-1, keepdim=True).clamp_min(1e-8)
    new_bank = new_bank / norms.view(cfg.num_bases, 1, 1)
    # Rescale alpha by atom norms (both already on same device)
    alpha = alpha * norms.squeeze(-1)
    
    # Safer parameter update using torch.no_grad()
    with torch.no_grad():
        bank_module.bases.copy_(new_bank)

    # Batched adapter refit using normal equations + Cholesky -----------
    # Make the nonlinearity explicit: first form the linear mixture then apply
    # the bank's activation.  ``alpha`` is ``E×m`` and bases ``m×r×d`` ->
    # ``E×r×d``.
    mixed_linear = torch.einsum('em,mrd->erd', alpha, bank_module.bases)
    mixed = bank_module.activation(mixed_linear)
    
    # Ensure W and A are on the correct device
    W = W.to(device)
    A = A.to(device)
    
    # Solve A_i @ B_i = W_i for A_i using normal equations
    # A_i = W_i @ B_i^T @ (B_i @ B_i^T + lambda_A * I)^{-1}
    E, r, d = mixed.shape
    p = W.shape[1]
    
    # Compute Gram matrices: G_i = B_i @ B_i^T for all experts (E × r × r)
    G = torch.einsum('erd,esd->ers', mixed, mixed)  # E × r × r
    
    # Compute cross-terms: R_i = W_i @ B_i^T for all experts (E × p × r)  
    R = torch.einsum('epd,erd->epr', W, mixed)  # E × p × r
    
    # Add regularization
    reg_A = cfg.lambda_A * torch.eye(r, device=mixed.device, dtype=mixed.dtype)
    G = G + reg_A.unsqueeze(0)  # E × r × r
    
    # Batched solve using Cholesky (more stable than general solve for PD matrices)
    if mixed.device.type == "mps":
        # Move to CPU for batched operations
        G_cpu = G.cpu().float()  # E × r × r
        R_cpu = R.cpu().float()  # E × p × r
        try:
            L = torch.linalg.cholesky(G_cpu)  # E × r × r
            A_cpu = torch.cholesky_solve(R_cpu.transpose(-1, -2), L).transpose(-1, -2)  # E × p × r
            A = A_cpu.to(mixed.device, mixed.dtype)
        except RuntimeError:
            # Fallback to general solve if Cholesky fails
            A_cpu = torch.linalg.solve(G_cpu, R_cpu.transpose(-1, -2)).transpose(-1, -2)
            A = A_cpu.to(mixed.device, mixed.dtype)
    else:
        G_f32 = G.float()  # E × r × r
        R_f32 = R.float()  # E × p × r
        try:
            L = torch.linalg.cholesky(G_f32)  # E × r × r
            A_f32 = torch.cholesky_solve(R_f32.transpose(-1, -2), L).transpose(-1, -2)  # E × p × r
            A = A_f32.to(mixed.dtype)
        except RuntimeError:
            # Fallback to general solve if Cholesky fails
            A_f32 = torch.linalg.solve(G_f32, R_f32.transpose(-1, -2)).transpose(-1, -2)
            A = A_f32.to(mixed.dtype)

    # Reconstruction loss -------------------------------------------------
    recon = torch.einsum("epr,erd->epd", A, mixed)
    diff = recon - W
    loss = (diff ** 2).mean()

    # Temperature update --------------------------------------------------
    support = (alpha > cfg.epsilon).sum(dim=-1).float()
    med = support.median().item()
    
    # More conservative temperature annealing with both min and max bounds
    T_new = T * math.exp(cfg.eta * (med / cfg.target_support - 1.0))
    T = max(min(T_new, cfg.max_temperature), cfg.min_temperature)

    # Collect comprehensive metrics (only when needed)
    recon_error_norm = diff.norm().item()
    W_frobenius_norm = W.norm().item()
    relative_error = (
        recon_error_norm / W_frobenius_norm if W_frobenius_norm > 0 else recon_error_norm
    )

    metrics = {'recon/relative_frobenius_error': relative_error}
    if log_metrics:
        recon_mse = loss.item()
        metrics['loss/total'] = recon_mse

        # B. Coding step (α: entmax/sparsemax) - Keep on device for speed
        support_sizes = (alpha > cfg.epsilon).sum(dim=-1).float()
        # Better entropy calculation avoiding small bias
        entropy_vals = -torch.where(alpha > 0, alpha * alpha.log(), torch.zeros_like(alpha)).sum(dim=-1)
        pruned_frac = (alpha < cfg.epsilon).float().mean().item()

        metrics.update({
            'alpha/median_support': support_sizes.median().item(),
            'alpha/mean_support': support_sizes.mean().item(),
            'alpha/p95_support': support_sizes.quantile(0.95).item(),
            'alpha/entropy_mean': entropy_vals.mean().item(),
            'alpha/epsilon_prune_frac': pruned_frac,
            'alpha/temp': T,
            'alpha/target_support': cfg.target_support,
            'alpha/support_error': support_sizes.median().item() - cfg.target_support,
        })
        
        # C. Dictionary step (bank B: MOD update) - Simplified for speed
        bank_data = bank_module.bases.detach()
        bank_flat = bank_data.view(bank_data.shape[0], -1)
        atom_norms = bank_flat.norm(dim=-1)
        
        # Skip expensive coherence matrix computation - only compute norms
        metrics.update({
            'bank/atom_norm_mean': atom_norms.mean().item(),
            'bank/atom_norm_max_dev': (atom_norms - 1.0).abs().max().item(),
        })
        
        # D. A refit stability - Simplified for speed (skip expensive condition numbers)
        recon_error = torch.norm(recon - W, dim=(1, 2)) / torch.norm(W, dim=(1, 2))
        metrics.update({
            'A/refit_resid_rel_mean': recon_error.mean().item(),
        })
        
        # E. Health checks
        nan_infs = 0
        for tensor in [alpha, bank_module.bases, A]:
            nan_infs += torch.isnan(tensor).sum().item()
            nan_infs += torch.isinf(tensor).sum().item()
        
        metrics.update({
            'health/nan_infs': nan_infs,
        })

    return alpha.detach(), A, float(loss.detach()), T, med, metrics


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
        
        # Use warm start method
        A0, B0, energy, seeding_metrics = create_warm_start(W, cfg.rank, cfg.seeding_config)
        
        bank, alpha0 = build_initial_bank(B0, cfg)
        # Extract hidden_dim from the B0 tensor shape
        _, _, hidden_dim = B0.shape
        bank_module = GloBEBank(cfg.num_bases, cfg.rank, hidden_dim)
        with torch.no_grad():
            bank_module.bases.copy_(bank)
        alpha = alpha0.clone()  # Use alpha directly, not logits
        A = A0.clone()
        T = cfg.temperature
        loss = 0.0
        for step in range(cfg.steps):
            alpha, A, loss, T, _, _ = am_step(W, B0, bank_module, alpha, A, T, cfg, step=step, log_metrics=False)
        results[family] = {"bank": bank_module.bases.detach(), "codes": alpha, "adapters": A}
        
        # Combine metrics
        family_metrics = {
            "loss": float(loss),
            "energy_captured_mean": float(energy.mean()),
            "final_temperature": float(T),
        }
        family_metrics.update({f"seeding_{k}": v for k, v in seeding_metrics.items()})
        metrics[family] = family_metrics

    return results, normalizer, metrics


# End of file

