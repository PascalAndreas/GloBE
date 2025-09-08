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
import time

import torch
from torch import Tensor
import torch.nn.functional as F

from globe.modules.globe_bank import GloBEBank
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
    target_support: int = 12
    epsilon: float = 1e-4  # For support calculation compatibility

    # Device and precision
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float32
    
    # JW-MOD parameters
    tau: float = 1.0  # Softmax temperature (mutable)
    lambda_A: float = 1e-4  # Ridge for A-refit
    lambda_B: float = 1e-4  # Ridge for MOD
    lambda_T: float = 1e-4  # Ridge for pre-act target backsolve
    j_min: float = 1e-3  # Jacobian floor
    eta: float = 0.5  # JW-MOD step damping
    # route_w_start removed - now handled by λ/β scheduling in fit_banks.py
    
    # Temperature control
    temp_control_freq: int = 5  # Update temperature every K steps
    temp_control_eta: float = 0.1  # Temperature adjustment rate
    
    # Activation homotopy (adaptive)
    activation_lambda: float = 0.0  # Current activation strength [0,1]
    lambda_step: float = 0.2  # How much to increase λ per transition (more aggressive)
    lambda_schedule_freq: int = 2  # Evaluate λ increase every K steps (more frequent)
    lambda_backoff_threshold: float = 0.05  # Backoff if relF worsens by this much
    
    # Route blending (adaptive)
    route_beta: float = 0.0  # Route-L to Route-W blending [0,1]
    beta_step: float = 0.2  # How much to increase β per transition
    min_relF_for_route_blend: float = 0.05  # Start route blending when relF < this
    
    # Quality gates for transitions
    min_relF_improvement: float = 1e-4  # Minimum improvement to increase λ
    max_condition_number: float = 1e5  # Don't transition if conditioning is bad (more permissive)
    relF_ema_beta: float = 0.9  # EMA smoothing for relF
    
    # Warm start / seeding configuration
    seeding_config: Optional[SeedingConfig] = None


def update_temperature(cfg: InitConfig, seff_median: float, step: int) -> None:
    """Update temperature for softmax coding based on effective support."""
    if step > 0 and step % cfg.temp_control_freq == 0:
        target_support = cfg.target_support
        # Exponential temperature update
        adjustment = cfg.temp_control_eta * (seff_median / target_support - 1)
        cfg.tau *= math.exp(adjustment)
        # Clamp temperature to reasonable bounds
        cfg.tau = max(0.1, min(10.0, cfg.tau))


def update_adaptive_schedule(cfg: InitConfig, metrics: Dict[str, float], step: int, 
                           relF_history: List[float]) -> Dict[str, str]:
    """Update activation lambda and route beta based on quality metrics."""
    actions = []
    
    # Debug: minimal step info (can be removed in production)
    if step % 4 == 0:  # Only show occasionally to reduce clutter
        actions.append(f"s{step}")
    
    # Update relF EMA
    current_relF = metrics.get('recon/relative_frobenius_error', 1.0)
    if not hasattr(cfg, '_relF_ema'):
        cfg._relF_ema = current_relF
    else:
        cfg._relF_ema = cfg.relF_ema_beta * cfg._relF_ema + (1 - cfg.relF_ema_beta) * current_relF
    
    # Track best relF for backoff detection
    if not hasattr(cfg, '_best_relF'):
        cfg._best_relF = current_relF
    else:
        cfg._best_relF = min(cfg._best_relF, current_relF)
    
    # Check for backoff conditions
    relF_worsened = current_relF > cfg._best_relF * (1 + cfg.lambda_backoff_threshold)
    condition_bad = metrics.get('mod/cond_AAt', 0) > cfg.max_condition_number
    
    if relF_worsened or condition_bad:
        # Backoff lambda
        if cfg.activation_lambda > 0:
            cfg.activation_lambda = max(0.0, 0.5 * cfg.activation_lambda)
            actions.append(f"λ_backoff→{cfg.activation_lambda:.2f}")
        return {"actions": "; ".join(actions)}
    
    # Consider increasing lambda (every lambda_schedule_freq steps)
    # Note: step is 0-indexed, so step 1 in display = step 0 in code
    if (step + 1) % cfg.lambda_schedule_freq == 0 and cfg.activation_lambda < 1.0:
        # Check quality gates - be more permissive for lambda increases
        relF_good = cfg._relF_ema <= 0.5 or len(relF_history) < 3  # Allow lambda increase when relF is reasonable
        if len(relF_history) >= 3:
            recent_improvement = relF_history[-3] - relF_history[-1]
            improving = recent_improvement >= cfg.min_relF_improvement
        else:
            improving = True
            
        condition_ok = metrics.get('mod/cond_AAt', 0) <= cfg.max_condition_number
        drift_ok = metrics.get('bank/drift_rel_mean', 0) < 1e-3
        
        # Be more permissive in early steps when we don't have good drift estimates
        early_steps = step < 5
        if early_steps:
            condition_ok = True  # Ignore conditioning in early steps
            drift_ok = True      # Ignore drift in early steps
        
        if (relF_good or improving) and condition_ok and drift_ok:
            cfg.activation_lambda = min(1.0, cfg.activation_lambda + cfg.lambda_step)
            actions.append(f"λ_increase→{cfg.activation_lambda:.2f}")
        else:
            # Debug: why didn't we increase lambda?
            reasons = []
            if not relF_good: reasons.append("relF_bad")
            if not improving: reasons.append("not_improving")
            if not condition_ok: reasons.append("bad_condition")
            if not drift_ok: reasons.append("bad_drift")
            actions.append(f"λ_skip({','.join(reasons)})")
    
    # Consider increasing route beta (only after lambda is substantial)
    if (cfg.activation_lambda >= 0.5 and cfg.route_beta < 1.0 and 
        cfg._relF_ema < cfg.min_relF_for_route_blend):
        cfg.route_beta = min(1.0, cfg.route_beta + cfg.beta_step)
        actions.append(f"β_increase→{cfg.route_beta:.2f}")
    
    return {"actions": "; ".join(actions) if actions else "stable"}


def _resolve_device(cfg: InitConfig) -> torch.device:
    if cfg.device is not None:
        return cfg.device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def phi_lambda(z: Tensor, lam: float) -> Tensor:
    """Homotopy activation: (1-λ)z + λ*SiLU(z), λ ∈ [0,1]"""
    if lam == 0.0:
        return z  # Pure linear
    elif lam == 1.0:
        return F.silu(z)  # Pure SiLU
    else:
        return (1 - lam) * z + lam * F.silu(z)  # Blended


def _silu_prime(x: Tensor) -> Tensor:
    """Compute SiLU derivative: φ'(x) = σ(x) * (1 + x * (1 - σ(x)))"""
    sigmoid_x = torch.sigmoid(x)
    return sigmoid_x * (1 + x * (1 - sigmoid_x))


def _chol_solve_batched(G: Tensor, Y: Tensor) -> Tensor:
    """Batched Cholesky solve with CPU fallback for MPS.
    
    Args:
        G: Batched Gram matrices (E × r × r)
        Y: Batched RHS (E × r × d)
        
    Returns:
        Solution X such that G @ X = Y (E × r × d)
    """
    device = G.device
    E, r, _ = G.shape
    _, _, d = Y.shape
    
    if device.type == "mps":
        # Move to CPU for batched operations
        G_cpu = G.cpu().float()
        Y_cpu = Y.cpu().float()
        
        # Solve one expert at a time
        X_list = []
        for i in range(E):
            try:
                L_i = torch.linalg.cholesky(G_cpu[i])  # r × r
                X_i = torch.cholesky_solve(Y_cpu[i], L_i)  # r × d
            except RuntimeError:
                # Fallback to general solve
                X_i = torch.linalg.solve(G_cpu[i], Y_cpu[i])  # r × d
            X_list.append(X_i)
        X_cpu = torch.stack(X_list)  # E × r × d
        return X_cpu.to(device, G.dtype)
    else:
        G_f32 = G.float()
        Y_f32 = Y.float()
        
        # Solve one expert at a time
        X_list = []
        for i in range(E):
            try:
                L_i = torch.linalg.cholesky(G_f32[i])  # r × r
                X_i = torch.cholesky_solve(Y_f32[i], L_i)  # r × d
            except RuntimeError:
                # Fallback to general solve
                X_i = torch.linalg.solve(G_f32[i], Y_f32[i])  # r × d
            X_list.append(X_i)
        X_f32 = torch.stack(X_list)  # E × r × d
        return X_f32.to(G.dtype)


def _chol_solve_batched_transpose(G: Tensor, R: Tensor) -> Tensor:
    """Solve G @ A^T = R^T for A, returning A (not A^T).
    
    Args:
        G: Batched Gram matrices (E × r × r)
        R: Batched RHS (E × p × r)
        
    Returns:
        A such that A @ G = R (E × p × r)
    """
    device = G.device
    E, r, _ = G.shape
    _, p, _ = R.shape
    
    if device.type == "mps":
        # Move to CPU for batched operations
        G_cpu = G.cpu().float()
        R_cpu = R.cpu().float()
        
        # Solve one expert at a time: G @ A^T = R^T, so A^T = G^-1 @ R^T
        A_list = []
        for i in range(E):
            try:
                L_i = torch.linalg.cholesky(G_cpu[i])  # r × r
                At_i = torch.cholesky_solve(R_cpu[i].T, L_i)  # r × p
                A_i = At_i.T  # p × r
            except RuntimeError:
                # Fallback to general solve
                At_i = torch.linalg.solve(G_cpu[i], R_cpu[i].T)  # r × p
                A_i = At_i.T  # p × r
            A_list.append(A_i)
        A_cpu = torch.stack(A_list)  # E × p × r
        return A_cpu.to(device, G.dtype)
    else:
        G_f32 = G.float()
        R_f32 = R.float()
        
        # Solve one expert at a time
        A_list = []
        for i in range(E):
            try:
                L_i = torch.linalg.cholesky(G_f32[i])  # r × r
                At_i = torch.cholesky_solve(R_f32[i].T, L_i)  # r × p
                A_i = At_i.T  # p × r
            except RuntimeError:
                # Fallback to general solve
                At_i = torch.linalg.solve(G_f32[i], R_f32[i].T)  # r × p
                A_i = At_i.T  # p × r
            A_list.append(A_i)
        A_f32 = torch.stack(A_list)  # E × p × r
        return A_f32.to(G.dtype)


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
    cfg: InitConfig,
    step: int = 0,
    log_metrics: bool = False,
    # Pre-computed buffers for performance
    I_r: Optional[Tensor] = None,
    I_m: Optional[Tensor] = None,
    bank_unit_flat: Optional[Tensor] = None,
    # Hyperparameter values passed from fit_banks
    activation_lambda: float = 0.0,
    route_beta: float = 0.0,
    use_weight_aware_targets: bool = False,
) -> Tuple[Tensor, Tensor, float, float, Dict[str, float]]:
    """Minimal alternating minimization step.

    Args:
        W: ``E × p × d`` expert weight tensor.
        B_targets: ``E × r × d`` warm start targets.
        bank_module: ``GloBEBank`` instance holding learnable bases.
        alpha: Current mixture weights (ignored, recomputed).
        A: Current adapters ``E × p × r``.
        cfg: Configuration.
        step: Current iteration step.
        log_metrics: Whether to compute detailed metrics.
        I_r: Pre-computed identity matrix (r × r) for performance.
        I_m: Pre-computed identity matrix (m × m) for performance.
        bank_unit_flat: Pre-computed normalized bank for coding (m × rd).
        activation_lambda: Current lambda value for activation homotopy.
        route_beta: Current beta value for target blending.
        use_weight_aware_targets: Whether to use weight-aware targets (Route-W).

    Returns:
        Tuple ``(alpha, A, loss, median_support, metrics)``.
    """
    # Start timing
    step_start = time.perf_counter()
    
    # Disable autograd for performance
    with torch.inference_mode():
        device = bank_module.bases.device
        E, p, d = W.shape
        m, r, _ = bank_module.bases.shape
        
        # Use pre-computed identity tensors or create once
        if I_r is None:
            I_r = torch.eye(r, device=device, dtype=W.dtype)
        if I_m is None:
            I_m = torch.eye(m, device=device, dtype=W.dtype)
        
        # Keep everything fp32 on MPS to avoid repeated casting
        if device.type == "mps" or W.dtype != torch.float32:
            W32 = W.float()
            A32 = A.float()
            I_r32 = I_r.float()
            I_m32 = I_m.float()
            working_dtype = torch.float32
        else:
            W32 = W
            A32 = A
            I_r32 = I_r
            I_m32 = I_m
            working_dtype = W.dtype
        
        # Step 1: Build coding targets S_i (linear space) with blended routing
        coding_start = time.perf_counter()
        
        # Targets are ALWAYS updated every step based on current parameters
        S_L = B_targets.to(working_dtype) if device.type == "mps" or W.dtype != torch.float32 else B_targets
        
        # Compute weight-aware targets if requested or if beta > 0
        if use_weight_aware_targets or route_beta > 0:
            # Route-W: weight-aware targets
            Gt = torch.einsum('epr,eps->ers', A32, A32) + cfg.lambda_T * I_r32.unsqueeze(0)  # E × r × r
            Yt = torch.einsum('epr,epd->erd', A32, W32)  # E × r × d
            S_W = _chol_solve_batched(Gt, Yt)
            
            if use_weight_aware_targets and route_beta == 0:
                # Pure weight-aware targets (e.g., for λ=0 after initial steps)
                S = S_W
            else:
                # Blend targets: S = (1-β)*S_L + β*S_W
                S = (1 - route_beta) * S_L + route_beta * S_W
        else:
            # Pure Route-L (warm start targets)
            S = S_L
        
        coding_time = (time.perf_counter() - coding_start) * 1000
        
        # Step 2: Coding - use ridge regression for linear phase, softmax for nonlinear
        if activation_lambda == 0.0 or getattr(cfg, 'use_ridge_codes', False):
            # Linear phase or forced ridge: use ridge regression for signed codes
            # Solve: alpha* = argmin ||vec(S) - D*alpha||^2 + lambda*||alpha||^2
            # where D is the bank flattened to (rd) x m
            D_flat = bank_module.bases.view(m, -1).T.to(working_dtype)  # (rd) × m
            
            # Ridge regression for each expert
            alpha_list = []
            for i in range(E):
                S_i_flat = S[i].view(-1)  # (rd)
                # Normal equations: (D^T D + lambda I) alpha = D^T S
                # Use smaller regularization for ridge codes to allow better fit
                ridge_lambda = 1e-6 if activation_lambda == 0.0 else cfg.lambda_A
                DTD = D_flat.T @ D_flat + ridge_lambda * I_m32  # m × m
                DTS = D_flat.T @ S_i_flat  # m
                try:
                    alpha_i = torch.linalg.solve(DTD, DTS)  # m
                except:
                    # Fallback with more regularization
                    DTD += 1e-4 * I_m32
                    alpha_i = torch.linalg.solve(DTD, DTS)
                alpha_list.append(alpha_i)
            
            alpha = torch.stack(alpha_list)  # E × m (can be signed!)
            A_codes = alpha.T  # m × E
            
        else:
            # Nonlinear phase: use softmax coding (nonnegative, sum to 1)
            # Use pre-computed normalized bank if available
            if bank_unit_flat is not None:
                D_norm = bank_unit_flat  # m × (rd), already normalized
            else:
                D = bank_module.bases.view(m, -1).to(working_dtype)  # m × (rd)
                D_norm = D / (D.norm(dim=-1, keepdim=True).clamp_min(1e-8))
            
            X = S.view(E, -1)  # E × (rd)
            X_norm = X / (X.norm(dim=-1, keepdim=True).clamp_min(1e-8))
            C = X_norm @ D_norm.T  # E × m (cosine scores)
            
            # Temperature-controlled softmax
            alpha = F.softmax(C / cfg.tau, dim=-1)  # E × m (simplex)
            A_codes = alpha.T  # m × E
        
        # Step 3: Mix bases and apply activation
        M_lin = torch.einsum('em,mrd->erd', alpha.to(bank_module.bases.dtype), bank_module.bases)  # E × r × d
        M_lin32 = M_lin.to(working_dtype)
        M32 = phi_lambda(M_lin32, activation_lambda)  # Homotopy activation
        
        # Step 4: Adapter refit (batched, ridge) - always refit for proper dictionary learning
        refit_start = time.perf_counter()
        
        # Use einsum for now - batched matmul dimension mismatch needs more careful handling
        # G = M^T @ M where M is E × r × d, so G should be E × r × r
        G = torch.einsum('erd,esd->ers', M32, M32) + cfg.lambda_A * I_r32.unsqueeze(0)  # E × r × r
        # R = W @ M^T where W is E × p × d and M^T is E × d × r, so R should be E × p × r
        R = torch.einsum('epd,erd->epr', W32, M32)  # E × p × r
        A32_new = _chol_solve_batched_transpose(G, R)
        
        refit_time = (time.perf_counter() - refit_start) * 1000
        
        # Step 5: Residual and Jacobian (φ-aware)
        residual_start = time.perf_counter()
        
        E_out = A32_new @ M32 - W32  # E × p × d (output residual)
        
        if activation_lambda == 0.0:
            # Linear case: Jacobian is 1 everywhere
            J = torch.ones_like(M_lin32)
        elif activation_lambda < 1.0:
            # Blended Jacobian for homotopy: J = (1-λ) * I + λ * φ'(M_lin)
            J_linear = torch.ones_like(M_lin32)
            J_silu = _silu_prime(M_lin32)
            J_blended = (1 - activation_lambda) * J_linear + activation_lambda * J_silu
            J = torch.clamp(J_blended, min=cfg.j_min)
        else:
            # Full SiLU case: compute actual Jacobian
            J = _silu_prime(M_lin32)
            J = torch.clamp(J, min=cfg.j_min)
        
        # Step 6: Pull residual back to pre-activation space
        # Use einsum for now - these operations work correctly
        # Gt = A^T @ A where A is E × p × r, so A^T is E × r × p, and Gt should be E × r × r
        Gt = torch.einsum('epr,eps->ers', A32_new, A32_new) + cfg.lambda_T * I_r32.unsqueeze(0)  # E × r × r
        # Yt = A^T @ E_out where A^T is E × r × p and E_out is E × p × d, so Yt should be E × r × d
        Yt = torch.einsum('epr,epd->erd', A32_new, E_out)  # E × r × d
        U = _chol_solve_batched(Gt, Yt)
        
        # 6b: Undo Jacobian: Δ_i = U_i / J_i (elementwise)
        Delta = U / J  # E × r × d (pre-activation correction targets)
        
        residual_time = (time.perf_counter() - residual_start) * 1000
        
        # Step 7: Simplified JW-MOD bank update
        bank_solve_start = time.perf_counter()
        
        # More efficient RHS computation avoiding giant flattening
        # rhs (m×r×d) = einsum('me,erd->mrd', A_codes, Delta)
        rhs = torch.einsum('me,erd->mrd', A_codes.to(working_dtype), Delta)  # m × r × d
        rhs_flat = rhs.view(m, -1)  # m × (rd)
        
        Mmat = A_codes.to(working_dtype) @ A_codes.to(working_dtype).T + cfg.lambda_B * I_m32
        
        try:
            DeltaB_flat = torch.linalg.solve(Mmat, rhs_flat)  # m × (rd)
        except RuntimeError:
            # Add regularization if singular
            Mmat += 1e-6 * I_m32
            DeltaB_flat = torch.linalg.solve(Mmat, rhs_flat)  # m × (rd)
        
        DeltaB = DeltaB_flat.view(m, r, d)  # m × r × d
        
        bank_solve_time = (time.perf_counter() - bank_solve_start) * 1000
        
        # Step 8: Line search without renormalizing inside loop
        line_search_start = time.perf_counter()
        
        # Try full step first - evaluate WITHOUT renormalization
        eta = cfg.eta
        bank_trial = bank_module.bases + eta * DeltaB.to(bank_module.bases.dtype)
        
        # Quick stability check
        bank_change_norm = (DeltaB.to(bank_module.bases.dtype) * eta).norm()
        bank_norm = bank_module.bases.norm()
        
        line_search_trials = 1
        
        # If change is too large, use half step
        if bank_change_norm > 0.1 * bank_norm:
            eta *= 0.5
            bank_trial = bank_module.bases + eta * DeltaB.to(bank_module.bases.dtype)
            line_search_trials = 2
        
        line_search_time = (time.perf_counter() - line_search_start) * 1000
        
        # Accept the step and normalize atoms ONCE
        with torch.no_grad():
            bank_module.bases.copy_(bank_trial)
            # Normalize atoms after accepting
            norms = bank_module.bases.view(m, -1).norm(dim=-1, keepdim=True).clamp_min(1e-8)
            bank_module.bases.div_(norms.view(m, 1, 1))
        
        # CRITICAL FIX: Always recompute alpha with normalized bank before loss evaluation
        if cfg.activation_lambda == 0.0 or getattr(cfg, 'use_ridge_codes', False):
            # Ridge coding: recompute with normalized bank
            D_flat_norm = bank_module.bases.view(m, -1).T.to(working_dtype)  # (rd) × m
            alpha_final_list = []
            ridge_lambda = 1e-6 if cfg.activation_lambda == 0.0 else cfg.lambda_A
            for i in range(E):
                S_i_flat = S[i].view(-1).to(working_dtype)  # (rd)
                DTD = D_flat_norm.T @ D_flat_norm + ridge_lambda * I_m32  # m × m
                DTS = D_flat_norm.T @ S_i_flat  # m
                try:
                    alpha_i = torch.linalg.solve(DTD, DTS)  # m
                except:
                    DTD += 1e-4 * I_m32
                    alpha_i = torch.linalg.solve(DTD, DTS)
                alpha_final_list.append(alpha_i)
            alpha_final = torch.stack(alpha_final_list)  # E × m
        else:
            # Softmax coding: recompute with normalized bank
            D_unit = bank_module.bases.view(m, -1).to(working_dtype)
            D_unit = D_unit / (D_unit.norm(dim=-1, keepdim=True).clamp_min(1e-8))
            X_unit = S.view(E, -1)
            X_unit = X_unit / (X_unit.norm(dim=-1, keepdim=True).clamp_min(1e-8))
            alpha_final = F.softmax((X_unit @ D_unit.T) / cfg.tau, dim=-1)
        
        # Recompute final reconstruction with normalized bank and recomputed alpha
        M_lin_final = torch.einsum('em,mrd->erd', alpha_final.to(bank_module.bases.dtype), bank_module.bases)
        M_final = phi_lambda(M_lin_final, cfg.activation_lambda)  # Use homotopy activation
        W_hat = A32_new.to(bank_module.bases.dtype) @ M_final
        
        # Convert back to original dtype
        A_final = A32_new.to(W.dtype)
        
        # Step 9: Loss & metrics
        diff = W_hat - W
        mse = (diff ** 2).mean()
        relF = (diff.norm() / W.norm().clamp_min(1e-12)).item()
        
        # Enhanced softmax metrics
        entropy_vals = -(alpha.clamp_min(1e-12) * alpha.clamp_min(1e-12).log()).sum(-1)
        entropy_median = entropy_vals.median().item()
        entropy_mean = entropy_vals.mean().item()
        
        # Effective support from median entropy
        seff_median = math.exp(entropy_median)
        
        # Top-k mass metrics
        topk8_mass = alpha.topk(k=min(8, m), dim=-1).values.sum(-1).mean().item()
        topk16_mass = alpha.topk(k=min(16, m), dim=-1).values.sum(-1).mean().item()
        
        total_time = (time.perf_counter() - step_start) * 1000
        
        # Basic metrics with adaptive controller info
        metrics = {
            'recon/relative_frobenius_error': relF,
            'recon/mse': mse.item(),
            'recon/rMSE_ema': getattr(cfg, '_relF_ema', relF),
            'alpha/entropy_median': entropy_median,
            'alpha/entropy_mean': entropy_mean,
            'alpha/seff_median': seff_median,
            'alpha/top8_mass': topk8_mass,
            'alpha/top16_mass': topk16_mass,
            'alpha/tau': cfg.tau,
            # Adaptive controller metrics
            'adaptive/lambda': cfg.activation_lambda,
            'adaptive/beta_route': cfg.route_beta,
            'adaptive/phase': 0 if cfg.activation_lambda == 0 else (1 if cfg.activation_lambda < 1 else 2),
        }
        
        if log_metrics:
            # Detailed metrics
            bank_norms = bank_module.bases.view(m, -1).norm(dim=-1)
            
            # Compute conditioning numbers for adaptive controller
            try:
                # Condition number of A^T A (for quality gates)
                AAt = A32_new.transpose(-2, -1) @ A32_new  # E × r × r
                cond_AAt = torch.linalg.cond(AAt).mean().item()
            except:
                cond_AAt = 1e6  # High value if computation fails
            
            # Bank drift (if we have previous bank)
            if hasattr(cfg, '_prev_bank'):
                bank_drift = (bank_module.bases - cfg._prev_bank).norm() / cfg._prev_bank.norm().clamp_min(1e-12)
                bank_drift_rel = bank_drift.item()
            else:
                bank_drift_rel = 0.0
            cfg._prev_bank = bank_module.bases.clone().detach()
            
            # Bank coherence (off-diagonal correlations)
            bank_flat = bank_module.bases.view(m, -1)
            bank_unit = bank_flat / bank_flat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            coherence_matrix = bank_unit @ bank_unit.T
            # Zero out diagonal and take max off-diagonal
            coherence_offdiag = coherence_matrix - torch.eye(m, device=coherence_matrix.device)
            coherence_max = coherence_offdiag.abs().max().item()
            
            metrics.update({
                # Alpha statistics
                'alpha/max_weight': alpha_final.max().item(),
                'alpha/min_weight': alpha_final.min().item(),
                'alpha/seff_target': cfg.target_support,  # For temperature control
                
                # Bank health
                'bank/atom_norm_mean': bank_norms.mean().item(),
                'bank/atom_norm_std': bank_norms.std().item(),
                'bank/atom_norm_max_dev': (bank_norms - 1).abs().max().item(),
                'bank/drift_rel_mean': bank_drift_rel,
                'bank/coherence_offdiag_max': coherence_max,
                
                # Adapter quality & conditioning
                'A/refit_resid_rel_mean': (torch.norm(diff, dim=(1, 2)) / torch.norm(W, dim=(1, 2)).clamp_min(1e-12)).mean().item(),
                'mod/cond_AAt': cond_AAt,
                'A/refit_cond_mean': cond_AAt,  # Alias for compatibility
                
                # Step info
                'step/eta_used': eta,
                
                # Performance timing
                'time/coding_ms': coding_time,
                'time/A_refit_ms': refit_time,
                'time/residual_and_backsolve_ms': residual_time,
                'time/bank_solve_ms': bank_solve_time,
                'time/line_search_ms': line_search_time,
                'time/total_ms': total_time,
                
                # Performance counters
                'count/line_search_trials': line_search_trials,
            })
            
            # Health checks
            nan_infs = 0
            for tensor in [alpha_final, bank_module.bases, A_final]:
                nan_infs += torch.isnan(tensor).sum().item()
                nan_infs += torch.isinf(tensor).sum().item()
            metrics['health/nan_infs'] = nan_infs
        
        return alpha_final.detach(), A_final, float(mse.detach()), seff_median, metrics


# End of file
