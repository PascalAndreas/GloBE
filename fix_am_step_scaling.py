#!/usr/bin/env python3
"""
Fix for the scaling issue in am_step.

The problem: 
1. Bases are normalized to unit norm, losing scale information
2. Alpha coefficients are probability distributions (sum=1), but don't account for weight magnitude

Solution:
1. Don't normalize bases to unit norm, or if we do, rescale alpha to compensate
2. Use proper scaling for alpha that accounts for the magnitude of the reconstruction

This script provides a fixed version of the key functions.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Dict
from globe.init.init_bank import InitConfig, _chol_solve_batched, _chol_solve_batched_transpose, _silu_prime


def build_initial_bank_fixed(B0: torch.Tensor, cfg: InitConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fixed version of build_initial_bank that preserves scale information.
    """
    from globe.init.init_bank import _resolve_device, _kmeans_pytorch
    
    device = _resolve_device(cfg)
    E, r, d = B0.shape
    flat = B0.view(E, -1)

    centroids = _kmeans_pytorch(flat, cfg.num_bases, iters=10)  # num_bases × (r*d)
    bank = centroids.view(cfg.num_bases, r, d).to(device, cfg.dtype)  # num_bases × r × d

    # Solve NNLS approximately using least squares + clamp
    bank_flat = bank.view(cfg.num_bases, -1).T  # RD × m
    
    # Move to CPU for lstsq if on MPS (not supported), and ensure float32
    if bank_flat.device.type == "mps":
        bank_flat_cpu = bank_flat.cpu().float()
        flat_cpu = flat.cpu().float()
        codes = torch.linalg.lstsq(bank_flat_cpu, flat_cpu.T).solution.T
        codes = codes.to(device, cfg.dtype)
    else:
        bank_flat_f32 = bank_flat.float()
        flat_f32 = flat.float()
        codes = torch.linalg.lstsq(bank_flat_f32, flat_f32.T).solution.T
        codes = codes.to(cfg.dtype)
    
    codes = codes.clamp_min(cfg.epsilon)
    
    # FIXED: Don't normalize codes to probability distributions
    # Instead, keep the natural scale from the least squares solution
    # This preserves the magnitude information needed for reconstruction
    
    return bank, codes.to(device, cfg.dtype)


def am_step_fixed(
    W: torch.Tensor,
    B_targets: torch.Tensor,
    bank_module,  # GloBEBank
    alpha: torch.Tensor,
    A: torch.Tensor,
    cfg: InitConfig,
    step: int = 0,
    log_metrics: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, float, float, Dict[str, float]]:
    """
    Fixed version of am_step that handles scaling correctly.
    """
    device = bank_module.bases.device
    E, p, d = W.shape
    m, r, _ = bank_module.bases.shape
    
    # Create identity tensors
    I_r = torch.eye(r, device=device, dtype=W.dtype)
    I_m = torch.eye(m, device=device, dtype=W.dtype)
    
    # Step 1: Build coding targets S_i (linear space)
    if step < cfg.route_w_start:
        # Route-L: use warm start targets
        S = B_targets  # E × r × d
    else:
        # Route-W: weight-aware targets S = (A^T A + λ_T I)^-1 A^T W
        Gt = torch.einsum('epr,eps->ers', A, A) + cfg.lambda_T * I_r.unsqueeze(0)  # E × r × r
        Yt = torch.einsum('epr,epd->erd', A, W)  # E × r × d
        S = _chol_solve_batched(Gt, Yt)  # E × r × d
    
    # Step 2: Coding (with proper scaling instead of softmax normalization)
    D = bank_module.bases.view(m, -1).T  # (rd) × m
    X = S.view(E, -1)  # E × (rd)
    C = X @ D  # E × m (scores)
    
    # FIXED: Use a more appropriate normalization that preserves scale
    # Option 1: Use softmax but with a larger temperature to preserve more scale info
    # Option 2: Use L1 normalization instead of softmax
    # Option 3: Don't normalize at all, use raw scores with regularization
    
    # Let's try option 2: L1 normalization with positivity constraint
    alpha_raw = F.relu(C / cfg.tau)  # Ensure positivity
    alpha_sum = alpha_raw.sum(dim=-1, keepdim=True).clamp_min(cfg.epsilon)
    alpha = alpha_raw / alpha_sum  # L1 normalize
    
    A_codes = alpha.T  # m × E
    
    # Step 3: Mix + activation (the model you'll deploy)
    M_lin = torch.einsum('em,mrd->erd', alpha, bank_module.bases)  # E × r × d
    M = bank_module.activation(M_lin)  # φ(M_lin)
    
    # Step 4: Adapter refit (batched, ridge)
    G = torch.einsum('erd,esd->ers', M, M) + cfg.lambda_A * I_r.unsqueeze(0)  # E × r × r
    R = torch.einsum('epd,erd->epr', W, M)  # E × p × r
    A = _chol_solve_batched_transpose(G, R)  # E × p × r
    
    # Step 5: Residual and Jacobian (φ-aware)
    E_out = torch.einsum('epr,erd->epd', A, M) - W  # E × p × d (output residual)
    J = _silu_prime(M_lin).clamp_min(cfg.j_min)  # E × r × d (Jacobian, clamped)
    
    # Step 6: Pull residual back to pre-activation space
    # 6a: Backsolve through A_i: U_i = (A_i^T A_i + λ_T I)^-1 A_i^T E_i
    Gt = torch.einsum('epr,eps->ers', A, A) + cfg.lambda_T * I_r.unsqueeze(0)  # E × r × r
    Yt = torch.einsum('epr,epd->erd', A, E_out)  # E × r × d
    U = _chol_solve_batched(Gt, Yt)  # E × r × d
    
    # 6b: Undo Jacobian: Δ_i = U_i / J_i (elementwise)
    Delta = U / J  # E × r × d (pre-activation correction targets)
    X_delta = Delta.view(E, -1).T  # (rd) × E
    
    # Step 7: JW-MOD bank update (closed-form)
    Mmat = A_codes @ A_codes.T + cfg.lambda_B * I_m  # m × m
    rhs = (A_codes @ X_delta.T).T  # (rd) × m
    
    # Solve for ΔB
    if device.type == "mps":
        Mmat_cpu = Mmat.cpu().float()
        rhs_cpu = rhs.cpu().float()
        try:
            DeltaB_flat_cpu = torch.linalg.solve(Mmat_cpu, rhs_cpu.T).T
        except RuntimeError:
            Mmat_cpu += 1e-6 * torch.eye(m, device='cpu', dtype=torch.float32)
            DeltaB_flat_cpu = torch.linalg.solve(Mmat_cpu, rhs_cpu.T).T
        DeltaB_flat = DeltaB_flat_cpu.to(device, W.dtype)
    else:
        Mmat_f32 = Mmat.float()
        rhs_f32 = rhs.float()
        try:
            DeltaB_flat_f32 = torch.linalg.solve(Mmat_f32, rhs_f32.T).T
        except RuntimeError:
            Mmat_f32 += 1e-6 * torch.eye(m, device=device, dtype=torch.float32)
            DeltaB_flat_f32 = torch.linalg.solve(Mmat_f32, rhs_f32.T).T
        DeltaB_flat = DeltaB_flat_f32.to(W.dtype)
    
    # Reshape and apply damped update
    DeltaB = DeltaB_flat.T.view(m, r, d)  # m × r × d
    new_bank = bank_module.bases + cfg.eta * DeltaB
    
    # FIXED: Option 1 - Don't normalize bases at all
    # This preserves scale information in the bases
    # new_bank = new_bank  # No normalization
    
    # FIXED: Option 2 - Normalize bases but rescale alpha to compensate
    # Compute the scale factors before normalization
    old_norms = bank_module.bases.view(m, -1).norm(dim=-1, keepdim=True).clamp_min(1e-8)
    new_norms = new_bank.view(m, -1).norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scale_factors = new_norms / old_norms  # m × 1
    
    # Normalize the bases
    new_bank = new_bank / new_norms.view(m, 1, 1)
    
    # Rescale alpha to compensate for the normalization
    # This preserves the overall scale of the reconstruction
    alpha = alpha * scale_factors.view(1, m)  # E × m
    
    # Update bank parameters
    with torch.no_grad():
        bank_module.bases.copy_(new_bank)
    
    # Step 8: Loss & metrics
    W_hat = torch.einsum('epr,erd->epd', A, M)
    diff = W_hat - W
    mse = (diff ** 2).mean()
    relF = (diff.norm() / W.norm().clamp_min(1e-12)).item()
    
    # Support statistics
    support = (alpha > cfg.epsilon).sum(dim=-1).float()
    med = support.median().item()
    
    # Basic metrics
    metrics = {
        'recon/relative_frobenius_error': relF,
        'loss/mse': mse.item(),
    }
    
    if log_metrics:
        # Entropy (since α is L1 normalized, not softmax)
        entropy_vals = -(alpha * alpha.clamp_min(1e-12).log()).sum(-1)
        
        metrics.update({
            'alpha/entropy_mean': entropy_vals.mean().item(),
            'alpha/median_support': support.median().item(),
            'alpha/mean_support': support.mean().item(),
            'bank/atom_norm_max_dev': (new_bank.view(m, -1).norm(dim=-1) - 1).abs().max().item(),
            'A/refit_resid_rel_mean': (torch.norm(diff, dim=(1, 2)) / torch.norm(W, dim=(1, 2)).clamp_min(1e-12)).mean().item(),
            'scale/alpha_mean': alpha.mean().item(),
            'scale/alpha_max': alpha.max().item(),
            'scale/W_norm': W.norm().item(),
            'scale/W_hat_norm': W_hat.norm().item(),
            'scale/scale_ratio': (W_hat.norm() / W.norm()).item(),
        })
        
        # Health checks
        nan_infs = 0
        for tensor in [alpha, new_bank, A]:
            nan_infs += torch.isnan(tensor).sum().item()
            nan_infs += torch.isinf(tensor).sum().item()
        metrics['health/nan_infs'] = nan_infs
    
    return alpha.detach(), A, float(mse.detach()), med, metrics


def am_step_fixed_v2(
    W: torch.Tensor,
    B_targets: torch.Tensor,
    bank_module,  # GloBEBank
    alpha: torch.Tensor,
    A: torch.Tensor,
    cfg: InitConfig,
    step: int = 0,
    log_metrics: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, float, float, Dict[str, float]]:
    """
    Alternative fixed version that doesn't normalize bases at all.
    This is simpler and should preserve scale naturally.
    """
    device = bank_module.bases.device
    E, p, d = W.shape
    m, r, _ = bank_module.bases.shape
    
    # Create identity tensors
    I_r = torch.eye(r, device=device, dtype=W.dtype)
    I_m = torch.eye(m, device=device, dtype=W.dtype)
    
    # Step 1: Build coding targets S_i (linear space)
    if step < cfg.route_w_start:
        S = B_targets  # E × r × d
    else:
        Gt = torch.einsum('epr,eps->ers', A, A) + cfg.lambda_T * I_r.unsqueeze(0)  # E × r × r
        Yt = torch.einsum('epr,epd->erd', A, W)  # E × r × d
        S = _chol_solve_batched(Gt, Yt)  # E × r × d
    
    # Step 2: Coding with non-negative least squares approach
    D = bank_module.bases.view(m, -1).T  # (rd) × m
    X = S.view(E, -1)  # E × (rd)
    
    # Solve for alpha using non-negative least squares
    # X ≈ D @ alpha.T, so alpha.T ≈ D^+ @ X.T where D^+ is pseudoinverse
    # But we want non-negative solutions, so we use a different approach
    
    # Method 1: Use the scoring approach but with better scaling
    C = X @ D  # E × m (scores)
    alpha = F.relu(C) + cfg.epsilon  # Ensure positivity, E × m
    # Don't normalize to sum=1, keep the natural scale
    
    A_codes = alpha.T  # m × E
    
    # Step 3: Mix + activation
    M_lin = torch.einsum('em,mrd->erd', alpha, bank_module.bases)  # E × r × d
    M = bank_module.activation(M_lin)  # φ(M_lin)
    
    # Step 4: Adapter refit
    G = torch.einsum('erd,esd->ers', M, M) + cfg.lambda_A * I_r.unsqueeze(0)  # E × r × r
    R = torch.einsum('epd,erd->epr', W, M)  # E × p × r
    A = _chol_solve_batched_transpose(G, R)  # E × p × r
    
    # Step 5: Residual and Jacobian
    E_out = torch.einsum('epr,erd->epd', A, M) - W  # E × p × d
    J = _silu_prime(M_lin).clamp_min(cfg.j_min)  # E × r × d
    
    # Step 6: Pull residual back to pre-activation space
    Gt = torch.einsum('epr,eps->ers', A, A) + cfg.lambda_T * I_r.unsqueeze(0)  # E × r × r
    Yt = torch.einsum('epr,epd->erd', A, E_out)  # E × r × d
    U = _chol_solve_batched(Gt, Yt)  # E × r × d
    
    Delta = U / J  # E × r × d
    X_delta = Delta.view(E, -1).T  # (rd) × E
    
    # Step 7: Bank update
    Mmat = A_codes @ A_codes.T + cfg.lambda_B * I_m  # m × m
    rhs = (A_codes @ X_delta.T).T  # (rd) × m
    
    # Solve for ΔB
    if device.type == "mps":
        Mmat_cpu = Mmat.cpu().float()
        rhs_cpu = rhs.cpu().float()
        try:
            DeltaB_flat_cpu = torch.linalg.solve(Mmat_cpu, rhs_cpu.T).T
        except RuntimeError:
            Mmat_cpu += 1e-6 * torch.eye(m, device='cpu', dtype=torch.float32)
            DeltaB_flat_cpu = torch.linalg.solve(Mmat_cpu, rhs_cpu.T).T
        DeltaB_flat = DeltaB_flat_cpu.to(device, W.dtype)
    else:
        Mmat_f32 = Mmat.float()
        rhs_f32 = rhs.float()
        try:
            DeltaB_flat_f32 = torch.linalg.solve(Mmat_f32, rhs_f32.T).T
        except RuntimeError:
            Mmat_f32 += 1e-6 * torch.eye(m, device=device, dtype=torch.float32)
            DeltaB_flat_f32 = torch.linalg.solve(Mmat_f32, rhs_f32.T).T
        DeltaB_flat = DeltaB_flat_f32.to(W.dtype)
    
    # Apply update without normalization
    DeltaB = DeltaB_flat.T.view(m, r, d)  # m × r × d
    new_bank = bank_module.bases + cfg.eta * DeltaB
    
    # FIXED: Don't normalize bases - this preserves scale
    # Update bank parameters
    with torch.no_grad():
        bank_module.bases.copy_(new_bank)
    
    # Step 8: Loss & metrics
    W_hat = torch.einsum('epr,erd->epd', A, M)
    diff = W_hat - W
    mse = (diff ** 2).mean()
    relF = (diff.norm() / W.norm().clamp_min(1e-12)).item()
    
    # Support statistics
    support = (alpha > cfg.epsilon).sum(dim=-1).float()
    med = support.median().item()
    
    # Basic metrics
    metrics = {
        'recon/relative_frobenius_error': relF,
        'loss/mse': mse.item(),
    }
    
    if log_metrics:
        # Entropy computation for non-normalized alpha
        alpha_normalized = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        entropy_vals = -(alpha_normalized * alpha_normalized.clamp_min(1e-12).log()).sum(-1)
        
        metrics.update({
            'alpha/entropy_mean': entropy_vals.mean().item(),
            'alpha/median_support': support.median().item(),
            'alpha/mean_support': support.mean().item(),
            'bank/atom_norm_mean': new_bank.view(m, -1).norm(dim=-1).mean().item(),
            'bank/atom_norm_std': new_bank.view(m, -1).norm(dim=-1).std().item(),
            'A/refit_resid_rel_mean': (torch.norm(diff, dim=(1, 2)) / torch.norm(W, dim=(1, 2)).clamp_min(1e-12)).mean().item(),
            'scale/alpha_mean': alpha.mean().item(),
            'scale/alpha_max': alpha.max().item(),
            'scale/alpha_sum_mean': alpha.sum(dim=-1).mean().item(),
            'scale/W_norm': W.norm().item(),
            'scale/W_hat_norm': W_hat.norm().item(),
            'scale/scale_ratio': (W_hat.norm() / W.norm()).item(),
        })
        
        # Health checks
        nan_infs = 0
        for tensor in [alpha, new_bank, A]:
            nan_infs += torch.isnan(tensor).sum().item()
            nan_infs += torch.isinf(tensor).sum().item()
        metrics['health/nan_infs'] = nan_infs
    
    return alpha.detach(), A, float(mse.detach()), med, metrics
