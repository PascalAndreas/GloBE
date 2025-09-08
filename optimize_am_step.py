#!/usr/bin/env python3
"""
Performance-optimized version of am_step that maintains GPT-5 fix benefits
while addressing major performance bottlenecks.

Key optimizations:
1. Simplified line search (1-2 trials max)
2. In-place operations where possible  
3. Reduced fp32 conversions
4. Memory-efficient tensor management
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Dict
from globe.init.init_bank import InitConfig, _chol_solve_batched, _chol_solve_batched_transpose, _silu_prime


def am_step_optimized(
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
    Performance-optimized am_step with GPT-5 fixes but better efficiency.
    """
    device = bank_module.bases.device
    E, p, d = W.shape
    m, r, _ = bank_module.bases.shape
    
    # Create identity tensors once
    I_r = torch.eye(r, device=device, dtype=W.dtype)
    I_m = torch.eye(m, device=device, dtype=W.dtype)
    
    # Step 1: Build coding targets S_i (selective fp32 usage)
    if step < cfg.route_w_start:
        S = B_targets  # E × r × d
    else:
        # Only use fp32 if we detect numerical issues
        try:
            Gt = torch.einsum('epr,eps->ers', A, A) + cfg.lambda_T * I_r.unsqueeze(0)
            Yt = torch.einsum('epr,epd->erd', A, W)
            S = _chol_solve_batched(Gt, Yt)
        except RuntimeError:
            # Fallback to fp32 only if needed
            Gt32 = torch.einsum('epr,eps->ers', A.float(), A.float()) + cfg.lambda_T * I_r.float().unsqueeze(0)
            Yt32 = torch.einsum('epr,epd->erd', A.float(), W.float())
            S = _chol_solve_batched(Gt32, Yt32).to(W.dtype)
    
    # Step 2: Scale-invariant coding (keep this - it's the key improvement)
    D = bank_module.bases.view(m, -1)  # m × (rd)
    X = S.view(E, -1)  # E × (rd)
    
    # Normalize for cosine similarity (fp32 only for normalization)
    with torch.cuda.amp.autocast(enabled=False):
        Dn = F.normalize(D.float(), dim=-1)  # m × (rd)
        Xn = F.normalize(X.float(), dim=-1)  # E × (rd)
        C = (Xn @ Dn.T).to(W.dtype)  # E × m (cosine scores)
    
    alpha = F.softmax(C / cfg.tau, dim=-1)  # E × m
    A_codes = alpha.T  # m × E
    
    # Step 3: Mix + activation
    M_lin = torch.einsum('em,mrd->erd', alpha, bank_module.bases)
    M = bank_module.activation(M_lin)
    
    # Step 4: Adapter refit (selective fp32)
    try:
        G = torch.einsum('erd,esd->ers', M, M) + cfg.lambda_A * I_r.unsqueeze(0)
        R = torch.einsum('epd,erd->epr', W, M)
        A = _chol_solve_batched_transpose(G, R)
    except RuntimeError:
        # Fallback to fp32 only if needed
        G32 = torch.einsum('erd,esd->ers', M.float(), M.float()) + cfg.lambda_A * I_r.float().unsqueeze(0)
        R32 = torch.einsum('epd,erd->epr', W.float(), M.float())
        A = _chol_solve_batched_transpose(G32, R32).to(W.dtype)
    
    # Step 5: Residual and Jacobian
    E_out = torch.einsum('epr,erd->epd', A, M) - W
    J = _silu_prime(M_lin).clamp_min(max(cfg.j_min, 1e-2))  # Keep aggressive clamping
    
    # Step 6: Backsolve (selective fp32)
    try:
        Gt = torch.einsum('epr,eps->ers', A, A) + cfg.lambda_T * I_r.unsqueeze(0)
        Yt = torch.einsum('epr,epd->erd', A, E_out)
        U = _chol_solve_batched(Gt, Yt)
    except RuntimeError:
        # Fallback to fp32 only if needed
        Gt32 = torch.einsum('epr,eps->ers', A.float(), A.float()) + cfg.lambda_T * I_r.float().unsqueeze(0)
        Yt32 = torch.einsum('epr,epd->erd', A.float(), E_out.float())
        U = _chol_solve_batched(Gt32, Yt32).to(W.dtype)
    
    Delta = U / J
    X_delta = Delta.view(E, -1).T  # (rd) × E
    
    # Step 7: Simplified JW-MOD update (reduced line search)
    try:
        Mmat = A_codes @ A_codes.T + cfg.lambda_B * I_m
        rhs = (A_codes @ X_delta.T).T
        DeltaB_flat = torch.linalg.solve(Mmat, rhs.T).T
    except RuntimeError:
        # Fallback to fp32 with regularization
        Mmat32 = (A_codes.float() @ A_codes.float().T + cfg.lambda_B * I_m.float())
        rhs32 = (A_codes.float() @ X_delta.float().T).T
        
        # Add regularization if singular
        try:
            DeltaB_flat = torch.linalg.solve(Mmat32, rhs32.T).T.to(W.dtype)
        except RuntimeError:
            Mmat32 += 1e-6 * torch.eye(m, device=device, dtype=torch.float32)
            DeltaB_flat = torch.linalg.solve(Mmat32, rhs32.T).T.to(W.dtype)
    
    DeltaB = DeltaB_flat.T.view(m, r, d)
    
    # Simplified line search (max 2 trials to balance stability vs speed)
    W_hat_old = torch.einsum('epr,erd->epd', A, M)
    old_relF = ((W_hat_old - W).norm() / W.norm().clamp_min(1e-12)).item()
    
    # Try full step first
    eta = cfg.eta
    new_bank = bank_module.bases + eta * DeltaB
    
    # Normalize
    norms = new_bank.view(m, -1).norm(dim=-1, keepdim=True).clamp_min(1e-8)
    new_bank = new_bank / norms.view(m, 1, 1)
    
    # Quick check: compute new loss
    with torch.no_grad():
        bank_module.bases.copy_(new_bank)
        M_lin_new = torch.einsum('em,mrd->erd', alpha, bank_module.bases)
        M_new = bank_module.activation(M_lin_new)
        W_hat_new = torch.einsum('epr,erd->epd', A, M_new)
        new_relF = ((W_hat_new - W).norm() / W.norm().clamp_min(1e-12)).item()
    
    # If step is bad, try half step (only 1 retry for performance)
    if new_relF > old_relF * 1.01:  # Allow small increase
        eta *= 0.5
        new_bank = bank_module.bases + eta * DeltaB
        norms = new_bank.view(m, -1).norm(dim=-1, keepdim=True).clamp_min(1e-8)
        new_bank = new_bank / norms.view(m, 1, 1)
        
        with torch.no_grad():
            bank_module.bases.copy_(new_bank)
            M_lin_new = torch.einsum('em,mrd->erd', alpha, bank_module.bases)
            M_new = bank_module.activation(M_lin_new)
            W_hat_new = torch.einsum('epr,erd->epd', A, M_new)
            new_relF = ((W_hat_new - W).norm() / W.norm().clamp_min(1e-12)).item()
    
    # Use the new values
    M = M_new
    W_hat = W_hat_new
    relF = new_relF
    
    # Step 8: Efficient metrics computation
    diff = W_hat - W
    mse = (diff ** 2).mean()
    
    # Meaningful statistics
    entropy_vals = -(alpha.clamp_min(1e-12) * alpha.clamp_min(1e-12).log()).sum(-1)
    entropy_mean = entropy_vals.mean().item()
    topk_mass = alpha.topk(k=min(8, m), dim=-1).values.sum(-1).mean().item()
    
    metrics = {
        'recon/relative_frobenius_error': relF,
        'loss/mse': mse.item(),
        'alpha/entropy_mean': entropy_mean,
        'alpha/top8_mass': topk_mass,
    }
    
    if log_metrics:
        metrics.update({
            'alpha/max_weight': alpha.max().item(),
            'alpha/min_weight': alpha.min().item(),
            'step/eta_used': eta,
        })
    
    return alpha.detach(), A, float(mse.detach()), entropy_mean, metrics


def am_step_ultra_fast(
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
    Ultra-fast version: Keep cosine coding but remove line search for speed.
    Use this if you need maximum performance and can accept slightly less stability.
    """
    device = bank_module.bases.device
    E, p, d = W.shape
    m, r, _ = bank_module.bases.shape
    
    I_r = torch.eye(r, device=device, dtype=W.dtype)
    I_m = torch.eye(m, device=device, dtype=W.dtype)
    
    # Step 1: Targets (no fp32 fallback for speed)
    if step < cfg.route_w_start:
        S = B_targets
    else:
        Gt = torch.einsum('epr,eps->ers', A, A) + cfg.lambda_T * I_r.unsqueeze(0)
        Yt = torch.einsum('epr,epd->erd', A, W)
        S = _chol_solve_batched(Gt, Yt)
    
    # Step 2: Cosine coding (keep this key improvement)
    D = bank_module.bases.view(m, -1)
    X = S.view(E, -1)
    
    # Fast normalization
    D_norm = D / (D.norm(dim=-1, keepdim=True).clamp_min(1e-8))
    X_norm = X / (X.norm(dim=-1, keepdim=True).clamp_min(1e-8))
    C = X_norm @ D_norm.T
    
    alpha = F.softmax(C / cfg.tau, dim=-1)
    A_codes = alpha.T
    
    # Step 3-4: Standard operations
    M_lin = torch.einsum('em,mrd->erd', alpha, bank_module.bases)
    M = bank_module.activation(M_lin)
    
    G = torch.einsum('erd,esd->ers', M, M) + cfg.lambda_A * I_r.unsqueeze(0)
    R = torch.einsum('epd,erd->epr', W, M)
    A = _chol_solve_batched_transpose(G, R)
    
    # Step 5-6: Jacobian and backsolve
    E_out = torch.einsum('epr,erd->epd', A, M) - W
    J = _silu_prime(M_lin).clamp_min(1e-2)
    
    Gt = torch.einsum('epr,eps->ers', A, A) + cfg.lambda_T * I_r.unsqueeze(0)
    Yt = torch.einsum('epr,epd->erd', A, E_out)
    U = _chol_solve_batched(Gt, Yt)
    
    Delta = U / J
    X_delta = Delta.view(E, -1).T
    
    # Step 7: Direct bank update (no line search)
    Mmat = A_codes @ A_codes.T + cfg.lambda_B * I_m
    rhs = (A_codes @ X_delta.T).T
    
    try:
        DeltaB_flat = torch.linalg.solve(Mmat, rhs.T).T
    except RuntimeError:
        Mmat += 1e-6 * torch.eye(m, device=device, dtype=W.dtype)
        DeltaB_flat = torch.linalg.solve(Mmat, rhs.T).T
    
    DeltaB = DeltaB_flat.T.view(m, r, d)
    new_bank = bank_module.bases + cfg.eta * DeltaB
    
    # Normalize
    norms = new_bank.view(m, -1).norm(dim=-1, keepdim=True).clamp_min(1e-8)
    new_bank = new_bank / norms.view(m, 1, 1)
    
    with torch.no_grad():
        bank_module.bases.copy_(new_bank)
    
    # Step 8: Final loss (with updated bank)
    M_lin_final = torch.einsum('em,mrd->erd', alpha, bank_module.bases)
    M_final = bank_module.activation(M_lin_final)
    W_hat = torch.einsum('epr,erd->epd', A, M_final)
    
    diff = W_hat - W
    mse = (diff ** 2).mean()
    relF = (diff.norm() / W.norm().clamp_min(1e-12)).item()
    
    # Fast metrics
    entropy_mean = -(alpha.clamp_min(1e-12) * alpha.clamp_min(1e-12).log()).sum(-1).mean().item()
    
    metrics = {
        'recon/relative_frobenius_error': relF,
        'loss/mse': mse.item(),
        'alpha/entropy_mean': entropy_mean,
    }
    
    return alpha.detach(), A, float(mse.detach()), entropy_mean, metrics
