#!/usr/bin/env python3
"""
Debug script for AlternatingBankTrainer am_step issues.

This script implements the four debugging tests suggested:
1. Identity φ ablation: set φ=identity for 3–5 iters to test JW wiring
2. Pure A-refit bound: solve A exactly with current α,B to check if α/B are correct  
3. Plain MOD only for 10 iterations with warm targets to test coding/MOD solve
4. J stats: log mean/min of J and % elements at floor

The main issue: MSE gets very small but relative Frobenius error hovers around 0.95
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional
import json

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from globe.data.naming_qwen15 import extract_expert_weights
from globe.train.fit_banks import AlternatingBankTrainer
from globe.init.init_bank import (
    InitConfig, create_warm_start, build_initial_bank, 
    _silu_prime, _chol_solve_batched, _chol_solve_batched_transpose
)
from globe.init.warm_start import SeedingMethod, SeedingConfig
from globe.modules.globe_bank import GloBEBank


def identity_activation(x):
    """Identity activation function for testing."""
    return x


def debug_am_step_identity_phi(
    W: torch.Tensor,
    B_targets: torch.Tensor,
    bank_module: GloBEBank,
    alpha: torch.Tensor,
    A: torch.Tensor,
    cfg: InitConfig,
    num_identity_steps: int = 5,
) -> Dict[str, Any]:
    """
    Test 1: Identity φ ablation
    Set φ=identity for 3-5 iters. If relF plummets, your JW wiring (J or Δ construction) is the culprit.
    """
    print(f"\n🔬 DEBUG TEST 1: Identity φ ablation ({num_identity_steps} steps)")
    
    # Store original activation
    original_activation = bank_module.activation
    
    # Temporarily replace with identity
    bank_module.activation = identity_activation
    
    results = []
    current_alpha = alpha.clone()
    current_A = A.clone()
    
    for step in range(num_identity_steps):
        # Run am_step with identity activation
        from globe.init.init_bank import am_step
        current_alpha, current_A, loss, med, metrics = am_step(
            W, B_targets, bank_module, current_alpha, current_A, cfg, 
            step=step, log_metrics=True
        )
        
        rel_frob = metrics.get("recon/relative_frobenius_error", float("nan"))
        results.append({
            'step': step,
            'loss': loss,
            'rel_frob': rel_frob,
            'median_support': med,
            'metrics': metrics.copy()
        })
        
        print(f"   Step {step+1}: Loss={loss:.6f}, RelFrob={rel_frob:.6f}, Support={med:.1f}")
    
    # Restore original activation
    bank_module.activation = original_activation
    
    # Analysis
    initial_rel_frob = results[0]['rel_frob']
    final_rel_frob = results[-1]['rel_frob']
    rel_frob_improvement = initial_rel_frob - final_rel_frob
    
    print(f"   📊 Identity φ Results:")
    print(f"      Initial RelFrob: {initial_rel_frob:.6f}")
    print(f"      Final RelFrob: {final_rel_frob:.6f}")
    print(f"      Improvement: {rel_frob_improvement:.6f}")
    
    if rel_frob_improvement > 0.1:  # Significant improvement
        print(f"   ⚠️  FINDING: RelFrob improved significantly with identity φ!")
        print(f"      This suggests JW wiring (J or Δ construction) may be problematic.")
    else:
        print(f"   ✅ Identity φ didn't dramatically improve RelFrob.")
        print(f"      JW wiring appears to be working correctly.")
    
    return {
        'test_name': 'identity_phi_ablation',
        'results': results,
        'analysis': {
            'initial_rel_frob': initial_rel_frob,
            'final_rel_frob': final_rel_frob,
            'improvement': rel_frob_improvement,
            'jw_wiring_suspect': rel_frob_improvement > 0.1
        }
    }


def debug_pure_a_refit_bound(
    W: torch.Tensor,
    bank_module: GloBEBank,
    alpha: torch.Tensor,
    cfg: InitConfig,
) -> Dict[str, Any]:
    """
    Test 2: Pure A-refit bound
    With current α,B, solve A exactly → relF_Aonly. If that's already >0.8, your α/B are garbage.
    """
    print(f"\n🔬 DEBUG TEST 2: Pure A-refit bound")
    
    device = bank_module.bases.device
    E, p, d = W.shape
    m, r, _ = bank_module.bases.shape
    
    # Step 1: Mix with current alpha and bases (with activation)
    M_lin = torch.einsum('em,mrd->erd', alpha, bank_module.bases)  # E × r × d
    M = bank_module.activation(M_lin)  # φ(M_lin)
    
    # Step 2: Solve for optimal A exactly (ridge regression)
    I_r = torch.eye(r, device=device, dtype=W.dtype)
    G = torch.einsum('erd,esd->ers', M, M) + cfg.lambda_A * I_r.unsqueeze(0)  # E × r × r
    R = torch.einsum('epd,erd->epr', W, M)  # E × p × r
    
    # Solve for A exactly
    A_optimal = _chol_solve_batched_transpose(G, R)  # E × p × r
    
    # Step 3: Compute reconstruction error with optimal A
    W_hat = torch.einsum('epr,erd->epd', A_optimal, M)
    diff = W_hat - W
    mse = (diff ** 2).mean()
    rel_frob = (diff.norm() / W.norm().clamp_min(1e-12)).item()
    
    print(f"   📊 Pure A-refit Results:")
    print(f"      MSE with optimal A: {mse:.8f}")
    print(f"      RelFrob with optimal A: {rel_frob:.6f}")
    
    if rel_frob > 0.8:
        print(f"   ⚠️  FINDING: RelFrob with optimal A is {rel_frob:.6f} > 0.8!")
        print(f"      This suggests current α/B (coding/bases) are problematic.")
        print(f"      Check coding scores and MOD solve.")
    else:
        print(f"   ✅ RelFrob with optimal A is reasonable ({rel_frob:.6f} ≤ 0.8).")
        print(f"      Current α/B appear to be working correctly.")
    
    # Additional diagnostics
    alpha_entropy = -(alpha * alpha.clamp_min(1e-12).log()).sum(-1).mean().item()
    alpha_max_weight = alpha.max(-1)[0].mean().item()
    alpha_sparsity = (alpha > 1e-3).sum(-1).float().mean().item()
    
    print(f"   📈 Alpha diagnostics:")
    print(f"      Mean entropy: {alpha_entropy:.4f}")
    print(f"      Mean max weight: {alpha_max_weight:.4f}")
    print(f"      Mean sparsity: {alpha_sparsity:.1f} bases")
    
    return {
        'test_name': 'pure_a_refit_bound',
        'optimal_mse': mse.item(),
        'optimal_rel_frob': rel_frob,
        'alpha_diagnostics': {
            'entropy': alpha_entropy,
            'max_weight': alpha_max_weight,
            'sparsity': alpha_sparsity
        },
        'analysis': {
            'alpha_b_problematic': rel_frob > 0.8
        }
    }


def debug_plain_mod_only(
    W: torch.Tensor,
    B_targets: torch.Tensor,
    bank_module: GloBEBank,
    cfg: InitConfig,
    num_mod_steps: int = 10,
) -> Dict[str, Any]:
    """
    Test 3: Plain MOD only
    Run MOD only for 10 iterations with warm targets. If relF doesn't move, 
    your coding scores or MOD solve are wrong (shapes/transpose).
    """
    print(f"\n🔬 DEBUG TEST 3: Plain MOD only ({num_mod_steps} steps)")
    
    device = bank_module.bases.device
    E, p, d = W.shape
    m, r, _ = bank_module.bases.shape
    
    # Use warm start targets as fixed coding targets
    S = B_targets  # E × r × d
    
    results = []
    
    for step in range(num_mod_steps):
        # Step 1: Coding (compute alpha from fixed targets S)
        D = bank_module.bases.view(m, -1).T  # (rd) × m
        X = S.view(E, -1)  # E × (rd)
        C = X @ D  # E × m (scores)
        alpha = F.softmax(C / cfg.tau, dim=-1)  # E × m (simplex)
        A_codes = alpha.T  # m × E
        
        # Step 2: Compute reconstruction error (no A adaptation, just direct)
        M_lin = torch.einsum('em,mrd->erd', alpha, bank_module.bases)  # E × r × d
        M = bank_module.activation(M_lin)
        
        # For MOD-only test, assume A = I (or some fixed simple mapping)
        # Actually, let's use the targets directly for comparison
        W_hat_direct = M_lin  # Direct comparison with linear mixing
        diff_direct = W_hat_direct - S  # Compare mixed bases to targets
        mse_direct = (diff_direct ** 2).mean()
        rel_frob_direct = (diff_direct.norm() / S.norm().clamp_min(1e-12)).item()
        
        # Step 3: MOD update (bank update only)
        # Compute residual in target space
        residual = S - M_lin  # E × r × d (target - current mixing)
        X_delta = residual.view(E, -1).T  # (rd) × E
        
        # MOD bank update (same as in am_step but simplified)
        I_m = torch.eye(m, device=device, dtype=W.dtype)
        Mmat = A_codes @ A_codes.T + cfg.lambda_B * I_m  # m × m
        rhs = (A_codes @ X_delta.T).T  # (rd) × m
        
        # Solve for ΔB
        try:
            DeltaB_flat = torch.linalg.solve(Mmat.float(), rhs.float().T).T.to(W.dtype)
        except RuntimeError:
            # Add regularization if singular
            Mmat_reg = Mmat.float() + 1e-6 * torch.eye(m, device=device, dtype=torch.float32)
            DeltaB_flat = torch.linalg.solve(Mmat_reg, rhs.float().T).T.to(W.dtype)
        
        # Apply update
        DeltaB = DeltaB_flat.T.view(m, r, d)  # m × r × d
        new_bank = bank_module.bases + cfg.eta * DeltaB
        
        # Normalize atoms
        norms = new_bank.view(m, -1).norm(dim=-1, keepdim=True).clamp_min(1e-8)
        new_bank = new_bank / norms.view(m, 1, 1)
        
        # Update bank
        with torch.no_grad():
            bank_module.bases.copy_(new_bank)
        
        results.append({
            'step': step,
            'mse_direct': mse_direct.item(),
            'rel_frob_direct': rel_frob_direct,
            'alpha_entropy': -(alpha * alpha.clamp_min(1e-12).log()).sum(-1).mean().item(),
            'delta_norm': DeltaB.norm().item(),
        })
        
        print(f"   Step {step+1}: MSE={mse_direct:.8f}, RelFrob={rel_frob_direct:.6f}, ΔB_norm={DeltaB.norm():.6f}")
    
    # Analysis
    initial_rel_frob = results[0]['rel_frob_direct']
    final_rel_frob = results[-1]['rel_frob_direct']
    rel_frob_change = final_rel_frob - initial_rel_frob
    
    print(f"   📊 Plain MOD Results:")
    print(f"      Initial RelFrob: {initial_rel_frob:.6f}")
    print(f"      Final RelFrob: {final_rel_frob:.6f}")
    print(f"      Change: {rel_frob_change:.6f}")
    
    if abs(rel_frob_change) < 0.01:  # Very little change
        print(f"   ⚠️  FINDING: RelFrob barely changed with MOD-only!")
        print(f"      This suggests coding scores or MOD solve may be wrong.")
        print(f"      Check tensor shapes and transpose operations.")
    else:
        print(f"   ✅ MOD-only produced reasonable RelFrob changes.")
        print(f"      Coding and MOD solve appear to be working.")
    
    return {
        'test_name': 'plain_mod_only',
        'results': results,
        'analysis': {
            'initial_rel_frob': initial_rel_frob,
            'final_rel_frob': final_rel_frob,
            'change': rel_frob_change,
            'coding_mod_suspect': abs(rel_frob_change) < 0.01
        }
    }


def debug_j_statistics(
    W: torch.Tensor,
    bank_module: GloBEBank,
    alpha: torch.Tensor,
    A: torch.Tensor,
    cfg: InitConfig,
) -> Dict[str, Any]:
    """
    Test 4: J statistics
    Log mean/min of J and % elements at the floor. If a large fraction are at j_min,
    try a slightly larger τ to reduce saturation, or a larger j_min (e.g., 3e-2).
    """
    print(f"\n🔬 DEBUG TEST 4: J statistics analysis")
    
    # Compute current state
    M_lin = torch.einsum('em,mrd->erd', alpha, bank_module.bases)  # E × r × d
    M = bank_module.activation(M_lin)  # φ(M_lin)
    
    # Compute Jacobian
    J_raw = _silu_prime(M_lin)  # E × r × d (raw Jacobian)
    J_clamped = J_raw.clamp_min(cfg.j_min)  # E × r × d (clamped)
    
    # Statistics
    j_min_count = (J_raw <= cfg.j_min).sum().item()
    j_total_count = J_raw.numel()
    j_min_fraction = j_min_count / j_total_count
    
    j_raw_mean = J_raw.mean().item()
    j_raw_min = J_raw.min().item()
    j_raw_max = J_raw.max().item()
    j_raw_std = J_raw.std().item()
    
    j_clamped_mean = J_clamped.mean().item()
    j_clamped_min = J_clamped.min().item()
    
    print(f"   📊 Jacobian Statistics:")
    print(f"      Raw J - Mean: {j_raw_mean:.6f}, Min: {j_raw_min:.6f}, Max: {j_raw_max:.6f}, Std: {j_raw_std:.6f}")
    print(f"      Clamped J - Mean: {j_clamped_mean:.6f}, Min: {j_clamped_min:.6f}")
    print(f"      Elements at floor (≤{cfg.j_min}): {j_min_count}/{j_total_count} ({j_min_fraction:.2%})")
    
    # Analysis
    if j_min_fraction > 0.3:  # More than 30% at floor
        print(f"   ⚠️  FINDING: {j_min_fraction:.1%} of J elements are at the floor!")
        print(f"      Consider:")
        print(f"      - Increasing τ from {cfg.tau} to reduce saturation")
        print(f"      - Increasing j_min from {cfg.j_min} to ~3e-2")
    else:
        print(f"   ✅ J statistics look reasonable ({j_min_fraction:.1%} at floor).")
    
    # Additional activation statistics
    m_lin_mean = M_lin.mean().item()
    m_lin_std = M_lin.std().item()
    m_mean = M.mean().item()
    m_std = M.std().item()
    
    print(f"   📈 Activation Statistics:")
    print(f"      M_lin - Mean: {m_lin_mean:.6f}, Std: {m_lin_std:.6f}")
    print(f"      M (post-φ) - Mean: {m_mean:.6f}, Std: {m_std:.6f}")
    
    return {
        'test_name': 'j_statistics',
        'jacobian_stats': {
            'raw_mean': j_raw_mean,
            'raw_min': j_raw_min,
            'raw_max': j_raw_max,
            'raw_std': j_raw_std,
            'clamped_mean': j_clamped_mean,
            'clamped_min': j_clamped_min,
            'floor_fraction': j_min_fraction,
            'floor_count': j_min_count,
            'total_count': j_total_count,
        },
        'activation_stats': {
            'pre_activation_mean': m_lin_mean,
            'pre_activation_std': m_lin_std,
            'post_activation_mean': m_mean,
            'post_activation_std': m_std,
        },
        'analysis': {
            'j_floor_problematic': j_min_fraction > 0.3,
            'recommended_tau': cfg.tau * 1.5 if j_min_fraction > 0.3 else cfg.tau,
            'recommended_j_min': 3e-2 if j_min_fraction > 0.3 else cfg.j_min,
        }
    }


def analyze_mse_vs_relf_discrepancy(
    W: torch.Tensor,
    bank_module: GloBEBank,
    alpha: torch.Tensor,
    A: torch.Tensor,
) -> Dict[str, Any]:
    """
    Investigate why MSE is small but relative Frobenius error stays around 0.95.
    This suggests a scaling or normalization issue.
    """
    print(f"\n🔬 DEBUG: MSE vs RelF discrepancy analysis")
    
    # Compute reconstruction
    M_lin = torch.einsum('em,mrd->erd', alpha, bank_module.bases)  # E × r × d
    M = bank_module.activation(M_lin)  # φ(M_lin)
    W_hat = torch.einsum('epr,erd->epd', A, M)  # E × p × d
    
    # Compute errors
    diff = W_hat - W  # E × p × d
    mse = (diff ** 2).mean()
    frobenius_error = diff.norm()
    frobenius_norm_W = W.norm()
    rel_frob = (frobenius_error / frobenius_norm_W.clamp_min(1e-12)).item()
    
    # Scaling analysis
    W_scale = W.norm()
    W_hat_scale = W_hat.norm()
    scale_ratio = W_hat_scale / W_scale
    
    # Element-wise statistics
    W_mean = W.mean().item()
    W_std = W.std().item()
    W_hat_mean = W_hat.mean().item()
    W_hat_std = W_hat.std().item()
    
    diff_mean = diff.mean().item()
    diff_std = diff.std().item()
    
    print(f"   📊 MSE vs RelF Analysis:")
    print(f"      MSE: {mse:.8f}")
    print(f"      Frobenius error: {frobenius_error:.6f}")
    print(f"      ||W||_F: {frobenius_norm_W:.6f}")
    print(f"      RelF: {rel_frob:.6f}")
    print(f"   📏 Scale Analysis:")
    print(f"      ||W||: {W_scale:.6f}")
    print(f"      ||W_hat||: {W_hat_scale:.6f}")
    print(f"      Scale ratio: {scale_ratio:.6f}")
    print(f"   📈 Distribution Analysis:")
    print(f"      W: mean={W_mean:.6f}, std={W_std:.6f}")
    print(f"      W_hat: mean={W_hat_mean:.6f}, std={W_hat_std:.6f}")
    print(f"      Diff: mean={diff_mean:.6f}, std={diff_std:.6f}")
    
    # Potential issues
    issues = []
    if abs(scale_ratio - 1.0) > 0.1:
        issues.append(f"Scale mismatch: W_hat is {scale_ratio:.2f}x the scale of W")
    
    if abs(diff_mean) > 0.01:
        issues.append(f"Large mean bias: {diff_mean:.6f}")
    
    if rel_frob > 0.8 and mse < 1e-4:
        issues.append("MSE is tiny but RelF is large - suggests scaling/normalization problem")
    
    if issues:
        print(f"   ⚠️  FINDINGS:")
        for issue in issues:
            print(f"      - {issue}")
    else:
        print(f"   ✅ No obvious scaling issues detected.")
    
    return {
        'test_name': 'mse_vs_relf_analysis',
        'metrics': {
            'mse': mse.item(),
            'frobenius_error': frobenius_error.item(),
            'frobenius_norm_W': frobenius_norm_W.item(),
            'rel_frob': rel_frob,
            'scale_ratio': scale_ratio.item(),
        },
        'distributions': {
            'W_mean': W_mean,
            'W_std': W_std,
            'W_hat_mean': W_hat_mean,
            'W_hat_std': W_hat_std,
            'diff_mean': diff_mean,
            'diff_std': diff_std,
        },
        'issues': issues
    }


def run_comprehensive_debug(
    model_name: str = "Qwen/Qwen1.5-MoE-A2.7B",
    layer_idx: int = 0,
    family: str = "up",
    device: str = "auto",
    rank: int = 64,
    num_bases: int = 16,
    num_test_experts: int = 8,
) -> Dict[str, Any]:
    """Run all debugging tests on a specific layer and family."""
    
    print(f"🚀 Starting comprehensive AlternatingBankTrainer debugging")
    print(f"   Model: {model_name}")
    print(f"   Layer: {layer_idx}")
    print(f"   Family: {family}")
    print(f"   Test experts: {num_test_experts}")
    print(f"   Rank: {rank}, Bases: {num_bases}")
    
    # Auto-detect device
    if device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)
    
    print(f"   Device: {device}")
    
    # Extract expert weights
    print(f"📥 Loading expert weights...")
    expert_weights = extract_expert_weights(model_name, max_experts=num_test_experts)
    
    if family not in expert_weights:
        raise ValueError(f"No weights found for family {family}. Available: {list(expert_weights.keys())}")
    
    weights_list = expert_weights[family][:num_test_experts]
    W = torch.stack(weights_list).to(device, dtype=torch.float32)
    
    print(f"   Loaded weights: {W.shape}")
    
    # Set up configuration
    seeding_config = SeedingConfig(method=SeedingMethod.TS_PCA)
    cfg = InitConfig(
        rank=rank,
        num_bases=num_bases,
        device=device,
        dtype=torch.float32,
        target_support=8,
        epsilon=1e-4,
        seeding_config=seeding_config,
        tau=1.0,
        lambda_A=1e-4,
        lambda_B=1e-4,
        lambda_T=1e-4,
        j_min=1e-3,
        eta=0.5,
        route_w_start=5,
    )
    
    # Create warm start
    print(f"🌱 Creating warm start...")
    A0, B0, energy, seeding_metrics = create_warm_start(W, rank, seeding_config)
    bank0, alpha0 = build_initial_bank(B0, cfg)
    
    # Create bank module
    _, _, hidden_dim = W.shape
    bank_module = GloBEBank(num_bases, rank, hidden_dim, activation="silu").to(device, dtype=torch.float32)
    with torch.no_grad():
        bank_module.bases.copy_(bank0)
    
    alpha = alpha0.clone()
    A = A0.to(device, dtype=torch.float32)
    
    print(f"   Initial energy: {energy.mean():.6f} ± {energy.std():.6f}")
    
    # Run all debugging tests
    debug_results = {}
    
    # Test 1: Identity φ ablation
    debug_results['identity_phi'] = debug_am_step_identity_phi(
        W, B0, bank_module, alpha, A, cfg, num_identity_steps=5
    )
    
    # Test 2: Pure A-refit bound
    debug_results['pure_a_refit'] = debug_pure_a_refit_bound(
        W, bank_module, alpha, cfg
    )
    
    # Test 3: Plain MOD only
    debug_results['plain_mod'] = debug_plain_mod_only(
        W, B0, bank_module, cfg, num_mod_steps=10
    )
    
    # Test 4: J statistics
    debug_results['j_statistics'] = debug_j_statistics(
        W, bank_module, alpha, A, cfg
    )
    
    # Additional: MSE vs RelF analysis
    debug_results['mse_vs_relf'] = analyze_mse_vs_relf_discrepancy(
        W, bank_module, alpha, A
    )
    
    # Summary
    print(f"\n📋 DEBUG SUMMARY:")
    
    # Collect findings
    findings = []
    if debug_results['identity_phi']['analysis']['jw_wiring_suspect']:
        findings.append("🔴 Identity φ test: JW wiring (J or Δ construction) may be problematic")
    
    if debug_results['pure_a_refit']['analysis']['alpha_b_problematic']:
        findings.append("🔴 Pure A-refit test: α/B (coding/bases) appear problematic")
    
    if debug_results['plain_mod']['analysis']['coding_mod_suspect']:
        findings.append("🔴 Plain MOD test: Coding scores or MOD solve may be wrong")
    
    if debug_results['j_statistics']['analysis']['j_floor_problematic']:
        findings.append("🔴 J statistics: Too many elements at floor, consider increasing τ or j_min")
    
    if debug_results['mse_vs_relf']['issues']:
        findings.append(f"🔴 MSE vs RelF: {'; '.join(debug_results['mse_vs_relf']['issues'])}")
    
    if not findings:
        findings.append("✅ No major issues detected in debugging tests")
    
    for finding in findings:
        print(f"   {finding}")
    
    return {
        'config': {
            'model_name': model_name,
            'layer_idx': layer_idx,
            'family': family,
            'device': str(device),
            'rank': rank,
            'num_bases': num_bases,
            'num_test_experts': num_test_experts,
        },
        'debug_results': debug_results,
        'summary_findings': findings,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug AlternatingBankTrainer am_step")
    parser.add_argument("--model", default="Qwen/Qwen1.5-MoE-A2.7B", help="Model name")
    parser.add_argument("--layer", type=int, default=0, help="Layer index to test")
    parser.add_argument("--family", choices=["up", "gate"], default="up", help="Weight family")
    parser.add_argument("--device", default="auto", help="Device to use")
    parser.add_argument("--rank", type=int, default=64, help="Rank for decomposition")
    parser.add_argument("--num-bases", type=int, default=16, help="Number of bases")
    parser.add_argument("--num-experts", type=int, default=8, help="Number of experts to test")
    parser.add_argument("--save-results", help="Save results to JSON file")
    
    args = parser.parse_args()
    
    try:
        results = run_comprehensive_debug(
            model_name=args.model,
            layer_idx=args.layer,
            family=args.family,
            device=args.device,
            rank=args.rank,
            num_bases=args.num_bases,
            num_test_experts=args.num_experts,
        )
        
        if args.save_results:
            with open(args.save_results, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to {args.save_results}")
        
    except Exception as e:
        print(f"❌ Error during debugging: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
