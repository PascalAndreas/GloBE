#!/usr/bin/env python3
"""
Test script to compare original am_step vs fixed versions.
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from typing import Dict, Any

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from globe.data.naming_qwen15 import extract_expert_weights
from globe.init.init_bank import (
    InitConfig, create_warm_start, build_initial_bank, am_step
)
from globe.init.warm_start import SeedingMethod, SeedingConfig
from globe.modules.globe_bank import GloBEBank
from fix_am_step_scaling import am_step_fixed, am_step_fixed_v2, build_initial_bank_fixed


def test_am_step_comparison(
    model_name: str = "Qwen/Qwen1.5-MoE-A2.7B",
    family: str = "up",
    num_experts: int = 4,
    rank: int = 32,
    num_bases: int = 8,
    num_steps: int = 10,
    device: str = "auto"
):
    """Compare original vs fixed am_step implementations."""
    
    print(f"🔬 Testing am_step fixes")
    print(f"   Experts: {num_experts}, Rank: {rank}, Bases: {num_bases}")
    
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
    
    # Load data
    expert_weights = extract_expert_weights(model_name, max_experts=num_experts)
    weights_list = expert_weights[family][:num_experts]
    W = torch.stack(weights_list).to(device, dtype=torch.float32)
    
    print(f"   Loaded weights: {W.shape}")
    print(f"   Original weight norm: {W.norm():.6f}")
    
    # Configuration
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
    A0, B0, energy, seeding_metrics = create_warm_start(W, rank, seeding_config)
    print(f"   Initial energy: {energy.mean():.6f} ± {energy.std():.6f}")
    
    # Test 1: Original implementation
    print(f"\n📊 TEST 1: Original implementation")
    bank0_orig, alpha0_orig = build_initial_bank(B0, cfg)
    
    _, _, hidden_dim = W.shape
    bank_module_orig = GloBEBank(num_bases, rank, hidden_dim, activation="silu").to(device, dtype=torch.float32)
    with torch.no_grad():
        bank_module_orig.bases.copy_(bank0_orig)
    
    alpha_orig = alpha0_orig.clone()
    A_orig = A0.to(device, dtype=torch.float32)
    
    print(f"   Initial alpha sum: {alpha_orig.sum(dim=-1).mean():.6f}")
    print(f"   Initial bank norm: {bank_module_orig.bases.norm():.6f}")
    
    orig_results = []
    for step in range(num_steps):
        alpha_orig, A_orig, loss, med, metrics = am_step(
            W, B0, bank_module_orig, alpha_orig, A_orig, cfg, 
            step=step, log_metrics=True
        )
        
        rel_frob = metrics.get("recon/relative_frobenius_error", float("nan"))
        scale_ratio = metrics.get('scale/scale_ratio', float("nan"))
        
        orig_results.append({
            'step': step,
            'loss': loss,
            'rel_frob': rel_frob,
            'alpha_sum_mean': alpha_orig.sum(dim=-1).mean().item(),
            'bank_norm': bank_module_orig.bases.norm().item(),
        })
        
        print(f"   Step {step+1}: Loss={loss:.6f}, RelFrob={rel_frob:.6f}, αSum={alpha_orig.sum(dim=-1).mean():.4f}")
    
    # Test 2: Fixed implementation v1 (with basis normalization + alpha rescaling)
    print(f"\n📊 TEST 2: Fixed implementation v1 (normalize bases + rescale alpha)")
    bank0_fix1, alpha0_fix1 = build_initial_bank_fixed(B0, cfg)
    
    bank_module_fix1 = GloBEBank(num_bases, rank, hidden_dim, activation="silu").to(device, dtype=torch.float32)
    with torch.no_grad():
        bank_module_fix1.bases.copy_(bank0_fix1)
    
    alpha_fix1 = alpha0_fix1.clone()
    A_fix1 = A0.to(device, dtype=torch.float32)
    
    print(f"   Initial alpha sum: {alpha_fix1.sum(dim=-1).mean():.6f}")
    print(f"   Initial bank norm: {bank_module_fix1.bases.norm():.6f}")
    
    fix1_results = []
    for step in range(num_steps):
        alpha_fix1, A_fix1, loss, med, metrics = am_step_fixed(
            W, B0, bank_module_fix1, alpha_fix1, A_fix1, cfg, 
            step=step, log_metrics=True
        )
        
        rel_frob = metrics.get("recon/relative_frobenius_error", float("nan"))
        scale_ratio = metrics.get('scale/scale_ratio', float("nan"))
        
        fix1_results.append({
            'step': step,
            'loss': loss,
            'rel_frob': rel_frob,
            'scale_ratio': scale_ratio,
            'alpha_sum_mean': alpha_fix1.sum(dim=-1).mean().item(),
            'bank_norm': bank_module_fix1.bases.norm().item(),
        })
        
        print(f"   Step {step+1}: Loss={loss:.6f}, RelFrob={rel_frob:.6f}, Scale={scale_ratio:.4f}, αSum={alpha_fix1.sum(dim=-1).mean():.4f}")
    
    # Test 3: Fixed implementation v2 (no basis normalization)
    print(f"\n📊 TEST 3: Fixed implementation v2 (no basis normalization)")
    bank0_fix2, alpha0_fix2 = build_initial_bank_fixed(B0, cfg)
    
    bank_module_fix2 = GloBEBank(num_bases, rank, hidden_dim, activation="silu").to(device, dtype=torch.float32)
    with torch.no_grad():
        bank_module_fix2.bases.copy_(bank0_fix2)
    
    alpha_fix2 = alpha0_fix2.clone()
    A_fix2 = A0.to(device, dtype=torch.float32)
    
    print(f"   Initial alpha sum: {alpha_fix2.sum(dim=-1).mean():.6f}")
    print(f"   Initial bank norm: {bank_module_fix2.bases.norm():.6f}")
    
    fix2_results = []
    for step in range(num_steps):
        alpha_fix2, A_fix2, loss, med, metrics = am_step_fixed_v2(
            W, B0, bank_module_fix2, alpha_fix2, A_fix2, cfg, 
            step=step, log_metrics=True
        )
        
        rel_frob = metrics.get("recon/relative_frobenius_error", float("nan"))
        scale_ratio = metrics.get('scale/scale_ratio', float("nan"))
        
        fix2_results.append({
            'step': step,
            'loss': loss,
            'rel_frob': rel_frob,
            'scale_ratio': scale_ratio,
            'alpha_sum_mean': alpha_fix2.sum(dim=-1).mean().item(),
            'bank_norm': bank_module_fix2.bases.norm().item(),
        })
        
        print(f"   Step {step+1}: Loss={loss:.6f}, RelFrob={rel_frob:.6f}, Scale={scale_ratio:.4f}, αSum={alpha_fix2.sum(dim=-1).mean():.4f}")
    
    # Summary comparison
    print(f"\n📋 COMPARISON SUMMARY:")
    
    orig_final = orig_results[-1]
    fix1_final = fix1_results[-1]
    fix2_final = fix2_results[-1]
    
    print(f"   Original:    RelFrob={orig_final['rel_frob']:.6f}, Loss={orig_final['loss']:.6f}")
    print(f"   Fixed v1:    RelFrob={fix1_final['rel_frob']:.6f}, Loss={fix1_final['loss']:.6f}, Scale={fix1_final['scale_ratio']:.4f}")
    print(f"   Fixed v2:    RelFrob={fix2_final['rel_frob']:.6f}, Loss={fix2_final['loss']:.6f}, Scale={fix2_final['scale_ratio']:.4f}")
    
    # Determine best approach
    best_rel_frob = min(orig_final['rel_frob'], fix1_final['rel_frob'], fix2_final['rel_frob'])
    
    if fix1_final['rel_frob'] == best_rel_frob:
        print(f"   🏆 WINNER: Fixed v1 (normalize bases + rescale alpha)")
    elif fix2_final['rel_frob'] == best_rel_frob:
        print(f"   🏆 WINNER: Fixed v2 (no basis normalization)")
    else:
        print(f"   🏆 WINNER: Original (no improvement needed)")
    
    return {
        'original': orig_results,
        'fixed_v1': fix1_results,
        'fixed_v2': fix2_results,
        'summary': {
            'best_approach': 'fixed_v1' if fix1_final['rel_frob'] == best_rel_frob else 'fixed_v2' if fix2_final['rel_frob'] == best_rel_frob else 'original',
            'best_rel_frob': best_rel_frob,
            'improvement': orig_final['rel_frob'] - best_rel_frob,
        }
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test am_step fixes")
    parser.add_argument("--model", default="Qwen/Qwen1.5-MoE-A2.7B", help="Model name")
    parser.add_argument("--family", choices=["up", "gate"], default="up", help="Weight family")
    parser.add_argument("--device", default="auto", help="Device to use")
    parser.add_argument("--rank", type=int, default=32, help="Rank for decomposition")
    parser.add_argument("--num-bases", type=int, default=8, help="Number of bases")
    parser.add_argument("--num-experts", type=int, default=4, help="Number of experts to test")
    parser.add_argument("--num-steps", type=int, default=10, help="Number of AM steps")
    
    args = parser.parse_args()
    
    try:
        results = test_am_step_comparison(
            model_name=args.model,
            family=args.family,
            num_experts=args.num_experts,
            rank=args.rank,
            num_bases=args.num_bases,
            num_steps=args.num_steps,
            device=args.device,
        )
        
        print(f"\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
