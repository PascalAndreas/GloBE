#!/usr/bin/env python3
"""
Test script for comparing different warm start methods for GloBE bank initialization.

This script evaluates all available seeding methods on a small subset of experts
to understand their performance characteristics before running full training.
"""

import os
import sys
import time
from pathlib import Path
import torch
import argparse
from typing import Dict, Any, List
# import pandas as pd  # Optional for simple testing

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from globe.data.naming_qwen15 import extract_expert_weights
from globe.init.warm_start import SeedingMethod, SeedingConfig, get_seeding_method_info, create_warm_start_proxies
from globe.init.init_bank import create_warm_start, build_initial_bank, InitConfig
from globe.init.zscore import normalize_expert_families


def get_model_dtype(model_name: str) -> torch.dtype:
    """Get the appropriate dtype for the model."""
    if "qwen" in model_name.lower():
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        else:
            return torch.float16
    return torch.float32


def analyze_support_stats(codes: torch.Tensor, epsilon: float = 1e-4) -> Dict[str, float]:
    """Analyze support statistics for mixture codes."""
    support_sizes = (codes > epsilon).sum(dim=-1).float()
    return {
        "mean_support": support_sizes.mean().item(),
        "median_support": support_sizes.median().item(),
        "p95_support": support_sizes.quantile(0.95).item(),
        "min_support": support_sizes.min().item(),
        "max_support": support_sizes.max().item(),
        "sparsity": (codes < epsilon).float().mean().item(),
    }


def test_seeding_method(
    weights: torch.Tensor,
    seeding_method: SeedingMethod,
    rank: int,
    num_bases: int,
    device: torch.device,
    dtype: torch.dtype,
    epsilon: float = 1e-4
) -> Dict[str, Any]:
    """Test a single seeding method and return comprehensive stats."""
    
    seeding_config = SeedingConfig(method=seeding_method)
    
    # Time the warm start creation
    start_time = time.time()
    try:
        A0, B0, energy, seeding_metrics = create_warm_start(weights, rank, seeding_config)
        warm_start_time = time.time() - start_time
        
        # Build initial bank and codes
        cfg = InitConfig(
            rank=rank,
            num_bases=num_bases,
            device=device,
            dtype=dtype,
            epsilon=epsilon
        )
        
        bank_start_time = time.time()
        bank0, codes0 = build_initial_bank(B0, cfg)
        bank_time = time.time() - bank_start_time
        
        total_time = warm_start_time + bank_time
        
        # Analyze support statistics
        support_stats = analyze_support_stats(codes0, epsilon)
        
        # Compute reconstruction quality
        E, p, d = weights.shape
        reconstructions = []
        for i in range(E):
            mixed_basis = torch.einsum('m,mrd->rd', codes0[i], bank0)
            reconstruction = torch.matmul(A0[i], mixed_basis)
            reconstructions.append(reconstruction)
        
        reconstructed_weights = torch.stack(reconstructions)
        mse = torch.nn.functional.mse_loss(reconstructed_weights, weights).item()
        relative_error = (torch.norm(reconstructed_weights - weights) / torch.norm(weights)).item()
        
        results = {
            "method": seeding_method.value,
            "success": True,
            "warm_start_time": warm_start_time,
            "bank_time": bank_time,
            "total_time": total_time,
            "initial_mse": mse,
            "initial_relative_error": relative_error,
            "energy_captured_mean": energy.mean().item(),
            "energy_captured_std": energy.std().item(),
            **support_stats,
            **{f"seeding_{k}": v for k, v in seeding_metrics.items() if isinstance(v, (int, float))}
        }
        
        return results
        
    except Exception as e:
        return {
            "method": seeding_method.value,
            "success": False,
            "error": str(e),
            "warm_start_time": 0,
            "bank_time": 0,
            "total_time": 0,
        }


def main():
    parser = argparse.ArgumentParser(description="Test different warm start methods")
    parser.add_argument("--model", default="Qwen/Qwen1.5-MoE-A2.7B", help="Model name or path")
    parser.add_argument("--max-experts", type=int, default=32, help="Max experts to test (for speed)")
    parser.add_argument("--rank-ratio", type=float, default=0.75, help="Rank as ratio of projection dim")
    parser.add_argument("--basis-ratio", type=float, default=0.5, help="Num bases as ratio of experts")
    parser.add_argument("--device", default="auto", help="Device to use")
    parser.add_argument("--families", nargs="+", default=["up"], help="Families to test")
    parser.add_argument("--epsilon", type=float, default=1e-4, help="Sparsity threshold")
    
    args = parser.parse_args()
    
    print("🌟 GloBE Warm Start Method Comparison")
    print("=" * 60)
    
    # Auto-detect device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"🖥️  Using CUDA device: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            print(f"🖥️  Using MPS device")
        else:
            device = torch.device("cpu")
            print(f"🖥️  Using CPU device")
    else:
        device = torch.device(args.device)
    
    dtype = get_model_dtype(args.model)
    print(f"🔢 Using dtype: {dtype}")
    
    # Extract expert weights
    print(f"\n📤 Extracting expert weights (max: {args.max_experts})...")
    expert_weights = extract_expert_weights(
        args.model,
        include_shared=False,
        max_experts=args.max_experts
    )
    
    # Normalize experts
    normalized_weights, normalizer = normalize_expert_families(expert_weights)
    
    # Get available seeding methods
    available_methods = list(SeedingMethod)
    method_info = get_seeding_method_info()
    
    print(f"\n🧪 Testing {len(available_methods)} seeding methods on {len(args.families)} families...")
    print(f"   Methods: {[m.value for m in available_methods]}")
    print(f"   Families: {args.families}")
    
    all_results = []
    
    for family in args.families:
        if family not in normalized_weights:
            print(f"⚠️  Skipping family '{family}' - not found in weights")
            continue
            
        weights = torch.stack(normalized_weights[family]).to(device, dtype)
        E, p, d = weights.shape
        
        # Calculate dimensions
        rank = int(args.rank_ratio * p)
        num_bases = int(args.basis_ratio * E)
        
        print(f"\n📊 Testing family: {family.upper()}")
        print(f"   Shape: {weights.shape} (E={E}, p={p}, d={d})")
        print(f"   Rank: {rank} ({args.rank_ratio:.1%} of projection dim)")
        print(f"   Num bases: {num_bases} ({args.basis_ratio:.1%} of experts)")
        print(f"   Epsilon: {args.epsilon}")
        
        family_results = []
        
        for method in available_methods:
            print(f"\n   🔬 Testing {method.value}...")
            info = method_info[method.value]
            print(f"      {info['description']}")
            print(f"      Speed: {info['speed']}, Quality: {info['quality']}, MPS: {info['mps_friendly']}")
            
            result = test_seeding_method(
                weights, method, rank, num_bases, device, dtype, args.epsilon
            )
            result["family"] = family
            result["num_experts"] = E
            result["projection_dim"] = p
            result["hidden_dim"] = d
            result["rank"] = rank
            result["num_bases"] = num_bases
            
            if result["success"]:
                print(f"      ✅ Success! Time: {result['total_time']:.3f}s")
                print(f"         MSE: {result['initial_mse']:.6f}, Rel Error: {result['initial_relative_error']:.6f}")
                print(f"         Support: mean={result['mean_support']:.1f}, median={result['median_support']:.1f}")
                print(f"         Sparsity: {result['sparsity']:.3f}")
            else:
                print(f"      ❌ Failed: {result['error']}")
            
            family_results.append(result)
            all_results.append(result)
    
    # Create summary table
    print(f"\n📋 Summary Table")
    print("=" * 100)
    
    successful_results = [r for r in all_results if r["success"]]
    if successful_results:
        # Simple table without pandas
        print(f"{'Method':<15} {'Time(s)':<8} {'MSE':<10} {'RelErr':<8} {'MeanSupp':<9} {'MedianSupp':<11} {'Sparsity':<9}")
        print("-" * 80)
        
        for result in successful_results:
            print(f"{result['method']:<15} "
                  f"{result['total_time']:<8.3f} "
                  f"{result['initial_mse']:<10.6f} "
                  f"{result['initial_relative_error']:<8.4f} "
                  f"{result['mean_support']:<9.1f} "
                  f"{result['median_support']:<11.1f} "
                  f"{result['sparsity']:<9.3f}")
        
        # Find best methods per metric
        print(f"\n🏆 Best Methods by Metric:")
        family_results = [r for r in successful_results if r["family"] == args.families[0]]
        if family_results:
            fastest = min(family_results, key=lambda x: x['total_time'])
            lowest_mse = min(family_results, key=lambda x: x['initial_mse'])
            highest_support = max(family_results, key=lambda x: x['mean_support'])
            best_energy = max(family_results, key=lambda x: x['energy_captured_mean'])
            
            print(f"   Fastest: {fastest['method']} ({fastest['total_time']:.3f}s)")
            print(f"   Lowest MSE: {lowest_mse['method']} ({lowest_mse['initial_mse']:.6f})")
            print(f"   Highest Support: {highest_support['method']} ({highest_support['mean_support']:.1f})")
            print(f"   Best Energy: {best_energy['method']} ({best_energy['energy_captured_mean']:.4f})")
    else:
        print("❌ No successful results to display")
    
    print(f"\n✅ Warm start method comparison completed!")


if __name__ == "__main__":
    main()
