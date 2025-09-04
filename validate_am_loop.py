#!/usr/bin/env python3
"""
Quick validation script for the Alternating Minimization loop.

This script creates synthetic expert weights and validates the AM loop
without requiring model downloads. Useful for debugging and development.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from globe.train.fit_banks import AlternatingBankTrainer, fold_normalization_into_adapters
from globe.init.zscore import normalize_expert_families


def create_synthetic_experts(
    num_experts: int = 64,
    intermediate_dim: int = 1024,  # p
    hidden_dim: int = 512,         # d
    rank: int = 64,                # r
    noise_level: float = 0.1,
) -> dict:
    """Create synthetic expert weights with known low-rank structure."""
    print(f"🔧 Creating {num_experts} synthetic experts...")
    print(f"   - Dimensions: {intermediate_dim} × {hidden_dim}")
    print(f"   - True rank: {rank}")
    print(f"   - Noise level: {noise_level}")
    
    torch.manual_seed(42)  # For reproducibility
    
    up_experts = []
    gate_experts = []
    
    # Create some shared low-rank components to make the problem interesting
    num_components = rank // 4
    shared_components_up = torch.randn(num_components, rank, hidden_dim)
    shared_components_gate = torch.randn(num_components, rank, hidden_dim)
    
    for i in range(num_experts):
        # Create low-rank structure with some shared components
        A_up = torch.randn(intermediate_dim, rank) * 0.5
        A_gate = torch.randn(intermediate_dim, rank) * 0.5
        
        # Mix some shared components
        mixing_weights = torch.softmax(torch.randn(num_components), dim=0)
        B_up = torch.einsum('c,crd->rd', mixing_weights, shared_components_up)
        B_gate = torch.einsum('c,crd->rd', mixing_weights, shared_components_gate)
        
        # Add some expert-specific components
        B_up += torch.randn(rank, hidden_dim) * 0.3
        B_gate += torch.randn(rank, hidden_dim) * 0.3
        
        # Create expert weights
        W_up = torch.matmul(A_up, B_up) + torch.randn(intermediate_dim, hidden_dim) * noise_level
        W_gate = torch.matmul(A_gate, B_gate) + torch.randn(intermediate_dim, hidden_dim) * noise_level
        
        up_experts.append(W_up)
        gate_experts.append(W_gate)
    
    return {"up": up_experts, "gate": gate_experts}


def validate_am_loop():
    """Validate the alternating minimization loop."""
    print("🧪 Validating Alternating Minimization Loop")
    print("=" * 50)
    
    # Create synthetic data
    hidden_dim = 128
    intermediate_dim = 256
    rank = 32
    
    expert_weights = create_synthetic_experts(
        num_experts=32,     # Smaller for quick testing
        intermediate_dim=intermediate_dim,
        hidden_dim=hidden_dim,
        rank=rank,
        noise_level=0.05,
    )
    
    # Test normalization
    print(f"\n📊 Testing z-score normalization...")
    normalized_weights, normalizer = normalize_expert_families(expert_weights)
    
    # Check normalization worked
    for family in ['up', 'gate']:
        original_stack = torch.stack(expert_weights[family])
        normalized_stack = torch.stack(normalized_weights[family])
        
        print(f"   {family.upper()} family:")
        print(f"   - Original mean: {original_stack.mean():.6f}")
        print(f"   - Normalized mean: {normalized_stack.mean():.6f}")
        print(f"   - Original std: {original_stack.std():.6f}")
        print(f"   - Normalized std: {normalized_stack.std():.6f}")
    
    # Create trainer
    print(f"\n🚀 Testing bank training...")
    trainer = AlternatingBankTrainer(
        rank=rank,
        num_bases=16,
        device=torch.device("cpu"),  # Use CPU for quick testing
        normalize_experts=True
    )
    
    # Configure per-family settings to match our synthetic data
    family_configs = {
        "up": {"rank": rank, "num_bases": 16},
        "gate": {"rank": rank, "num_bases": 16},
    }
    
    # Train banks
    results, normalizer = trainer.train(
        expert_weights,
        family_configs=family_configs,
        num_steps=25,
        temperature_init=0.1,  # Lower initial temperature
        target_support=4,      # Lower target support for small bank
        activation="silu",
        log_wandb=False,
    )
    
    print(f"✅ Training completed!")
    
    # Test reconstruction quality
    print(f"\n🔍 Testing reconstruction quality...")
    device = torch.device("cpu")
    
    total_mse = 0.0
    total_relative_error = 0.0
    
    for family in ['up', 'gate']:
        print(f"\n📊 {family.upper()} reconstruction:")
        
        # Get components
        bank = results[family]['bank'].to(device)
        codes = results[family]['codes'].to(device)
        adapters = results[family]['adapters'].to(device)
        
        # Original weights (normalized)
        original_weights = torch.stack(normalized_weights[family]).to(device)
        
        # Reconstruct
        num_experts = codes.shape[0]
        reconstructions = []
        
        for i in range(num_experts):
            mixed_linear = torch.einsum('m,mrd->rd', codes[i], bank)
            mixed_basis = F.silu(mixed_linear)
            reconstruction = torch.matmul(adapters[i], mixed_basis)
            reconstructions.append(reconstruction)
        
        reconstructed_weights = torch.stack(reconstructions)
        
        # Compute metrics
        mse = torch.nn.functional.mse_loss(reconstructed_weights, original_weights).item()
        relative_error = (torch.norm(reconstructed_weights - original_weights) / 
                         torch.norm(original_weights)).item()
        
        total_mse += mse
        total_relative_error += relative_error
        
        print(f"   - MSE: {mse:.8f}")
        print(f"   - Relative error: {relative_error:.6f}")
        
        # Sparsity analysis
        sparsity = (codes < 1e-6).float().mean().item()
        support_sizes = (codes > 1e-6).sum(dim=1).float()
        print(f"   - Sparsity: {sparsity:.3f}")
        print(f"   - Mean support: {support_sizes.mean().item():.1f}")
        print(f"   - Support std: {support_sizes.std().item():.1f}")
    
    # Overall quality assessment
    print(f"\n📈 Overall Quality Assessment:")
    avg_mse = total_mse / 2
    avg_relative_error = total_relative_error / 2
    
    print(f"   - Average MSE: {avg_mse:.8f}")
    print(f"   - Average relative error: {avg_relative_error:.6f}")
    
    # Quality thresholds for synthetic data
    if avg_mse < 1e-3 and avg_relative_error < 0.1:
        print(f"✅ PASS: Reconstruction quality is good!")
    elif avg_mse < 1e-2 and avg_relative_error < 0.2:
        print(f"⚠️  MARGINAL: Reconstruction quality is acceptable")
    else:
        print(f"❌ FAIL: Reconstruction quality is poor")
        print(f"   This might indicate issues with the AM loop")
    
    # Test folding
    print(f"\n🔄 Testing normalization folding...")
    folded_results = fold_normalization_into_adapters(results, normalizer)

    for family in ['up', 'gate']:
        if family in folded_results:
            print(f"   - {family.upper()} adapters folded")
    
    print(f"\n🎉 Validation completed!")
    
    return {
        "avg_mse": avg_mse,
        "avg_relative_error": avg_relative_error,
        "passed": avg_mse < 1e-3 and avg_relative_error < 0.1
    }


if __name__ == "__main__":
    try:
        results = validate_am_loop()
        if results["passed"]:
            print(f"\n✅ All validations passed! The AM loop is working correctly.")
            sys.exit(0)
        else:
            print(f"\n⚠️  Some validations failed. Check the output above.")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
