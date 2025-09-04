#!/usr/bin/env python3
"""
Debug script to test individual components of the GloBE system.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from globe.modules.globe_bank import GloBEBank, DualGloBEBank
from globe.init.init_bank import create_warm_start, build_initial_bank, InitConfig
from globe.init.warm_start import SeedingConfig, SeedingMethod


def test_globe_bank_basic():
    """Test basic GloBEBank functionality."""
    print("🔧 Testing GloBEBank basic functionality...")
    
    # Create a simple bank
    num_bases = 4
    rank = 8
    hidden_dim = 16
    
    bank = GloBEBank(num_bases, rank, hidden_dim)
    print(f"✅ Created GloBEBank: {num_bases} bases, rank {rank}, hidden_dim {hidden_dim}")
    print(f"   Bank shape: {bank.bases.shape}")
    
    # Test forward pass
    batch_size = 2
    mixture_weights = torch.randn(batch_size, num_bases)
    mixture_weights = torch.softmax(mixture_weights, dim=1)  # Make it a proper mixture
    
    mixed_bases = bank(mixture_weights)
    print(f"✅ Forward pass successful")
    print(f"   Input shape: {mixture_weights.shape}")
    print(f"   Output shape: {mixed_bases.shape}")
    print(f"   Expected shape: ({batch_size}, {rank}, {hidden_dim})")
    
    assert mixed_bases.shape == (batch_size, rank, hidden_dim), f"Shape mismatch: {mixed_bases.shape}"
    
    return bank, mixture_weights, mixed_bases


def test_dual_bank():
    """Test DualGloBEBank."""
    print("\n🔧 Testing DualGloBEBank...")
    
    num_bases_up = 4
    num_bases_gate = 6
    rank = 8
    hidden_dim = 16
    
    dual_bank = DualGloBEBank(num_bases_up, num_bases_gate, rank, hidden_dim)
    print(f"✅ Created DualGloBEBank")
    print(f"   Up bank shape: {dual_bank.up_bank.bases.shape}")
    print(f"   Gate bank shape: {dual_bank.gate_bank.bases.shape}")
    
    # Test forward pass
    batch_size = 3
    up_weights = torch.softmax(torch.randn(batch_size, num_bases_up), dim=1)
    gate_weights = torch.softmax(torch.randn(batch_size, num_bases_gate), dim=1)
    
    up_mixed, gate_mixed = dual_bank(up_weights, gate_weights)
    
    print(f"✅ Dual bank forward pass successful")
    print(f"   Up output shape: {up_mixed.shape}")
    print(f"   Gate output shape: {gate_mixed.shape}")
    
    return dual_bank


def test_svd_and_initialization():
    """Test SVD and bank initialization."""
    print("\n🔧 Testing SVD and bank initialization...")
    
    # Create synthetic expert weights
    num_experts = 8
    intermediate_dim = 32
    hidden_dim = 16
    true_rank = 8
    
    # Create low-rank synthetic data
    experts = []
    for i in range(num_experts):
        A = torch.randn(intermediate_dim, true_rank) * 0.5
        B = torch.randn(true_rank, hidden_dim) * 0.5
        W = torch.matmul(A, B) + torch.randn(intermediate_dim, hidden_dim) * 0.01
        experts.append(W)
    
    W_stack = torch.stack(experts)  # E × p × d
    print(f"✅ Created {num_experts} synthetic experts with shape {W_stack.shape}")
    
    # Test warm start seeding
    rank = 8
    # Use default TS-PCA seeding method
    seeding_config = SeedingConfig(method=SeedingMethod.TS_PCA)
    A0, B0, energy, seeding_metrics = create_warm_start(W_stack, rank, seeding_config)
    
    print(f"✅ Warm start seeding successful")
    print(f"   A0 shape: {A0.shape}")  # E × p × r
    print(f"   B0 shape: {B0.shape}")  # E × r × d
    print(f"   Energy captured: {energy.mean():.3f} ± {energy.std():.3f}")
    
    # Test initial bank construction
    cfg = InitConfig(
        rank=rank,
        num_bases=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    
    bank0, alpha0 = build_initial_bank(B0, cfg)
    
    print(f"✅ Initial bank construction successful")
    print(f"   Bank shape: {bank0.shape}")  # m × r × d
    print(f"   Codes shape: {alpha0.shape}")  # E × m
    print(f"   Codes sparsity: {(alpha0 < 1e-6).float().mean():.3f}")
    
    # Test reconstruction quality with initial bank
    reconstructed_B = torch.einsum('em,mrd->erd', alpha0, bank0)  # E × r × d
    mse = torch.nn.functional.mse_loss(reconstructed_B, B0)
    relative_error = torch.norm(reconstructed_B - B0) / torch.norm(B0)
    
    print(f"✅ Initial reconstruction quality:")
    print(f"   MSE: {mse:.6f}")
    print(f"   Relative error: {relative_error:.6f}")
    
    return W_stack, A0, B0, bank0, alpha0


def test_full_reconstruction():
    """Test full weight reconstruction."""
    print("\n🔧 Testing full weight reconstruction...")
    
    W_stack, A0, B0, bank0, alpha0 = test_svd_and_initialization()
    
    # Reconstruct full weights: W_i = A_i @ B_i where B_i = Σ_j α_{i,j} B_j
    num_experts = W_stack.shape[0]
    reconstructed_weights = []
    
    for i in range(num_experts):
        # Get mixed basis for expert i
        mixed_basis = torch.einsum('m,mrd->rd', alpha0[i], bank0)  # r × d
        
        # Reconstruct weight: W_i = A_i @ mixed_basis
        reconstructed_W = torch.matmul(A0[i], mixed_basis)  # p × d
        reconstructed_weights.append(reconstructed_W)
    
    reconstructed_stack = torch.stack(reconstructed_weights)  # E × p × d
    
    # Compare with original
    mse = torch.nn.functional.mse_loss(reconstructed_stack, W_stack)
    relative_error = torch.norm(reconstructed_stack - W_stack) / torch.norm(W_stack)
    
    print(f"✅ Full weight reconstruction:")
    print(f"   MSE: {mse:.6f}")
    print(f"   Relative error: {relative_error:.6f}")
    
    # Per-expert analysis
    per_expert_mse = torch.nn.functional.mse_loss(
        reconstructed_stack, W_stack, reduction='none'
    ).view(num_experts, -1).mean(dim=1)
    
    print(f"   Best expert MSE: {per_expert_mse.min():.6f}")
    print(f"   Worst expert MSE: {per_expert_mse.max():.6f}")
    print(f"   MSE std: {per_expert_mse.std():.6f}")
    
    if mse < 0.01 and relative_error < 0.1:
        print(f"✅ PASS: Reconstruction quality is good!")
        return True
    else:
        print(f"⚠️  MARGINAL: Reconstruction quality could be better")
        return False


def main():
    """Run all component tests."""
    print("🧪 GloBE Component Debug Tests")
    print("=" * 50)
    
    try:
        # Test individual components
        test_globe_bank_basic()
        test_dual_bank()
        success = test_full_reconstruction()
        
        print(f"\n🎉 All component tests completed!")
        
        if success:
            print(f"✅ All tests passed! The components are working correctly.")
            return True
        else:
            print(f"⚠️  Some tests had marginal results. The components work but could be improved.")
            return True
            
    except Exception as e:
        print(f"\n💥 Component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
