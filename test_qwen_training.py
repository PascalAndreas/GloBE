#!/usr/bin/env python3
"""
Test script for GloBE bank training on Qwen1.5-MoE-A2.7B

This script downloads the model, extracts routed expert weights, 
and trains global basis banks using the alternating minimization workflow.
"""

import os
import sys
from pathlib import Path
import torch
import argparse
import json
import wandb
from typing import Dict, Any

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from globe.data.naming_qwen15 import extract_expert_weights, extract_all_expert_info
from globe.train.fit_banks import AlternatingBankTrainer, fold_normalization_into_adapters, build_dual_globe_bank
from globe.init.init_bank import InitConfig
from globe.init.warm_start import SeedingMethod, SeedingConfig, get_seeding_method_info


def get_model_dtype(model_name: str) -> torch.dtype:
    """Get the appropriate dtype for the model."""
    # Qwen models typically use bf16
    if "qwen" in model_name.lower():
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        else:
            return torch.float16  # Fallback for non-CUDA or older hardware
    return torch.float32





def train_banks_minimal_test(
    expert_weights: Dict[str, torch.Tensor],
    num_steps: int = 5,
    rank_ratio: float = 0.75,  # rank = rank_ratio * projection_dim
    basis_ratio: float = 0.5,  # num_bases = basis_ratio * num_experts
    device: str = "auto",
    model_name: str = "Qwen/Qwen1.5-MoE-A2.7B",
    log_wandb: bool = False,
    seeding_method: str = "ts_pca",
    # Conservative hyperparameters
    temperature_init: float = 1.5,
    min_temperature: float = 1.0,
    target_support: int = 16,
    lambda_A: float = 1e-5,
    lambda_B: float = 1e-4,
    epsilon: float = 1e-4,
) -> Dict[str, Any]:
    """Run a minimal bank training test."""
    # Get dimensions from first expert
    up_dim, hidden_dim = expert_weights['up'][0].shape  # projection_dim × hidden_dim
    gate_dim = expert_weights['gate'][0].shape[0]
    num_experts = len(expert_weights['up'])
    
    # Calculate scaled parameters
    rank = int(rank_ratio * up_dim)  # rank = rank_ratio * projection_dim
    num_bases = int(basis_ratio * num_experts)  # num_bases = basis_ratio * num_experts
    
    # Parse seeding method
    try:
        seeding_enum = SeedingMethod(seeding_method.lower())
    except ValueError:
        available_methods = list(SeedingMethod)
        raise ValueError(f"Invalid seeding method '{seeding_method}'. Available: {[m.value for m in available_methods]}")
    
    seeding_config = SeedingConfig(method=seeding_enum)
    method_info = get_seeding_method_info()[seeding_method.lower()]
    
    print(f"\n🚀 Starting minimal bank training test...")
    print(f"   - Steps: {num_steps}")
    print(f"   - Rank: {rank} ({rank_ratio:.1%} of projection dim)")
    print(f"   - Num bases: {num_bases} ({basis_ratio:.1%} of experts)")
    print(f"   - Seeding method: {method_info['name']}")
    print(f"   - Method description: {method_info['description']}")
    print(f"   - Speed: {method_info['speed']}, Quality: {method_info['quality']}, MPS-friendly: {method_info['mps_friendly']}")
    print(f"   - Temperature: {temperature_init} (min: {min_temperature})")
    print(f"   - Target support: {target_support}")
    print(f"   - Regularization: λ_A={lambda_A}, λ_B={lambda_B}")
    print(f"   - Epsilon: {epsilon}")
    
    # Auto-detect device
    if device == "auto":
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
        device = torch.device(device)
        print(f"🖥️  Using specified device: {device}")
    
    # Get appropriate dtype for the model
    model_dtype = get_model_dtype(model_name)
    print(f"🔢 Using dtype: {model_dtype}")
    
    print(f"📏 Expert dimensions:")
    print(f"   - Up: {up_dim} × {hidden_dim}")
    print(f"   - Gate: {gate_dim} × {hidden_dim}")
    print(f"   - Number of experts: {num_experts}")
    print(f"📐 Computed parameters:")
    print(f"   - Rank: {rank} (ratio: {rank_ratio:.2f})")
    print(f"   - Num bases: {num_bases} (ratio: {basis_ratio:.2f})")
    
    # Create trainer with appropriate dtype and seeding method
    trainer = AlternatingBankTrainer(
        rank=rank,
        num_bases=num_bases,
        device=device,
        dtype=model_dtype,
        normalize_experts=True,
        seeding_config=seeding_config
    )
    
    # Configure per-family settings
    family_configs = {
        "up": {"rank": rank, "num_bases": num_bases},
        "gate": {"rank": rank, "num_bases": num_bases},
    }
    
    print(f"🔄 Training banks...")
    try:
        # Train banks with conservative hyperparameters
        results, normalizer = trainer.train(
            expert_weights,
            family_configs=family_configs,
            num_steps=num_steps,
            temperature_init=temperature_init,
            target_support=target_support,
            activation="silu",
            log_wandb=log_wandb,
            min_temperature=min_temperature,
            lambda_A=lambda_A,
            lambda_B=lambda_B,
            epsilon=epsilon,
        )
        
        print(f"✅ Training completed!")
        
        # Analyze results
        for family in ['up', 'gate']:
            if family in results:
                bank_shape = results[family]['bank'].shape
                codes_shape = results[family]['codes'].shape
                adapters_shape = results[family]['adapters'].shape
                print(f"📊 {family.upper()} results:")
                print(f"   - Bank shape: {bank_shape}")
                print(f"   - Codes shape: {codes_shape}")
                print(f"   - Adapters shape: {adapters_shape}")
                
                # Check sparsity
                codes = results[family]['codes']
                sparsity = (codes < 1e-6).float().mean().item()
                support_sizes = (codes > 1e-6).sum(dim=1).float()
                print(f"   - Sparsity: {sparsity:.3f}")
                print(f"   - Mean support size: {support_sizes.mean().item():.1f}")
        
        return {
            "results": results,
            "normalizer": normalizer,
            "device": device,
            "config": {
                "rank": rank,
                "rank_ratio": rank_ratio,
                "num_bases": num_bases,
                "basis_ratio": basis_ratio,
                "num_steps": num_steps,
                "num_experts": num_experts,
                "projection_dim": up_dim,
                "seeding_method": seeding_method,
            }
        }
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise


def test_reconstruction_quality(training_output: Dict[str, Any], expert_weights: Dict[str, torch.Tensor]):
    """Test reconstruction quality of trained banks with NaN detection."""
    print(f"\n🔍 Testing reconstruction quality...")
    
    results = training_output["results"]
    device = training_output["device"]
    
    for family in ['up', 'gate']:
        if family not in results:
            continue
            
        print(f"\n📊 {family.upper()} reconstruction:")
        
        # Get components
        bank = results[family]['bank'].to(device)  # m × r × d
        codes = results[family]['codes'].to(device)  # E × m
        adapters = results[family]['adapters'].to(device)  # E × p × r
        
        # Debug: Check for NaN/Inf in components
        bank_nans = torch.isnan(bank).sum().item()
        codes_nans = torch.isnan(codes).sum().item()
        adapters_nans = torch.isnan(adapters).sum().item()
        
        if bank_nans > 0 or codes_nans > 0 or adapters_nans > 0:
            print(f"   ⚠️  NaN detected in components: bank={bank_nans}, codes={codes_nans}, adapters={adapters_nans}")
        
        # Check for extreme values
        bank_max = bank.abs().max().item()
        codes_max = codes.abs().max().item()
        adapters_max = adapters.abs().max().item()
        print(f"   📊 Max absolute values: bank={bank_max:.6f}, codes={codes_max:.6f}, adapters={adapters_max:.6f}")
        
        # Check codes sparsity
        codes_sparsity = (codes < 1e-6).float().mean().item()
        codes_zeros = (codes == 0).float().mean().item()
        print(f"   📊 Codes sparsity: {codes_sparsity:.3f}, exact zeros: {codes_zeros:.3f}")
        
        # Original weights - ensure consistent dtype with bank
        original_weights = torch.stack(expert_weights[family]).to(device, dtype=bank.dtype)  # E × p × d
        
        # Reconstruct: W_i ≈ A_i @ (Σ_j α_{i,j} B_j)
        num_experts = codes.shape[0]
        reconstructions = []
        
        for i in range(num_experts):
            # Mix bases: Σ_j α_{i,j} B_j
            mixed_basis = torch.einsum('m,mrd->rd', codes[i], bank)  # r × d
            
            # Check for NaN in mixed basis
            if torch.isnan(mixed_basis).any():
                print(f"   ⚠️  NaN in mixed_basis for expert {i}")
                print(f"      codes[{i}] range: [{codes[i].min():.6f}, {codes[i].max():.6f}]")
                print(f"      codes[{i}] sum: {codes[i].sum():.6f}")
            
            # Apply adapter: A_i @ mixed_basis
            reconstruction = torch.matmul(adapters[i], mixed_basis)  # p × d
            
            # Check for NaN in reconstruction
            if torch.isnan(reconstruction).any():
                print(f"   ⚠️  NaN in reconstruction for expert {i}")
                print(f"      adapter[{i}] range: [{adapters[i].min():.6f}, {adapters[i].max():.6f}]")
            
            reconstructions.append(reconstruction)
        
        reconstructed_weights = torch.stack(reconstructions)  # E × p × d
        
        # Final NaN check
        recon_nans = torch.isnan(reconstructed_weights).sum().item()
        if recon_nans > 0:
            print(f"   ❌ {recon_nans} NaN values in final reconstruction!")
            return  # Skip metrics if we have NaN
        
        # Compute metrics with safe division
        mse = torch.nn.functional.mse_loss(reconstructed_weights, original_weights).item()
        
        # Safe relative error computation
        diff_norm = torch.norm(reconstructed_weights - original_weights)
        orig_norm = torch.norm(original_weights)
        if orig_norm > 0:
            relative_error = (diff_norm / orig_norm).item()
        else:
            relative_error = float('inf') if diff_norm > 0 else 0.0
        
        print(f"   - MSE: {mse:.6f}")
        print(f"   - Relative error: {relative_error:.6f}")
        
        # Per-expert analysis
        per_expert_mse = torch.nn.functional.mse_loss(
            reconstructed_weights, original_weights, reduction='none'
        ).view(num_experts, -1).mean(dim=1)
        
        print(f"   - Best expert MSE: {per_expert_mse.min().item():.6f}")
        print(f"   - Worst expert MSE: {per_expert_mse.max().item():.6f}")
        print(f"   - MSE std: {per_expert_mse.std().item():.6f}")


def test_dual_bank_creation(training_output: Dict[str, Any]):
    """Test DualGloBEBank creation from results."""
    print(f"\n🔧 Testing DualGloBEBank creation...")
    
    try:
        dual_bank = build_dual_globe_bank(training_output["results"], activation="silu")
        print(f"✅ DualGloBEBank created successfully!")
        print(f"   - Up bank shape: {dual_bank.up_bank.bases.shape}")
        print(f"   - Gate bank shape: {dual_bank.gate_bank.bases.shape}")
        
        # Test forward pass
        batch_size = 4
        num_bases_up = dual_bank.up_bank.bases.shape[0]
        num_bases_gate = dual_bank.gate_bank.bases.shape[0]
        
        up_weights = torch.randn(batch_size, num_bases_up)
        gate_weights = torch.randn(batch_size, num_bases_gate)
        
        up_mixed, gate_mixed = dual_bank(up_weights, gate_weights)
        print(f"   - Test forward pass successful!")
        print(f"   - Up mixed shape: {up_mixed.shape}")
        print(f"   - Gate mixed shape: {gate_mixed.shape}")
        
        return dual_bank
        
    except Exception as e:
        print(f"❌ DualGloBEBank creation failed: {e}")
        raise


def save_results(training_output: Dict[str, Any], output_dir: str = "test_output"):
    """Save training results."""
    print(f"\n💾 Saving results to {output_dir}/...")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    results = training_output["results"]
    normalizer = training_output["normalizer"]
    config = training_output["config"]
    
    # Save individual family results
    for family in ['up', 'gate']:
        if family in results:
            torch.save(results[family], output_path / f"{family}_bank.pt")
            print(f"✅ Saved {family}_bank.pt")
    
    # Save combined results
    torch.save({
        "results": results,
        "normalizer": normalizer,
        "config": config,
    }, output_path / "globe_banks_combined.pt")
    print(f"✅ Saved globe_banks_combined.pt")
    
    # Save DualGloBEBank
    if "up" in results and "gate" in results:
        dual_bank = build_dual_globe_bank(results, activation="silu")
        torch.save(dual_bank.state_dict(), output_path / "dual_globe_bank.pt")
        print(f"✅ Saved dual_globe_bank.pt")
    
    # Save config as JSON
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"✅ Saved config.json")


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test GloBE bank training on Qwen1.5-MoE-A2.7B")
    parser.add_argument("--model", default="Qwen/Qwen1.5-MoE-A2.7B", help="Model name or path")
    parser.add_argument("--steps", type=int, default=10, help="Number of training steps")
    parser.add_argument("--rank-ratio", type=float, default=0.75, help="Rank as ratio of projection dimension (default: 0.75)")
    parser.add_argument("--basis-ratio", type=float, default=0.5, help="Number of bases as ratio of experts (default: 0.5)")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cpu, cuda, mps)")
    parser.add_argument("--output-dir", default="test_output", help="Output directory")
    parser.add_argument("--skip-download", action="store_true", help="Skip model download/analysis")
    parser.add_argument("--force-download", action="store_true", help="Force re-download even if cached")
    parser.add_argument("--max-experts", type=int, default=128, help="Limit number of experts (default: 128 for memory efficiency)")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching (for memory-constrained systems)")
    parser.add_argument("--wandb-project", default="globe-bank-training", help="Wandb project name")
    parser.add_argument("--wandb-name", default=None, help="Wandb run name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--seeding-method", default="ts_pca", choices=["svd", "ts_pca", "left_gram_pca", "spherical_kmeans", "residual_greedy", "hybrid"], help="Seeding method for bank initialization")
    # Conservative hyperparameters for stability
    parser.add_argument("--temperature-init", type=float, default=1.5, help="Initial temperature (lower for stability)")
    parser.add_argument("--min-temperature", type=float, default=1.0, help="Minimum temperature to prevent support collapse")
    parser.add_argument("--target-support", type=int, default=16, help="Target support size (higher for stability)")
    parser.add_argument("--lambda-A", type=float, default=1e-5, help="Adapter regularization (higher for stability)")
    parser.add_argument("--lambda-B", type=float, default=1e-4, help="Bank regularization")
    parser.add_argument("--epsilon", type=float, default=1e-4, help="Sparsity threshold")
    
    args = parser.parse_args()
    
    print("🌍 GloBE Bank Training Test")
    print("=" * 50)
    
    # Show available seeding methods
    if args.seeding_method == "svd":
        print("⚠️  Using SVD seeding - this will be slow on Mac but provides high quality baseline")
    else:
        method_info = get_seeding_method_info()[args.seeding_method]
        print(f"🚀 Using {method_info['name']} seeding method")
        print(f"   Speed: {method_info['speed']}, Quality: {method_info['quality']}, MPS-friendly: {method_info['mps_friendly']}")
    print()
    
    # Initialize wandb if enabled
    if not args.no_wandb:
        wandb_name = args.wandb_name or f"test-{args.steps}steps-rr{args.rank_ratio:.2f}-br{args.basis_ratio:.2f}"
        try:
            # Try online mode first
            wandb.init(
                project=args.wandb_project,
                name=wandb_name,
                config={
                    "model": args.model,
                    "steps": args.steps,
                    "rank_ratio": args.rank_ratio,
                    "basis_ratio": args.basis_ratio,
                    "max_experts": args.max_experts,
                    "device": args.device,
                    "seeding_method": args.seeding_method,
                },
                tags=["test", "qwen", "globe"],
                mode="online"
            )
            print(f"📊 Wandb logging enabled (online): {wandb_name}")
        except Exception as e:
            print(f"⚠️  Online WandB failed ({e}), falling back to offline mode...")
            wandb.init(
                project=args.wandb_project,
                name=wandb_name,
                config={
                    "model": args.model,
                    "steps": args.steps,
                    "rank_ratio": args.rank_ratio,
                    "basis_ratio": args.basis_ratio,
                    "max_experts": args.max_experts,
                    "device": args.device,
                    "seeding_method": args.seeding_method,
                },
                tags=["test", "qwen", "globe"],
                mode="offline"
            )
            print(f"📊 Wandb logging enabled (offline): {wandb_name}")
            print(f"   💡 You can sync later with: wandb sync wandb/offline-run-*")
    
    try:
        # Step 1: Extract expert weights with built-in caching and subset functionality
        print(f"📤 Extracting routed expert weights (max: {args.max_experts})...")
        expert_weights = extract_expert_weights(
            args.model, 
            include_shared=False,
            force_download=args.force_download,
            max_experts=args.max_experts
        )
        
        print(f"✅ Extracted {len(expert_weights['up'])} routed Up experts")
        print(f"✅ Extracted {len(expert_weights['gate'])} routed Gate experts")
        
        # Step 3: Train banks
        training_output = train_banks_minimal_test(
            expert_weights,
            num_steps=args.steps,
            rank_ratio=args.rank_ratio,
            basis_ratio=args.basis_ratio,
            device=args.device,
            model_name=args.model,
            log_wandb=not args.no_wandb,
            seeding_method=args.seeding_method,
            temperature_init=args.temperature_init,
            min_temperature=args.min_temperature,
            target_support=args.target_support,
            lambda_A=args.lambda_A,
            lambda_B=args.lambda_B,
            epsilon=args.epsilon,
        )
        
        # Step 4: Test reconstruction quality
        test_reconstruction_quality(training_output, expert_weights)
        
        # Step 5: Test DualGloBEBank creation
        dual_bank = test_dual_bank_creation(training_output)
        
        # Step 6: Save results
        save_results(training_output, args.output_dir)
        
        print(f"\n🎉 All tests completed successfully!")
        print(f"📁 Results saved to: {args.output_dir}/")
        
        if not args.no_wandb:
            wandb.finish()
            print(f"📊 Wandb run completed")
        
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        if not args.no_wandb:
            wandb.finish()
        sys.exit(1)


if __name__ == "__main__":
    main()
