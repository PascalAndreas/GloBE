#!/usr/bin/env python3
"""
Minimal test script for GloBE bank training - focused purely on training loop.

This script is stripped down to focus only on getting the training loop working.
No bank building, no integration code - just extract weights, train, and log metrics.
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

from globe.data.naming_qwen15 import extract_expert_weights
from globe.train.fit_banks import AlternatingBankTrainer
from globe.init.warm_start import SeedingMethod, SeedingConfig, get_seeding_method_info


def get_model_dtype(model_name: str) -> torch.dtype:
    """Get the appropriate dtype for the model."""
    if "qwen" in model_name.lower():
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        else:
            return torch.float16
    return torch.float32


def train_minimal(
    expert_weights: Dict[str, torch.Tensor],
    num_steps: int = 10,
    rank_ratio: float = 0.75,
    basis_ratio: float = 0.5,
    device: str = "auto",
    model_name: str = "Qwen/Qwen1.5-MoE-A2.7B",
    log_wandb: bool = False,
    seeding_method: str = "ts_pca",
    # Training hyperparameters
    target_support: int = 16,
    epsilon: float = 1e-4,
    tau: float = 1.0,
    lambda_A: float = 1e-4,
    lambda_B: float = 1e-4,
    lambda_T: float = 1e-4,
    j_min: float = 1e-3,
    eta: float = 0.5,
    # route_w_start removed - superseded by λ/β scheduling
    temp_control_freq: int = 5,
    temp_control_eta: float = 0.1,
) -> Dict[str, Any]:
    """Run minimal training test focused on core loop."""
    
    # Get dimensions from first available family
    first_family = list(expert_weights.keys())[0]
    up_dim, hidden_dim = expert_weights[first_family][0].shape
    num_experts = len(expert_weights[first_family])
    
    # Calculate parameters
    rank = int(rank_ratio * up_dim)
    num_bases = int(basis_ratio * num_experts)
    
    # Parse seeding method
    try:
        seeding_enum = SeedingMethod(seeding_method.lower())
    except ValueError:
        available_methods = list(SeedingMethod)
        raise ValueError(f"Invalid seeding method '{seeding_method}'. Available: {[m.value for m in available_methods]}")
    
    seeding_config = SeedingConfig(method=seeding_enum)
    method_info = get_seeding_method_info()[seeding_method.lower()]
    
    print(f"🚀 Starting minimal training test...")
    print(f"   - Steps: {num_steps}")
    print(f"   - Families: {list(expert_weights.keys())}")
    print(f"   - Experts per family: {[len(weights) for weights in expert_weights.values()]}")
    print(f"   - Rank: {rank} ({rank_ratio:.1%} of projection dim)")
    print(f"   - Num bases: {num_bases} ({basis_ratio:.1%} of experts)")
    print(f"   - Seeding: {method_info['name']}")
    
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
    
    model_dtype = get_model_dtype(model_name)
    print(f"🔢 Using dtype: {model_dtype}")
    
    # Create trainer
    trainer = AlternatingBankTrainer(
        rank=rank,
        num_bases=num_bases,
        device=device,
        dtype=model_dtype,
        normalize_experts=False,
        seeding_config=seeding_config
    )
    
    # Configure families
    family_configs = {}
    for family in expert_weights.keys():
        if len(expert_weights[family]) > 0:
            family_configs[family] = {"rank": rank, "num_bases": num_bases}
    
    print(f"🔄 Training families: {list(family_configs.keys())}")
    
    # Train
    try:
        results = trainer.train(
            expert_weights,
            family_configs=family_configs,
            num_steps=num_steps,
            target_support=target_support,
            activation="silu",
            log_wandb=log_wandb,
            epsilon=epsilon,
            tau=tau,
            lambda_A=lambda_A,
            lambda_B=lambda_B,
            lambda_T=lambda_T,
            j_min=j_min,
            eta=eta,
            # route_w_start removed
            temp_control_freq=temp_control_freq,
            temp_control_eta=temp_control_eta,
        )
        
        print(f"✅ Training completed!")
        
        # Quick analysis
        for family in results.keys():
            bank_shape = results[family]['bank'].shape
            codes_shape = results[family]['codes'].shape
            adapters_shape = results[family]['adapters'].shape
            print(f"📊 {family.upper()} results:")
            print(f"   - Bank: {bank_shape}")
            print(f"   - Codes: {codes_shape}")
            print(f"   - Adapters: {adapters_shape}")
            
            codes = results[family]['codes']
            sparsity = (codes < 1e-6).float().mean().item()
            support_sizes = (codes > 1e-6).sum(dim=1).float()
            print(f"   - Sparsity: {sparsity:.3f}")
            print(f"   - Mean support: {support_sizes.mean().item():.1f}")
        
        return {
            "results": results,
            "device": device,
            "config": {
                "rank": rank,
                "num_bases": num_bases,
                "num_steps": num_steps,
                "families": list(expert_weights.keys()),
                "seeding_method": seeding_method,
            }
        }
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Minimal GloBE training test")
    parser.add_argument("--model", default="Qwen/Qwen1.5-MoE-A2.7B", help="Model name or path")
    parser.add_argument("--steps", type=int, default=10, help="Number of training steps")
    parser.add_argument("--rank-ratio", type=float, default=0.75, help="Rank as ratio of projection dimension")
    parser.add_argument("--basis-ratio", type=float, default=0.5, help="Number of bases as ratio of experts")
    parser.add_argument("--device", default="auto", help="Device to use")
    parser.add_argument("--force-download", action="store_true", help="Force re-download model")
    
    # Expert sampling
    parser.add_argument("--layers", type=str, default=None, help="Extract experts from specific layers (e.g., '12' or '12,13,14')")
    parser.add_argument("--max-experts", type=int, default=128, help="Limit number of experts")
    parser.add_argument("--sample-method", default="first_n", choices=["first_n", "evenly_spaced"], help="Sampling method")
    parser.add_argument("--families", default="up", choices=["up", "gate", "both"], help="Which families to train")
    
    # Wandb
    parser.add_argument("--wandb-project", default="globe-minimal-training", help="Wandb project")
    parser.add_argument("--wandb-name", default=None, help="Wandb run name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb")
    
    # Training params
    parser.add_argument("--seeding-method", default="ts_pca", 
                       choices=["svd", "ts_pca", "left_gram_pca", "spherical_kmeans", "residual_greedy", "hybrid"],
                       help="Seeding method")
    parser.add_argument("--target-support", type=int, default=16, help="Target support size")
    parser.add_argument("--tau", type=float, default=1.0, help="Softmax temperature")
    parser.add_argument("--lambda-A", type=float, default=1e-4, help="Ridge for A-refit")
    parser.add_argument("--lambda-B", type=float, default=1e-4, help="Ridge for MOD")
    parser.add_argument("--lambda-T", type=float, default=1e-4, help="Ridge for target backsolve")
    
    args = parser.parse_args()
    
    print("🌍 GloBE Minimal Training Test")
    print("=" * 50)
    
    # Initialize wandb if enabled
    if not args.no_wandb:
        wandb_name = args.wandb_name or f"minimal-{args.steps}steps-{args.families}"
        try:
            wandb.init(
                project=args.wandb_project,
                name=wandb_name,
                config=vars(args),
                tags=["minimal", "training-loop"],
                mode="online"
            )
            print(f"📊 Wandb enabled: {wandb_name}")
        except Exception as e:
            print(f"⚠️  Wandb failed, using offline mode: {e}")
            wandb.init(
                project=args.wandb_project,
                name=wandb_name,
                config=vars(args),
                tags=["minimal", "training-loop"],
                mode="offline"
            )
    
    try:
        # Parse layers argument
        layers = None
        if args.layers is not None:
            try:
                layers = [int(x.strip()) for x in args.layers.split(',')]
            except ValueError:
                print(f"❌ Invalid layers format: {args.layers}. Use format like '12' or '12,13,14'")
                sys.exit(1)
        
        # Extract expert weights
        layer_info = ""
        if layers is not None:
            if len(layers) == 1:
                layer_info = f" from layer {layers[0]}"
            else:
                layer_info = f" from layers {layers}"
        print(f"📤 Extracting expert weights{layer_info}...")
        
        expert_weights = extract_expert_weights(
            args.model,
            include_shared=False,
            force_download=args.force_download,
            max_experts=args.max_experts,
            layers=layers,
            sample_method=args.sample_method
        )
        
        # Filter families
        if args.families == "up":
            expert_weights = {"up": expert_weights["up"]}
        elif args.families == "gate":
            expert_weights = {"gate": expert_weights["gate"]}
        # "both" keeps both
        
        total_experts = sum(len(weights) for weights in expert_weights.values())
        families_str = ", ".join([f"{len(weights)} {family.upper()}" for family, weights in expert_weights.items()])
        print(f"✅ Using {total_experts} experts: {families_str}")
        
        # Train
        training_output = train_minimal(
            expert_weights,
            num_steps=args.steps,
            rank_ratio=args.rank_ratio,
            basis_ratio=args.basis_ratio,
            device=args.device,
            model_name=args.model,
            log_wandb=not args.no_wandb,
            seeding_method=args.seeding_method,
            target_support=args.target_support,
            tau=args.tau,
            lambda_A=args.lambda_A,
            lambda_B=args.lambda_B,
            lambda_T=args.lambda_T,
        )
        
        print(f"🎉 Training completed successfully!")
        
        if not args.no_wandb:
            wandb.finish()
            
    except Exception as e:
        print(f"💥 Training failed: {e}")
        import traceback
        traceback.print_exc()
        
        if not args.no_wandb:
            wandb.finish()
        sys.exit(1)


if __name__ == "__main__":
    main()
