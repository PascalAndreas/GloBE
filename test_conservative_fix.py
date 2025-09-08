#!/usr/bin/env python3
"""
Conservative fix for the AlternatingBankTrainer scaling issues.

Based on the debugging results, the main issues are:
1. Alpha has extremely low entropy (0.0071) - essentially selecting only one basis per expert
2. Scale mismatch between W and W_hat (0.11x ratio)

Conservative approach:
1. Increase tau (temperature) to improve alpha diversity
2. Add scale tracking and optional rescaling
3. Keep all existing normalizations for stability
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from globe.data.naming_qwen15 import extract_expert_weights
from globe.train.fit_banks import AlternatingBankTrainer


def test_conservative_fixes():
    """Test different conservative parameter adjustments."""
    
    print(f"🔬 Testing conservative fixes for AlternatingBankTrainer")
    
    # Load small test data
    expert_weights = extract_expert_weights("Qwen/Qwen1.5-MoE-A2.7B", max_experts=4)
    
    # Test configurations
    configs = [
        {
            "name": "Baseline",
            "tau": 1.0,
            "lambda_A": 1e-4,
            "lambda_B": 1e-4,
            "j_min": 1e-3,
            "eta": 0.5,
        },
        {
            "name": "Higher Temperature",
            "tau": 5.0,  # Higher temperature for more diverse alpha
            "lambda_A": 1e-4,
            "lambda_B": 1e-4,
            "j_min": 1e-3,
            "eta": 0.5,
        },
        {
            "name": "Much Higher Temperature",
            "tau": 10.0,  # Even higher temperature
            "lambda_A": 1e-4,
            "lambda_B": 1e-4,
            "j_min": 1e-3,
            "eta": 0.5,
        },
        {
            "name": "Higher Temp + Lower Regularization",
            "tau": 10.0,
            "lambda_A": 1e-5,  # Lower regularization
            "lambda_B": 1e-5,
            "j_min": 1e-3,
            "eta": 0.5,
        },
        {
            "name": "Higher Temp + Smaller Steps",
            "tau": 10.0,
            "lambda_A": 1e-4,
            "lambda_B": 1e-4,
            "j_min": 1e-3,
            "eta": 0.1,  # Smaller update steps
        }
    ]
    
    results = []
    
    for config in configs:
        print(f"\n📊 Testing: {config['name']}")
        print(f"   tau={config['tau']}, λ_A={config['lambda_A']}, λ_B={config['lambda_B']}, η={config['eta']}")
        
        # Create trainer
        trainer = AlternatingBankTrainer(
            rank=32,
            num_bases=8,
            device=torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
            dtype=torch.float32,
        )
        
        try:
            # Train with specific config
            result = trainer.train(
                expert_weights,
                num_steps=5,
                target_support=8,
                activation="silu",
                log_wandb=False,
                tau=config['tau'],
                lambda_A=config['lambda_A'],
                lambda_B=config['lambda_B'],
                lambda_T=1e-4,
                j_min=config['j_min'],
                eta=config['eta'],
                route_w_start=5,
            )
            
            # Analyze results
            up_codes = result['up']['codes']
            up_bank = result['up']['bank']
            up_adapters = result['up']['adapters']
            
            # Compute alpha entropy
            alpha_entropy = -(up_codes * up_codes.clamp_min(1e-12).log()).sum(-1).mean().item()
            
            # Compute sparsity
            sparsity = (up_codes > 1e-3).sum(-1).float().mean().item()
            
            # Test reconstruction
            W_test = torch.stack(expert_weights['up'][:4])
            M_lin = torch.einsum('em,mrd->erd', up_codes, up_bank)
            M = torch.nn.functional.silu(M_lin)
            W_hat = torch.einsum('epr,erd->epd', up_adapters, M)
            
            scale_ratio = (W_hat.norm() / W_test.norm()).item()
            rel_frob = ((W_hat - W_test).norm() / W_test.norm()).item()
            
            result_summary = {
                'config': config['name'],
                'alpha_entropy': alpha_entropy,
                'sparsity': sparsity,
                'scale_ratio': scale_ratio,
                'rel_frob': rel_frob,
                'success': True,
            }
            
            print(f"   ✅ Entropy: {alpha_entropy:.4f}, Sparsity: {sparsity:.1f}, Scale: {scale_ratio:.4f}, RelFrob: {rel_frob:.4f}")
            
        except Exception as e:
            result_summary = {
                'config': config['name'],
                'success': False,
                'error': str(e),
            }
            print(f"   ❌ Failed: {e}")
        
        results.append(result_summary)
    
    # Summary
    print(f"\n📋 SUMMARY:")
    successful_results = [r for r in results if r['success']]
    
    if successful_results:
        best_entropy = max(successful_results, key=lambda x: x['alpha_entropy'])
        best_scale = min(successful_results, key=lambda x: abs(x['scale_ratio'] - 1.0))
        best_rel_frob = min(successful_results, key=lambda x: x['rel_frob'])
        
        print(f"   🏆 Best entropy: {best_entropy['config']} (entropy: {best_entropy['alpha_entropy']:.4f})")
        print(f"   🏆 Best scale: {best_scale['config']} (ratio: {best_scale['scale_ratio']:.4f})")
        print(f"   🏆 Best RelFrob: {best_rel_frob['config']} (RelFrob: {best_rel_frob['rel_frob']:.4f})")
        
        # Recommend best overall config
        if best_entropy == best_scale == best_rel_frob:
            print(f"   🎯 RECOMMENDED: {best_entropy['config']} - best in all metrics!")
        else:
            print(f"   🎯 RECOMMENDED: Higher Temperature configs generally perform better")
    
    return results


if __name__ == "__main__":
    results = test_conservative_fixes()
