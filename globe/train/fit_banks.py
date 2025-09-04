"""Training script for learning global basis banks.

This module provides the CLI entry point for fitting banks using the
alternating minimisation workflow described in
``globe_bank_initialization_training_workflow.md``.  Heavy lifting of
SVD warm starts and initial bank construction lives in
``globe.init.init_bank``.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import hydra
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
import wandb

from globe.init.init_bank import (
    InitConfig,
    create_warm_start,
    build_initial_bank,
    am_step,
)
from globe.init.warm_start import SeedingConfig, SeedingMethod
from globe.modules.globe_bank import GloBEBank, DualGloBEBank
from globe.modules.globe_mixer import SparseMixer
from globe.init.zscore import normalize_expert_families, ZScoreNormalizer
from globe.data.naming_qwen15 import extract_expert_weights


try:  # entmax is optional
    from entmax import entmax15
except Exception:  # pragma: no cover - fallback
    entmax15 = None


class AlternatingBankTrainer:
    """Bank training via alternating minimisation using ``GloBEBank`` modules."""

    def __init__(
        self,
        rank: int,
        num_bases: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        normalize_experts: bool = True,
        seeding_method: SeedingMethod = SeedingMethod.TS_PCA,
        seeding_config: Optional[SeedingConfig] = None,
    ) -> None:
        self.rank = rank
        self.num_bases = num_bases
        self.normalize_experts = normalize_experts
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = device
        self.dtype = dtype
        
        # Set up seeding configuration
        if seeding_config is None:
            seeding_config = SeedingConfig(method=seeding_method)
        self.seeding_config = seeding_config

    # ------------------------------------------------------------------
    def train(
        self,
        expert_weights: Dict[str, List[torch.Tensor]],
        family_configs: Optional[Dict[str, Dict]] = None,
        num_steps: int = 25,
        temperature_init: float = 1.0,
        target_support: int = 12,
        activation: str = "silu",
        log_wandb: bool = True,
    ) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Optional[ZScoreNormalizer]]:
        """Train banks for the provided expert weight families.
        
        Args:
            expert_weights: Dictionary of expert weights per family
            family_configs: Optional per-family configurations (rank, num_bases)
            Other args: Training hyperparameters
        
        Returns:
            Tuple of (results, normalizer) where results contains trained banks/codes/adapters
            and normalizer contains z-score statistics for folding back into adapters.
        """

        # Step 1: Z-score normalization per workflow
        normalizer = None
        working_weights = expert_weights
        
        if self.normalize_experts:
            working_weights, normalizer = normalize_expert_families(expert_weights)
            if log_wandb and wandb.run is not None:
                # Log normalization stats with explicit step to avoid conflicts
                zscore_metrics = {}
                for family in working_weights.keys():
                    if family in normalizer.stats:
                        stats = normalizer.stats[family]
                        zscore_metrics.update({
                            f"{family}/zscore_mean_norm": torch.norm(stats.mean).item(),
                            f"{family}/zscore_std_mean": stats.std.mean().item(),
                            f"{family}/zscore_std_std": stats.std.std().item(),
                        })
                # Log z-score stats at step -1 to avoid conflicts with AM loop steps
                wandb.log(zscore_metrics, step=-1)

        results: Dict[str, Dict[str, torch.Tensor]] = {}
        global_step = 0  # Global step counter across all families

        for family, weights in working_weights.items():
            if not weights:
                continue

            W = torch.stack(weights).to(self.device, self.dtype)

            # Use family-specific config if provided, otherwise use defaults
            if family_configs and family in family_configs:
                family_config = family_configs[family]
                rank = family_config.get("rank", self.rank)
                num_bases = family_config.get("num_bases", self.num_bases)
            else:
                rank = self.rank
                num_bases = self.num_bases

            cfg = InitConfig(
                rank=rank,
                num_bases=num_bases,
                device=self.device,
                dtype=self.dtype,
                target_support=target_support,
                temperature=temperature_init,
                seeding_config=self.seeding_config,
            )

            A0, B0, energy, seeding_metrics = create_warm_start(W, rank, self.seeding_config)
            bank0, alpha0 = build_initial_bank(B0, cfg)
            
            # Get hidden dimension from the weight tensor
            _, _, hidden_dim = W.shape  # E × p × d
            bank_module = GloBEBank(num_bases, rank, hidden_dim, activation=activation).to(
                self.device, self.dtype
            )
            with torch.no_grad():
                bank_module.bases.copy_(bank0)

            # Use alpha directly instead of logits to avoid entmax drift
            alpha = alpha0.clone()
            A = A0.to(self.device, self.dtype)
            T = temperature_init
            loss = 0.0
            

            # Log initial energy capture and seeding metrics
            if log_wandb and wandb.run is not None:
                seeding_log = {
                    f"{family}/initial_energy_captured": energy.mean().item(),
                    f"{family}/initial_energy_std": energy.std().item(),
                }
                # Add seeding-specific metrics
                for key, value in seeding_metrics.items():
                    if isinstance(value, (int, float)):
                        seeding_log[f"{family}/seeding_{key}"] = value
                    elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], (int, float)):
                        # Log first few values of lists
                        seeding_log[f"{family}/seeding_{key}_first"] = value[0]
                        if len(value) > 1:
                            seeding_log[f"{family}/seeding_{key}_mean"] = sum(value) / len(value)
                
                wandb.log(seeding_log)

            for step in range(num_steps):
                alpha, A, loss, T, med, metrics = am_step(
                    W, B0, bank_module, alpha, A, T, cfg, step=step, 
                    log_metrics=log_wandb, log_frequency=3  # Log every 3 steps for speed
                )
                current_global_step = global_step + step
                
                if log_wandb and metrics and wandb.run is not None:
                    # Log family-specific metrics with proper prefixes
                    family_metrics = {}
                    for key, value in metrics.items():
                        family_metrics[f"{family}/{key}"] = value
                    
                    # Add basic metrics for backward compatibility
                    family_metrics.update({
                        f"{family}/loss": float(loss),
                        f"{family}/median_support": med,
                        f"{family}/temperature": T,
                        f"{family}/step": step,
                    })
                    
                    wandb.log(family_metrics, step=current_global_step)

            # Update global step counter for next family
            global_step += num_steps
            
            # alpha is already the final mixture weights (no need to convert from logits)
            final_alpha = alpha
            
            # Store results (adapters will be folded later if normalization was used)
            results[family] = {
                "bank": bank_module.bases.detach(),
                "codes": final_alpha.detach(),
                "adapters": A.detach(),
            }

        return results, normalizer


def fold_normalization_into_adapters(
    results: Dict[str, Dict[str, torch.Tensor]], 
    normalizer: ZScoreNormalizer
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Fold z-score normalization back into adapter matrices as per workflow step 11.
    
    This modifies the adapters so that A_folded = A_normalized * std + mean,
    effectively undoing the normalization at the adapter level.
    """
    if normalizer is None:
        return results
    
    folded_results = {}
    for family, family_results in results.items():
        if family not in normalizer.stats:
            folded_results[family] = family_results
            continue
            
        stats = normalizer.stats[family]
        adapters = family_results["adapters"]  # E × p × r
        
        # Fold normalization: A_folded = A * std + mean (broadcasted appropriately)
        # adapters: E × p × r, stats.std: p × d, stats.mean: p × d
        # We need to be careful about dimensions here
        
        # For the folding to work correctly, we need to think about the reconstruction:
        # Original: W_i = A_i @ B_mixed where B_mixed comes from normalized bases
        # After folding: W_i = A_folded @ B_mixed where A_folded accounts for denormalization
        
        # Since the bases were trained on normalized data, the adapters need to be scaled
        # to account for the fact that they'll be applied to the same bases but we want
        # the final result to be in the original scale.
        
        # The correct folding depends on how the reconstruction works:
        # If W_normalized = A @ B and W_original = W_normalized * std + mean,
        # then we need A_folded such that A_folded @ B = A @ B * std + mean
        # This is not straightforward because std and mean are per-position, not scalars.
        
        # For now, we'll store the normalization info separately and handle folding
        # during the actual weight reconstruction in the GloBEFFN forward pass.
        folded_results[family] = {
            **family_results,
            "normalization_stats": {
                "mean": stats.mean,
                "std": stats.std,
            }
        }
    
    return folded_results


def build_dual_globe_bank(
    results: Dict[str, Dict[str, torch.Tensor]], 
    activation: str = "silu"
) -> DualGloBEBank:
    """Build a DualGloBEBank from training results.
    
    Args:
        results: Training results containing banks for 'up' and 'gate' families
        activation: Activation function for the banks
        
    Returns:
        DualGloBEBank instance with trained parameters
    """
    if "up" not in results or "gate" not in results:
        raise ValueError("Results must contain both 'up' and 'gate' families")
    
    up_bank_data = results["up"]["bank"]  # m_up × r × d
    gate_bank_data = results["gate"]["bank"]  # m_gate × r × d
    
    num_bases_up, rank, hidden_dim = up_bank_data.shape
    num_bases_gate = gate_bank_data.shape[0]
    
    # Create dual bank
    dual_bank = DualGloBEBank(
        num_bases_up=num_bases_up,
        num_bases_gate=num_bases_gate,
        rank=rank,
        hidden_dim=hidden_dim,
        activation=activation
    )
    
    # Load trained parameters
    dual_bank.up_bank.bases.data.copy_(up_bank_data)
    dual_bank.gate_bank.bases.data.copy_(gate_bank_data)
    
    return dual_bank


def create_globe_ffn_from_results(
    results: Dict[str, Dict[str, torch.Tensor]],
    layer_idx: int,
    hidden_dim: int,
    intermediate_dim: int,
    activation: str = "silu",
    sparsity_config: Optional[Dict] = None,
) -> "GloBEFFN":
    """Create a complete GloBEFFN from training results.
    
    Args:
        results: Training results from AlternatingBankTrainer
        layer_idx: Layer index for the FFN
        hidden_dim: Hidden dimension (d)
        intermediate_dim: Intermediate FFN dimension (p)
        activation: Activation function
        sparsity_config: Configuration for SparseMixer
        
    Returns:
        GloBEFFN instance ready for inference
    """
    from ..modules.ffn_globe import GloBEFFN  # Import here to avoid circular imports
    
    if "up" not in results or "gate" not in results:
        raise ValueError("Results must contain both 'up' and 'gate' families")
    
    # Extract dimensions from results
    up_bank_data = results["up"]["bank"]  # m_up × r × d
    gate_bank_data = results["gate"]["bank"]  # m_gate × r × d
    up_codes = results["up"]["codes"]  # E × m_up
    gate_codes = results["gate"]["codes"]  # E × m_gate
    up_adapters = results["up"]["adapters"]  # E × p × r
    gate_adapters = results["gate"]["adapters"]  # E × p × r
    
    num_experts = up_codes.shape[0]
    num_bases_up, rank, _ = up_bank_data.shape
    num_bases_gate = gate_bank_data.shape[0]
    
    # Create global banks
    global_banks = build_dual_globe_bank(results, activation)
    
    # Create sparse mixer
    if sparsity_config is None:
        sparsity_config = {"type": "entmax", "temperature": 1.0}
    
    sparse_mixer = SparseMixer(
        sparsity_type=sparsity_config.get("type", "entmax"),
        temperature=sparsity_config.get("temperature", 1.0),
        l1_weight=sparsity_config.get("l1_weight", 1e-4),
    )
    
    # Create GloBEFFN
    globe_ffn = GloBEFFN(
        layer_idx=layer_idx,
        num_experts=num_experts,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_bases_up=num_bases_up,
        num_bases_gate=num_bases_gate,
        rank=rank,
        global_banks=global_banks,
        sparse_mixer=sparse_mixer,
        activation=activation,
    )
    
    # Load trained parameters
    globe_ffn.up_adapters.data.copy_(up_adapters)
    globe_ffn.gate_adapters.data.copy_(gate_adapters)
    globe_ffn.up_mixture_logits.data.copy_(up_codes.log())  # Convert codes back to logits
    globe_ffn.gate_mixture_logits.data.copy_(gate_codes.log())
    
    # Note: down_projections need to be initialized separately as they're not trained
    # in the current workflow (they remain as dense expert-specific weights)
    
    return globe_ffn


# ---------------------------------------------------------------------------
# Hydra entry point ---------------------------------------------------------


@hydra.main(version_base=None, config_path="../../config", config_name="default")
def main(cfg: DictConfig) -> None:
    """Main training entry point with Hydra configuration."""

    if cfg.wandb.mode != "disabled":
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.run_name,
            tags=cfg.wandb.tags,
            config=dict(cfg),
            mode=cfg.wandb.mode,
        )

    torch.manual_seed(cfg.seed)

    expert_weights = extract_expert_weights(cfg.model.name_or_path)

    up_dim = expert_weights["up"][0].shape[0]
    gate_dim = expert_weights["gate"][0].shape[0]
    rank_up = cfg.bank.r if cfg.bank.r else up_dim // 2
    rank_gate = cfg.bank.r if cfg.bank.r else gate_dim // 2

    # Train both families together to share normalization
    trainer = AlternatingBankTrainer(
        rank=rank_up,  # Default rank (will be overridden per family)
        num_bases=cfg.bank.m_up,  # Default num_bases (will be overridden per family)
        device=cfg.device,
        normalize_experts=True,
    )
    
    # Configure per-family settings
    family_configs = {
        "up": {"rank": rank_up, "num_bases": cfg.bank.m_up},
        "gate": {"rank": rank_gate, "num_bases": cfg.bank.m_gate},
    }
    
    # Train all families at once to ensure consistent normalization
    all_results, normalizer = trainer.train(
        expert_weights,
        family_configs=family_configs,
        num_steps=cfg.train.steps,
        temperature_init=cfg.bank.init.temperature,
        target_support=cfg.bank.init.target_support,
        activation=cfg.bank.activation.type,
        log_wandb=cfg.wandb.mode != "disabled",
    )
    
    # Fold normalization back into adapters
    results = fold_normalization_into_adapters(all_results, normalizer)
    
    # Save per-family checkpoints
    for family in ["up", "gate"]:
        if family in results:
            out_path = os.path.join(os.getcwd(), f"{family}_bank.pt")
            torch.save(results[family], out_path)
    
    # Save combined results for GloBEFFN initialization
    combined_path = os.path.join(os.getcwd(), "globe_banks_combined.pt")
    torch.save({
        "results": results,
        "normalizer": normalizer,
        "config": dict(cfg),
    }, combined_path)
    
    # Build and save DualGloBEBank for easy loading
    if "up" in results and "gate" in results:
        dual_bank = build_dual_globe_bank(results, activation=cfg.bank.activation.type)
        dual_bank_path = os.path.join(os.getcwd(), "dual_globe_bank.pt")
        torch.save(dual_bank.state_dict(), dual_bank_path)

    if cfg.wandb.mode != "disabled":
        wandb.finish()


if __name__ == "__main__":
    main()

