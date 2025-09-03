"""Training script for GloBE bank initialization with gradient descent.

This script handles the full training orchestration including:
- Hydra configuration management
- WandB logging
- Gradient-based alternating optimization
- Support for unified vs separate banks
- Checkpointing and early stopping
"""

import hydra
from omegaconf import DictConfig
import torch
import wandb
from typing import Dict, List, Optional

from globe.init.init_bank import InitConfig, BankInitializer
from globe.data.naming_qwen15 import extract_expert_weights  # TODO: implement


class GradientBankTrainer:
    """Gradient-based bank training with PyTorch optimization."""
    
    def __init__(
        self,
        rank: int,
        num_bases: int,
        unified_bank: bool = False,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.rank = rank
        self.num_bases = num_bases
        self.unified_bank = unified_bank
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        
    def train(
        self,
        expert_weights: Dict[str, List[torch.Tensor]],
        num_steps: int = 1000,
        learning_rate: float = 1e-3,
        temperature_init: float = 1.0,
        target_support: int = 12,
        log_wandb: bool = True,
    ):
        """Train banks using gradient descent with alternating optimization."""
        # TODO: Implement gradient-based training
        # 1. Initialize banks and codes
        # 2. Create optimizers for logits
        # 3. Training loop with:
        #    - Forward pass through entmax
        #    - Reconstruction loss
        #    - Dictionary update (differentiable)
        #    - Temperature annealing
        #    - WandB logging
        pass


@hydra.main(version_base=None, config_path="../../config", config_name="default")
def main(cfg: DictConfig):
    """Main training entry point with Hydra configuration."""
    
    # Initialize WandB
    if cfg.wandb.mode != "disabled":
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.run_name,
            tags=cfg.wandb.tags,
            config=dict(cfg),
        )
    
    # Load model and extract expert weights
    expert_weights = extract_expert_weights(cfg.model.name_or_path)
    
    # Determine ranks based on expert dimensions
    up_dim = expert_weights["up"][0].shape[0]
    gate_dim = expert_weights["gate"][0].shape[0]
    
    rank_up = cfg.bank.r if cfg.bank.r else up_dim // 2
    rank_gate = cfg.bank.r if cfg.bank.r else gate_dim // 2
    
    # Create trainer based on bank architecture choice
    if cfg.bank.unified:
        # Train single unified bank for both Gate and Up
        trainer = GradientBankTrainer(
            rank=min(rank_up, rank_gate),  # Use smaller rank
            num_bases=cfg.bank.m_unified,
            unified_bank=True,
            device=cfg.device,
        )
        # Combine expert weights
        combined_weights = {"unified": expert_weights["up"] + expert_weights["gate"]}
        trainer.train(combined_weights, **cfg.train)
    else:
        # Train separate banks for Gate and Up
        for family in ["up", "gate"]:
            trainer = GradientBankTrainer(
                rank=rank_up if family == "up" else rank_gate,
                num_bases=cfg.bank[f"m_{family}"],
                unified_bank=False,
                device=cfg.device,
            )
            family_weights = {family: expert_weights[family]}
            trainer.train(family_weights, **cfg.train)
    
    # Finalize WandB
    if cfg.wandb.mode != "disabled":
        wandb.finish()


if __name__ == "__main__":
    main()
