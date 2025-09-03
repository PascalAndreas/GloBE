"""Training script for learning global basis banks.

This module provides the CLI entry point for fitting banks using the
alternating minimisation workflow described in
``globe_bank_initialization_training_workflow.md``.  Heavy lifting of
SVD warm starts and initial bank construction lives in
``globe.init.init_bank``.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import hydra
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
import wandb

from globe.init.init_bank import (
    InitConfig,
    truncated_svd,
    build_initial_bank,
    am_step,
)
from globe.modules.globe_bank import GloBEBank
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
    ) -> None:
        self.rank = rank
        self.num_bases = num_bases
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = device
        self.dtype = dtype

    # ------------------------------------------------------------------
    def train(
        self,
        expert_weights: Dict[str, List[torch.Tensor]],
        num_steps: int = 25,
        temperature_init: float = 1.0,
        target_support: int = 12,
        activation: str = "silu",
        log_wandb: bool = True,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Train banks for the provided expert weight families."""

        results: Dict[str, Dict[str, torch.Tensor]] = {}

        for family, weights in expert_weights.items():
            if not weights:
                continue

            W = torch.stack(weights).to(self.device, self.dtype)

            cfg = InitConfig(
                rank=self.rank,
                num_bases=self.num_bases,
                device=self.device,
                dtype=self.dtype,
                target_support=target_support,
                temperature=temperature_init,
            )

            A0, B0, _ = truncated_svd(W, self.rank)
            bank0, alpha0 = build_initial_bank(B0, cfg)
            bank_module = GloBEBank(self.num_bases, self.rank, activation=activation).to(
                self.device, self.dtype
            )
            bank_module.bases.data.copy_(bank0)

            alpha_logits = alpha0.clamp_min(cfg.epsilon).log()
            A = A0.to(self.device, self.dtype)
            T = temperature_init
            loss = 0.0

            for step in range(num_steps):
                alpha_logits, A, loss, T, med = am_step(
                    W, B0, bank_module, alpha_logits, A, T, cfg
                )
                if log_wandb:
                    wandb.log(
                        {
                            f"{family}/loss": float(loss),
                            f"{family}/median_support": med,
                            f"{family}/temperature": T,
                        },
                        step=step,
                    )

            final_alpha = (
                entmax15(alpha_logits / T, dim=-1)
                if entmax15 is not None
                else F.softmax(alpha_logits / T, dim=-1)
            )
            results[family] = {
                "bank": bank_module.bases.detach(),
                "codes": final_alpha.detach(),
                "adapters": A.detach(),
            }

        return results


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

    results: Dict[str, Dict[str, torch.Tensor]] = {}
    for family, rank, m in [
        ("up", rank_up, cfg.bank.m_up),
        ("gate", rank_gate, cfg.bank.m_gate),
    ]:
        trainer = AlternatingBankTrainer(rank=rank, num_bases=m, device=cfg.device)
        family_weights = {family: expert_weights[family]}
        family_results = trainer.train(
            family_weights,
            num_steps=cfg.train.steps,
            temperature_init=cfg.bank.init.temperature,
            target_support=cfg.bank.init.target_support,
            activation=cfg.bank.activation.type,
            log_wandb=cfg.wandb.mode != "disabled",
        )
        results.update(family_results)

        # Save per-family checkpoint
        out_path = os.path.join(os.getcwd(), f"{family}_bank.pt")
        torch.save(family_results[family], out_path)

    if cfg.wandb.mode != "disabled":
        wandb.finish()


if __name__ == "__main__":
    main()

