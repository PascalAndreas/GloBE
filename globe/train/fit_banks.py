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

import torch
import torch.nn.functional as F
import wandb

from globe.init.init_bank import (
    InitConfig,
    create_warm_start,
    build_initial_bank,
    am_step,
    update_temperature,
    update_adaptive_schedule,
)
from globe.init.warm_start import SeedingConfig, SeedingMethod
from globe.modules.globe_bank import GloBEBank, DualGloBEBank
# Legacy imports removed - SparseMixer and extract_expert_weights not needed here


class AlternatingBankTrainer:
    """Bank training via alternating minimisation using ``GloBEBank`` modules."""

    def __init__(
        self,
        rank: int,
        num_bases: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        normalize_experts: bool = False,
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
        
        # Pre-allocate performance buffers
        self._identity_buffers = {}

    def _get_identity_buffers(self, rank: int, num_bases: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get or create cached identity matrices."""
        key = (rank, num_bases, self.device, self.dtype)
        if key not in self._identity_buffers:
            I_r = torch.eye(rank, device=self.device, dtype=self.dtype)
            I_m = torch.eye(num_bases, device=self.device, dtype=self.dtype)
            self._identity_buffers[key] = (I_r, I_m)
        return self._identity_buffers[key]

    # ------------------------------------------------------------------
    def train(
        self,
        expert_weights: Dict[str, List[torch.Tensor]],
        family_configs: Optional[Dict[str, Dict]] = None,
        num_steps: int = 25,
        target_support: int = 12,
        activation: str = "silu",
        log_wandb: bool = True,
        # JW-MOD parameters
        tau: float = 1.0,
        lambda_A: float = 1e-4,
        lambda_B: float = 1e-4,
        lambda_T: float = 1e-4,
        j_min: float = 1e-3,
        eta: float = 0.5,
        # route_w_start removed - superseded by λ/β scheduling
        epsilon: float = 1e-4,  # For support calculation compatibility
        # Temperature control
        temp_control_freq: int = 5,
        temp_control_eta: float = 0.1,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Train banks for the provided expert weight families with GPT-5 optimizations.
        
        Args:
            expert_weights: Dictionary of expert weights per family
            family_configs: Optional per-family configurations (rank, num_bases)
            Other args: Training hyperparameters including temperature control
        
        Returns:
            Dictionary containing trained banks/codes/adapters per family.
        """

        # Work directly with expert weights (no normalization)
        working_weights = expert_weights

        results: Dict[str, Dict[str, torch.Tensor]] = {}

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
                epsilon=epsilon,
                seeding_config=self.seeding_config,
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

            A0, B0, energy, seeding_metrics = create_warm_start(W, rank, self.seeding_config)
            bank0, alpha0 = build_initial_bank(B0, cfg)
            
            # Get hidden dimension from the weight tensor
            _, _, hidden_dim = W.shape  # E × p × d
            bank_module = GloBEBank(num_bases, rank, hidden_dim, activation=activation).to(
                self.device, self.dtype
            )
            with torch.no_grad():
                bank_module.bases.copy_(bank0)

            # Use alpha directly as mixture weights
            alpha = alpha0.clone()
            A = A0.to(self.device, self.dtype)
            
            # Get pre-computed buffers for performance
            I_r, I_m = self._get_identity_buffers(rank, num_bases)
            
            # Pre-compute normalized bank for coding efficiency
            bank_unit_flat = None
            
            # Track relF history for adaptive scheduling
            relF_history = []

            # Log initial energy capture and seeding metrics
            if log_wandb and wandb.run is not None:
                # Log seeding metrics to console only
                seeding_summary = []
                seeding_summary.append(f"energy={energy.mean().item():.3f}±{energy.std().item():.3f}")
                for key, value in seeding_metrics.items():
                    if isinstance(value, (int, float)):
                        seeding_summary.append(f"{key}={value:.3f}" if isinstance(value, float) else f"{key}={value}")
                    elif isinstance(value, list) and len(value) > 0:
                        seeding_summary.append(f"{key}_mean={sum(value) / len(value):.3f}")
                
                print(f"      Seeding: {', '.join(seeding_summary)}")

            print(f"      Training {family.upper()} family:")
            
            # Calculate global step offset to avoid WandB step conflicts
            global_step_offset = getattr(self, '_wandb_step_counter', 0)
            
            for step in range(num_steps):
                # Simple progress bar
                progress = (step + 1) / num_steps
                bar_length = 30
                filled_length = int(bar_length * progress)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                print(f"\r      [{bar}] {step+1:2d}/{num_steps} steps | ", end="", flush=True)
                
                # Update pre-computed normalized bank for coding
                if step > 0:  # After first update
                    with torch.no_grad():
                        D = bank_module.bases.view(num_bases, -1).to(cfg.dtype if cfg.dtype == torch.float32 else torch.float32)
                        bank_unit_flat = D / (D.norm(dim=-1, keepdim=True).clamp_min(1e-8))
                
                # Determine current hyperparameter values based on step and phase
                current_lambda = cfg.activation_lambda
                current_beta = cfg.route_beta
                
                # Decide if we should use weight-aware targets
                # For λ=0, after initial steps, switch to weight-aware targets
                use_weight_aware = (current_lambda == 0.0 and step >= 2)
                
                alpha, A, loss, seff_median, metrics = am_step(
                    W, B0, bank_module, alpha, A, cfg, step=step,
                    log_metrics=log_wandb,
                    I_r=I_r, I_m=I_m, bank_unit_flat=bank_unit_flat,
                    activation_lambda=current_lambda,
                    route_beta=current_beta,
                    use_weight_aware_targets=use_weight_aware
                )
                
                # Temperature control
                update_temperature(cfg, seff_median, step)
                
                # Adaptive scheduling for activation homotopy and route blending
                relF_history.append(metrics.get('recon/relative_frobenius_error', 1.0))
                adaptive_actions = update_adaptive_schedule(cfg, metrics, step, relF_history)
                
                # Update hyperparameters based on adaptive scheduling
                # This ensures the next iteration uses the updated values
                # (cfg is modified in-place by update_adaptive_schedule)

                # Update progress bar with current metrics (including adaptive info)
                rel_frob = metrics.get("recon/relative_frobenius_error", float("nan"))
                lambda_val = cfg.activation_lambda
                beta_val = cfg.route_beta
                phase = metrics.get("adaptive/phase", 0)
                phase_name = ["Linear", "Transition", "Nonlinear"][int(phase)]
                tau_display = f"τ={cfg.tau:.2f}"
                adaptive_display = f"λ={lambda_val:.2f},β={beta_val:.2f},{phase_name}"
                actions_display = adaptive_actions.get("actions", "")
                if actions_display and actions_display != "stable":
                    adaptive_display += f"[{actions_display}]"
                print(
                    f"Loss: {loss:.4f} | RelFrob: {rel_frob:.4f} | Seff: {seff_median:.1f} | {adaptive_display} | {tau_display}",
                    end="",
                    flush=True,
                )
                
                if log_wandb and metrics and wandb.run is not None:
                    # Prefix metrics with family name and log with proper step offset
                    family_metrics = {f"{family}/{k}": v for k, v in metrics.items()}
                    current_step = global_step_offset + step
                    wandb.log(family_metrics, step=current_step)
            
            # Complete the progress bar (including final adaptive state)
            final_rel = metrics.get("recon/relative_frobenius_error", float("nan"))
            final_lambda = cfg.activation_lambda
            final_beta = cfg.route_beta
            final_phase = metrics.get("adaptive/phase", 0)
            final_phase_name = ["Linear", "Transition", "Nonlinear"][int(final_phase)]
            final_tau = f"τ={cfg.tau:.2f}"
            final_adaptive = f"λ={final_lambda:.2f},β={final_beta:.2f},{final_phase_name}"
            print(
                f"\r      [{'█' * bar_length}] {num_steps:2d}/{num_steps} steps | Final - Loss: {loss:.4f} | RelFrob: {final_rel:.4f} | Seff: {seff_median:.1f} | {final_adaptive} | {final_tau}"
            )
            
            # Update step counter for next family
            if not hasattr(self, '_wandb_step_counter'):
                self._wandb_step_counter = 0
            self._wandb_step_counter += num_steps
            print(f"      ✅ {family.upper()} training completed!")
            
            # Store final mixture weights
            final_alpha = alpha
            
            # Store results
            results[family] = {
                "bank": bank_module.bases.detach(),
                "codes": final_alpha.detach(),
                "adapters": A.detach(),
            }

        return results




# Legacy functions removed - focusing only on core bank training
# build_dual_globe_bank and create_globe_ffn_from_results moved to test files if needed


# End of file

