"""Training loop for GloBE reconstruction.

This module implements the main training loop for learning global basis banks
and sparse mixture coefficients through weight-only reconstruction.
"""

from typing import Dict, List, Optional, Tuple, Any, Iterator
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import numpy as np
import wandb
import time
from pathlib import Path
import json

from .recon_objective import CombinedLoss, ReconstructionMetrics
from ..modules.globe_bank import DualGloBEBank
from ..modules.globe_mixer import SparseMixer, TemperatureScheduler
from ..init.zscore import ZScoreNormalizer


class ExpertWeightDataset(Dataset):
    """Dataset for expert weights."""
    
    def __init__(
        self,
        expert_weights: Dict[str, List[Tensor]],
        expert_indices: Optional[List[Tuple[int, int]]] = None,
    ):
        """Initialize expert weight dataset.
        
        Args:
            expert_weights: Dictionary mapping family to expert weight lists
            expert_indices: Optional list of (layer, expert) indices
        """
        self.expert_weights = expert_weights
        self.families = list(expert_weights.keys())
        
        # Generate expert indices if not provided
        if expert_indices is None:
            self.expert_indices = []
            for family in self.families:
                for i, _ in enumerate(expert_weights[family]):
                    self.expert_indices.append((0, i))  # Assuming single layer for simplicity
        else:
            self.expert_indices = expert_indices
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.expert_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """Get expert weights by index.
        
        Args:
            idx: Index in dataset
            
        Returns:
            Dictionary with expert weights per family
        """
        layer_idx, expert_idx = self.expert_indices[idx]
        
        item = {"layer_idx": layer_idx, "expert_idx": expert_idx}
        
        for family in self.families:
            if expert_idx < len(self.expert_weights[family]):
                item[family] = self.expert_weights[family][expert_idx]
        
        return item


class GloBETrainer:
    """Main trainer for GloBE reconstruction."""
    
    def __init__(
        self,
        dual_bank: DualGloBEBank,
        sparse_mixer: SparseMixer,
        loss_fn: CombinedLoss,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        normalizer: Optional[ZScoreNormalizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        grad_clip: Optional[float] = None,
        log_interval: int = 100,
        checkpoint_interval: int = 1000,
        validation_interval: int = 500,
    ):
        """Initialize GloBE trainer.
        
        Args:
            dual_bank: Dual global basis bank
            sparse_mixer: Sparse mixture controller
            loss_fn: Combined loss function
            optimizer: Optimizer
            device: Training device
            normalizer: Optional weight normalizer
            scheduler: Optional learning rate scheduler
            grad_clip: Gradient clipping threshold
            log_interval: Logging interval
            checkpoint_interval: Checkpoint saving interval
            validation_interval: Validation interval
        """
        self.dual_bank = dual_bank
        self.sparse_mixer = sparse_mixer
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.normalizer = normalizer
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.log_interval = log_interval
        self.checkpoint_interval = checkpoint_interval
        self.validation_interval = validation_interval
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Metrics tracking
        self.metrics_tracker = ReconstructionMetrics()
        self.train_losses = []
        self.val_losses = []
        
        # Move models to device
        self.dual_bank.to(device)
        self.sparse_mixer.to(device)
        self.loss_fn.to(device)
        self.metrics_tracker.to(device)
    
    def train_step(
        self,
        batch: Dict[str, Tensor],
        mixture_logits: Dict[str, Tensor],
        adapter_matrices: Dict[str, Tensor],
    ) -> Dict[str, float]:
        """Perform a single training step.
        
        Args:
            batch: Batch of expert weights
            mixture_logits: Mixture logits per family
            adapter_matrices: Adapter matrices per family
            
        Returns:
            Dictionary with step metrics
        """
        self.optimizer.zero_grad()
        
        batch_size = len(batch["expert_idx"])
        step_losses = {}
        step_metrics = {}
        
        # Process each family
        for family in ["up", "gate"]:
            if family not in batch:
                continue
            
            target_weights = batch[family].to(self.device)  # batch × p × d
            
            # Get mixture logits and adapter matrices for this batch
            expert_indices = batch["expert_idx"]
            family_mixture_logits = mixture_logits[family][expert_indices]  # batch × num_bases
            family_adapters = adapter_matrices[family][expert_indices]  # batch × p × rank
            
            # Compute sparse mixture weights
            mixture_weights, mixer_stats = self.sparse_mixer(family_mixture_logits)
            
            # Get mixed basis vectors
            if family == "up":
                mixed_bases, _ = self.dual_bank(mixture_weights, torch.zeros_like(mixture_weights))
            else:  # gate
                _, mixed_bases = self.dual_bank(torch.zeros_like(mixture_weights), mixture_weights)
            
            # Reconstruct weights: A_i @ φ(Σ_j α_{i,j} B_j)
            # mixed_bases: batch × rank
            # family_adapters: batch × p × rank
            reconstructed = torch.bmm(
                family_adapters,  # batch × p × rank
                mixed_bases.unsqueeze(-1)  # batch × rank × 1
            ).squeeze(-1)  # batch × p
            
            # Reshape to match target
            if target_weights.dim() == 3:  # batch × p × d
                reconstructed = reconstructed.unsqueeze(-1).expand(-1, -1, target_weights.shape[-1])
            
            # Compute losses
            family_losses = self.loss_fn(
                reconstructed,
                target_weights,
                mixture_weights,
                family_adapters,
                getattr(self.dual_bank, f"{family}_bank").bases,
            )
            
            # Accumulate losses
            for key, value in family_losses.items():
                step_losses[f"{family}_{key}"] = value.item()
            
            # Compute metrics
            family_metrics = self.metrics_tracker(reconstructed, target_weights, mixture_weights)
            for key, value in family_metrics.items():
                step_metrics[f"{family}_{key}"] = value
            
            # Add mixer stats
            for key, value in mixer_stats.items():
                step_metrics[f"{family}_mixer_{key}"] = value
        
        # Combined loss for backpropagation
        total_loss = sum(loss for key, loss in step_losses.items() if "total" in key)
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                list(self.dual_bank.parameters()) + 
                list(mixture_logits["up"]) + list(mixture_logits["gate"]) +
                list(adapter_matrices["up"]) + list(adapter_matrices["gate"]),
                self.grad_clip
            )
        
        # Optimizer step
        self.optimizer.step()
        
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Combine step losses and metrics
        step_info = {**step_losses, **step_metrics}
        step_info["total_loss"] = total_loss
        step_info["learning_rate"] = self.optimizer.param_groups[0]["lr"]
        
        return step_info
    
    def validate(
        self,
        val_loader: DataLoader,
        mixture_logits: Dict[str, Tensor],
        adapter_matrices: Dict[str, Tensor],
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """Run validation.
        
        Args:
            val_loader: Validation data loader
            mixture_logits: Mixture logits per family
            adapter_matrices: Adapter matrices per family
            max_batches: Maximum number of batches to validate
            
        Returns:
            Dictionary with validation metrics
        """
        self.dual_bank.eval()
        
        val_losses = []
        val_metrics = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                
                # Run forward pass (similar to train_step but without gradients)
                batch_metrics = self._validate_batch(batch, mixture_logits, adapter_matrices)
                val_losses.append(batch_metrics.get("total_loss", 0.0))
                val_metrics.append(batch_metrics)
        
        self.dual_bank.train()
        
        # Aggregate validation metrics
        if val_metrics:
            aggregated = {}
            for key in val_metrics[0].keys():
                values = [m[key] for m in val_metrics if key in m]
                if values:
                    aggregated[f"val_{key}"] = np.mean(values)
            
            aggregated["val_loss"] = np.mean(val_losses)
            return aggregated
        
        return {"val_loss": float('inf')}
    
    def _validate_batch(
        self,
        batch: Dict[str, Tensor],
        mixture_logits: Dict[str, Tensor],
        adapter_matrices: Dict[str, Tensor],
    ) -> Dict[str, float]:
        """Validate a single batch."""
        # Similar to train_step but without gradients and optimizer steps
        batch_metrics = {}
        
        for family in ["up", "gate"]:
            if family not in batch:
                continue
            
            target_weights = batch[family].to(self.device)
            expert_indices = batch["expert_idx"]
            family_mixture_logits = mixture_logits[family][expert_indices]
            family_adapters = adapter_matrices[family][expert_indices]
            
            mixture_weights, mixer_stats = self.sparse_mixer(family_mixture_logits)
            
            if family == "up":
                mixed_bases, _ = self.dual_bank(mixture_weights, torch.zeros_like(mixture_weights))
            else:
                _, mixed_bases = self.dual_bank(torch.zeros_like(mixture_weights), mixture_weights)
            
            reconstructed = torch.bmm(
                family_adapters,
                mixed_bases.unsqueeze(-1)
            ).squeeze(-1)
            
            if target_weights.dim() == 3:
                reconstructed = reconstructed.unsqueeze(-1).expand(-1, -1, target_weights.shape[-1])
            
            family_losses = self.loss_fn(
                reconstructed, target_weights, mixture_weights, family_adapters,
                getattr(self.dual_bank, f"{family}_bank").bases,
            )
            
            for key, value in family_losses.items():
                batch_metrics[f"{family}_{key}"] = value.item()
        
        batch_metrics["total_loss"] = sum(
            loss for key, loss in batch_metrics.items() if "total" in key
        )
        
        return batch_metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 10,
        mixture_logits: Optional[Dict[str, Tensor]] = None,
        adapter_matrices: Optional[Dict[str, Tensor]] = None,
        save_dir: Optional[Path] = None,
    ) -> Dict[str, List[float]]:
        """Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            num_epochs: Number of training epochs
            mixture_logits: Learnable mixture logits per family
            adapter_matrices: Learnable adapter matrices per family
            save_dir: Directory to save checkpoints
            
        Returns:
            Dictionary with training history
        """
        if mixture_logits is None or adapter_matrices is None:
            raise ValueError("Must provide mixture_logits and adapter_matrices")
        
        history = {"train_loss": [], "val_loss": []}
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_losses = []
            
            # Training phase
            self.dual_bank.train()
            for batch_idx, batch in enumerate(train_loader):
                step_info = self.train_step(batch, mixture_logits, adapter_matrices)
                epoch_losses.append(step_info["total_loss"])
                
                # Logging
                if self.step % self.log_interval == 0:
                    self._log_metrics(step_info, prefix="train")
                
                # Validation
                if (val_loader is not None and 
                    self.step % self.validation_interval == 0):
                    val_metrics = self.validate(val_loader, mixture_logits, adapter_matrices)
                    self._log_metrics(val_metrics, prefix="val")
                    
                    # Save best model
                    val_loss = val_metrics.get("val_loss", float('inf'))
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        if save_dir is not None:
                            self.save_checkpoint(save_dir / "best_model.pt", 
                                               mixture_logits, adapter_matrices)
                
                # Checkpointing
                if (save_dir is not None and 
                    self.step % self.checkpoint_interval == 0):
                    self.save_checkpoint(save_dir / f"checkpoint_step_{self.step}.pt",
                                       mixture_logits, adapter_matrices)
                
                self.step += 1
            
            # End of epoch
            avg_train_loss = np.mean(epoch_losses)
            history["train_loss"].append(avg_train_loss)
            
            if val_loader is not None:
                val_metrics = self.validate(val_loader, mixture_logits, adapter_matrices)
                val_loss = val_metrics.get("val_loss", float('inf'))
                history["val_loss"].append(val_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}")
        
        return history
    
    def _log_metrics(self, metrics: Dict[str, float], prefix: str = "") -> None:
        """Log metrics to wandb and console."""
        if wandb.run is not None:
            log_dict = {f"{prefix}_{k}" if prefix else k: v for k, v in metrics.items()}
            log_dict["step"] = self.step
            wandb.log(log_dict)
        
        # Console logging for key metrics
        if "total_loss" in metrics:
            print(f"Step {self.step}, {prefix} loss: {metrics['total_loss']:.6f}")
    
    def save_checkpoint(
        self,
        filepath: Path,
        mixture_logits: Dict[str, Tensor],
        adapter_matrices: Dict[str, Tensor],
    ) -> None:
        """Save training checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            mixture_logits: Mixture logits to save
            adapter_matrices: Adapter matrices to save
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "step": self.step,
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
            "dual_bank_state_dict": self.dual_bank.state_dict(),
            "mixture_logits": mixture_logits,
            "adapter_matrices": adapter_matrices,
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        if self.normalizer is not None:
            checkpoint["normalizer_stats"] = self.normalizer.get_stats()
        
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: Path) -> Dict[str, Tensor]:
        """Load training checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            
        Returns:
            Dictionary with mixture_logits and adapter_matrices
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        
        self.dual_bank.load_state_dict(checkpoint["dual_bank_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if self.normalizer is not None and "normalizer_stats" in checkpoint:
            self.normalizer.stats = checkpoint["normalizer_stats"]
        
        return {
            "mixture_logits": checkpoint["mixture_logits"],
            "adapter_matrices": checkpoint["adapter_matrices"],
        }
