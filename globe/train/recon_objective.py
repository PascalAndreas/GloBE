"""Reconstruction objective and loss functions for GloBE training.

This module implements the reconstruction loss and various regularizers
used during GloBE training to learn global basis banks and sparse mixtures.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ReconstructionLoss(nn.Module):
    """Main reconstruction loss for GloBE training."""
    
    def __init__(
        self,
        mse_weight: float = 1.0,
        l1_mixture_weight: float = 1e-4,
        l2_adapter_weight: float = 1e-4,
        orthogonal_bank_weight: float = 0.0,
        spectral_bank_weight: float = 0.0,
        diversity_weight: float = 0.0,
        reduction: str = "mean",
    ):
        """Initialize reconstruction loss.
        
        Args:
            mse_weight: Weight for MSE reconstruction loss
            l1_mixture_weight: Weight for L1 regularization on mixture coefficients
            l2_adapter_weight: Weight for L2 regularization on adapter matrices
            orthogonal_bank_weight: Weight for orthogonality regularization on banks
            spectral_bank_weight: Weight for spectral regularization on banks
            diversity_weight: Weight for diversity regularization on mixtures
            reduction: Reduction method ("mean", "sum", "none")
        """
        super().__init__()
        
        self.mse_weight = mse_weight
        self.l1_mixture_weight = l1_mixture_weight
        self.l2_adapter_weight = l2_adapter_weight
        self.orthogonal_bank_weight = orthogonal_bank_weight
        self.spectral_bank_weight = spectral_bank_weight
        self.diversity_weight = diversity_weight
        self.reduction = reduction
    
    def forward(
        self,
        reconstructed: Tensor,
        target: Tensor,
        mixture_weights: Optional[Tensor] = None,
        adapter_matrices: Optional[Tensor] = None,
        basis_banks: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Compute reconstruction loss and regularizers.
        
        Args:
            reconstructed: Reconstructed expert weights ∈ R^{batch×p×d}
            target: Target expert weights ∈ R^{batch×p×d}
            mixture_weights: Mixture coefficients ∈ R^{batch×num_bases}
            adapter_matrices: Adapter matrices ∈ R^{batch×p×rank}
            basis_banks: Basis bank ∈ R^{d×num_bases}
            
        Returns:
            Dictionary with loss components
        """
        losses = {}
        
        # Main reconstruction loss (MSE)
        mse_loss = F.mse_loss(reconstructed, target, reduction=self.reduction)
        losses["mse"] = mse_loss
        losses["weighted_mse"] = self.mse_weight * mse_loss
        
        total_loss = losses["weighted_mse"]
        
        # L1 regularization on mixture weights
        if mixture_weights is not None and self.l1_mixture_weight > 0:
            l1_mixture = torch.mean(torch.abs(mixture_weights))
            losses["l1_mixture"] = l1_mixture
            losses["weighted_l1_mixture"] = self.l1_mixture_weight * l1_mixture
            total_loss = total_loss + losses["weighted_l1_mixture"]
        
        # L2 regularization on adapter matrices
        if adapter_matrices is not None and self.l2_adapter_weight > 0:
            l2_adapter = torch.mean(adapter_matrices ** 2)
            losses["l2_adapter"] = l2_adapter
            losses["weighted_l2_adapter"] = self.l2_adapter_weight * l2_adapter
            total_loss = total_loss + losses["weighted_l2_adapter"]
        
        # Orthogonality regularization on basis banks
        if basis_banks is not None and self.orthogonal_bank_weight > 0:
            # Compute B^T B - I and penalize off-diagonal elements
            gram = torch.matmul(basis_banks.T, basis_banks)  # num_bases × num_bases
            identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
            ortho_loss = torch.sum((gram - identity) ** 2)
            losses["orthogonal_bank"] = ortho_loss
            losses["weighted_orthogonal_bank"] = self.orthogonal_bank_weight * ortho_loss
            total_loss = total_loss + losses["weighted_orthogonal_bank"]
        
        # Spectral regularization on basis banks
        if basis_banks is not None and self.spectral_bank_weight > 0:
            spectral_norm = torch.norm(basis_banks, p=2)
            losses["spectral_bank"] = spectral_norm
            losses["weighted_spectral_bank"] = self.spectral_bank_weight * spectral_norm
            total_loss = total_loss + losses["weighted_spectral_bank"]
        
        # Diversity regularization on mixtures
        if mixture_weights is not None and self.diversity_weight > 0:
            # Encourage different experts to use different basis combinations
            # Compute pairwise cosine similarities and penalize high similarities
            normalized_mixtures = F.normalize(mixture_weights, p=2, dim=1)
            similarity_matrix = torch.matmul(normalized_mixtures, normalized_mixtures.T)
            
            # Zero out diagonal (self-similarities)
            mask = torch.eye(similarity_matrix.shape[0], device=similarity_matrix.device, dtype=torch.bool)
            similarity_matrix = similarity_matrix.masked_fill(mask, 0)
            
            diversity_loss = torch.mean(similarity_matrix ** 2)
            losses["diversity"] = diversity_loss
            losses["weighted_diversity"] = self.diversity_weight * diversity_loss
            total_loss = total_loss + losses["weighted_diversity"]
        
        losses["total"] = total_loss
        
        return losses


class SparsityLoss(nn.Module):
    """Sparsity-related losses for mixture coefficients."""
    
    def __init__(
        self,
        target_sparsity: float = 0.8,
        sparsity_weight: float = 1e-3,
        support_target: int = 12,
        support_weight: float = 1e-3,
    ):
        """Initialize sparsity loss.
        
        Args:
            target_sparsity: Target sparsity level (0-1)
            sparsity_weight: Weight for sparsity loss
            support_target: Target support size (number of non-zero elements)
            support_weight: Weight for support size loss
        """
        super().__init__()
        
        self.target_sparsity = target_sparsity
        self.sparsity_weight = sparsity_weight
        self.support_target = support_target
        self.support_weight = support_weight
    
    def forward(self, mixture_weights: Tensor, eps: float = 1e-8) -> Dict[str, Tensor]:
        """Compute sparsity-related losses.
        
        Args:
            mixture_weights: Mixture coefficients ∈ R^{batch×num_bases}
            eps: Small epsilon for numerical stability
            
        Returns:
            Dictionary with sparsity loss components
        """
        losses = {}
        
        # Current sparsity (fraction of near-zero elements)
        current_sparsity = (mixture_weights.abs() < eps).float().mean()
        losses["current_sparsity"] = current_sparsity
        
        # Sparsity loss (encourage target sparsity level)
        sparsity_loss = (current_sparsity - self.target_sparsity) ** 2
        losses["sparsity_loss"] = sparsity_loss
        losses["weighted_sparsity_loss"] = self.sparsity_weight * sparsity_loss
        
        # Support size (number of non-zero elements per expert)
        support_sizes = (mixture_weights.abs() > eps).sum(dim=1).float()  # batch
        mean_support = support_sizes.mean()
        losses["mean_support_size"] = mean_support
        
        # Support size loss
        support_loss = (mean_support - self.support_target) ** 2
        losses["support_loss"] = support_loss
        losses["weighted_support_loss"] = self.support_weight * support_loss
        
        # Combined sparsity loss
        total_sparsity_loss = losses["weighted_sparsity_loss"] + losses["weighted_support_loss"]
        losses["total_sparsity"] = total_sparsity_loss
        
        return losses


class ReconstructionMetrics(nn.Module):
    """Metrics for evaluating reconstruction quality."""
    
    def __init__(self):
        """Initialize reconstruction metrics."""
        super().__init__()
    
    def forward(
        self, 
        reconstructed: Tensor, 
        target: Tensor,
        mixture_weights: Optional[Tensor] = None,
    ) -> Dict[str, float]:
        """Compute reconstruction metrics.
        
        Args:
            reconstructed: Reconstructed weights ∈ R^{batch×p×d}
            target: Target weights ∈ R^{batch×p×d}
            mixture_weights: Mixture coefficients ∈ R^{batch×num_bases}
            
        Returns:
            Dictionary with reconstruction metrics
        """
        metrics = {}
        
        with torch.no_grad():
            # Basic reconstruction metrics
            mse = F.mse_loss(reconstructed, target).item()
            mae = F.l1_loss(reconstructed, target).item()
            
            # Relative errors
            target_norm = torch.norm(target, p="fro")
            error_norm = torch.norm(reconstructed - target, p="fro")
            relative_error = (error_norm / (target_norm + 1e-8)).item()
            
            metrics.update({
                "mse": mse,
                "mae": mae,
                "rmse": mse ** 0.5,
                "relative_error": relative_error,
                "target_norm": target_norm.item(),
                "reconstruction_norm": torch.norm(reconstructed, p="fro").item(),
            })
            
            # Per-sample metrics
            batch_size = reconstructed.shape[0]
            if batch_size > 1:
                per_sample_mse = F.mse_loss(reconstructed, target, reduction="none")
                per_sample_mse = per_sample_mse.view(batch_size, -1).mean(dim=1)
                
                metrics.update({
                    "mse_std": per_sample_mse.std().item(),
                    "mse_min": per_sample_mse.min().item(),
                    "mse_max": per_sample_mse.max().item(),
                })
            
            # Mixture-related metrics
            if mixture_weights is not None:
                sparsity = (mixture_weights.abs() < 1e-6).float().mean().item()
                support_sizes = (mixture_weights.abs() > 1e-6).sum(dim=1).float()
                
                metrics.update({
                    "mixture_sparsity": sparsity,
                    "mean_support_size": support_sizes.mean().item(),
                    "std_support_size": support_sizes.std().item(),
                    "min_support_size": support_sizes.min().item(),
                    "max_support_size": support_sizes.max().item(),
                    "mixture_l1_norm": torch.mean(torch.abs(mixture_weights)).item(),
                    "mixture_l2_norm": torch.mean(mixture_weights ** 2).item(),
                })
            
            # Spectral properties
            try:
                # Compute singular values of target and reconstruction
                target_flat = target.view(batch_size, -1)  # batch × (p*d)
                recon_flat = reconstructed.view(batch_size, -1)  # batch × (p*d)
                
                target_svd = torch.svd(target_flat)
                recon_svd = torch.svd(recon_flat)
                
                # Effective rank (ratio of sum of singular values to max singular value)
                target_eff_rank = (target_svd.S.sum(dim=1) / (target_svd.S.max(dim=1).values + 1e-8)).mean().item()
                recon_eff_rank = (recon_svd.S.sum(dim=1) / (recon_svd.S.max(dim=1).values + 1e-8)).mean().item()
                
                metrics.update({
                    "target_effective_rank": target_eff_rank,
                    "reconstruction_effective_rank": recon_eff_rank,
                    "spectral_error": F.mse_loss(recon_svd.S, target_svd.S).item(),
                })
                
            except Exception:
                # Skip spectral metrics if SVD fails
                pass
        
        return metrics


class CombinedLoss(nn.Module):
    """Combined loss function with all components."""
    
    def __init__(
        self,
        reconstruction_loss: ReconstructionLoss,
        sparsity_loss: Optional[SparsityLoss] = None,
    ):
        """Initialize combined loss.
        
        Args:
            reconstruction_loss: Main reconstruction loss
            sparsity_loss: Optional sparsity loss
        """
        super().__init__()
        
        self.reconstruction_loss = reconstruction_loss
        self.sparsity_loss = sparsity_loss
    
    def forward(
        self,
        reconstructed: Tensor,
        target: Tensor,
        mixture_weights: Optional[Tensor] = None,
        adapter_matrices: Optional[Tensor] = None,
        basis_banks: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Compute combined loss.
        
        Args:
            reconstructed: Reconstructed expert weights
            target: Target expert weights
            mixture_weights: Mixture coefficients
            adapter_matrices: Adapter matrices
            basis_banks: Basis banks
            
        Returns:
            Dictionary with all loss components
        """
        # Main reconstruction loss
        recon_losses = self.reconstruction_loss(
            reconstructed, target, mixture_weights, adapter_matrices, basis_banks
        )
        
        # Sparsity loss
        if self.sparsity_loss is not None and mixture_weights is not None:
            sparsity_losses = self.sparsity_loss(mixture_weights)
            recon_losses.update(sparsity_losses)
            recon_losses["total"] = recon_losses["total"] + sparsity_losses["total_sparsity"]
        
        return recon_losses
