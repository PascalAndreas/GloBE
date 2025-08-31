"""Sparse mixture controller for GloBE.

This module implements entmax/sparsemax-based sparse mixture computation
with temperature annealing for learning token-independent basis combinations.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor
import math

try:
    from entmax import entmax15, sparsemax, entmax_bisect
    ENTMAX_AVAILABLE = True
except ImportError:
    ENTMAX_AVAILABLE = False


class TemperatureScheduler:
    """Temperature scheduler for annealing sparse mixtures."""
    
    def __init__(
        self,
        initial_temp: float = 2.0,
        target_support: int = 12,
        annealing_rate: float = 0.05,
        min_temp: float = 0.1,
        support_ema_decay: float = 0.99,
    ):
        """Initialize temperature scheduler.
        
        Args:
            initial_temp: Initial temperature
            target_support: Target median support size
            annealing_rate: Rate of temperature annealing
            min_temp: Minimum temperature
            support_ema_decay: EMA decay for support size tracking
        """
        self.initial_temp = initial_temp
        self.target_support = target_support
        self.annealing_rate = annealing_rate
        self.min_temp = min_temp
        self.support_ema_decay = support_ema_decay
        
        self.current_temp = initial_temp
        self.support_ema = None
        self.step_count = 0
    
    def update(self, current_support: float) -> float:
        """Update temperature based on current support size.
        
        Args:
            current_support: Current median support size
            
        Returns:
            Updated temperature
        """
        # Update EMA of support size
        if self.support_ema is None:
            self.support_ema = current_support
        else:
            self.support_ema = (
                self.support_ema_decay * self.support_ema +
                (1 - self.support_ema_decay) * current_support
            )
        
        # Anneal temperature if support is above target
        if self.support_ema > self.target_support:
            self.current_temp = max(
                self.min_temp,
                self.current_temp * (1 - self.annealing_rate)
            )
        
        self.step_count += 1
        return self.current_temp
    
    def get_temp(self) -> float:
        """Get current temperature."""
        return self.current_temp
    
    def reset(self) -> None:
        """Reset scheduler state."""
        self.current_temp = self.initial_temp
        self.support_ema = None
        self.step_count = 0


class SparseMixer(nn.Module):
    """Sparse mixture computation with entmax/sparsemax and temperature annealing."""
    
    def __init__(
        self,
        sparsity_map: str = "entmax",
        alpha: float = 1.5,
        temperature_scheduler: Optional[TemperatureScheduler] = None,
        l1_reg: float = 1e-4,
        eps_prune: float = 1e-4,
    ):
        """Initialize sparse mixer.
        
        Args:
            sparsity_map: Sparsity mapping ("entmax", "sparsemax", "softmax")
            alpha: Alpha parameter for entmax (ignored for sparsemax)
            temperature_scheduler: Temperature scheduler for annealing
            l1_reg: L1 regularization strength on mixture weights
            eps_prune: Epsilon for pruning small coefficients
        """
        super().__init__()
        
        if sparsity_map in ["entmax", "sparsemax"] and not ENTMAX_AVAILABLE:
            raise ImportError("entmax package required for sparse mappings")
        
        self.sparsity_map = sparsity_map
        self.alpha = alpha
        self.temperature_scheduler = temperature_scheduler
        self.l1_reg = l1_reg
        self.eps_prune = eps_prune
        
        # Choose sparsity function
        if sparsity_map == "entmax":
            if alpha == 1.5:
                self.sparse_fn = entmax15
            else:
                self.sparse_fn = lambda x, dim: entmax_bisect(x, alpha, dim)
        elif sparsity_map == "sparsemax":
            self.sparse_fn = sparsemax
        elif sparsity_map == "softmax":
            self.sparse_fn = lambda x, dim: torch.softmax(x, dim=dim)
        else:
            raise ValueError(f"Unknown sparsity mapping: {sparsity_map}")
    
    def forward(self, logits: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        """Compute sparse mixture weights.
        
        Args:
            logits: Raw logits ∈ R^{batch×num_bases}
            
        Returns:
            Tuple of (mixture_weights, stats_dict)
        """
        batch_size, num_bases = logits.shape
        
        # Apply temperature scaling
        if self.temperature_scheduler is not None:
            temp = self.temperature_scheduler.get_temp()
            scaled_logits = logits / temp
        else:
            scaled_logits = logits
            temp = 1.0
        
        # Compute sparse mixture weights
        mixture_weights = self.sparse_fn(scaled_logits, dim=-1)
        
        # Epsilon pruning
        if self.eps_prune > 0:
            mask = mixture_weights.abs() > self.eps_prune
            mixture_weights = mixture_weights * mask
            
            # Renormalize if using softmax (others are already normalized)
            if self.sparsity_map == "softmax":
                mixture_weights = mixture_weights / (mixture_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Compute statistics
        with torch.no_grad():
            support_sizes = (mixture_weights > 0).sum(dim=-1).float()  # batch
            mean_support = support_sizes.mean().item()
            median_support = support_sizes.median().item()
            max_weight = mixture_weights.max().item()
            min_nonzero_weight = mixture_weights[mixture_weights > 0].min().item() if (mixture_weights > 0).any() else 0.0
            
            # Update temperature scheduler
            if self.temperature_scheduler is not None and self.training:
                self.temperature_scheduler.update(median_support)
        
        stats = {
            "mean_support": mean_support,
            "median_support": median_support,
            "temperature": temp,
            "max_weight": max_weight,
            "min_nonzero_weight": min_nonzero_weight,
            "sparsity": 1.0 - (mixture_weights > 0).float().mean().item(),
        }
        
        return mixture_weights, stats
    
    def get_regularization_loss(self, mixture_weights: Tensor) -> Tensor:
        """Compute L1 regularization loss on mixture weights.
        
        Args:
            mixture_weights: Mixture weights ∈ R^{batch×num_bases}
            
        Returns:
            L1 regularization loss
        """
        if self.l1_reg > 0:
            return self.l1_reg * torch.mean(torch.abs(mixture_weights))
        return torch.tensor(0.0, device=mixture_weights.device)
    
    def get_support_histogram(self, mixture_weights: Tensor, num_bins: int = 20) -> Dict[str, Tensor]:
        """Get histogram of support sizes.
        
        Args:
            mixture_weights: Mixture weights ∈ R^{batch×num_bases}
            num_bins: Number of histogram bins
            
        Returns:
            Dictionary with histogram data
        """
        with torch.no_grad():
            support_sizes = (mixture_weights > 0).sum(dim=-1).float()
            min_support = support_sizes.min().item()
            max_support = support_sizes.max().item()
            
            if max_support > min_support:
                hist = torch.histc(
                    support_sizes, 
                    bins=num_bins, 
                    min=min_support, 
                    max=max_support
                )
                bin_edges = torch.linspace(min_support, max_support, num_bins + 1)
            else:
                hist = torch.zeros(num_bins)
                bin_edges = torch.zeros(num_bins + 1)
            
            return {
                "histogram": hist,
                "bin_edges": bin_edges,
                "min_support": min_support,
                "max_support": max_support,
            }
    
    def extra_repr(self) -> str:
        """Extra representation string."""
        return (
            f"sparsity_map={self.sparsity_map}, alpha={self.alpha}, "
            f"l1_reg={self.l1_reg}, eps_prune={self.eps_prune}"
        )
