"""Global basis banks for GloBE.

This module implements global Up and Gate projection basis banks that are shared
across all MoE layers, enabling parameter compression and efficient inference.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GloBEBank(nn.Module):
    """Global basis bank for Up or Gate projections.
    
    A learnable bank of basis vectors B ∈ R^{m×r} that can be linearly combined
    to approximate expert projections: W_i ≈ A_i @ φ(Σ_j α_{i,j} B_j)
    """
    
    def __init__(
        self,
        num_bases: int,
        rank: int,
        hidden_dim: int,
        activation: str = "silu",
        init_method: str = "xavier_uniform",
        orthogonal_reg: float = 0.0,
        spectral_reg: float = 0.0,
    ):
        """Initialize the global basis bank.
        
        Args:
            num_bases: Number of basis vectors (m)
            rank: Rank/dimension of each basis vector (r)
            hidden_dim: Hidden dimension (d)
            activation: Activation function ("silu", "tanh", "relu", "gelu")
            init_method: Initialization method for basis vectors
            orthogonal_reg: Orthogonality regularization strength
            spectral_reg: Spectral regularization strength
        """
        super().__init__()
        
        self.num_bases = num_bases
        self.rank = rank
        self.hidden_dim = hidden_dim
        self.orthogonal_reg = orthogonal_reg
        self.spectral_reg = spectral_reg
        
        # Basis bank B ∈ R^{m×r×d} - each basis is a r×d matrix
        self.bases = nn.Parameter(torch.empty(num_bases, rank, hidden_dim))
        
        # Activation function
        if activation == "silu":
            self.activation = F.silu
        elif activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        self._init_parameters(init_method)
    
    def _init_parameters(self, method: str) -> None:
        """Initialize basis parameters."""
        if method == "xavier_uniform":
            nn.init.xavier_uniform_(self.bases)
        elif method == "xavier_normal":
            nn.init.xavier_normal_(self.bases)
        elif method == "kaiming_uniform":
            nn.init.kaiming_uniform_(self.bases, nonlinearity="relu")
        elif method == "orthogonal":
            nn.init.orthogonal_(self.bases)
        else:
            nn.init.normal_(self.bases, std=0.02)
    
    def forward(self, mixture_weights: Tensor) -> Tensor:
        """Compute basis mixture.
        
        Args:
            mixture_weights: Mixture weights α ∈ R^{batch×m}
            
        Returns:
            Mixed basis matrices φ(Σ_j α_j B_j) ∈ R^{batch×r×d}
        """
        # Linear combination: batch×m @ m×r×d -> batch×r×d
        # Using einsum for clarity: batch×m, m×r×d -> batch×r×d
        mixed_bases = torch.einsum('bm,mrd->brd', mixture_weights, self.bases)
        
        # Apply activation
        return self.activation(mixed_bases)
    
    def get_regularization_loss(self) -> Tensor:
        """Compute regularization losses for the basis bank.
        
        Returns:
            Combined regularization loss
        """
        loss = torch.tensor(0.0, device=self.bases.device)
        
        # Orthogonality regularization: encourage orthogonal bases
        if self.orthogonal_reg > 0:
            # Compute B^T B - I and penalize off-diagonal elements
            gram = torch.matmul(self.bases.T, self.bases)  # r×r
            identity = torch.eye(self.rank, device=gram.device)
            ortho_loss = torch.sum((gram - identity) ** 2)
            loss = loss + self.orthogonal_reg * ortho_loss
        
        # Spectral regularization: penalize large singular values
        if self.spectral_reg > 0:
            # Compute spectral norm (largest singular value)
            spectral_norm = torch.norm(self.bases, p=2)
            loss = loss + self.spectral_reg * spectral_norm
        
        return loss
    
    def get_basis_stats(self) -> Dict[str, float]:
        """Get statistics about the basis bank.
        
        Returns:
            Dictionary with basis statistics
        """
        with torch.no_grad():
            bases = self.bases
            
            # Compute singular values
            _, s, _ = torch.svd(bases)
            
            stats = {
                "spectral_norm": torch.max(s).item(),
                "condition_number": (torch.max(s) / torch.min(s)).item(),
                "effective_rank": (torch.sum(s) ** 2 / torch.sum(s ** 2)).item(),
                "frobenius_norm": torch.norm(bases, p="fro").item(),
                "mean_abs_weight": torch.mean(torch.abs(bases)).item(),
                "std_weight": torch.std(bases).item(),
            }
            
            # Orthogonality measure
            if bases.shape[0] >= bases.shape[1]:  # More bases than rank
                gram = torch.matmul(bases.T, bases)
                identity = torch.eye(self.rank, device=gram.device)
                ortho_error = torch.norm(gram - identity, p="fro").item()
                stats["orthogonality_error"] = ortho_error
            
            return stats
    
    def extra_repr(self) -> str:
        """Extra representation string."""
        return f"num_bases={self.num_bases}, rank={self.rank}"


class DualGloBEBank(nn.Module):
    """Dual global basis banks for Up and Gate projections.
    
    Manages separate basis banks for Up and Gate projections with shared
    configuration and utilities.
    """
    
    def __init__(
        self,
        num_bases_up: int,
        num_bases_gate: int,
        rank: int,
        hidden_dim: int,
        activation: str = "silu",
        **bank_kwargs,
    ):
        """Initialize dual basis banks.
        
        Args:
            num_bases_up: Number of basis vectors for Up projection
            num_bases_gate: Number of basis vectors for Gate projection  
            rank: Rank/dimension of basis vectors
            hidden_dim: Hidden dimension
            activation: Activation function
            **bank_kwargs: Additional arguments for GloBEBank
        """
        super().__init__()
        
        self.up_bank = GloBEBank(
            num_bases_up, rank, hidden_dim, activation, **bank_kwargs
        )
        self.gate_bank = GloBEBank(
            num_bases_gate, rank, hidden_dim, activation, **bank_kwargs
        )
    
    def forward(
        self, 
        up_weights: Tensor, 
        gate_weights: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Compute basis mixtures for both projections.
        
        Args:
            up_weights: Up mixture weights ∈ R^{batch×m_up}
            gate_weights: Gate mixture weights ∈ R^{batch×m_gate}
            
        Returns:
            Tuple of (up_mixed, gate_mixed) basis vectors
        """
        up_mixed = self.up_bank(up_weights)
        gate_mixed = self.gate_bank(gate_weights)
        return up_mixed, gate_mixed
    
    def get_regularization_loss(self) -> Tensor:
        """Get combined regularization loss from both banks."""
        return (
            self.up_bank.get_regularization_loss() +
            self.gate_bank.get_regularization_loss()
        )
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics from both banks."""
        return {
            "up_bank": self.up_bank.get_basis_stats(),
            "gate_bank": self.gate_bank.get_basis_stats(),
        }
