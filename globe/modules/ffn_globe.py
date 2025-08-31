"""GloBE FFN module for MoE replacement.

This module implements the GloBE FFN that replaces standard MoE FFNs,
using global basis banks and sparse mixtures for Up/Gate projections
while keeping Down projections dense.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .globe_bank import DualGloBEBank
from .globe_mixer import SparseMixer
from .precompose_cache import PrecompositionCache


class GloBEFFN(nn.Module):
    """GloBE FFN module: A_i · φ(Σ_j α_{i,j} B_j) for Up/Gate, Down stays dense.
    
    This module replaces a standard MoE FFN layer with a GloBE variant that:
    1. Uses global basis banks for Up and Gate projections
    2. Learns sparse, token-independent mixture weights per expert
    3. Keeps Down projections as dense expert-specific weights
    4. Supports precomposition and caching for efficient inference
    """
    
    def __init__(
        self,
        layer_idx: int,
        num_experts: int,
        hidden_dim: int,
        intermediate_dim: int,
        num_bases_up: int,
        num_bases_gate: int,
        rank: int,
        global_banks: DualGloBEBank,
        sparse_mixer: SparseMixer,
        cache: Optional[PrecompositionCache] = None,
        activation: str = "silu",
    ):
        """Initialize GloBE FFN module.
        
        Args:
            layer_idx: Layer index in the model
            num_experts: Number of experts in this layer
            hidden_dim: Hidden dimension (d)
            intermediate_dim: Intermediate FFN dimension (p) 
            num_bases_up: Number of Up projection bases
            num_bases_gate: Number of Gate projection bases
            rank: Rank of basis vectors (r)
            global_banks: Shared global basis banks
            sparse_mixer: Sparse mixture controller
            cache: Optional precomposition cache
            activation: Activation function for FFN
        """
        super().__init__()
        
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.rank = rank
        
        # Global basis banks (shared across layers)
        self.global_banks = global_banks
        
        # Sparse mixture controller
        self.sparse_mixer = sparse_mixer
        
        # Precomposition cache (optional)
        self.cache = cache
        
        # Expert-specific parameters
        # A_i matrices for Up projection: R^{p×r} per expert
        self.up_adapters = nn.Parameter(torch.empty(num_experts, intermediate_dim, rank))
        
        # A_i matrices for Gate projection: R^{p×r} per expert  
        self.gate_adapters = nn.Parameter(torch.empty(num_experts, intermediate_dim, rank))
        
        # Mixture logits for Up projection: R^{m_up} per expert
        self.up_mixture_logits = nn.Parameter(torch.empty(num_experts, num_bases_up))
        
        # Mixture logits for Gate projection: R^{m_gate} per expert
        self.gate_mixture_logits = nn.Parameter(torch.empty(num_experts, num_bases_gate))
        
        # Dense Down projections: R^{d×p} per expert (unchanged from MoE)
        self.down_projections = nn.Parameter(torch.empty(num_experts, hidden_dim, intermediate_dim))
        
        # Activation function
        if activation == "silu":
            self.activation = F.silu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        self._init_parameters()
    
    def _init_parameters(self) -> None:
        """Initialize parameters."""
        # Initialize adapters
        nn.init.xavier_uniform_(self.up_adapters)
        nn.init.xavier_uniform_(self.gate_adapters)
        
        # Initialize mixture logits (small random values)
        nn.init.normal_(self.up_mixture_logits, std=0.01)
        nn.init.normal_(self.gate_mixture_logits, std=0.01)
        
        # Initialize down projections
        nn.init.xavier_uniform_(self.down_projections)
    
    def forward(
        self, 
        hidden_states: Tensor,
        expert_indices: Tensor,
        expert_weights: Tensor,
    ) -> Tuple[Tensor, Dict[str, any]]:
        """Forward pass through GloBE FFN.
        
        Args:
            hidden_states: Input hidden states ∈ R^{seq×d}
            expert_indices: Selected expert indices ∈ R^{seq×topk}
            expert_weights: Expert routing weights ∈ R^{seq×topk}
            
        Returns:
            Tuple of (output_states, info_dict)
        """
        seq_len, hidden_dim = hidden_states.shape
        topk = expert_indices.shape[1]
        
        # Flatten for expert processing
        flat_hidden = hidden_states.view(-1, hidden_dim)  # (seq×1)×d
        flat_indices = expert_indices.view(-1)  # seq×topk
        flat_weights = expert_weights.view(-1)  # seq×topk
        
        outputs = []
        stats = {"up_mixer_stats": {}, "gate_mixer_stats": {}, "cache_stats": {}}
        
        # Process each selected expert
        unique_experts = torch.unique(flat_indices)
        
        for expert_idx in unique_experts:
            expert_idx = expert_idx.item()
            
            # Find tokens routed to this expert
            expert_mask = (flat_indices == expert_idx)
            if not expert_mask.any():
                continue
                
            expert_tokens = flat_hidden[expert_mask]  # n_tokens×d
            expert_routing_weights = flat_weights[expert_mask]  # n_tokens
            
            # Get expert output
            expert_output, expert_stats = self._forward_expert(
                expert_tokens, expert_idx, expert_routing_weights
            )
            
            # Accumulate outputs and stats
            outputs.append((expert_output, expert_mask, expert_routing_weights))
            for key, value in expert_stats.items():
                if key not in stats:
                    stats[key] = {}
                stats[key][f"expert_{expert_idx}"] = value
        
        # Combine expert outputs
        final_output = torch.zeros_like(flat_hidden)
        for expert_output, expert_mask, routing_weights in outputs:
            final_output[expert_mask] += expert_output * routing_weights.unsqueeze(-1)
        
        # Reshape back to sequence format
        final_output = final_output.view(seq_len, hidden_dim)
        
        return final_output, stats
    
    def _forward_expert(
        self, 
        tokens: Tensor, 
        expert_idx: int,
        routing_weights: Tensor,
    ) -> Tuple[Tensor, Dict[str, any]]:
        """Forward pass for a single expert.
        
        Args:
            tokens: Tokens for this expert ∈ R^{n_tokens×d}
            expert_idx: Expert index
            routing_weights: Routing weights for these tokens
            
        Returns:
            Tuple of (expert_output, expert_stats)
        """
        n_tokens, hidden_dim = tokens.shape
        
        # Check cache first if available
        if self.cache is not None:
            cached_up, cached_gate = self.cache.get(self.layer_idx, expert_idx)
            if cached_up is not None and cached_gate is not None:
                # Use cached precomposed weights
                up_proj = torch.matmul(tokens, cached_up.T)  # n_tokens×p
                gate_proj = torch.matmul(tokens, cached_gate.T)  # n_tokens×p
                
                cache_stats = {"cache_hit": True}
            else:
                # Cache miss: compute and cache
                up_proj, gate_proj, cache_stats = self._compute_and_cache_expert(
                    tokens, expert_idx
                )
        else:
            # No cache: direct computation
            up_proj, gate_proj, cache_stats = self._compute_expert_projections(
                tokens, expert_idx
            )
        
        # Apply activation and element-wise multiplication
        intermediate = self.activation(gate_proj) * up_proj  # n_tokens×p
        
        # Down projection (dense)
        down_weight = self.down_projections[expert_idx]  # d×p
        output = torch.matmul(intermediate, down_weight.T)  # n_tokens×d
        
        expert_stats = {
            "cache_stats": cache_stats,
            "mean_activation": intermediate.mean().item(),
            "activation_sparsity": (intermediate == 0).float().mean().item(),
        }
        
        return output, expert_stats
    
    def _compute_expert_projections(
        self, 
        tokens: Tensor, 
        expert_idx: int
    ) -> Tuple[Tensor, Tensor, Dict[str, any]]:
        """Compute Up and Gate projections for an expert.
        
        Args:
            tokens: Input tokens ∈ R^{n_tokens×d}
            expert_idx: Expert index
            
        Returns:
            Tuple of (up_proj, gate_proj, stats)
        """
        # Get mixture logits for this expert
        up_logits = self.up_mixture_logits[expert_idx:expert_idx+1]  # 1×m_up
        gate_logits = self.gate_mixture_logits[expert_idx:expert_idx+1]  # 1×m_gate
        
        # Compute sparse mixture weights
        up_weights, up_mixer_stats = self.sparse_mixer(up_logits)  # 1×m_up
        gate_weights, gate_mixer_stats = self.sparse_mixer(gate_logits)  # 1×m_gate
        
        # Get mixed basis vectors from global banks
        up_mixed, gate_mixed = self.global_banks(up_weights, gate_weights)  # 1×r each
        
        # Apply adapter matrices: A_i @ φ(Σ_j α_{i,j} B_j)
        up_adapter = self.up_adapters[expert_idx]  # p×r
        gate_adapter = self.gate_adapters[expert_idx]  # p×r
        
        up_weight = torch.matmul(up_adapter, up_mixed.T)  # p×1 -> p
        gate_weight = torch.matmul(gate_adapter, gate_mixed.T)  # p×1 -> p
        
        # Project tokens
        up_proj = torch.matmul(tokens, up_weight.unsqueeze(0).T)  # n_tokens×p
        gate_proj = torch.matmul(tokens, gate_weight.unsqueeze(0).T)  # n_tokens×p
        
        stats = {
            "up_mixer_stats": up_mixer_stats,
            "gate_mixer_stats": gate_mixer_stats,
            "cache_hit": False,
        }
        
        return up_proj, gate_proj, stats
    
    def _compute_and_cache_expert(
        self, 
        tokens: Tensor, 
        expert_idx: int
    ) -> Tuple[Tensor, Tensor, Dict[str, any]]:
        """Compute expert projections and update cache.
        
        Args:
            tokens: Input tokens ∈ R^{n_tokens×d}
            expert_idx: Expert index
            
        Returns:
            Tuple of (up_proj, gate_proj, stats)
        """
        # Compute projections
        up_proj, gate_proj, stats = self._compute_expert_projections(tokens, expert_idx)
        
        # Precompose weights for caching
        if self.cache is not None:
            # Get the composed weight matrices
            up_logits = self.up_mixture_logits[expert_idx:expert_idx+1]
            gate_logits = self.gate_mixture_logits[expert_idx:expert_idx+1]
            
            up_weights, _ = self.sparse_mixer(up_logits)
            gate_weights, _ = self.sparse_mixer(gate_logits)
            
            up_mixed, gate_mixed = self.global_banks(up_weights, gate_weights)
            
            up_adapter = self.up_adapters[expert_idx]
            gate_adapter = self.gate_adapters[expert_idx]
            
            # Precomposed weights: W̃_i = A_i @ φ(Σ_j α_{i,j} B_j)
            up_composed = torch.matmul(up_adapter, up_mixed.T).squeeze()  # p
            gate_composed = torch.matmul(gate_adapter, gate_mixed.T).squeeze()  # p
            
            # Cache the composed weights
            self.cache.put(self.layer_idx, expert_idx, up_composed, gate_composed)
            stats["cache_stats"]["cached"] = True
        
        return up_proj, gate_proj, stats
    
    def get_regularization_loss(self) -> Tensor:
        """Get regularization losses from all components."""
        loss = torch.tensor(0.0, device=self.up_adapters.device)
        
        # Global bank regularization
        loss = loss + self.global_banks.get_regularization_loss()
        
        # Mixture weight regularization (L1)
        # Process all experts at once for efficiency
        all_up_weights, _ = self.sparse_mixer(self.up_mixture_logits)
        all_gate_weights, _ = self.sparse_mixer(self.gate_mixture_logits)
        
        loss = loss + self.sparse_mixer.get_regularization_loss(all_up_weights)
        loss = loss + self.sparse_mixer.get_regularization_loss(all_gate_weights)
        
        return loss
    
    def extra_repr(self) -> str:
        """Extra representation string."""
        return (
            f"layer_idx={self.layer_idx}, num_experts={self.num_experts}, "
            f"hidden_dim={self.hidden_dim}, intermediate_dim={self.intermediate_dim}, "
            f"rank={self.rank}"
        )
