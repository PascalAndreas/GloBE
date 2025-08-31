"""Z-score normalization utilities for GloBE.

This module provides utilities for normalizing expert weights using
z-score normalization per family (Up, Gate, Down) to improve training stability.
"""

from typing import Dict, List, Tuple, Optional
import torch
from torch import Tensor
import json
from pathlib import Path


class ZScoreNormalizer:
    """Z-score normalization for expert weight families."""
    
    def __init__(self, eps: float = 1e-8):
        """Initialize z-score normalizer.
        
        Args:
            eps: Small epsilon for numerical stability
        """
        self.eps = eps
        self.stats: Dict[str, Dict[str, float]] = {}
    
    def fit(self, expert_weights: Dict[str, List[Tensor]]) -> None:
        """Compute normalization statistics for each family.
        
        Args:
            expert_weights: Dictionary mapping family names to lists of expert tensors
        """
        self.stats = {}
        
        for family_name, weight_list in expert_weights.items():
            if not weight_list:
                continue
            
            # Stack all weights in the family
            stacked_weights = torch.stack(weight_list, dim=0)  # num_experts × p × d
            
            # Compute global mean and std across all experts and dimensions
            global_mean = torch.mean(stacked_weights).item()
            global_std = torch.std(stacked_weights).item()
            
            # Compute per-expert statistics for analysis
            per_expert_means = torch.mean(stacked_weights, dim=(1, 2))  # num_experts
            per_expert_stds = torch.std(stacked_weights, dim=(1, 2))  # num_experts
            
            self.stats[family_name] = {
                "global_mean": global_mean,
                "global_std": max(global_std, self.eps),  # Avoid division by zero
                "per_expert_mean_avg": torch.mean(per_expert_means).item(),
                "per_expert_mean_std": torch.std(per_expert_means).item(),
                "per_expert_std_avg": torch.mean(per_expert_stds).item(),
                "per_expert_std_std": torch.std(per_expert_stds).item(),
                "num_experts": len(weight_list),
                "weight_shape": list(weight_list[0].shape),
                "total_parameters": sum(w.numel() for w in weight_list),
            }
    
    def transform(self, expert_weights: Dict[str, List[Tensor]]) -> Dict[str, List[Tensor]]:
        """Apply z-score normalization to expert weights.
        
        Args:
            expert_weights: Dictionary mapping family names to lists of expert tensors
            
        Returns:
            Dictionary with normalized expert weights
        """
        if not self.stats:
            raise ValueError("Must call fit() before transform()")
        
        normalized_weights = {}
        
        for family_name, weight_list in expert_weights.items():
            if family_name not in self.stats:
                # If no stats for this family, return as-is
                normalized_weights[family_name] = weight_list
                continue
            
            stats = self.stats[family_name]
            mean = stats["global_mean"]
            std = stats["global_std"]
            
            # Normalize each weight tensor
            normalized_list = []
            for weight in weight_list:
                normalized_weight = (weight - mean) / std
                normalized_list.append(normalized_weight)
            
            normalized_weights[family_name] = normalized_list
        
        return normalized_weights
    
    def inverse_transform(self, normalized_weights: Dict[str, List[Tensor]]) -> Dict[str, List[Tensor]]:
        """Apply inverse z-score normalization to recover original scale.
        
        Args:
            normalized_weights: Dictionary with normalized expert weights
            
        Returns:
            Dictionary with denormalized expert weights
        """
        if not self.stats:
            raise ValueError("Must call fit() before inverse_transform()")
        
        denormalized_weights = {}
        
        for family_name, weight_list in normalized_weights.items():
            if family_name not in self.stats:
                # If no stats for this family, return as-is
                denormalized_weights[family_name] = weight_list
                continue
            
            stats = self.stats[family_name]
            mean = stats["global_mean"]
            std = stats["global_std"]
            
            # Denormalize each weight tensor
            denormalized_list = []
            for weight in weight_list:
                denormalized_weight = weight * std + mean
                denormalized_list.append(denormalized_weight)
            
            denormalized_weights[family_name] = denormalized_list
        
        return denormalized_weights
    
    def transform_single(self, weight: Tensor, family_name: str) -> Tensor:
        """Transform a single weight tensor.
        
        Args:
            weight: Weight tensor to normalize
            family_name: Family name for normalization stats
            
        Returns:
            Normalized weight tensor
        """
        if family_name not in self.stats:
            return weight
        
        stats = self.stats[family_name]
        mean = stats["global_mean"]
        std = stats["global_std"]
        
        return (weight - mean) / std
    
    def inverse_transform_single(self, normalized_weight: Tensor, family_name: str) -> Tensor:
        """Inverse transform a single weight tensor.
        
        Args:
            normalized_weight: Normalized weight tensor
            family_name: Family name for normalization stats
            
        Returns:
            Denormalized weight tensor
        """
        if family_name not in self.stats:
            return normalized_weight
        
        stats = self.stats[family_name]
        mean = stats["global_mean"]
        std = stats["global_std"]
        
        return normalized_weight * std + mean
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get normalization statistics.
        
        Returns:
            Dictionary with normalization statistics per family
        """
        return self.stats.copy()
    
    def save_stats(self, filepath: Path) -> None:
        """Save normalization statistics to file.
        
        Args:
            filepath: Path to save statistics
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    def load_stats(self, filepath: Path) -> None:
        """Load normalization statistics from file.
        
        Args:
            filepath: Path to load statistics from
        """
        filepath = Path(filepath)
        
        with open(filepath, 'r') as f:
            self.stats = json.load(f)
    
    def get_summary(self) -> Dict[str, any]:
        """Get summary of normalization statistics.
        
        Returns:
            Summary dictionary
        """
        if not self.stats:
            return {"error": "No statistics available"}
        
        summary = {
            "num_families": len(self.stats),
            "families": list(self.stats.keys()),
            "total_experts": sum(s["num_experts"] for s in self.stats.values()),
            "total_parameters": sum(s["total_parameters"] for s in self.stats.values()),
        }
        
        # Add per-family summaries
        for family_name, stats in self.stats.items():
            family_summary = {
                "mean": stats["global_mean"],
                "std": stats["global_std"],
                "num_experts": stats["num_experts"],
                "shape": stats["weight_shape"],
                "parameters": stats["total_parameters"],
                "expert_mean_variation": stats["per_expert_mean_std"],
                "expert_std_variation": stats["per_expert_std_std"],
            }
            summary[f"{family_name}_summary"] = family_summary
        
        return summary


def normalize_expert_families(
    expert_weights: Dict[str, List[Tensor]],
    save_stats_path: Optional[Path] = None,
) -> Tuple[Dict[str, List[Tensor]], ZScoreNormalizer]:
    """Convenience function to normalize expert weight families.
    
    Args:
        expert_weights: Dictionary mapping family names to expert weight lists
        save_stats_path: Optional path to save normalization statistics
        
    Returns:
        Tuple of (normalized_weights, normalizer)
    """
    normalizer = ZScoreNormalizer()
    normalizer.fit(expert_weights)
    
    if save_stats_path is not None:
        normalizer.save_stats(save_stats_path)
    
    normalized_weights = normalizer.transform(expert_weights)
    
    return normalized_weights, normalizer


def load_and_denormalize(
    normalized_weights: Dict[str, List[Tensor]],
    stats_path: Path,
) -> Dict[str, List[Tensor]]:
    """Load normalization stats and denormalize weights.
    
    Args:
        normalized_weights: Dictionary with normalized weights
        stats_path: Path to normalization statistics file
        
    Returns:
        Dictionary with denormalized weights
    """
    normalizer = ZScoreNormalizer()
    normalizer.load_stats(stats_path)
    
    return normalizer.inverse_transform(normalized_weights)
