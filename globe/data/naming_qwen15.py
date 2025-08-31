"""Tensor name patterns for Qwen1.5-MoE models.

This module provides utilities to map between HuggingFace tensor names
and GloBE internal representations for Qwen1.5-MoE models.
"""

from typing import Dict, List, Tuple, Optional
import re


class Qwen15MoETensorNaming:
    """Tensor naming patterns for Qwen1.5-MoE models."""
    
    def __init__(self):
        """Initialize naming patterns."""
        # Regex patterns for different tensor types
        self.patterns = {
            "moe_gate": r"model\.layers\.(\d+)\.mlp\.gate\.weight",
            "expert_up": r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.up_proj\.weight",
            "expert_gate": r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.gate_proj\.weight", 
            "expert_down": r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.down_proj\.weight",
            "shared_up": r"model\.layers\.(\d+)\.mlp\.shared_experts\.up_proj\.weight",
            "shared_gate": r"model\.layers\.(\d+)\.mlp\.shared_experts\.gate_proj\.weight",
            "shared_down": r"model\.layers\.(\d+)\.mlp\.shared_experts\.down_proj\.weight",
        }
    
    def parse_tensor_name(self, name: str) -> Optional[Dict[str, any]]:
        """Parse a tensor name and return its components.
        
        Args:
            name: Tensor name from HuggingFace model
            
        Returns:
            Dictionary with parsed components or None if not matched
        """
        for tensor_type, pattern in self.patterns.items():
            match = re.match(pattern, name)
            if match:
                result = {"type": tensor_type, "name": name}
                if tensor_type == "moe_gate":
                    result["layer"] = int(match.group(1))
                elif tensor_type.startswith("expert_"):
                    result["layer"] = int(match.group(1))
                    result["expert"] = int(match.group(2))
                elif tensor_type.startswith("shared_"):
                    result["layer"] = int(match.group(1))
                return result
        return None
    
    def get_expert_tensors(self, layer: int, expert: int) -> Dict[str, str]:
        """Get tensor names for a specific expert.
        
        Args:
            layer: Layer index
            expert: Expert index
            
        Returns:
            Dictionary mapping projection type to tensor name
        """
        return {
            "up": f"model.layers.{layer}.mlp.experts.{expert}.up_proj.weight",
            "gate": f"model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight",
            "down": f"model.layers.{layer}.mlp.experts.{expert}.down_proj.weight",
        }
    
    def get_shared_tensors(self, layer: int) -> Dict[str, str]:
        """Get tensor names for shared experts in a layer.
        
        Args:
            layer: Layer index
            
        Returns:
            Dictionary mapping projection type to tensor name
        """
        return {
            "up": f"model.layers.{layer}.mlp.shared_experts.up_proj.weight",
            "gate": f"model.layers.{layer}.mlp.shared_experts.gate_proj.weight", 
            "down": f"model.layers.{layer}.mlp.shared_experts.down_proj.weight",
        }
    
    def get_moe_gate_tensor(self, layer: int) -> str:
        """Get MoE gate tensor name for a layer.
        
        Args:
            layer: Layer index
            
        Returns:
            Gate tensor name
        """
        return f"model.layers.{layer}.mlp.gate.weight"
    
    def group_tensors_by_family(self, tensor_names: List[str]) -> Dict[str, List[Tuple[str, Dict]]]:
        """Group tensors by family (up, gate, down).
        
        Args:
            tensor_names: List of tensor names
            
        Returns:
            Dictionary mapping family to list of (name, parsed_info) tuples
        """
        families = {"up": [], "gate": [], "down": []}
        
        for name in tensor_names:
            parsed = self.parse_tensor_name(name)
            if parsed is None:
                continue
                
            tensor_type = parsed["type"]
            if "up" in tensor_type:
                families["up"].append((name, parsed))
            elif "gate" in tensor_type:
                families["gate"].append((name, parsed))
            elif "down" in tensor_type:
                families["down"].append((name, parsed))
                
        return families
