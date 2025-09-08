"""Tensor name patterns for Qwen1.5-MoE models.

This module provides utilities to map between HuggingFace tensor names
and GloBE internal representations for Qwen1.5-MoE models.
Includes local caching functionality to avoid re-downloading models.
"""

from typing import Dict, List, Tuple, Optional
import re
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM
from huggingface_hub import snapshot_download

# Cache directory for downloaded models
CACHE_DIR = Path.home() / ".cache" / "globe_models"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


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


def get_model_cache_path(model_name: str) -> Path:
    """Get the local cache path for a model."""
    # Replace slashes and special chars for filesystem safety
    safe_name = model_name.replace("/", "_").replace(":", "_")
    return CACHE_DIR / safe_name


def is_model_cached(model_name: str) -> bool:
    """Check if a model is already cached locally."""
    cache_path = get_model_cache_path(model_name)
    # Check if cache exists and has safetensors files
    if cache_path.exists():
        safetensors_files = list(cache_path.glob("*.safetensors"))
        return len(safetensors_files) > 0
    return False


def download_and_cache_model(model_name: str, force_download: bool = False) -> Path:
    """Download and cache a model locally, or return existing cache path."""
    cache_path = get_model_cache_path(model_name)
    
    if not force_download and is_model_cached(model_name):
        print(f"✅ Using cached model at {cache_path}")
        return cache_path
    
    print(f"📥 Downloading {model_name} to {cache_path}")
    print("   This may take a few minutes for the initial download...")
    
    # Download using huggingface_hub (more efficient than AutoModel)
    try:
        snapshot_download(
            repo_id=model_name,
            local_dir=cache_path,
            local_dir_use_symlinks=False,  # Use actual files, not symlinks
            resume_download=True,  # Resume if interrupted
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.md", "*.py"],  # Include model files
            ignore_patterns=["*.msgpack", "*.h5"],  # Skip unnecessary formats
        )
        print(f"✅ Model cached successfully at {cache_path}")
        return cache_path
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        raise


def load_model_from_cache(model_name: str, force_download: bool = False):
    """Load a model from cache, downloading if necessary."""
    cache_path = download_and_cache_model(model_name, force_download)
    
    print(f"🔄 Loading model from {cache_path}")
    model = AutoModelForCausalLM.from_pretrained(
        cache_path,
        trust_remote_code=True,
        torch_dtype="auto",  # Use model's native dtype
    )
    
    return model


def extract_expert_weights(
    model_name_or_path: str, 
    include_shared: bool = False,
    force_download: bool = False,
    max_experts: Optional[int] = None,
    layers: Optional[List[int]] = None,
    sample_method: str = "first_n"
) -> Dict[str, List[torch.Tensor]]:
    """Load a HuggingFace model and extract expert weights grouped by family.

    Args:
        model_name_or_path: Path to HuggingFace model or local path
        include_shared: If True, include shared experts. If False (default), 
                       only extract routed experts for bank training.
        force_download: If True, force re-download even if cached
        max_experts: If provided, limit to this many experts (for memory efficiency)
        layers: If provided, extract experts only from these layer indices (e.g., [12, 13, 14])
        sample_method: How to sample experts when max_experts is set:
                      - "first_n": Take first n experts
                      - "evenly_spaced": Take evenly spaced experts across all available

    Returns:
        Dictionary with 'up' and 'gate' families containing expert weight tensors.
        Only routed experts are included by default since shared experts should 
        remain dense and not be decomposed into banks.
    """

    # Check if it's a local path or a model name that needs downloading
    if Path(model_name_or_path).exists():
        print(f"📂 Loading from local path: {model_name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
    else:
        # Use our caching system
        model = load_model_from_cache(model_name_or_path, force_download)
    naming = Qwen15MoETensorNaming()

    families: Dict[str, List[torch.Tensor]] = {"up": [], "gate": []}
    for name, param in model.named_parameters():
        info = naming.parse_tensor_name(name)
        if info is None:
            continue
        ttype = info["type"]
        
        # Skip if layers are specified and this isn't one of the right layers
        if layers is not None and "layer" in info and info["layer"] not in layers:
            continue
        
        # Skip shared experts unless explicitly requested
        if ttype.startswith("shared_") and not include_shared:
            continue
            
        # Only include routed experts by default
        if ttype.startswith("expert_"):
            if "up" in ttype:
                families["up"].append(param.detach().to("cpu"))
            elif "gate" in ttype:
                families["gate"].append(param.detach().to("cpu"))
        # Include shared experts only if requested
        elif include_shared and ttype.startswith("shared_"):
            if "up" in ttype:
                families["up"].append(param.detach().to("cpu"))
            elif "gate" in ttype:
                families["gate"].append(param.detach().to("cpu"))
    
    # Apply expert limit if requested (for memory efficiency)
    if max_experts is not None:
        print(f"🔧 Limiting to {max_experts} experts using {sample_method} method")
        for family in families:
            if len(families[family]) > max_experts:
                original_count = len(families[family])
                
                if sample_method == "first_n":
                    # Take first n experts
                    families[family] = families[family][:max_experts]
                elif sample_method == "evenly_spaced":
                    # Take evenly spaced experts to maintain representativeness
                    indices = torch.linspace(0, len(families[family]) - 1, max_experts).long()
                    families[family] = [families[family][i] for i in indices]
                else:
                    raise ValueError(f"Unknown sample_method: {sample_method}. Use 'first_n' or 'evenly_spaced'")
                
                print(f"   - {family}: {original_count} → {len(families[family])} experts")
    
    # Report extraction results
    if layers is not None:
        if len(layers) == 1:
            print(f"✅ Extracted from layer {layers[0]}: {len(families['up'])} UP, {len(families['gate'])} Gate experts")
        else:
            print(f"✅ Extracted from layers {layers}: {len(families['up'])} UP, {len(families['gate'])} Gate experts")
    else:
        print(f"✅ Extracted from all layers: {len(families['up'])} UP, {len(families['gate'])} Gate experts")
                
    return families


def extract_all_expert_info(model_name_or_path: str, force_download: bool = False) -> Dict[str, any]:
    """Extract complete expert information for debugging and analysis.
    
    Args:
        model_name_or_path: Model name or local path
        force_download: If True, force re-download even if cached
    
    Returns:
        Dictionary with routed_experts, shared_experts, and model_info
    """
    # Check if it's a local path or a model name that needs downloading
    if Path(model_name_or_path).exists():
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
    else:
        model = load_model_from_cache(model_name_or_path, force_download)
    naming = Qwen15MoETensorNaming()
    
    routed_experts = {"up": [], "gate": [], "down": []}
    shared_experts = {"up": [], "gate": [], "down": []}
    moe_gates = []
    
    expert_counts_per_layer = {}
    
    for name, param in model.named_parameters():
        info = naming.parse_tensor_name(name)
        if info is None:
            continue
            
        ttype = info["type"]
        
        if ttype.startswith("expert_"):
            layer = info["layer"]
            expert_idx = info["expert"]
            
            # Track expert counts per layer
            if layer not in expert_counts_per_layer:
                expert_counts_per_layer[layer] = set()
            expert_counts_per_layer[layer].add(expert_idx)
            
            if "up" in ttype:
                routed_experts["up"].append(param.detach().to("cpu"))
            elif "gate" in ttype:
                routed_experts["gate"].append(param.detach().to("cpu"))
            elif "down" in ttype:
                routed_experts["down"].append(param.detach().to("cpu"))
                
        elif ttype.startswith("shared_"):
            if "up" in ttype:
                shared_experts["up"].append(param.detach().to("cpu"))
            elif "gate" in ttype:
                shared_experts["gate"].append(param.detach().to("cpu"))
            elif "down" in ttype:
                shared_experts["down"].append(param.detach().to("cpu"))
                
        elif ttype == "moe_gate":
            moe_gates.append(param.detach().to("cpu"))
    
    # Convert sets to counts
    expert_counts = {layer: len(experts) for layer, experts in expert_counts_per_layer.items()}
    
    return {
        "routed_experts": routed_experts,
        "shared_experts": shared_experts,
        "moe_gates": moe_gates,
        "expert_counts_per_layer": expert_counts,
        "total_routed_experts": len(routed_experts["up"]),
        "num_layers_with_shared": len(shared_experts["up"]),
        "model_info": {
            "name": model_name_or_path,
            "num_parameters": sum(p.numel() for p in model.parameters()),
        }
    }
