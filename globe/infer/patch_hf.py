"""HuggingFace model patching utilities for GloBE.

This module provides utilities to patch HuggingFace MoE models with GloBE
replacements, enabling drop-in usage with existing model architectures.
"""

from typing import Dict, List, Optional, Tuple, Any, Type
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.models.qwen2_moe import Qwen2MoeSparseMoeBlock
import warnings
from pathlib import Path

from ..modules.ffn_globe import GloBEFFN
from ..modules.globe_bank import DualGloBEBank  
from ..modules.globe_mixer import SparseMixer, TemperatureScheduler
from ..modules.precompose_cache import PrecompositionCache
from ..data.naming_qwen15 import Qwen15MoETensorNaming


class GloBEPatcher:
    """Patcher for replacing MoE layers with GloBE equivalents."""
    
    def __init__(
        self,
        model_name: str = "Qwen1.5-MoE-A2.7B",
        cache_config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ):
        """Initialize GloBE patcher.
        
        Args:
            model_name: Name of the model to patch
            cache_config: Configuration for precomposition cache
            device: Device for patched components
        """
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize cache if configured
        self.cache = None
        if cache_config and cache_config.get("enabled", True):
            self.cache = PrecompositionCache(
                capacity_per_layer=cache_config.get("S_per_layer", 16),
                dtype=getattr(torch, cache_config.get("dtype", "bfloat16")),
                device=self.device,
                max_memory_gb=cache_config.get("max_memory_gb"),
                track_stats=cache_config.get("track_stats", True),
            )
        
        # Model-specific configurations
        self.tensor_naming = Qwen15MoETensorNaming()
        
        # Track patched layers
        self.patched_layers: Dict[str, GloBEFFN] = {}
        self.original_layers: Dict[str, nn.Module] = {}
    
    def patch_model(
        self,
        model: PreTrainedModel,
        globe_weights_path: Path,
        layer_indices: Optional[List[int]] = None,
    ) -> PreTrainedModel:
        """Patch MoE layers in a HuggingFace model with GloBE.
        
        Args:
            model: HuggingFace model to patch
            globe_weights_path: Path to saved GloBE weights
            layer_indices: Optional list of layer indices to patch (patch all if None)
            
        Returns:
            Patched model
        """
        # Load GloBE weights
        globe_data = torch.load(globe_weights_path, map_location=self.device)
        
        # Extract model configuration
        model_config = self._extract_model_config(model)
        
        # Find MoE layers to patch
        moe_layers = self._find_moe_layers(model)
        
        if layer_indices is not None:
            moe_layers = {k: v for k, v in moe_layers.items() if k in layer_indices}
        
        # Patch each MoE layer
        for layer_idx, moe_layer in moe_layers.items():
            print(f"Patching layer {layer_idx}...")
            
            # Create GloBE replacement
            globe_ffn = self._create_globe_ffn(
                layer_idx, model_config, globe_data, moe_layer
            )
            
            # Replace the layer
            self._replace_layer(model, layer_idx, moe_layer, globe_ffn)
        
        print(f"Successfully patched {len(moe_layers)} MoE layers with GloBE")
        return model
    
    def _extract_model_config(self, model: PreTrainedModel) -> Dict[str, Any]:
        """Extract configuration from HuggingFace model."""
        config = model.config
        
        return {
            "hidden_dim": config.hidden_size,
            "intermediate_dim": config.moe_intermediate_size,
            "num_experts": config.num_experts,
            "num_experts_per_tok": config.num_experts_per_tok,
            "num_layers": config.num_hidden_layers,
        }
    
    def _find_moe_layers(self, model: PreTrainedModel) -> Dict[int, nn.Module]:
        """Find MoE layers in the model."""
        moe_layers = {}
        
        # Walk through model to find MoE layers
        for name, module in model.named_modules():
            # Look for MoE blocks (model-specific)
            if isinstance(module, Qwen2MoeSparseMoeBlock):
                # Extract layer index from name
                parts = name.split(".")
                for part in parts:
                    if part.isdigit():
                        layer_idx = int(part)
                        moe_layers[layer_idx] = module
                        break
        
        return moe_layers
    
    def _create_globe_ffn(
        self,
        layer_idx: int,
        model_config: Dict[str, Any],
        globe_data: Dict[str, Any],
        original_moe: nn.Module,
    ) -> GloBEFFN:
        """Create GloBE FFN to replace MoE layer."""
        
        # Extract GloBE components from saved data
        dual_bank = globe_data["dual_bank"]
        mixture_logits = globe_data["mixture_logits"]
        adapter_matrices = globe_data["adapter_matrices"]
        sparsity_config = globe_data.get("sparsity_config", {})
        
        # Create sparse mixer
        temp_scheduler = TemperatureScheduler(
            initial_temp=sparsity_config.get("T0", 2.0),
            target_support=sparsity_config.get("target_support", 12),
            annealing_rate=sparsity_config.get("eta", 0.05),
            min_temp=sparsity_config.get("min_temp", 0.1),
        )
        
        sparse_mixer = SparseMixer(
            sparsity_map=sparsity_config.get("map", "entmax"),
            alpha=sparsity_config.get("alpha", 1.5),
            temperature_scheduler=temp_scheduler,
            l1_reg=sparsity_config.get("l1", 1e-4),
            eps_prune=sparsity_config.get("eps_prune", 1e-4),
        )
        
        # Create GloBE FFN
        globe_ffn = GloBEFFN(
            layer_idx=layer_idx,
            num_experts=model_config["num_experts"],
            hidden_dim=model_config["hidden_dim"],
            intermediate_dim=model_config["intermediate_dim"],
            num_bases_up=dual_bank.up_bank.num_bases,
            num_bases_gate=dual_bank.gate_bank.num_bases,
            rank=dual_bank.up_bank.rank,
            global_banks=dual_bank,
            sparse_mixer=sparse_mixer,
            cache=self.cache,
        )
        
        # Load layer-specific weights
        layer_mixture_logits = {
            "up": mixture_logits["up"],
            "gate": mixture_logits["gate"],
        }
        layer_adapters = {
            "up": adapter_matrices["up"],
            "gate": adapter_matrices["gate"],
        }
        
        # Set the learned parameters
        with torch.no_grad():
            globe_ffn.up_mixture_logits.copy_(layer_mixture_logits["up"])
            globe_ffn.gate_mixture_logits.copy_(layer_mixture_logits["gate"])
            globe_ffn.up_adapters.copy_(layer_adapters["up"])
            globe_ffn.gate_adapters.copy_(layer_adapters["gate"])
            
            # Copy dense down projections from original MoE
            if hasattr(original_moe, "experts"):
                for i, expert in enumerate(original_moe.experts):
                    if hasattr(expert, "down_proj"):
                        globe_ffn.down_projections[i].copy_(expert.down_proj.weight)
        
        return globe_ffn
    
    def _replace_layer(
        self,
        model: PreTrainedModel,
        layer_idx: int,
        original_layer: nn.Module,
        globe_layer: GloBEFFN,
    ) -> None:
        """Replace original MoE layer with GloBE layer."""
        
        # Store original layer for potential restoration
        self.original_layers[f"layer_{layer_idx}"] = original_layer
        
        # Find the parent module and attribute name
        parent_module = None
        attr_name = None
        
        for name, module in model.named_modules():
            for child_name, child_module in module.named_children():
                if child_module is original_layer:
                    parent_module = module
                    attr_name = child_name
                    break
            if parent_module is not None:
                break
        
        if parent_module is None:
            raise ValueError(f"Could not find parent of layer {layer_idx}")
        
        # Replace the layer
        setattr(parent_module, attr_name, globe_layer)
        self.patched_layers[f"layer_{layer_idx}"] = globe_layer
        
        # Move to correct device
        globe_layer.to(self.device)
    
    def restore_original_layers(self, model: PreTrainedModel) -> PreTrainedModel:
        """Restore original MoE layers.
        
        Args:
            model: Patched model
            
        Returns:
            Model with original layers restored
        """
        for layer_name, original_layer in self.original_layers.items():
            # Find and replace the GloBE layer with original
            for name, module in model.named_modules():
                for child_name, child_module in module.named_children():
                    if isinstance(child_module, GloBEFFN):
                        layer_idx = child_module.layer_idx
                        if f"layer_{layer_idx}" == layer_name:
                            setattr(module, child_name, original_layer)
                            break
        
        # Clear tracking
        self.patched_layers.clear()
        self.original_layers.clear()
        
        return model
    
    def get_patch_info(self) -> Dict[str, Any]:
        """Get information about patched layers.
        
        Returns:
            Dictionary with patch information
        """
        info = {
            "num_patched_layers": len(self.patched_layers),
            "patched_layer_names": list(self.patched_layers.keys()),
            "model_name": self.model_name,
            "cache_enabled": self.cache is not None,
        }
        
        if self.cache is not None:
            info["cache_stats"] = self.cache.get_stats()
            info["cache_memory"] = self.cache.get_memory_usage()
        
        # Add per-layer information
        layer_info = {}
        for layer_name, globe_layer in self.patched_layers.items():
            layer_info[layer_name] = {
                "layer_idx": globe_layer.layer_idx,
                "num_experts": globe_layer.num_experts,
                "rank": globe_layer.rank,
                "num_bases_up": globe_layer.global_banks.up_bank.num_bases,
                "num_bases_gate": globe_layer.global_banks.gate_bank.num_bases,
            }
        
        info["layer_details"] = layer_info
        
        return info
    
    def warmup_cache(
        self,
        model: PreTrainedModel,
        dataloader: torch.utils.data.DataLoader,
        num_tokens: int = 2048,
    ) -> Dict[str, Any]:
        """Warm up precomposition cache.
        
        Args:
            model: Model with patched GloBE layers
            dataloader: Data loader for warmup
            num_tokens: Number of tokens for warmup
            
        Returns:
            Warmup statistics
        """
        if self.cache is None:
            return {"error": "No cache configured"}
        
        return self.cache.warmup(model, dataloader, num_tokens)
    
    def benchmark_inference(
        self,
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        num_runs: int = 10,
    ) -> Dict[str, float]:
        """Benchmark inference performance.
        
        Args:
            model: Model to benchmark
            input_ids: Input token IDs
            num_runs: Number of benchmark runs
            
        Returns:
            Benchmark results
        """
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(input_ids)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                outputs = model(input_ids)
                end_time.record()
                
                torch.cuda.synchronize()
                elapsed = start_time.elapsed_time(end_time)  # milliseconds
                times.append(elapsed)
        
        # Calculate statistics
        times = torch.tensor(times)
        seq_len = input_ids.shape[1]
        
        results = {
            "mean_time_ms": times.mean().item(),
            "std_time_ms": times.std().item(),
            "min_time_ms": times.min().item(),
            "max_time_ms": times.max().item(),
            "tokens_per_second": seq_len * 1000 / times.mean().item(),
            "sequence_length": seq_len,
            "num_runs": num_runs,
        }
        
        # Add cache statistics if available
        if self.cache is not None:
            cache_stats = self.cache.get_stats()
            results["cache_hit_rate"] = cache_stats["global_stats"]["global_hit_rate"]
            results["cache_memory_mb"] = cache_stats["memory_stats"]["memory_usage_mb"]
        
        return results
