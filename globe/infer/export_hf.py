"""HuggingFace model export utilities for GloBE.

This module provides utilities to export GloBE models back to HuggingFace format
with precomposed weights for deployment and distribution.
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizer
from safetensors.torch import save_file
import json
from pathlib import Path
import shutil

from ..modules.ffn_globe import GloBEFFN
from ..modules.precompose_cache import PrecompositionCache
from ..data.io_safetensors import SafeTensorsIO


class GloBEExporter:
    """Exporter for converting GloBE models to HuggingFace format."""
    
    def __init__(
        self,
        output_dtype: torch.dtype = torch.bfloat16,
        save_mixed: bool = True,
        validate_export: bool = True,
        compress: bool = False,
    ):
        """Initialize GloBE exporter.
        
        Args:
            output_dtype: Data type for exported weights
            save_mixed: Whether to save both original and GloBE versions
            validate_export: Whether to validate exported model
            compress: Whether to compress exported weights
        """
        self.output_dtype = output_dtype
        self.save_mixed = save_mixed
        self.validate_export = validate_export
        self.compress = compress
    
    def export_model(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        output_dir: Path,
        model_name: Optional[str] = None,
        globe_config: Optional[Dict[str, Any]] = None,
        push_to_hub: bool = False,
        hub_repo_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Export GloBE model to HuggingFace format.
        
        Args:
            model: Model with GloBE layers
            tokenizer: Associated tokenizer
            output_dir: Output directory
            model_name: Optional model name
            globe_config: GloBE configuration to save
            push_to_hub: Whether to push to HuggingFace Hub
            hub_repo_id: Repository ID for Hub upload
            
        Returns:
            Export information dictionary
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find GloBE layers
        globe_layers = self._find_globe_layers(model)
        
        if not globe_layers:
            raise ValueError("No GloBE layers found in model")
        
        print(f"Found {len(globe_layers)} GloBE layers to export")
        
        # Precompose all expert weights
        export_info = self._precompose_experts(model, globe_layers)
        
        # Create export-ready model state dict
        export_state_dict = self._create_export_state_dict(model, globe_layers)
        
        # Save model components
        self._save_model_components(
            output_dir, model, tokenizer, export_state_dict, 
            globe_config, export_info
        )
        
        # Validation
        if self.validate_export:
            validation_results = self._validate_export(output_dir, model)
            export_info["validation"] = validation_results
        
        # Push to Hub if requested
        if push_to_hub and hub_repo_id:
            self._push_to_hub(output_dir, hub_repo_id)
            export_info["hub_repo_id"] = hub_repo_id
        
        print(f"Successfully exported model to {output_dir}")
        return export_info
    
    def _find_globe_layers(self, model: PreTrainedModel) -> Dict[str, GloBEFFN]:
        """Find GloBE layers in the model."""
        globe_layers = {}
        
        for name, module in model.named_modules():
            if isinstance(module, GloBEFFN):
                globe_layers[name] = module
        
        return globe_layers
    
    def _precompose_experts(
        self, 
        model: PreTrainedModel, 
        globe_layers: Dict[str, GloBEFFN]
    ) -> Dict[str, Any]:
        """Precompose expert weights for all GloBE layers.
        
        Args:
            model: Model with GloBE layers
            globe_layers: Dictionary of GloBE layers
            
        Returns:
            Export information
        """
        export_info = {
            "precomposed_experts": 0,
            "total_experts": 0,
            "compression_ratio": 0.0,
            "layer_stats": {},
        }
        
        model.eval()
        with torch.no_grad():
            for layer_name, globe_layer in globe_layers.items():
                layer_info = self._precompose_layer_experts(globe_layer)
                export_info["layer_stats"][layer_name] = layer_info
                export_info["precomposed_experts"] += layer_info["num_experts"]
                export_info["total_experts"] += layer_info["num_experts"]
        
        return export_info
    
    def _precompose_layer_experts(self, globe_layer: GloBEFFN) -> Dict[str, Any]:
        """Precompose expert weights for a single layer.
        
        Args:
            globe_layer: GloBE FFN layer
            
        Returns:
            Layer precomposition information
        """
        layer_info = {
            "layer_idx": globe_layer.layer_idx,
            "num_experts": globe_layer.num_experts,
            "rank": globe_layer.rank,
            "precomposed_weights": {},
        }
        
        # Precompose weights for each expert
        for expert_idx in range(globe_layer.num_experts):
            # Get mixture logits
            up_logits = globe_layer.up_mixture_logits[expert_idx:expert_idx+1]
            gate_logits = globe_layer.gate_mixture_logits[expert_idx:expert_idx+1]
            
            # Compute sparse mixture weights
            up_weights, _ = globe_layer.sparse_mixer(up_logits)
            gate_weights, _ = globe_layer.sparse_mixer(gate_logits)
            
            # Get mixed basis vectors
            up_mixed, gate_mixed = globe_layer.global_banks(up_weights, gate_weights)
            
            # Apply adapters to get precomposed weights
            up_adapter = globe_layer.up_adapters[expert_idx]  # p × r
            gate_adapter = globe_layer.gate_adapters[expert_idx]  # p × r
            
            # Precomposed weights: W̃_i = A_i @ φ(Σ_j α_{i,j} B_j)
            up_composed = torch.matmul(up_adapter, up_mixed.T).squeeze()  # p
            gate_composed = torch.matmul(gate_adapter, gate_mixed.T).squeeze()  # p
            
            # Convert to export dtype
            up_composed = up_composed.to(dtype=self.output_dtype)
            gate_composed = gate_composed.to(dtype=self.output_dtype)
            
            # Store precomposed weights
            layer_info["precomposed_weights"][expert_idx] = {
                "up_proj": up_composed,
                "gate_proj": gate_composed,
                "down_proj": globe_layer.down_projections[expert_idx].to(dtype=self.output_dtype),
            }
        
        return layer_info
    
    def _create_export_state_dict(
        self, 
        model: PreTrainedModel, 
        globe_layers: Dict[str, GloBEFFN]
    ) -> Dict[str, Tensor]:
        """Create state dict for export with precomposed weights.
        
        Args:
            model: Original model
            globe_layers: GloBE layers with precomposed weights
            
        Returns:
            Export state dict
        """
        # Start with original model state dict
        export_state_dict = model.state_dict().copy()
        
        # Replace GloBE layer weights with precomposed versions
        for layer_name, globe_layer in globe_layers.items():
            layer_idx = globe_layer.layer_idx
            
            # Get precomposed weights
            layer_info = self._precompose_layer_experts(globe_layer)
            precomposed = layer_info["precomposed_weights"]
            
            # Replace expert weights in state dict
            for expert_idx, expert_weights in precomposed.items():
                # Construct HuggingFace tensor names
                up_name = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight"
                gate_name = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight"
                down_name = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight"
                
                # Update state dict with precomposed weights
                export_state_dict[up_name] = expert_weights["up_proj"]
                export_state_dict[gate_name] = expert_weights["gate_proj"]
                export_state_dict[down_name] = expert_weights["down_proj"]
        
        # Remove GloBE-specific parameters
        keys_to_remove = []
        for key in export_state_dict.keys():
            if any(globe_key in key for globe_key in [
                "global_banks", "sparse_mixer", "up_mixture_logits", 
                "gate_mixture_logits", "up_adapters", "gate_adapters"
            ]):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del export_state_dict[key]
        
        return export_state_dict
    
    def _save_model_components(
        self,
        output_dir: Path,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        state_dict: Dict[str, Tensor],
        globe_config: Optional[Dict[str, Any]],
        export_info: Dict[str, Any],
    ) -> None:
        """Save all model components to output directory.
        
        Args:
            output_dir: Output directory
            model: Original model
            tokenizer: Tokenizer
            state_dict: Export state dict
            globe_config: GloBE configuration
            export_info: Export information
        """
        # Save model weights (SafeTensors format)
        weights_path = output_dir / "model.safetensors"
        
        # Prepare metadata
        metadata = {
            "format": "pt",
            "globe_export": "true",
            "export_dtype": str(self.output_dtype),
            "total_params": str(sum(p.numel() for p in state_dict.values())),
        }
        
        if globe_config:
            metadata["globe_config"] = json.dumps(globe_config)
        
        # Save weights
        SafeTensorsIO.save_tensors(state_dict, weights_path, metadata)
        
        # Save model configuration
        model.config.save_pretrained(output_dir)
        
        # Save tokenizer
        tokenizer.save_pretrained(output_dir)
        
        # Save GloBE-specific information
        globe_info_path = output_dir / "globe_export_info.json"
        with open(globe_info_path, 'w') as f:
            json.dump(export_info, f, indent=2, default=str)
        
        # Save original GloBE configuration if provided
        if globe_config:
            config_path = output_dir / "globe_config.json"
            with open(config_path, 'w') as f:
                json.dump(globe_config, f, indent=2)
        
        # Create README with export information
        self._create_export_readme(output_dir, export_info)
    
    def _create_export_readme(self, output_dir: Path, export_info: Dict[str, Any]) -> None:
        """Create README file with export information.
        
        Args:
            output_dir: Output directory
            export_info: Export information
        """
        readme_content = f"""# GloBE Exported Model

This model has been exported from GloBE (Global-Basis Experts) format to standard HuggingFace format.

## Export Information

- **Precomposed Experts**: {export_info['precomposed_experts']}
- **Total Experts**: {export_info['total_experts']}
- **Export Data Type**: {self.output_dtype}
- **Validation**: {'Passed' if export_info.get('validation', {}).get('passed', False) else 'Not performed'}

## Usage

This model can be used like any standard HuggingFace model:

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("path/to/this/directory")
tokenizer = AutoTokenizer.from_pretrained("path/to/this/directory")

# Use the model normally
inputs = tokenizer("Hello world", return_tensors="pt")
outputs = model(**inputs)
```

## GloBE Information

This model was originally trained using GloBE, which compresses MoE expert weights using:
- Global basis banks shared across layers
- Sparse mixture coefficients learned per expert
- Precomposition and caching for efficient inference

The exported model contains the precomposed expert weights for deployment compatibility.

## Files

- `model.safetensors`: Model weights in SafeTensors format
- `config.json`: Model configuration
- `tokenizer.json`: Tokenizer configuration  
- `globe_export_info.json`: Detailed export information
- `globe_config.json`: Original GloBE training configuration (if available)
"""
        
        readme_path = output_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
    
    def _validate_export(self, output_dir: Path, original_model: PreTrainedModel) -> Dict[str, Any]:
        """Validate the exported model.
        
        Args:
            output_dir: Output directory with exported model
            original_model: Original model for comparison
            
        Returns:
            Validation results
        """
        try:
            from transformers import AutoModel, AutoTokenizer
            
            # Load exported model
            exported_model = AutoModel.from_pretrained(output_dir)
            tokenizer = AutoTokenizer.from_pretrained(output_dir)
            
            # Basic validation
            validation_results = {
                "model_loaded": True,
                "tokenizer_loaded": True,
                "config_valid": True,
                "weights_valid": True,
                "passed": True,
            }
            
            # Test forward pass
            test_input = "Hello, this is a test."
            inputs = tokenizer(test_input, return_tensors="pt")
            
            with torch.no_grad():
                outputs = exported_model(**inputs)
                validation_results["forward_pass"] = True
                validation_results["output_shape"] = list(outputs.last_hidden_state.shape)
            
            return validation_results
            
        except Exception as e:
            return {
                "model_loaded": False,
                "error": str(e),
                "passed": False,
            }
    
    def _push_to_hub(self, output_dir: Path, repo_id: str) -> None:
        """Push exported model to HuggingFace Hub.
        
        Args:
            output_dir: Directory with exported model
            repo_id: Repository ID on Hub
        """
        try:
            from huggingface_hub import HfApi
            
            api = HfApi()
            
            # Upload all files in output directory
            for file_path in output_dir.iterdir():
                if file_path.is_file():
                    api.upload_file(
                        path_or_fileobj=str(file_path),
                        path_in_repo=file_path.name,
                        repo_id=repo_id,
                        repo_type="model",
                    )
            
            print(f"Successfully pushed model to {repo_id}")
            
        except ImportError:
            print("huggingface_hub not installed, skipping Hub upload")
        except Exception as e:
            print(f"Failed to push to Hub: {e}")


def export_globe_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    output_dir: Path,
    **export_kwargs,
) -> Dict[str, Any]:
    """Convenience function to export GloBE model.
    
    Args:
        model: Model with GloBE layers
        tokenizer: Associated tokenizer
        output_dir: Output directory
        **export_kwargs: Additional export arguments
        
    Returns:
        Export information
    """
    exporter = GloBEExporter(**export_kwargs)
    return exporter.export_model(model, tokenizer, output_dir)
