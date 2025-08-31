"""SafeTensors I/O utilities for GloBE.

This module provides utilities for loading and saving tensors using the
SafeTensors format, with support for memory-efficient loading and metadata handling.
"""

from typing import Dict, List, Optional, Union, Any
import torch
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file, load_file
import json


class SafeTensorsIO:
    """Utilities for SafeTensors I/O operations."""
    
    @staticmethod
    def load_tensors(
        filepath: Union[str, Path],
        tensor_names: Optional[List[str]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Load tensors from a SafeTensors file.
        
        Args:
            filepath: Path to SafeTensors file
            tensor_names: Specific tensor names to load (load all if None)
            device: Device to load tensors on
            
        Returns:
            Dictionary mapping tensor names to tensors
        """
        filepath = Path(filepath)
        tensors = {}
        
        with safe_open(filepath, framework="pt", device=str(device) if device else "cpu") as f:
            if tensor_names is None:
                tensor_names = f.keys()
                
            for name in tensor_names:
                if name in f.keys():
                    tensors[name] = f.get_tensor(name)
                    
        return tensors
    
    @staticmethod
    def save_tensors(
        tensors: Dict[str, torch.Tensor],
        filepath: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save tensors to a SafeTensors file.
        
        Args:
            tensors: Dictionary mapping tensor names to tensors
            filepath: Path to save SafeTensors file
            metadata: Optional metadata to include
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert metadata to strings for SafeTensors
        if metadata is not None:
            metadata = {k: json.dumps(v) if not isinstance(v, str) else v 
                       for k, v in metadata.items()}
        
        save_file(tensors, filepath, metadata=metadata)
    
    @staticmethod
    def get_tensor_info(filepath: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
        """Get information about tensors in a SafeTensors file.
        
        Args:
            filepath: Path to SafeTensors file
            
        Returns:
            Dictionary mapping tensor names to info (shape, dtype, etc.)
        """
        filepath = Path(filepath)
        info = {}
        
        with safe_open(filepath, framework="pt") as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                info[name] = {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "numel": tensor.numel(),
                    "size_bytes": tensor.numel() * tensor.element_size(),
                }
                
        return info
    
    @staticmethod
    def get_metadata(filepath: Union[str, Path]) -> Dict[str, Any]:
        """Get metadata from a SafeTensors file.
        
        Args:
            filepath: Path to SafeTensors file
            
        Returns:
            Dictionary with metadata
        """
        filepath = Path(filepath)
        
        with safe_open(filepath, framework="pt") as f:
            metadata = f.metadata()
            
        # Parse JSON strings back to objects
        if metadata is not None:
            parsed_metadata = {}
            for k, v in metadata.items():
                try:
                    parsed_metadata[k] = json.loads(v)
                except (json.JSONDecodeError, TypeError):
                    parsed_metadata[k] = v
            return parsed_metadata
        
        return {}
    
    @staticmethod
    def list_tensor_files(directory: Union[str, Path]) -> List[Path]:
        """List all SafeTensors files in a directory.
        
        Args:
            directory: Directory to search
            
        Returns:
            List of SafeTensors file paths
        """
        directory = Path(directory)
        return list(directory.glob("*.safetensors"))
    
    @staticmethod
    def merge_tensor_files(
        input_files: List[Union[str, Path]],
        output_file: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Merge multiple SafeTensors files into one.
        
        Args:
            input_files: List of input SafeTensors files
            output_file: Output SafeTensors file path
            metadata: Optional metadata for output file
        """
        all_tensors = {}
        
        for filepath in input_files:
            tensors = SafeTensorsIO.load_tensors(filepath)
            all_tensors.update(tensors)
        
        SafeTensorsIO.save_tensors(all_tensors, output_file, metadata)
