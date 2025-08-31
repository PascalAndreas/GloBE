"""Precomposition cache for GloBE inference.

This module implements an LRU cache for precomposed expert weights (W̃_i),
enabling dense-like inference performance by caching frequently used
expert compositions.
"""

from typing import Dict, Optional, Tuple, Any
import torch
from torch import Tensor
from collections import OrderedDict
import threading
import time


class LRUCache:
    """Thread-safe LRU cache implementation."""
    
    def __init__(self, capacity: int):
        """Initialize LRU cache.
        
        Args:
            capacity: Maximum number of items to cache
        """
        self.capacity = capacity
        self.cache = OrderedDict()
        self.lock = threading.RLock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get(self, key: Any) -> Optional[Any]:
        """Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                self.hits += 1
                return value
            else:
                self.misses += 1
                return None
    
    def put(self, key: Any, value: Any) -> None:
        """Put item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self.lock:
            if key in self.cache:
                # Update existing key
                self.cache.pop(key)
            elif len(self.cache) >= self.capacity:
                # Evict least recently used
                self.cache.popitem(last=False)
                self.evictions += 1
            
            self.cache[key] = value
    
    def clear(self) -> None:
        """Clear cache."""
        with self.lock:
            self.cache.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        with self.lock:
            return len(self.cache)
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
            
            return {
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "total_requests": total_requests,
                "hit_rate": hit_rate,
                "size": len(self.cache),
                "capacity": self.capacity,
            }
    
    def reset_stats(self) -> None:
        """Reset cache statistics."""
        with self.lock:
            self.hits = 0
            self.misses = 0
            self.evictions = 0


class PrecompositionCache:
    """Cache for precomposed expert weights with memory management."""
    
    def __init__(
        self,
        capacity_per_layer: int = 16,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
        max_memory_gb: Optional[float] = None,
        track_stats: bool = True,
    ):
        """Initialize precomposition cache.
        
        Args:
            capacity_per_layer: Cache capacity per layer
            dtype: Data type for cached tensors
            device: Device to store cached tensors
            max_memory_gb: Maximum memory usage in GB (None for no limit)
            track_stats: Whether to track detailed statistics
        """
        self.capacity_per_layer = capacity_per_layer
        self.dtype = dtype
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_memory_gb = max_memory_gb
        self.track_stats = track_stats
        
        # Layer-wise caches: layer_idx -> LRU cache
        self.layer_caches: Dict[int, LRUCache] = {}
        
        # Memory tracking
        self.memory_usage = 0  # bytes
        self.tensor_sizes: Dict[Tuple[int, int], int] = {}  # (layer, expert) -> size in bytes
        
        # Global statistics
        self.total_compositions = 0
        self.total_cache_time = 0.0
        
        # Thread safety
        self.lock = threading.RLock()
    
    def _get_layer_cache(self, layer_idx: int) -> LRUCache:
        """Get or create cache for a layer."""
        if layer_idx not in self.layer_caches:
            self.layer_caches[layer_idx] = LRUCache(self.capacity_per_layer)
        return self.layer_caches[layer_idx]
    
    def _estimate_tensor_size(self, tensor: Tensor) -> int:
        """Estimate tensor size in bytes."""
        return tensor.numel() * tensor.element_size()
    
    def _check_memory_limit(self, additional_bytes: int) -> bool:
        """Check if adding tensor would exceed memory limit."""
        if self.max_memory_gb is None:
            return True
        
        max_bytes = self.max_memory_gb * 1024**3
        return (self.memory_usage + additional_bytes) <= max_bytes
    
    def get(self, layer_idx: int, expert_idx: int) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """Get cached precomposed weights for an expert.
        
        Args:
            layer_idx: Layer index
            expert_idx: Expert index
            
        Returns:
            Tuple of (up_weight, gate_weight) or (None, None) if not cached
        """
        cache = self._get_layer_cache(layer_idx)
        key = expert_idx
        
        cached_data = cache.get(key)
        if cached_data is not None:
            up_weight, gate_weight = cached_data
            return up_weight, gate_weight
        
        return None, None
    
    def put(
        self, 
        layer_idx: int, 
        expert_idx: int, 
        up_weight: Tensor, 
        gate_weight: Tensor
    ) -> bool:
        """Cache precomposed weights for an expert.
        
        Args:
            layer_idx: Layer index
            expert_idx: Expert index
            up_weight: Precomposed up projection weight
            gate_weight: Precomposed gate projection weight
            
        Returns:
            True if successfully cached, False if memory limit exceeded
        """
        start_time = time.time() if self.track_stats else None
        
        with self.lock:
            # Convert to cache dtype and device
            up_cached = up_weight.to(dtype=self.dtype, device=self.device).detach()
            gate_cached = gate_weight.to(dtype=self.dtype, device=self.device).detach()
            
            # Estimate memory usage
            up_size = self._estimate_tensor_size(up_cached)
            gate_size = self._estimate_tensor_size(gate_cached)
            total_size = up_size + gate_size
            
            # Check memory limit
            if not self._check_memory_limit(total_size):
                return False
            
            # Cache the tensors
            cache = self._get_layer_cache(layer_idx)
            key = expert_idx
            
            # Remove old entry if exists
            old_data = cache.get(key)
            if old_data is not None:
                old_key = (layer_idx, expert_idx)
                if old_key in self.tensor_sizes:
                    self.memory_usage -= self.tensor_sizes[old_key]
                    del self.tensor_sizes[old_key]
            
            # Add new entry
            cache.put(key, (up_cached, gate_cached))
            self.tensor_sizes[(layer_idx, expert_idx)] = total_size
            self.memory_usage += total_size
            
            # Update statistics
            if self.track_stats:
                self.total_compositions += 1
                if start_time is not None:
                    self.total_cache_time += time.time() - start_time
            
            return True
    
    def evict_layer(self, layer_idx: int) -> None:
        """Evict all cached data for a layer.
        
        Args:
            layer_idx: Layer index to evict
        """
        with self.lock:
            if layer_idx in self.layer_caches:
                cache = self.layer_caches[layer_idx]
                
                # Update memory usage
                keys_to_remove = []
                for (l_idx, e_idx), size in self.tensor_sizes.items():
                    if l_idx == layer_idx:
                        self.memory_usage -= size
                        keys_to_remove.append((l_idx, e_idx))
                
                for key in keys_to_remove:
                    del self.tensor_sizes[key]
                
                # Clear cache
                cache.clear()
    
    def clear(self) -> None:
        """Clear all cached data."""
        with self.lock:
            for cache in self.layer_caches.values():
                cache.clear()
            
            self.tensor_sizes.clear()
            self.memory_usage = 0
    
    def warmup(
        self, 
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        num_tokens: int = 2048,
    ) -> Dict[str, Any]:
        """Warm up cache with popular experts.
        
        Args:
            model: Model to run warmup on
            dataloader: Data loader for warmup tokens
            num_tokens: Number of tokens to use for warmup
            
        Returns:
            Warmup statistics
        """
        model.eval()
        tokens_processed = 0
        expert_usage = {}  # (layer, expert) -> count
        
        with torch.no_grad():
            for batch in dataloader:
                if tokens_processed >= num_tokens:
                    break
                
                # Forward pass to trigger expert usage
                # This would need to be integrated with the actual model forward pass
                # to track which experts are used
                
                # Placeholder for actual warmup logic
                batch_tokens = batch.get("input_ids", batch).numel()
                tokens_processed += batch_tokens
        
        # Sort experts by usage and cache most popular ones
        popular_experts = sorted(
            expert_usage.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        cached_experts = 0
        for (layer_idx, expert_idx), usage_count in popular_experts:
            # This would trigger actual caching during model forward pass
            cached_experts += 1
        
        return {
            "tokens_processed": tokens_processed,
            "unique_experts_seen": len(expert_usage),
            "cached_experts": cached_experts,
            "cache_utilization": self.get_memory_usage()["utilization_pct"],
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self.lock:
            layer_stats = {}
            total_hits = 0
            total_misses = 0
            total_evictions = 0
            
            for layer_idx, cache in self.layer_caches.items():
                stats = cache.get_stats()
                layer_stats[f"layer_{layer_idx}"] = stats
                total_hits += stats["hits"]
                total_misses += stats["misses"]
                total_evictions += stats["evictions"]
            
            total_requests = total_hits + total_misses
            global_hit_rate = total_hits / total_requests if total_requests > 0 else 0.0
            
            memory_stats = self.get_memory_usage()
            
            return {
                "global_stats": {
                    "total_hits": total_hits,
                    "total_misses": total_misses,
                    "total_evictions": total_evictions,
                    "total_requests": total_requests,
                    "global_hit_rate": global_hit_rate,
                    "total_compositions": self.total_compositions,
                    "avg_cache_time_ms": (
                        self.total_cache_time * 1000 / self.total_compositions
                        if self.total_compositions > 0 else 0.0
                    ),
                },
                "layer_stats": layer_stats,
                "memory_stats": memory_stats,
            }
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics.
        
        Returns:
            Dictionary with memory statistics
        """
        with self.lock:
            max_memory = (
                self.max_memory_gb * 1024**3 
                if self.max_memory_gb is not None 
                else float('inf')
            )
            
            return {
                "memory_usage_bytes": self.memory_usage,
                "memory_usage_mb": self.memory_usage / (1024**2),
                "memory_usage_gb": self.memory_usage / (1024**3),
                "max_memory_gb": self.max_memory_gb,
                "utilization_pct": (
                    self.memory_usage / max_memory * 100
                    if max_memory != float('inf') else 0.0
                ),
                "num_cached_tensors": len(self.tensor_sizes),
                "num_layers_cached": len(self.layer_caches),
            }
    
    def reset_stats(self) -> None:
        """Reset all cache statistics."""
        with self.lock:
            for cache in self.layer_caches.values():
                cache.reset_stats()
            
            self.total_compositions = 0
            self.total_cache_time = 0.0
