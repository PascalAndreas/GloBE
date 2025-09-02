"""Latency and memory benchmarking utilities for GloBE.

This module provides comprehensive benchmarking tools for measuring
inference latency, memory usage, and cache performance of GloBE models.
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
import torch
from torch import Tensor
import time
import psutil
import gc
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..modules.ffn_globe import GloBEFFN
from ..modules.precompose_cache import PrecompositionCache


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking runs."""
    batch_sizes: List[int]
    sequence_lengths: List[int]
    num_warmup_runs: int
    num_benchmark_runs: int
    measure_memory: bool
    measure_cache_stats: bool
    output_dir: Optional[Path]


class LatencyBenchmark:
    """Latency benchmarking for GloBE models."""
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        precision: str = "bf16",
    ):
        """Initialize latency benchmark.

        Args:
            device: Device for benchmarking
            precision: Precision for benchmarking ("fp16", "bf16", "fp32", "auto")
        """
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

        self.device = torch.device(device)
        self.precision = precision

        # Set up precision
        if precision == "fp16":
            self.dtype = torch.float16
        elif precision == "bf16":
            self.dtype = torch.bfloat16
        elif precision == "fp32":
            self.dtype = torch.float32
        else:  # auto detection
            self.dtype = None
    
    def benchmark_model(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: BenchmarkConfig,
    ) -> Dict[str, Any]:
        """Comprehensive model benchmarking.
        
        Args:
            model: Model to benchmark
            tokenizer: Tokenizer for input generation
            config: Benchmark configuration
            
        Returns:
            Comprehensive benchmark results
        """
        results = {
            "config": {
                "device": str(self.device),
                "precision": self.precision,
                "batch_sizes": config.batch_sizes,
                "sequence_lengths": config.sequence_lengths,
                "num_warmup_runs": config.num_warmup_runs,
                "num_benchmark_runs": config.num_benchmark_runs,
            },
            "latency_results": {},
            "memory_results": {},
            "cache_results": {},
            "throughput_results": {},
        }

        # Infer precision from model if requested
        if self.dtype is None:
            self.dtype = next(model.parameters()).dtype
            self.precision = str(self.dtype).replace("torch.", "")

        # Move model to device and set precision
        model = model.to(self.device, dtype=self.dtype)
        model.eval()
        
        # Benchmark different configurations
        for batch_size in config.batch_sizes:
            for seq_len in config.sequence_lengths:
                config_name = f"batch_{batch_size}_seq_{seq_len}"
                print(f"Benchmarking {config_name}...")
                
                # Generate input
                input_data = self._generate_input(tokenizer, batch_size, seq_len)
                
                # Run benchmark
                config_results = self._benchmark_configuration(
                    model, input_data, config, config_name
                )
                
                results["latency_results"][config_name] = config_results["latency"]
                results["throughput_results"][config_name] = config_results["throughput"]
                
                if config.measure_memory:
                    results["memory_results"][config_name] = config_results["memory"]
                
                if config.measure_cache_stats:
                    results["cache_results"][config_name] = config_results["cache"]
        
        # Save results if output directory specified
        if config.output_dir:
            self._save_results(results, config.output_dir)

        return results
    
    def _generate_input(
        self, 
        tokenizer: PreTrainedTokenizer, 
        batch_size: int, 
        seq_len: int
    ) -> Dict[str, Tensor]:
        """Generate input for benchmarking.
        
        Args:
            tokenizer: Tokenizer
            batch_size: Batch size
            seq_len: Sequence length
            
        Returns:
            Input tensors
        """
        # Generate random text inputs
        sample_texts = [
            "This is a sample text for benchmarking the model performance. " * (seq_len // 10)
            for _ in range(batch_size)
        ]
        
        # Tokenize
        inputs = tokenizer(
            sample_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=seq_len,
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
    def _benchmark_configuration(
        self,
        model: PreTrainedModel,
        input_data: Dict[str, Tensor],
        config: BenchmarkConfig,
        config_name: str,
    ) -> Dict[str, Any]:
        """Benchmark a specific configuration.
        
        Args:
            model: Model to benchmark
            input_data: Input data
            config: Benchmark configuration
            config_name: Configuration name
            
        Returns:
            Configuration benchmark results
        """
        batch_size = input_data["input_ids"].shape[0]
        seq_len = input_data["input_ids"].shape[1]
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(config.num_warmup_runs):
                _ = model(**input_data)
        
        # Clear cache and collect garbage
        self._clear_cache()
        gc.collect()
        
        # Benchmark runs
        latencies = []
        memory_stats = []
        
        with torch.no_grad():
            for run_idx in range(config.num_benchmark_runs):
                # Memory before
                if config.measure_memory:
                    memory_before = self._get_memory_stats()
                
                # Time the forward pass
                start_time = self._get_precise_time()
                outputs = model(**input_data)
                end_time = self._get_precise_time()
                
                latency = end_time - start_time
                latencies.append(latency)
                
                # Memory after
                if config.measure_memory:
                    memory_after = self._get_memory_stats()
                    memory_stats.append({
                        "before": memory_before,
                        "after": memory_after,
                        "peak": self._get_peak_memory(),
                    })
        
        # Calculate statistics
        latencies = np.array(latencies)
        
        latency_stats = {
            "mean_ms": np.mean(latencies) * 1000,
            "std_ms": np.std(latencies) * 1000,
            "min_ms": np.min(latencies) * 1000,
            "max_ms": np.max(latencies) * 1000,
            "median_ms": np.median(latencies) * 1000,
            "p95_ms": np.percentile(latencies, 95) * 1000,
            "p99_ms": np.percentile(latencies, 99) * 1000,
        }
        
        # Throughput calculations
        mean_latency_s = np.mean(latencies)
        throughput_stats = {
            "tokens_per_second": (batch_size * seq_len) / mean_latency_s,
            "sequences_per_second": batch_size / mean_latency_s,
            "batches_per_second": 1.0 / mean_latency_s,
        }
        
        results = {
            "latency": latency_stats,
            "throughput": throughput_stats,
        }
        
        # Memory statistics
        if config.measure_memory and memory_stats:
            memory_results = self._analyze_memory_stats(memory_stats)
            results["memory"] = memory_results
        
        # Cache statistics
        if config.measure_cache_stats:
            cache_results = self._get_cache_stats(model)
            results["cache"] = cache_results
        
        return results

    def _get_precise_time(self) -> float:
        """Get precise time measurement."""
        if self._is_cuda():
            torch.cuda.synchronize()
        elif self._is_mps():
            torch.mps.synchronize()
        return time.perf_counter()
    
    def _get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        stats = {}

        # GPU memory
        if self._is_cuda():
            stats["gpu_allocated_mb"] = torch.cuda.memory_allocated() / 1024**2
            stats["gpu_reserved_mb"] = torch.cuda.memory_reserved() / 1024**2
            stats["gpu_max_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024**2
        elif self._is_mps():
            stats["gpu_allocated_mb"] = torch.mps.current_allocated_memory() / 1024**2
            stats["gpu_reserved_mb"] = torch.mps.driver_allocated_memory() / 1024**2

        # CPU memory
        process = psutil.Process()
        memory_info = process.memory_info()
        stats["cpu_rss_mb"] = memory_info.rss / 1024**2
        stats["cpu_vms_mb"] = memory_info.vms / 1024**2

        return stats
    
    def _get_peak_memory(self) -> Dict[str, float]:
        """Get peak memory usage."""
        stats = {}

        if self._is_cuda():
            stats["gpu_peak_mb"] = torch.cuda.max_memory_allocated() / 1024**2
        elif self._is_mps():
            stats["gpu_peak_mb"] = torch.mps.current_allocated_memory() / 1024**2

        return stats

    def _clear_cache(self) -> None:
        """Clear device cache if applicable."""
        if self._is_cuda():
            torch.cuda.empty_cache()
        elif self._is_mps():
            torch.mps.empty_cache()

    def _is_cuda(self) -> bool:
        return self.device.type == "cuda"

    def _is_mps(self) -> bool:
        return self.device.type == "mps"
    
    def _analyze_memory_stats(self, memory_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze memory statistics across runs."""
        if not memory_stats:
            return {}
        
        # Extract memory differences
        gpu_diffs = []
        cpu_diffs = []
        peak_usage = []
        
        for stats in memory_stats:
            before = stats["before"]
            after = stats["after"]
            
            if "gpu_allocated_mb" in before and "gpu_allocated_mb" in after:
                gpu_diff = after["gpu_allocated_mb"] - before["gpu_allocated_mb"]
                gpu_diffs.append(gpu_diff)
            
            if "cpu_rss_mb" in before and "cpu_rss_mb" in after:
                cpu_diff = after["cpu_rss_mb"] - before["cpu_rss_mb"]
                cpu_diffs.append(cpu_diff)
            
            if "peak" in stats and "gpu_peak_mb" in stats["peak"]:
                peak_usage.append(stats["peak"]["gpu_peak_mb"])
        
        results = {}
        
        if gpu_diffs:
            results["gpu_memory_diff"] = {
                "mean_mb": np.mean(gpu_diffs),
                "std_mb": np.std(gpu_diffs),
                "max_mb": np.max(gpu_diffs),
            }
        
        if cpu_diffs:
            results["cpu_memory_diff"] = {
                "mean_mb": np.mean(cpu_diffs),
                "std_mb": np.std(cpu_diffs),
                "max_mb": np.max(cpu_diffs),
            }
        
        if peak_usage:
            results["peak_gpu_usage"] = {
                "mean_mb": np.mean(peak_usage),
                "max_mb": np.max(peak_usage),
            }
        
        return results
    
    def _get_cache_stats(self, model: PreTrainedModel) -> Dict[str, Any]:
        """Get cache statistics from GloBE layers."""
        cache_stats = {}
        
        # Find GloBE layers and collect cache stats
        for name, module in model.named_modules():
            if isinstance(module, GloBEFFN) and module.cache is not None:
                layer_cache_stats = module.cache.get_stats()
                cache_stats[f"layer_{module.layer_idx}"] = layer_cache_stats
        
        # Aggregate cache statistics
        if cache_stats:
            total_hits = sum(
                stats["global_stats"]["total_hits"] 
                for stats in cache_stats.values()
            )
            total_requests = sum(
                stats["global_stats"]["total_requests"]
                for stats in cache_stats.values()
            )
            
            global_hit_rate = total_hits / total_requests if total_requests > 0 else 0.0
            
            cache_stats["global"] = {
                "total_hits": total_hits,
                "total_requests": total_requests,
                "global_hit_rate": global_hit_rate,
                "num_cached_layers": len(cache_stats),
            }
        
        return cache_stats
    
    def _save_results(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Save benchmark results to files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw results as JSON
        results_path = output_dir / "benchmark_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate summary report
        self._generate_summary_report(results, output_dir)
        
        print(f"Benchmark results saved to {output_dir}")
    
    def _generate_summary_report(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Generate human-readable summary report."""
        report_path = output_dir / "benchmark_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write("GloBE Model Benchmark Summary\n")
            f.write("=" * 50 + "\n\n")
            
            # Configuration
            config = results["config"]
            f.write("Configuration:\n")
            f.write(f"  Device: {config['device']}\n")
            f.write(f"  Precision: {config['precision']}\n")
            f.write(f"  Batch sizes: {config['batch_sizes']}\n")
            f.write(f"  Sequence lengths: {config['sequence_lengths']}\n")
            f.write(f"  Benchmark runs: {config['num_benchmark_runs']}\n\n")
            
            # Latency results
            f.write("Latency Results:\n")
            f.write("-" * 20 + "\n")
            for config_name, latency_stats in results["latency_results"].items():
                f.write(f"\n{config_name}:\n")
                f.write(f"  Mean: {latency_stats['mean_ms']:.2f} ms\n")
                f.write(f"  Std: {latency_stats['std_ms']:.2f} ms\n")
                f.write(f"  P95: {latency_stats['p95_ms']:.2f} ms\n")
                f.write(f"  P99: {latency_stats['p99_ms']:.2f} ms\n")
            
            # Throughput results
            f.write("\n\nThroughput Results:\n")
            f.write("-" * 20 + "\n")
            for config_name, throughput_stats in results["throughput_results"].items():
                f.write(f"\n{config_name}:\n")
                f.write(f"  Tokens/sec: {throughput_stats['tokens_per_second']:.2f}\n")
                f.write(f"  Sequences/sec: {throughput_stats['sequences_per_second']:.2f}\n")
            
            # Memory results
            if "memory_results" in results and results["memory_results"]:
                f.write("\n\nMemory Results:\n")
                f.write("-" * 20 + "\n")
                for config_name, memory_stats in results["memory_results"].items():
                    f.write(f"\n{config_name}:\n")
                    if "gpu_memory_diff" in memory_stats:
                        gpu_diff = memory_stats["gpu_memory_diff"]
                        f.write(f"  GPU memory diff: {gpu_diff['mean_mb']:.2f} ± {gpu_diff['std_mb']:.2f} MB\n")
                    if "peak_gpu_usage" in memory_stats:
                        peak = memory_stats["peak_gpu_usage"]
                        f.write(f"  Peak GPU usage: {peak['max_mb']:.2f} MB\n")
            
            # Cache results
            if "cache_results" in results and results["cache_results"]:
                f.write("\n\nCache Results:\n")
                f.write("-" * 20 + "\n")
                for config_name, cache_stats in results["cache_results"].items():
                    if "global" in cache_stats:
                        global_stats = cache_stats["global"]
                        f.write(f"\n{config_name}:\n")
                        f.write(f"  Hit rate: {global_stats['global_hit_rate']:.3f}\n")
                        f.write(f"  Total requests: {global_stats['total_requests']}\n")
                        f.write(f"  Cached layers: {global_stats['num_cached_layers']}\n")


def benchmark_globe_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    output_dir: Path,
    batch_sizes: List[int] = [1, 4, 8],
    sequence_lengths: List[int] = [128, 512, 1024],
    num_warmup_runs: int = 5,
    num_benchmark_runs: int = 20,
    device: Optional[torch.device] = None,
    precision: str = "bf16",
    measure_memory: bool = True,
    measure_cache_stats: bool = True,
) -> Dict[str, Any]:
    """Convenience function to benchmark GloBE model.
    
    Args:
        model: Model to benchmark
        tokenizer: Tokenizer
        output_dir: Output directory for results
        batch_sizes: Batch sizes to test
        sequence_lengths: Sequence lengths to test
        num_warmup_runs: Number of warmup runs
        num_benchmark_runs: Number of benchmark runs
        device: Device for benchmarking
        precision: Precision for benchmarking
        measure_memory: Whether to measure memory usage
        measure_cache_stats: Whether to measure cache statistics
        
    Returns:
        Benchmark results
    """
    config = BenchmarkConfig(
        batch_sizes=batch_sizes,
        sequence_lengths=sequence_lengths,
        num_warmup_runs=num_warmup_runs,
        num_benchmark_runs=num_benchmark_runs,
        measure_memory=measure_memory,
        measure_cache_stats=measure_cache_stats,
        output_dir=output_dir,
    )

    benchmark = LatencyBenchmark(device=device, precision=precision)
    return benchmark.benchmark_model(model, tokenizer, config)
