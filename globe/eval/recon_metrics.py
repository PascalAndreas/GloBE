"""Reconstruction metrics for evaluating GloBE quality.

This module provides comprehensive metrics for evaluating the quality
of GloBE reconstructions compared to original MoE expert weights.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import torch
from torch import Tensor
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class ReconstructionEvaluator:
    """Comprehensive evaluator for GloBE reconstruction quality."""

    def __init__(self, device: Optional[Union[str, torch.device]] = None):
        """Initialize reconstruction evaluator.

        Args:
            device: Device for computations
        """
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

        self.device = torch.device(device)
    
    def evaluate_reconstruction(
        self,
        original_weights: Dict[str, List[Tensor]],
        reconstructed_weights: Dict[str, List[Tensor]],
        mixture_coefficients: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Any]:
        """Comprehensive evaluation of reconstruction quality.
        
        Args:
            original_weights: Original expert weights by family
            reconstructed_weights: Reconstructed expert weights by family
            mixture_coefficients: Optional mixture coefficients
            
        Returns:
            Dictionary with comprehensive metrics
        """
        results = {
            "global_metrics": {},
            "family_metrics": {},
            "expert_metrics": {},
            "spectral_analysis": {},
            "sparsity_analysis": {},
        }
        
        # Global metrics across all families
        all_original = []
        all_reconstructed = []
        
        for family in original_weights.keys():
            if family in reconstructed_weights:
                orig_family = torch.cat([w.flatten() for w in original_weights[family]])
                recon_family = torch.cat([w.flatten() for w in reconstructed_weights[family]])
                all_original.append(orig_family)
                all_reconstructed.append(recon_family)
        
        if all_original:
            all_original = torch.cat(all_original)
            all_reconstructed = torch.cat(all_reconstructed)
            results["global_metrics"] = self._compute_basic_metrics(
                all_original, all_reconstructed
            )
        
        # Per-family metrics
        for family in original_weights.keys():
            if family in reconstructed_weights:
                family_metrics = self._evaluate_family(
                    original_weights[family], 
                    reconstructed_weights[family]
                )
                results["family_metrics"][family] = family_metrics
        
        # Spectral analysis
        results["spectral_analysis"] = self._analyze_spectral_properties(
            original_weights, reconstructed_weights
        )
        
        # Sparsity analysis
        if mixture_coefficients is not None:
            results["sparsity_analysis"] = self._analyze_sparsity(mixture_coefficients)
        
        return results
    
    def _compute_basic_metrics(self, original: Tensor, reconstructed: Tensor) -> Dict[str, float]:
        """Compute basic reconstruction metrics.
        
        Args:
            original: Original tensor
            reconstructed: Reconstructed tensor
            
        Returns:
            Dictionary with basic metrics
        """
        # Move to CPU for numpy operations
        orig_np = original.detach().cpu().numpy()
        recon_np = reconstructed.detach().cpu().numpy()
        
        # Basic error metrics
        mse = mean_squared_error(orig_np, recon_np)
        mae = mean_absolute_error(orig_np, recon_np)
        rmse = np.sqrt(mse)
        
        # Relative errors
        orig_norm = np.linalg.norm(orig_np)
        error_norm = np.linalg.norm(orig_np - recon_np)
        relative_error = error_norm / (orig_norm + 1e-8)
        
        # Correlation metrics
        correlation, _ = pearsonr(orig_np.flatten(), recon_np.flatten())
        rank_correlation, _ = spearmanr(orig_np.flatten(), recon_np.flatten())
        
        # R-squared
        ss_res = np.sum((orig_np - recon_np) ** 2)
        ss_tot = np.sum((orig_np - np.mean(orig_np)) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-8))
        
        # Signal-to-noise ratio
        signal_power = np.mean(orig_np ** 2)
        noise_power = np.mean((orig_np - recon_np) ** 2)
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-8))
        
        return {
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse),
            "relative_error": float(relative_error),
            "correlation": float(correlation),
            "rank_correlation": float(rank_correlation),
            "r_squared": float(r_squared),
            "snr_db": float(snr_db),
            "orig_norm": float(orig_norm),
            "recon_norm": float(np.linalg.norm(recon_np)),
        }
    
    def _evaluate_family(
        self, 
        original_family: List[Tensor], 
        reconstructed_family: List[Tensor]
    ) -> Dict[str, Any]:
        """Evaluate reconstruction for a specific weight family.
        
        Args:
            original_family: List of original expert weights
            reconstructed_family: List of reconstructed expert weights
            
        Returns:
            Family-specific metrics
        """
        if len(original_family) != len(reconstructed_family):
            return {"error": "Mismatched number of experts"}
        
        # Per-expert metrics
        expert_metrics = []
        for i, (orig, recon) in enumerate(zip(original_family, reconstructed_family)):
            if orig.shape != recon.shape:
                expert_metrics.append({"error": f"Shape mismatch: {orig.shape} vs {recon.shape}"})
                continue
            
            expert_metric = self._compute_basic_metrics(orig.flatten(), recon.flatten())
            expert_metric["expert_idx"] = i
            expert_metrics.append(expert_metric)
        
        # Aggregate family metrics
        valid_metrics = [m for m in expert_metrics if "error" not in m]
        if not valid_metrics:
            return {"error": "No valid expert metrics"}
        
        family_aggregate = {}
        for key in valid_metrics[0].keys():
            if key != "expert_idx":
                values = [m[key] for m in valid_metrics]
                family_aggregate[f"mean_{key}"] = np.mean(values)
                family_aggregate[f"std_{key}"] = np.std(values)
                family_aggregate[f"min_{key}"] = np.min(values)
                family_aggregate[f"max_{key}"] = np.max(values)
        
        return {
            "aggregate": family_aggregate,
            "per_expert": expert_metrics,
            "num_experts": len(valid_metrics),
        }
    
    def _analyze_spectral_properties(
        self,
        original_weights: Dict[str, List[Tensor]],
        reconstructed_weights: Dict[str, List[Tensor]],
    ) -> Dict[str, Any]:
        """Analyze spectral properties of weights.
        
        Args:
            original_weights: Original expert weights
            reconstructed_weights: Reconstructed expert weights
            
        Returns:
            Spectral analysis results
        """
        spectral_results = {}
        
        for family in original_weights.keys():
            if family not in reconstructed_weights:
                continue
            
            family_spectral = {
                "original": [],
                "reconstructed": [],
                "spectral_errors": [],
            }
            
            for orig, recon in zip(original_weights[family], reconstructed_weights[family]):
                if orig.shape != recon.shape:
                    continue
                
                # Compute SVD
                try:
                    orig_u, orig_s, orig_v = torch.svd(orig)
                    recon_u, recon_s, recon_v = torch.svd(recon)
                    
                    # Effective rank (ratio of sum to max singular value)
                    orig_eff_rank = (orig_s.sum() / (orig_s.max() + 1e-8)).item()
                    recon_eff_rank = (recon_s.sum() / (recon_s.max() + 1e-8)).item()
                    
                    # Spectral norm (largest singular value)
                    orig_spectral_norm = orig_s.max().item()
                    recon_spectral_norm = recon_s.max().item()
                    
                    # Condition number
                    orig_cond = (orig_s.max() / (orig_s.min() + 1e-8)).item()
                    recon_cond = (recon_s.max() / (recon_s.min() + 1e-8)).item()
                    
                    family_spectral["original"].append({
                        "effective_rank": orig_eff_rank,
                        "spectral_norm": orig_spectral_norm,
                        "condition_number": orig_cond,
                        "singular_values": orig_s.cpu().numpy(),
                    })
                    
                    family_spectral["reconstructed"].append({
                        "effective_rank": recon_eff_rank,
                        "spectral_norm": recon_spectral_norm,
                        "condition_number": recon_cond,
                        "singular_values": recon_s.cpu().numpy(),
                    })
                    
                    # Spectral error
                    min_len = min(len(orig_s), len(recon_s))
                    spectral_error = torch.norm(orig_s[:min_len] - recon_s[:min_len]).item()
                    family_spectral["spectral_errors"].append(spectral_error)
                    
                except Exception as e:
                    print(f"SVD failed for {family}: {e}")
                    continue
            
            # Aggregate spectral metrics
            if family_spectral["original"]:
                spectral_results[family] = self._aggregate_spectral_metrics(family_spectral)
        
        return spectral_results
    
    def _aggregate_spectral_metrics(self, family_spectral: Dict[str, List]) -> Dict[str, Any]:
        """Aggregate spectral metrics for a family."""
        aggregated = {}
        
        for data_type in ["original", "reconstructed"]:
            if not family_spectral[data_type]:
                continue
            
            metrics = family_spectral[data_type]
            
            # Aggregate scalar metrics
            for key in ["effective_rank", "spectral_norm", "condition_number"]:
                values = [m[key] for m in metrics]
                aggregated[f"{data_type}_{key}_mean"] = np.mean(values)
                aggregated[f"{data_type}_{key}_std"] = np.std(values)
        
        # Spectral error statistics
        if family_spectral["spectral_errors"]:
            errors = family_spectral["spectral_errors"]
            aggregated["spectral_error_mean"] = np.mean(errors)
            aggregated["spectral_error_std"] = np.std(errors)
            aggregated["spectral_error_max"] = np.max(errors)
        
        return aggregated
    
    def _analyze_sparsity(self, mixture_coefficients: Dict[str, Tensor]) -> Dict[str, Any]:
        """Analyze sparsity properties of mixture coefficients.
        
        Args:
            mixture_coefficients: Mixture coefficients by family
            
        Returns:
            Sparsity analysis results
        """
        sparsity_results = {}
        
        for family, coeffs in mixture_coefficients.items():
            # coeffs: num_experts × num_bases
            
            # Support sizes (number of non-zero elements per expert)
            support_sizes = (coeffs.abs() > 1e-6).sum(dim=1).float()
            
            # Sparsity (fraction of zero elements)
            sparsity = (coeffs.abs() <= 1e-6).float().mean()
            
            # Entropy (measure of concentration)
            # Add small epsilon to avoid log(0)
            coeffs_prob = coeffs.abs() + 1e-8
            coeffs_prob = coeffs_prob / coeffs_prob.sum(dim=1, keepdim=True)
            entropy = -torch.sum(coeffs_prob * torch.log(coeffs_prob), dim=1)
            
            # Gini coefficient (measure of inequality)
            gini_coeffs = []
            for i in range(coeffs.shape[0]):
                expert_coeffs = coeffs[i].abs().sort(descending=True)[0]
                n = len(expert_coeffs)
                index = torch.arange(1, n + 1, dtype=torch.float)
                gini = (2 * torch.sum(index * expert_coeffs) / torch.sum(expert_coeffs) - (n + 1)) / n
                gini_coeffs.append(gini.item())
            
            sparsity_results[family] = {
                "mean_support_size": support_sizes.mean().item(),
                "std_support_size": support_sizes.std().item(),
                "min_support_size": support_sizes.min().item(),
                "max_support_size": support_sizes.max().item(),
                "sparsity": sparsity.item(),
                "mean_entropy": entropy.mean().item(),
                "std_entropy": entropy.std().item(),
                "mean_gini": np.mean(gini_coeffs),
                "std_gini": np.std(gini_coeffs),
                "l1_norm_mean": coeffs.abs().sum(dim=1).mean().item(),
                "l2_norm_mean": (coeffs ** 2).sum(dim=1).sqrt().mean().item(),
                "max_coeff_mean": coeffs.abs().max(dim=1)[0].mean().item(),
            }
        
        return sparsity_results
    
    def generate_report(
        self,
        evaluation_results: Dict[str, Any],
        output_dir: Path,
        create_plots: bool = True,
    ) -> None:
        """Generate comprehensive evaluation report.
        
        Args:
            evaluation_results: Results from evaluate_reconstruction
            output_dir: Output directory for report
            create_plots: Whether to create visualization plots
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate text report
        self._generate_text_report(evaluation_results, output_dir)
        
        # Generate plots if requested
        if create_plots:
            self._generate_plots(evaluation_results, output_dir)
        
        print(f"Evaluation report saved to {output_dir}")
    
    def _generate_text_report(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Generate text-based evaluation report."""
        report_path = output_dir / "reconstruction_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("GloBE Reconstruction Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Global metrics
            if "global_metrics" in results:
                f.write("Global Metrics:\n")
                f.write("-" * 20 + "\n")
                for key, value in results["global_metrics"].items():
                    f.write(f"{key}: {value:.6f}\n")
                f.write("\n")
            
            # Family metrics
            if "family_metrics" in results:
                f.write("Per-Family Metrics:\n")
                f.write("-" * 20 + "\n")
                for family, metrics in results["family_metrics"].items():
                    f.write(f"\n{family.upper()} Family:\n")
                    if "aggregate" in metrics:
                        for key, value in metrics["aggregate"].items():
                            f.write(f"  {key}: {value:.6f}\n")
                f.write("\n")
            
            # Spectral analysis
            if "spectral_analysis" in results:
                f.write("Spectral Analysis:\n")
                f.write("-" * 20 + "\n")
                for family, metrics in results["spectral_analysis"].items():
                    f.write(f"\n{family.upper()} Family:\n")
                    for key, value in metrics.items():
                        f.write(f"  {key}: {value:.6f}\n")
                f.write("\n")
            
            # Sparsity analysis
            if "sparsity_analysis" in results:
                f.write("Sparsity Analysis:\n")
                f.write("-" * 20 + "\n")
                for family, metrics in results["sparsity_analysis"].items():
                    f.write(f"\n{family.upper()} Family:\n")
                    for key, value in metrics.items():
                        f.write(f"  {key}: {value:.6f}\n")
    
    def _generate_plots(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Generate visualization plots for evaluation results."""
        try:
            # Set up plotting style
            plt.style.use('seaborn-v0_8')
            
            # Plot global metrics
            if "global_metrics" in results:
                self._plot_global_metrics(results["global_metrics"], output_dir)
            
            # Plot family comparisons
            if "family_metrics" in results:
                self._plot_family_metrics(results["family_metrics"], output_dir)
            
            # Plot sparsity analysis
            if "sparsity_analysis" in results:
                self._plot_sparsity_analysis(results["sparsity_analysis"], output_dir)
            
        except Exception as e:
            print(f"Failed to generate plots: {e}")
    
    def _plot_global_metrics(self, global_metrics: Dict[str, float], output_dir: Path) -> None:
        """Plot global reconstruction metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Global Reconstruction Metrics', fontsize=16)
        
        # Error metrics
        error_metrics = ['mse', 'mae', 'rmse', 'relative_error']
        error_values = [global_metrics.get(m, 0) for m in error_metrics]
        axes[0, 0].bar(error_metrics, error_values)
        axes[0, 0].set_title('Error Metrics')
        axes[0, 0].set_yscale('log')
        
        # Correlation metrics
        corr_metrics = ['correlation', 'rank_correlation', 'r_squared']
        corr_values = [global_metrics.get(m, 0) for m in corr_metrics]
        axes[0, 1].bar(corr_metrics, corr_values)
        axes[0, 1].set_title('Correlation Metrics')
        axes[0, 1].set_ylim([0, 1])
        
        # Norms comparison
        norms = ['orig_norm', 'recon_norm']
        norm_values = [global_metrics.get(m, 0) for m in norms]
        axes[1, 0].bar(norms, norm_values)
        axes[1, 0].set_title('Norm Comparison')
        
        # SNR
        snr = global_metrics.get('snr_db', 0)
        axes[1, 1].bar(['SNR (dB)'], [snr])
        axes[1, 1].set_title('Signal-to-Noise Ratio')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'global_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_family_metrics(self, family_metrics: Dict[str, Any], output_dir: Path) -> None:
        """Plot per-family reconstruction metrics."""
        families = list(family_metrics.keys())
        
        if not families:
            return
        
        # Extract mean MSE and correlation for each family
        mse_values = []
        corr_values = []
        
        for family in families:
            if "aggregate" in family_metrics[family]:
                agg = family_metrics[family]["aggregate"]
                mse_values.append(agg.get("mean_mse", 0))
                corr_values.append(agg.get("mean_correlation", 0))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # MSE comparison
        ax1.bar(families, mse_values)
        ax1.set_title('Mean MSE by Family')
        ax1.set_ylabel('MSE')
        ax1.set_yscale('log')
        
        # Correlation comparison
        ax2.bar(families, corr_values)
        ax2.set_title('Mean Correlation by Family')
        ax2.set_ylabel('Correlation')
        ax2.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(output_dir / 'family_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_sparsity_analysis(self, sparsity_analysis: Dict[str, Any], output_dir: Path) -> None:
        """Plot sparsity analysis results."""
        families = list(sparsity_analysis.keys())
        
        if not families:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Sparsity Analysis', fontsize=16)
        
        # Support sizes
        support_means = [sparsity_analysis[f]["mean_support_size"] for f in families]
        support_stds = [sparsity_analysis[f]["std_support_size"] for f in families]
        axes[0, 0].bar(families, support_means, yerr=support_stds, capsize=5)
        axes[0, 0].set_title('Mean Support Size')
        axes[0, 0].set_ylabel('Support Size')
        
        # Sparsity levels
        sparsity_values = [sparsity_analysis[f]["sparsity"] for f in families]
        axes[0, 1].bar(families, sparsity_values)
        axes[0, 1].set_title('Sparsity Level')
        axes[0, 1].set_ylabel('Sparsity')
        axes[0, 1].set_ylim([0, 1])
        
        # Entropy
        entropy_values = [sparsity_analysis[f]["mean_entropy"] for f in families]
        axes[1, 0].bar(families, entropy_values)
        axes[1, 0].set_title('Mean Entropy')
        axes[1, 0].set_ylabel('Entropy')
        
        # Gini coefficient
        gini_values = [sparsity_analysis[f]["mean_gini"] for f in families]
        axes[1, 1].bar(families, gini_values)
        axes[1, 1].set_title('Mean Gini Coefficient')
        axes[1, 1].set_ylabel('Gini Coefficient')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'sparsity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
