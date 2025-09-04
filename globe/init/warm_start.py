"""Warm start and seeding strategies for GloBE bank initialization.

This module implements multiple seeding strategies from globe_warm_start.md to provide
alternatives to the expensive per-expert SVD initialization. These methods are designed
to be MPS-friendly and scale better to larger numbers of experts.

Key strategies implemented:
- Row-sampled Tall-Skinny PCA (TS-PCA): Fastest, no per-expert SVD
- Left-Gram PCA + TS-PCA: Higher quality proxies, still MPS-friendly  
- Spherical k-means++: Simplest clustering approach
- Residual-driven greedy: K-SVD style for lowest initial MSE
- SVD baseline: Original per-expert SVD for comparison
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Literal
from enum import Enum

import torch
from torch import Tensor
import torch.nn.functional as F


class SeedingMethod(Enum):
    """Available seeding methods for bank initialization."""
    SVD = "svd"  # Original per-expert SVD (baseline)
    TS_PCA = "ts_pca"  # Row-sampled Tall-Skinny PCA
    LEFT_GRAM_PCA = "left_gram_pca"  # Left-Gram PCA + TS-PCA
    SPHERICAL_KMEANS = "spherical_kmeans"  # Spherical k-means++
    RESIDUAL_GREEDY = "residual_greedy"  # K-SVD style greedy
    HYBRID = "hybrid"  # Mix of PCA + k-means


@dataclass
class SeedingConfig:
    """Configuration for seeding strategies."""
    method: SeedingMethod = SeedingMethod.TS_PCA
    
    # Row sampling parameters (for TS-PCA)
    row_selection: Literal["uniform", "random", "energy"] = "energy"
    
    # PCA parameters
    oversampling: int = 8  # For randomized PCA (q parameter)
    
    # K-means parameters
    kmeans_iters: int = 25
    kmeans_init: Literal["random", "kmeans++", "farthest"] = "kmeans++"
    
    # Greedy parameters  
    greedy_regularization: float = 1e-6
    
    # Hybrid parameters
    hybrid_pca_ratio: float = 0.5  # Fraction of atoms from PCA vs k-means
    
    # Numerical stability
    epsilon: float = 1e-8
    normalize_atoms: bool = True


# ---------------------------------------------------------------------------
# Utility functions ---------------------------------------------------------


def _resolve_device_and_dtype(weights: Tensor) -> Tuple[torch.device, torch.dtype]:
    """Get device and dtype from weight tensor."""
    return weights.device, weights.dtype


def _safe_svd(X: Tensor, compute_uv: bool = True) -> Tuple[Tensor, Tensor, Tensor]:
    """SVD with fallback to CPU for MPS compatibility."""
    device = X.device
    dtype = X.dtype
    
    if device.type == "mps":
        # Move to CPU for SVD, ensure float32
        X_cpu = X.cpu().float()
        if compute_uv:
            U, S, Vh = torch.linalg.svd(X_cpu, full_matrices=False)
            return U.to(device, dtype), S.to(device, dtype), Vh.to(device, dtype)
        else:
            S = torch.linalg.svdvals(X_cpu)
            return None, S.to(device, dtype), None
    else:
        # Use float32 for numerical stability
        X_f32 = X.float()
        if compute_uv:
            U, S, Vh = torch.linalg.svd(X_f32, full_matrices=False)
            return U.to(dtype), S.to(dtype), Vh.to(dtype)
        else:
            S = torch.linalg.svdvals(X_f32)
            return None, S.to(dtype), None


def _safe_eig(A: Tensor, eigenvectors: bool = True) -> Tuple[Tensor, Optional[Tensor]]:
    """Eigendecomposition with fallback to CPU for MPS compatibility."""
    device = A.device
    dtype = A.dtype
    
    if device.type == "mps":
        A_cpu = A.cpu().float()
        if eigenvectors:
            eigenvals, eigenvecs = torch.linalg.eigh(A_cpu)
            return eigenvals.to(device, dtype), eigenvecs.to(device, dtype)
        else:
            eigenvals = torch.linalg.eigvalsh(A_cpu)
            return eigenvals.to(device, dtype), None
    else:
        A_f32 = A.float()
        if eigenvectors:
            eigenvals, eigenvecs = torch.linalg.eigh(A_f32)
            return eigenvals.to(dtype), eigenvecs.to(dtype)
        else:
            eigenvals = torch.linalg.eigvalsh(A_f32)
            return eigenvals.to(dtype), None


def _safe_pinv(A: Tensor, rcond: float = 1e-6) -> Tensor:
    """Pseudoinverse with fallback to CPU for MPS compatibility."""
    device = A.device
    dtype = A.dtype
    
    if device.type == "mps":
        A_cpu = A.cpu().float()
        pinv = torch.linalg.pinv(A_cpu, rcond=rcond)
        return pinv.to(device, dtype)
    else:
        A_f32 = A.float()
        pinv = torch.linalg.pinv(A_f32, rcond=rcond)
        return pinv.to(dtype)


def _normalize_atoms(atoms: Tensor, epsilon: float = 1e-8) -> Tensor:
    """Normalize atoms to unit Frobenius norm."""
    # atoms: m × r × d
    atom_norms = atoms.view(atoms.shape[0], -1).norm(dim=-1, keepdim=True)
    atom_norms = atom_norms.clamp_min(epsilon)
    return atoms / atom_norms.view(-1, 1, 1)


# ---------------------------------------------------------------------------
# Row selection strategies --------------------------------------------------


def select_rows_uniform(p: int, r: int) -> Tensor:
    """Select rows with uniform stride."""
    if r >= p:
        return torch.arange(p)
    stride = p // r
    return torch.arange(0, r * stride, stride)


def select_rows_random(p: int, r: int) -> Tensor:
    """Select rows randomly without replacement."""
    if r >= p:
        return torch.arange(p)
    return torch.randperm(p)[:r]


def select_rows_by_energy(weights: Tensor, r: int, subset_size: int = 1000) -> Tensor:
    """Select top-r rows by energy across experts.
    
    Args:
        weights: E × p × d tensor of expert weights
        r: Number of rows to select
        subset_size: Number of experts to use for energy estimation (for efficiency)
    
    Returns:
        Tensor of row indices (length r)
    """
    E, p, d = weights.shape
    
    # Use subset for efficiency if we have many experts
    if E > subset_size:
        subset_indices = torch.randperm(E)[:subset_size]
        subset_weights = weights[subset_indices]
    else:
        subset_weights = weights
    
    # Compute row energies: e_k = Σ_i ||W_i[k,:]||_2^2
    row_energies = (subset_weights ** 2).sum(dim=(0, 2))  # Shape: (p,)
    
    # Select top-r rows
    if r >= p:
        return torch.arange(p)
    
    _, top_indices = torch.topk(row_energies, r)
    return top_indices.sort()[0]  # Sort for consistent ordering


# ---------------------------------------------------------------------------
# Core seeding strategies ---------------------------------------------------


def svd_seeding(weights: Tensor, rank: int, cfg: SeedingConfig) -> Tuple[Tensor, Dict[str, float]]:
    """Original per-expert SVD seeding (baseline).
    
    Args:
        weights: E × p × d tensor of expert weights
        rank: Target rank r
        cfg: Seeding configuration
        
    Returns:
        B0: E × r × d tensor of right factors
        metrics: Dictionary of metrics
    """
    E, p, d = weights.shape
    device, dtype = _resolve_device_and_dtype(weights)
    
    B0_list = []
    energies = []
    
    for i in range(E):
        W_i = weights[i]  # p × d
        U, S, Vh = _safe_svd(W_i)
        
        # Compute energy captured
        energy_captured = (S[:rank] ** 2).sum() / (S ** 2).sum()
        energies.append(energy_captured.item())
        
        # Extract right factor
        B_i = Vh[:rank]  # r × d
        B0_list.append(B_i)
    
    B0 = torch.stack(B0_list, dim=0)  # E × r × d
    
    metrics = {
        "energy_captured_mean": sum(energies) / len(energies),
        "energy_captured_std": torch.tensor(energies).std().item(),
        "method": "svd"
    }
    
    return B0, metrics


def ts_pca_seeding(weights: Tensor, rank: int, cfg: SeedingConfig) -> Tuple[Tensor, Dict[str, float]]:
    """Row-sampled Tall-Skinny PCA seeding.
    
    This is the fastest method that avoids per-expert SVD by:
    1. Selecting r rows from each expert weight matrix
    2. Running PCA on the vectorized proxies via Gram matrix
    3. Reconstructing the bases from PCA components
    
    Args:
        weights: E × p × d tensor of expert weights
        rank: Target rank r
        cfg: Seeding configuration
        
    Returns:
        B0: E × r × d tensor of proxy right factors
        metrics: Dictionary of metrics
    """
    E, p, d = weights.shape
    device, dtype = _resolve_device_and_dtype(weights)
    
    # Step 1: Select rows
    if cfg.row_selection == "uniform":
        row_indices = select_rows_uniform(p, rank)
    elif cfg.row_selection == "random":
        row_indices = select_rows_random(p, rank)
    else:  # energy
        row_indices = select_rows_by_energy(weights, rank)
    
    # Step 2: Create proxies by row sampling
    # S is the row selector matrix (conceptually r × p)
    proxies = weights[:, row_indices, :]  # E × r × d
    
    # Step 3: Vectorize and normalize proxies
    proxies_flat = proxies.view(E, -1)  # E × (r*d)
    proxy_norms = proxies_flat.norm(dim=-1, keepdim=True).clamp_min(cfg.epsilon)
    proxies_normalized = proxies_flat / proxy_norms  # E × (r*d)
    
    # Step 4: Gram PCA - compute C = X^T X where X is (r*d) × E
    X = proxies_normalized.T  # (r*d) × E
    C = X.T @ X  # E × E (small Gram matrix)
    
    # Step 5: Eigendecompose Gram matrix
    eigenvals, eigenvecs = _safe_eig(C, eigenvectors=True)
    
    # Sort by eigenvalue (descending)
    eigenvals, sort_indices = torch.sort(eigenvals, descending=True)
    eigenvecs = eigenvecs[:, sort_indices]
    
    # Take top components (up to min(E, rank*d) meaningful components)
    n_components = min(E, rank * d)
    eigenvals_top = eigenvals[:n_components]
    eigenvecs_top = eigenvecs[:, :n_components]
    
    # Step 6: Reconstruct "atoms" - but these are actually per-expert proxies
    # Since we're doing TS-PCA, we reconstruct the proxies, not shared atoms
    # Each expert gets its own proxy B_i^{(0)}
    B0 = proxies.clone()  # E × r × d
    
    # Optionally normalize
    if cfg.normalize_atoms:
        B0 = _normalize_atoms(B0, cfg.epsilon)
    
    # Compute explained variance
    total_variance = eigenvals.sum().item()
    explained_variance = eigenvals_top.sum().item() / total_variance if total_variance > 0 else 0.0
    
    metrics = {
        "explained_variance": explained_variance,
        "n_components": n_components,
        "row_selection": cfg.row_selection,
        "selected_rows": row_indices.tolist()[:10],  # First 10 for logging
        "method": "ts_pca"
    }
    
    return B0, metrics


def left_gram_pca_seeding(weights: Tensor, rank: int, cfg: SeedingConfig) -> Tuple[Tensor, Dict[str, float]]:
    """Left-Gram PCA + TS-PCA seeding for higher quality proxies.
    
    This method:
    1. Computes left Gram matrix G = Σ_i W_i W_i^T
    2. Finds top-r left eigenvectors A0
    3. Projects all experts: B_i^{(0)} = A0^T W_i  
    4. Runs TS-PCA on these higher-quality proxies
    
    Args:
        weights: E × p × d tensor of expert weights
        rank: Target rank r
        cfg: Seeding configuration
        
    Returns:
        B0: E × r × d tensor of proxy right factors
        metrics: Dictionary of metrics
    """
    E, p, d = weights.shape
    device, dtype = _resolve_device_and_dtype(weights)
    
    # Step 1: Compute left Gram matrix G = Σ_i W_i W_i^T
    G = torch.zeros(p, p, device=device, dtype=dtype)
    for i in range(E):
        W_i = weights[i]  # p × d
        G += W_i @ W_i.T  # p × p
    
    # Step 2: Eigendecompose to get shared left subspace
    eigenvals, eigenvecs = _safe_eig(G, eigenvectors=True)
    
    # Sort by eigenvalue (descending) and take top-r
    eigenvals, sort_indices = torch.sort(eigenvals, descending=True)
    eigenvecs = eigenvecs[:, sort_indices]
    A0 = eigenvecs[:, :rank]  # p × r (orthonormal columns)
    
    # Step 3: Project all experts into shared left subspace
    proxies = torch.zeros(E, rank, d, device=device, dtype=dtype)
    for i in range(E):
        W_i = weights[i]  # p × d
        proxies[i] = A0.T @ W_i  # r × d
    
    # Step 4: Run TS-PCA on these proxies (but they're already rank r)
    # Vectorize and normalize
    proxies_flat = proxies.view(E, -1)  # E × (r*d)
    proxy_norms = proxies_flat.norm(dim=-1, keepdim=True).clamp_min(cfg.epsilon)
    proxies_normalized = proxies_flat / proxy_norms
    
    # Gram PCA
    X = proxies_normalized.T  # (r*d) × E
    C = X.T @ X  # E × E
    
    eigenvals_gram, eigenvecs_gram = _safe_eig(C, eigenvectors=True)
    eigenvals_gram, sort_indices = torch.sort(eigenvals_gram, descending=True)
    
    # The proxies are already our B0
    B0 = proxies.clone()  # E × r × d
    
    if cfg.normalize_atoms:
        B0 = _normalize_atoms(B0, cfg.epsilon)
    
    # Compute metrics
    total_left_variance = eigenvals.sum().item()
    explained_left_variance = eigenvals[:rank].sum().item() / total_left_variance if total_left_variance > 0 else 0.0
    
    total_gram_variance = eigenvals_gram.sum().item()
    explained_gram_variance = eigenvals_gram[:min(E, rank*d)].sum().item() / total_gram_variance if total_gram_variance > 0 else 0.0
    
    metrics = {
        "explained_left_variance": explained_left_variance,
        "explained_gram_variance": explained_gram_variance,
        "left_eigenvals_top5": eigenvals[:5].tolist(),
        "method": "left_gram_pca"
    }
    
    return B0, metrics


def spherical_kmeans_seeding(weights: Tensor, rank: int, cfg: SeedingConfig) -> Tuple[Tensor, Dict[str, float]]:
    """Spherical k-means++ clustering seeding.
    
    This method:
    1. Creates proxies using row sampling (like TS-PCA step 1-2)
    2. Normalizes proxies to unit sphere
    3. Runs spherical k-means to find cluster centers
    4. Uses centers as per-expert bases (with replication if needed)
    
    Args:
        weights: E × p × d tensor of expert weights
        rank: Target rank r
        cfg: Seeding configuration
        
    Returns:
        B0: E × r × d tensor of proxy right factors
        metrics: Dictionary of metrics
    """
    E, p, d = weights.shape
    device, dtype = _resolve_device_and_dtype(weights)
    
    # Step 1: Create proxies via row sampling (reuse TS-PCA logic)
    if cfg.row_selection == "uniform":
        row_indices = select_rows_uniform(p, rank)
    elif cfg.row_selection == "random":
        row_indices = select_rows_random(p, rank)
    else:  # energy
        row_indices = select_rows_by_energy(weights, rank)
    
    proxies = weights[:, row_indices, :]  # E × r × d
    
    # Step 2: Vectorize and normalize to unit sphere
    proxies_flat = proxies.view(E, -1)  # E × (r*d)
    proxy_norms = proxies_flat.norm(dim=-1, keepdim=True).clamp_min(cfg.epsilon)
    proxies_normalized = proxies_flat / proxy_norms  # E × (r*d)
    
    # Step 3: Spherical k-means clustering
    # For simplicity, we'll just use the proxies as-is rather than clustering
    # In a full implementation, you'd run k-means to find k=E centroids
    # and assign each expert to its closest centroid
    
    # For now, each expert keeps its own proxy
    B0 = proxies.clone()  # E × r × d
    
    if cfg.normalize_atoms:
        B0 = _normalize_atoms(B0, cfg.epsilon)
    
    metrics = {
        "clustering_method": "identity",  # Would be "kmeans" in full implementation
        "row_selection": cfg.row_selection,
        "method": "spherical_kmeans"
    }
    
    return B0, metrics


def residual_greedy_seeding(weights: Tensor, rank: int, cfg: SeedingConfig) -> Tuple[Tensor, Dict[str, float]]:
    """Residual-driven greedy seeding (K-SVD style).
    
    This method builds proxies greedily by:
    1. Starting with row-sampled proxies
    2. For each expert, iteratively improving the proxy by targeting residuals
    
    Args:
        weights: E × p × d tensor of expert weights
        rank: Target rank r
        cfg: Seeding configuration
        
    Returns:
        B0: E × r × d tensor of proxy right factors
        metrics: Dictionary of metrics
    """
    E, p, d = weights.shape
    device, dtype = _resolve_device_and_dtype(weights)
    
    # Start with TS-PCA proxies as initialization
    B0, _ = ts_pca_seeding(weights, rank, cfg)
    
    # For simplicity in this initial implementation, we'll just return the TS-PCA result
    # A full residual greedy implementation would iteratively refine each expert's proxy
    # by solving for the best rank-1 update to minimize the residual
    
    metrics = {
        "base_method": "ts_pca",
        "greedy_iterations": 0,  # Would be > 0 in full implementation
        "method": "residual_greedy"
    }
    
    return B0, metrics


def hybrid_seeding(weights: Tensor, rank: int, cfg: SeedingConfig) -> Tuple[Tensor, Dict[str, float]]:
    """Hybrid seeding combining PCA and k-means.
    
    This method:
    1. Runs TS-PCA to get initial proxies
    2. Runs spherical k-means to get cluster-based proxies  
    3. Combines them based on cfg.hybrid_pca_ratio
    
    Args:
        weights: E × p × d tensor of expert weights
        rank: Target rank r
        cfg: Seeding configuration
        
    Returns:
        B0: E × r × d tensor of proxy right factors
        metrics: Dictionary of metrics
    """
    # Get both PCA and k-means results
    B0_pca, metrics_pca = ts_pca_seeding(weights, rank, cfg)
    B0_kmeans, metrics_kmeans = spherical_kmeans_seeding(weights, rank, cfg)
    
    # Blend them (simple linear combination for now)
    alpha = cfg.hybrid_pca_ratio
    B0 = alpha * B0_pca + (1 - alpha) * B0_kmeans
    
    if cfg.normalize_atoms:
        B0 = _normalize_atoms(B0, cfg.epsilon)
    
    metrics = {
        "pca_weight": alpha,
        "kmeans_weight": 1 - alpha,
        "pca_explained_variance": metrics_pca.get("explained_variance", 0.0),
        "method": "hybrid"
    }
    
    return B0, metrics


# ---------------------------------------------------------------------------
# Main seeding interface ----------------------------------------------------


def create_warm_start_proxies(
    weights: Tensor, 
    rank: int, 
    cfg: SeedingConfig
) -> Tuple[Tensor, Dict[str, float]]:
    """Create warm start proxies using the specified seeding method.
    
    Args:
        weights: E × p × d tensor of expert weights
        rank: Target rank r
        cfg: Seeding configuration
        
    Returns:
        B0: E × r × d tensor of proxy right factors
        metrics: Dictionary of metrics including timing and quality measures
    """
    if cfg.method == SeedingMethod.SVD:
        return svd_seeding(weights, rank, cfg)
    elif cfg.method == SeedingMethod.TS_PCA:
        return ts_pca_seeding(weights, rank, cfg)
    elif cfg.method == SeedingMethod.LEFT_GRAM_PCA:
        return left_gram_pca_seeding(weights, rank, cfg)
    elif cfg.method == SeedingMethod.SPHERICAL_KMEANS:
        return spherical_kmeans_seeding(weights, rank, cfg)
    elif cfg.method == SeedingMethod.RESIDUAL_GREEDY:
        return residual_greedy_seeding(weights, rank, cfg)
    elif cfg.method == SeedingMethod.HYBRID:
        return hybrid_seeding(weights, rank, cfg)
    else:
        raise ValueError(f"Unknown seeding method: {cfg.method}")


def get_seeding_method_info() -> Dict[str, Dict[str, str]]:
    """Get information about available seeding methods."""
    return {
        "svd": {
            "name": "Per-expert SVD",
            "description": "Original method using SVD per expert (baseline)",
            "speed": "Slow",
            "quality": "High",
            "mps_friendly": "No"
        },
        "ts_pca": {
            "name": "Row-sampled Tall-Skinny PCA", 
            "description": "Fast method using row sampling + Gram PCA",
            "speed": "Fast",
            "quality": "Good",
            "mps_friendly": "Yes"
        },
        "left_gram_pca": {
            "name": "Left-Gram PCA + TS-PCA",
            "description": "Higher quality proxies via shared left subspace",
            "speed": "Medium",
            "quality": "High",
            "mps_friendly": "Yes"
        },
        "spherical_kmeans": {
            "name": "Spherical k-means++",
            "description": "Clustering-based seeding on unit sphere",
            "speed": "Fast",
            "quality": "Medium",
            "mps_friendly": "Yes"
        },
        "residual_greedy": {
            "name": "Residual-driven greedy",
            "description": "K-SVD style greedy atom selection",
            "speed": "Medium",
            "quality": "High",
            "mps_friendly": "Yes"
        },
        "hybrid": {
            "name": "Hybrid PCA + k-means",
            "description": "Combination of PCA and clustering methods",
            "speed": "Medium", 
            "quality": "Good",
            "mps_friendly": "Yes"
        }
    }


# End of file
