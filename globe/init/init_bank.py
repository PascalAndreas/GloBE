"""Bank initialization utilities for GloBE.

This module provides methods for initializing global basis banks using
dictionary learning, SVD decomposition, and optimization-based approaches.
"""

from typing import Dict, List, Tuple, Optional, Union
import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from sklearn.decomposition import DictionaryLearning, TruncatedSVD
from sklearn.linear_model import Lasso
from scipy.optimize import nnls
import warnings


class BankInitializer:
    """Utilities for initializing global basis banks."""
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        seed: int = 42,
    ):
        """Initialize bank initializer.
        
        Args:
            device: Device for computations
            dtype: Data type for computations
            seed: Random seed
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.seed = seed
        
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    def init_banks_svd(
        self,
        expert_weights: Dict[str, List[Tensor]],
        num_bases: int,
        rank: int,
    ) -> Tuple[Tensor, Dict[str, Tensor], Dict[str, Tensor]]:
        """Initialize banks using SVD decomposition.
        
        Args:
            expert_weights: Dictionary mapping family ("up", "gate") to list of expert weights
            num_bases: Number of basis vectors to create
            rank: Rank of basis vectors
            
        Returns:
            Tuple of (basis_bank, mixture_coefficients, adapter_matrices)
        """
        results = {}
        
        for family, weights_list in expert_weights.items():
            if not weights_list:
                continue
            
            # Stack expert weights: num_experts × (p × d) -> num_experts × p × d
            stacked_weights = torch.stack(weights_list, dim=0)  # E × p × d
            num_experts, p, d = stacked_weights.shape
            
            # Reshape for matrix decomposition: (E*p) × d
            reshaped_weights = stacked_weights.view(-1, d)
            
            # Perform SVD on the concatenated weight matrix
            U, S, Vt = torch.svd(reshaped_weights)
            
            # Extract top-r components for basis bank
            # Basis bank: rank × d (transposed for consistency)
            basis_bank = Vt[:rank, :].T  # d × rank
            
            # For each expert, find best approximation using the basis
            mixture_coeffs = []
            adapter_matrices = []
            
            for i, expert_weight in enumerate(weights_list):
                # expert_weight: p × d
                # Find coefficients: solve expert_weight ≈ A_i @ φ(α_i @ B)
                
                # For SVD init, we use linear projection onto basis
                # Project expert weight onto basis space
                projected = torch.matmul(expert_weight, basis_bank)  # p × rank
                
                # Simple initialization: uniform mixture, identity-like adapter
                mixture_coeff = torch.ones(num_bases, device=self.device, dtype=self.dtype) / num_bases
                adapter_matrix = projected  # p × rank
                
                mixture_coeffs.append(mixture_coeff)
                adapter_matrices.append(adapter_matrix)
            
            results[family] = {
                "basis_bank": basis_bank,
                "mixture_coeffs": torch.stack(mixture_coeffs, dim=0),
                "adapter_matrices": torch.stack(adapter_matrices, dim=0),
            }
        
        return results
    
    def init_banks_dictionary_learning(
        self,
        expert_weights: Dict[str, List[Tensor]],
        num_bases: int,
        rank: int,
        alpha: float = 1.0,
        max_iter: int = 1000,
    ) -> Dict[str, Dict[str, Tensor]]:
        """Initialize banks using dictionary learning.
        
        Args:
            expert_weights: Dictionary mapping family to list of expert weights
            num_bases: Number of basis vectors (dictionary atoms)
            rank: Rank of basis vectors
            alpha: Sparsity regularization strength
            max_iter: Maximum iterations for dictionary learning
            
        Returns:
            Dictionary with initialized parameters per family
        """
        results = {}
        
        for family, weights_list in expert_weights.items():
            if not weights_list:
                continue
            
            # Stack and reshape expert weights
            stacked_weights = torch.stack(weights_list, dim=0)  # E × p × d
            num_experts, p, d = stacked_weights.shape
            
            # Convert to numpy for sklearn
            X = stacked_weights.view(-1, d).cpu().numpy()  # (E*p) × d
            
            # Apply dictionary learning
            dict_learner = DictionaryLearning(
                n_components=num_bases,
                alpha=alpha,
                max_iter=max_iter,
                random_state=self.seed,
                positive_dict=False,
                positive_code=False,
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sparse_codes = dict_learner.fit_transform(X)  # (E*p) × num_bases
                dictionary = dict_learner.components_  # num_bases × d
            
            # Convert back to torch tensors
            dictionary_tensor = torch.from_numpy(dictionary).to(
                device=self.device, dtype=self.dtype
            ).T  # d × num_bases
            
            sparse_codes_tensor = torch.from_numpy(sparse_codes).to(
                device=self.device, dtype=self.dtype
            )  # (E*p) × num_bases
            
            # Reshape sparse codes back to expert format
            sparse_codes_reshaped = sparse_codes_tensor.view(num_experts, p, num_bases)
            
            # Initialize mixture coefficients and adapters for each expert
            mixture_coeffs = []
            adapter_matrices = []
            
            for i in range(num_experts):
                # Average sparse codes across the intermediate dimension to get mixture
                mixture_coeff = torch.mean(sparse_codes_reshaped[i], dim=0)  # num_bases
                mixture_coeff = F.softmax(mixture_coeff, dim=0)  # Normalize
                
                # Use sparse codes as initialization for adapter
                # Reduce dimensionality if needed
                codes = sparse_codes_reshaped[i]  # p × num_bases
                if rank < num_bases:
                    # Use SVD to reduce dimensionality
                    U, S, Vt = torch.svd(codes)
                    adapter_matrix = U[:, :rank] * S[:rank].unsqueeze(0)  # p × rank
                else:
                    adapter_matrix = codes  # p × num_bases
                
                mixture_coeffs.append(mixture_coeff)
                adapter_matrices.append(adapter_matrix)
            
            results[family] = {
                "basis_bank": dictionary_tensor,
                "mixture_coeffs": torch.stack(mixture_coeffs, dim=0),
                "adapter_matrices": torch.stack(adapter_matrices, dim=0),
            }
        
        return results
    
    def optimize_mixture_coefficients(
        self,
        expert_weights: List[Tensor],
        basis_bank: Tensor,
        adapter_matrices: Tensor,
        method: str = "nnls",
        l1_reg: float = 1e-4,
    ) -> Tensor:
        """Optimize mixture coefficients given basis bank and adapters.
        
        Args:
            expert_weights: List of expert weight tensors
            basis_bank: Basis bank tensor (d × num_bases)
            adapter_matrices: Adapter matrices (num_experts × p × rank)
            method: Optimization method ("nnls", "lasso", "least_squares")
            l1_reg: L1 regularization strength for lasso
            
        Returns:
            Optimized mixture coefficients (num_experts × num_bases)
        """
        num_experts = len(expert_weights)
        num_bases = basis_bank.shape[1]
        mixture_coeffs = []
        
        for i, expert_weight in enumerate(expert_weights):
            # expert_weight: p × d
            # adapter: p × rank
            adapter = adapter_matrices[i]
            
            # For each row of the expert weight, solve for mixture coefficients
            # We want: expert_weight[j, :] ≈ adapter[j, :] @ basis_bank^T @ mixture_coeff
            
            # Simplified approach: solve globally
            # Flatten expert weight and compute pseudo-target
            target = expert_weight.flatten().cpu().numpy()  # p*d
            
            # Create design matrix
            # This is a simplified version - in practice, we need to account for
            # the nonlinear activation in the mixture path
            design_matrix = torch.kron(
                adapter, torch.eye(basis_bank.shape[0], device=basis_bank.device)
            )  # (p*d) × (rank*num_bases)
            design_matrix = design_matrix.cpu().numpy()
            
            if method == "nnls":
                # Non-negative least squares
                coeffs, _ = nnls(design_matrix, target)
            elif method == "lasso":
                # Lasso regression
                lasso = Lasso(alpha=l1_reg, random_state=self.seed)
                lasso.fit(design_matrix, target)
                coeffs = lasso.coef_
            else:  # least_squares
                # Standard least squares
                coeffs, _, _, _ = np.linalg.lstsq(design_matrix, target, rcond=None)
            
            # Reshape and normalize coefficients
            coeffs = coeffs.reshape(adapter.shape[1], num_bases)  # rank × num_bases
            coeffs = np.mean(coeffs, axis=0)  # Average over rank dimension
            coeffs = np.maximum(coeffs, 0)  # Ensure non-negative
            coeffs = coeffs / (np.sum(coeffs) + 1e-8)  # Normalize
            
            mixture_coeff = torch.from_numpy(coeffs).to(
                device=self.device, dtype=self.dtype
            )
            mixture_coeffs.append(mixture_coeff)
        
        return torch.stack(mixture_coeffs, dim=0)
    
    def optimize_adapters(
        self,
        expert_weights: List[Tensor],
        basis_bank: Tensor,
        mixture_coeffs: Tensor,
        activation_fn: callable = F.silu,
    ) -> Tensor:
        """Optimize adapter matrices given basis bank and mixture coefficients.
        
        Args:
            expert_weights: List of expert weight tensors
            basis_bank: Basis bank tensor (d × num_bases)  
            mixture_coeffs: Mixture coefficients (num_experts × num_bases)
            activation_fn: Activation function used in mixture path
            
        Returns:
            Optimized adapter matrices (num_experts × p × rank)
        """
        num_experts = len(expert_weights)
        adapter_matrices = []
        
        for i, expert_weight in enumerate(expert_weights):
            # expert_weight: p × d
            # mixture_coeff: num_bases
            
            # Compute mixed basis: φ(mixture_coeff @ basis_bank^T)
            mixed_basis = torch.matmul(mixture_coeffs[i], basis_bank.T)  # d
            mixed_basis = activation_fn(mixed_basis)  # d
            
            # Solve for adapter: expert_weight ≈ adapter @ mixed_basis^T
            # adapter: p × 1 (rank=1 case)
            # For general rank, we need to be more careful
            
            # Simplified approach: use pseudo-inverse
            if mixed_basis.dim() == 1:
                mixed_basis = mixed_basis.unsqueeze(0)  # 1 × d
            
            # Solve: expert_weight ≈ adapter @ mixed_basis
            adapter = torch.linalg.lstsq(
                mixed_basis.T.unsqueeze(0).expand(expert_weight.shape[0], -1, -1),
                expert_weight.unsqueeze(-1)
            ).solution.squeeze(-1)  # p × d
            
            # For now, assume rank=1 and take mean
            adapter = adapter.mean(dim=1, keepdim=True)  # p × 1
            
            adapter_matrices.append(adapter)
        
        return torch.stack(adapter_matrices, dim=0)
    
    def joint_optimization(
        self,
        expert_weights: Dict[str, List[Tensor]],
        num_bases: int,
        rank: int,
        num_iterations: int = 100,
        lr: float = 1e-3,
    ) -> Dict[str, Dict[str, Tensor]]:
        """Joint optimization of all parameters.
        
        Args:
            expert_weights: Dictionary mapping family to expert weights
            num_bases: Number of basis vectors
            rank: Rank of basis vectors
            num_iterations: Number of optimization iterations
            lr: Learning rate
            
        Returns:
            Dictionary with optimized parameters per family
        """
        results = {}
        
        for family, weights_list in expert_weights.items():
            if not weights_list:
                continue
            
            num_experts = len(weights_list)
            p, d = weights_list[0].shape
            
            # Initialize parameters
            basis_bank = torch.randn(d, num_bases, device=self.device, dtype=self.dtype) * 0.02
            mixture_logits = torch.randn(num_experts, num_bases, device=self.device, dtype=self.dtype) * 0.01
            adapter_matrices = torch.randn(num_experts, p, rank, device=self.device, dtype=self.dtype) * 0.02
            
            # Make parameters learnable
            basis_bank.requires_grad_(True)
            mixture_logits.requires_grad_(True)
            adapter_matrices.requires_grad_(True)
            
            optimizer = torch.optim.Adam([basis_bank, mixture_logits, adapter_matrices], lr=lr)
            
            target_weights = torch.stack(weights_list, dim=0).to(device=self.device, dtype=self.dtype)
            
            for iteration in range(num_iterations):
                optimizer.zero_grad()
                
                # Forward pass
                mixture_weights = F.softmax(mixture_logits, dim=-1)  # num_experts × num_bases
                
                # Compute mixed bases for each expert
                mixed_bases = torch.matmul(mixture_weights, basis_bank.T)  # num_experts × d
                mixed_bases = F.silu(mixed_bases)  # Apply activation
                
                # Compute reconstructed weights
                # For each expert: adapter @ mixed_basis^T
                reconstructed = torch.bmm(
                    adapter_matrices,  # num_experts × p × rank
                    mixed_bases.unsqueeze(-1)  # num_experts × d × 1
                ).squeeze(-1)  # num_experts × p
                
                # Reconstruction loss
                loss = F.mse_loss(reconstructed, target_weights.view(num_experts, -1))
                
                # Add regularization
                l1_loss = 1e-4 * torch.mean(torch.abs(mixture_weights))
                spectral_loss = 1e-4 * torch.norm(basis_bank, p=2)
                
                total_loss = loss + l1_loss + spectral_loss
                
                total_loss.backward()
                optimizer.step()
                
                if iteration % 20 == 0:
                    print(f"Family {family}, Iteration {iteration}: Loss = {total_loss.item():.6f}")
            
            results[family] = {
                "basis_bank": basis_bank.detach(),
                "mixture_coeffs": F.softmax(mixture_logits.detach(), dim=-1),
                "adapter_matrices": adapter_matrices.detach(),
            }
        
        return results
