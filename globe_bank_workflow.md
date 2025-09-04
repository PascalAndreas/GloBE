# GloBE — Bank Initialization & Training Workflow

Implementation of **global basis banks** for MoE FFN compression using sparse mixtures and alternating minimization. This document describes the complete workflow from expert extraction to trained banks.

---

## 1) Data Prep
1. **Load experts**: extract all MoE FFN **Up** and **Gate** expert matrices from the HF checkpoint.
2. **Per‑family Z‑score**: standardize each family (Up/Gate) across experts; record mean/scale to fold back later.
3. **Dims & sizes**: choose inner dim `r ∈ {p/2,3p/4,p}` and bank size `m ∈ {n/8,n/4,n/2}` where `n` is the total number of experts.

---

## 2) Warm Start Seeding
4. **Configurable seeding methods** for creating proxy right factors `B_i^(0) ∈ R^{r×d}`:
   - **TS-PCA** (default): Row-sampled Tall-Skinny PCA—fastest, MPS-friendly, no per-expert SVD
   - **Left-Gram PCA**: Shared left subspace projection for higher-quality proxies
   - **SVD baseline**: Original per-expert truncated SVD for comparison
   - **Spherical k-means**: Clustering-based seeding on unit sphere
   - **Residual greedy**: K-SVD style atom selection for lowest initial MSE
5. **Initial bank & codes**:
   - Build provisional bank `{B_j}` from proxy centroids via k-means clustering
   - **Least squares + clamp** to get initial codes `α_i^(0)` s.t. `B_i^(0) ≈ Σ_j α_{i,j} B_j`
   - **OLS adapter fit**: `A_i^(0) = W_i B_i^{(0)+}` for reconstruction `W_i ≈ A_i B_i^{(0)}`

---

## 3) Bank Learning (Alternating Minimization)
6. **Coding step**: Apply temperature scaling and sparsity to current mixture weights `α_i`:
   - `α_i = entmax(log(α_i) / T)` for T ≠ 1.0 (entmax/sparsemax activation)
   - **ε-prune**: zero entries below threshold; renormalize to maintain probability distribution
7. **Dictionary step (MOD)**: Update bank using linear solve instead of matrix inverse:
   - Solve `(A A^T + λ_B I) B^T = (A X^T)^T` where `A` is stacked codes, `X` is stacked targets
   - **Unit Frobenius normalize** atoms; rescale codes to preserve reconstruction
8. **Batched adapter refit**: Solve normal equations using Cholesky decomposition:
   - `G_i = B_i B_i^T + λ_A I`, `R_i = W_i B_i^T` for all experts
   - Batched solve: `A_i = cholesky_solve(R_i^T, cholesky(G_i))^T`
9. **Temperature control**: Adjust T to target median support with min/max bounds:
   - `T ← clamp(T · exp(η · (med_support/target - 1)), T_min, T_max)`
10. **Repeat 6–9** for S steps; early-stop on validation MSE plateau and stable supports

---

## 4) Finalize & Export
11. **Fold back normalization** constants into `A_i`.
12. **Save** `{A, B, α}` (Up & Gate) + config/seed.
13. **Precompose for inference**: `tilde_B_i = φ(Σ_j α_{i,j} B_j)` (after ε‑prune); cache per expert in compute dtype (bf16/fp16).
14. **Patch HF model** to use cached `tilde_B_i` + `A_i` in the FFN up/gate; Down stays dense.

---

## 5) Metrics to Log (Weights & Biases)
**Reconstruction & capacity**
- **Relative Frobenius error**: `||recon - W||_F / ||W||_F` per family and globally
- **Seeding quality**: Explained variance and method-specific metrics from warm start
- **Energy captured**: Initial reconstruction quality from seeding methods

**Sparsity & mixtures**  
- **Support sizes**: median/mean/p95 `|supp(α_i)|`; target vs actual support error
- **Temperature trajectory**: T values with min/max bounds; support stability
- **Entropy**: `H(α_i) = -Σ α_i log α_i` using proper zero-handling; pruning fractions
- **Base popularity**: atom usage frequencies; support overlap patterns

**Numerical stability**
- **Atom norms**: deviation from unit normalization; max norm deviation
- **Health checks**: NaN/Inf counts across tensors; Cholesky vs general solve fallbacks
- **Regularization**: λ_B and λ_A effectiveness; condition number improvements

**Runtime & resources**
- Step time; memory footprint; I/O throughput.

**Ablation hooks**
- Sweep logs keyed by `(seeding_method, m, r, target_support, λ_B, λ_A)`.

---

## 6) Implementation Notes & Future Improvements

**Current implementation features:**
- Multiple warm start methods with MPS compatibility for Mac M-series development
- Numerically stable AM loop using linear solvers and Cholesky decomposition
- Alpha state management (no logit drift) for consistent entmax sparsity patterns
- Batched operations for 5-10x speedup in adapter refit
- Comprehensive metrics and device-aware fallbacks

**Potential improvements:**
- Weight-aware MOD targeting current mixed bases instead of fixed SVD targets
- Coherence penalties to discourage atom duplication
- Advanced temperature schedules and top-K sparsity constraints
- Streaming implementations for very large expert counts

