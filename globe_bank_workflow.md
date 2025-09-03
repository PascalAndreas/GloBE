# GloBE — Bank Initialization & Training Workflow

Concise plan for initializing and training a **single global basis bank** (Up & Gate), with sparse, token‑independent mixtures and precomposition for inference. Includes key metrics to log.

---

## 1) Data Prep
1. **Load experts**: extract all MoE FFN **Up** and **Gate** expert matrices from the HF checkpoint.
2. **Per‑family Z‑score**: standardize each family (Up/Gate) across experts; record mean/scale to fold back later.
3. **Dims & sizes**: choose inner dim `r ∈ {p/2,3p/4,p}` and bank size `m ∈ {n/8,n/4,n/2}` where `n` is the total number of experts.

---

## 2) Warm Starts
4. **Truncated SVD (rank r)** per expert `W_i ∈ R^{p×d}`:
   - `W_i ≈ (U_r Σ_r)(V_r^T)` ⇒ set `A_i^(0)=U_r Σ_r ∈ R^{p×r}`, `B_i^(0)=V_r^T ∈ R^{r×d}`.
5. **Initial bank & codes**:
   - Build a provisional bank `{B_j}` from a few `B_i^(0)` centroids or random picks.
   - **NNLS (optional simplex)** per expert to get initial codes `α_i^(0)` s.t. `B_i^(0) ≈ Σ_j α_{i,j} B_j`.

---

## 3) Bank Learning (Alternating Minimization)
6. **Coding step** (sparse map): maintain logits `z_i`; compute `α_i = entmax(z_i / T)` (or sparsemax). Add tiny L1 if needed; **ε‑prune** very small entries.
7. **Dictionary step (MOD)**: vectorize targets `X` (stack of `B_i^(t)`), codes `A` (stack of `α_i`), then update bank:
   - `B ← X A^T (A A^T)^−1`, reshape to `r×d` atoms.
   - **Normalize atoms** (e.g., unit Frobenius/row norms) and rescale codes to fix gauge.
8. **Refit per‑expert A (OLS)**: `A_i ← argmin_A ||W_i − A · (Σ_j α_{i,j} B_j)||_F^2 = W_i · (Σ_j α_{i,j} B_j)^+`.
9. **Temperature control**: every K steps, adjust `T` to target a median support (e.g., 12):
   - `T ← T · exp(η · (median(|supp(α_i)|)/target − 1))`, with `η ∈ [0.02, 0.1]`.
10. **Repeat 6–9** for S steps; early‑stop on validation MSE plateau.

---

## 4) Finalize & Export
11. **Fold back normalization** constants into `A_i`.
12. **Save** `{A, B, α}` (Up & Gate) + config/seed.
13. **Precompose for inference**: `tilde_B_i = φ(Σ_j α_{i,j} B_j)` (after ε‑prune); cache per expert in compute dtype (bf16/fp16).
14. **Patch HF model** to use cached `tilde_B_i` + `A_i` in the FFN up/gate; Down stays dense.

---

## 5) Metrics to Log (Weights & Biases)
**Reconstruction & capacity**
- **MSE_F**: Frobenius MSE per family (Up/Gate) and per layer; global mean.
- **Energy captured**: \sum_{k≤r} σ_k^2 / \sum σ_k^2 (at init and periodically) for a sample of experts.
- **Effective rank** (95% energy) per family/layer (periodic).

**Sparsity & mixtures**
- **Support sizes**: mean/median/min/max |supp(α_i)|; histogram; **entropy** H(α_i).
- **Temperature** trajectory T; **L1 penalty** value; % of coefficients < ε (pruned).
- **Base popularity**: usage frequency per atom; Jaccard overlap of supports across families (Up vs Gate).

**Stability & diversity**
- **Atom norms** (after normalization); **coherence** between atoms (e.g., ||B^T B − I||_F).
- **Condition numbers** of `(A A^T)` in MOD updates; fallback count if regularized.

**Runtime & resources**
- Step time; memory footprint; I/O throughput.

**Ablation hooks**
- Sweep logs keyed by `(m, r, target_support, S_cache)`.

---

## 6) Stopping & Sanity Checks
- Validation MSE plateaus for N evaluations **and** support distribution stabilized.
- Random expert spot‑checks: reconstruct `W_i` and compare spectra & top singular vectors alignment.
- Compose→patch→forward on a tiny input: parity of activations between reconstructed FFN and original within tolerance.

