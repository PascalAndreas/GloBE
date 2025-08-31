# GloBE Implementation Plan

A drop‑in Global‑Basis Experts (GloBE) module for MoE FFNs that:
- uses a single global basis bank (Up & Gate) across all MoE layers,
- learns sparse, token‑independent mixtures via entmax/sparsemax with temperature annealing, and
- precomposes & caches `tilde_B_i` per expert for dense‑like inference.

---

## Decisions & Defaults (v1)
- Prototype model: `Qwen1.5-MoE-A2.7B` (modern, small enough for a 24‑GB Mac). Add larger targets later.
- Cache target: cache `tilde_B_i` in R^{r x d} (Up & Gate). If r≈p later, we may switch to caching `hat_W_i`.
- Sparsity: entmax (alpha≈1.5) or sparsemax with temperature annealing toward a target support; optional small L1 on alpha. No hard top‑s at inference; allow variable support; epsilon‑prune before precomposition.
- Banks: one global bank per family (Up, Gate). Down stays dense. No quantization in v1 (cache uses compute dtype bf16/fp16).
- Init dims: start with r = p; ablate r in {p, p/2}. Bank size m in {64, 128, 256}.

---

## What to reuse from MoBE
- Weight I/O and tensor naming helpers (mapping expert up/gate/down; Z‑score normalization per family).
- Reconstruction loop structure (e.g., train/train_group) and activation choices (SiLU/Tanh) in the mixture path.
- HF model patching patterns (checkpoint surgery utilities) as references for clean integration.

We change: per‑layer banks → global banks; softmax → entmax/sparsemax + temperature; add precompose + LRU cache for `tilde_B_i`.

---

## Repository skeleton (no code)
```
GloBE/
  README.md
  CITATION.cff
  LICENSE
  pyproject.toml   # torch, transformers, safetensors, hydra-core, wandb, entmax
  globe/
    __init__.py
    config/
      default.yaml
      model/qwen15_moe_a2_7b.yaml  # dims read from weights; router/topk
      bank/global.yaml             # m, r, activation
      sparsity/entmax.yaml         # alpha, T schedule, L1, epsilon-prune
      train/recon.yaml             # Adam, lr, steps, batch sizes
      cache/lru.yaml               # S, dtype: bf16/fp16, warmup tokens
      export/hf.yaml               # save options
    data/
      naming_qwen15.py             # tensor name patterns for Qwen1.5-MoE
      io_safetensors.py
    modules/
      globe_bank.py                # global Up/Gate banks (B_j)
      globe_mixer.py               # entmax/sparsemax + temperature controller
      ffn_globe.py                 # A_i · phi(sum alpha_j B_j) for Up/Gate; Down dense
      precompose_cache.py          # build & LRU-manage tilde_B_i per expert
    init/
      init_bank.py                 # dict learning/SVD init; NNLS alpha; LS A
      zscore.py                    # per-family normalization
    train/
      recon_objective.py           # MSE + regularizers
      loop.py                      # sample experts; update A, B, alpha; wandb logs
    infer/
      patch_hf.py                  # swap MoE FFN with GloBE in HF classes
      export_hf.py                 # write HF checkpoint with cached tilde_B_i
    eval/
      recon_metrics.py             # Frobenius MSE, spectra, effective rank
      latency_mem.py               # tok/s, VRAM, cache hit-rate
  scripts/
    fit_globe_up.sh
    fit_globe_gate.sh
    export_hf.sh
  third_party/
    mobe/                          # optional submodule (read‑only reference)
```

---

## Hydra config (key fields)
- `model`: { name, layers_moe, d (read), p (read), n_routed, n_shared, topk }
- `bank`: { m_up, m_gate, r, activation: {type: silu|tanh}, init: {method: svd|dict, seed} }
- `sparsity`: { map: entmax|sparsemax, alpha: 1.5, T0: 2.0, target_support: 12, eta: 0.05, l1: 1e-4, eps_prune: 1e-4 }
- `train`: { steps, batch_size, lr, adam: {beta1, beta2, eps, wd}, zscore: true }
- `cache`: { enabled: true, S_per_layer: 16, dtype: bf16|fp16, warmup_tokens: 2048 }
- `export`: { dtype: bf16, save_mixed: true }
- `wandb`: { project, entity, run_name, tags }

---

## Training procedure (weight‑only)
1) Load and normalize expert Up/Gate matrices (Z‑score per family); record scales.
2) Init global banks (Up & Gate separately): seed B_j via SVD/dictionary learning; NNLS for alpha_i; least squares for A_i.
3) Optimize (Adam): mini‑batch experts across layers; forward hat_W_i = A_i · phi(sum_j alpha_{i,j} B_j). Loss = MSE + small L1(alpha) + weight decay or spectral term on A,B. Apply entmax/sparsemax on logits z_i / T; anneal T toward target median support; epsilon‑prune tiny coeffs in‑loop for speed. Log MSE / support sizes / spectra to wandb.
4) Save A, B, alpha (Up & Gate) plus normalization metadata.

---

## Precomposition & cache (runtime)
- Compose once per expert: tilde_B_i = phi(sum_j alpha_{i,j} B_j) after epsilon‑prune. Store in compute dtype.
- LRU cache per layer (capacity S). Warm using a short calibration pass (~2k tokens). On miss: compose, insert, evict cold.
- Hot path: GEMM (r x d) then GEMM (p x r) per branch; Down stays dense.
- Export: HF checkpoint embedding cached tilde_B_i plus original A_i.

---

## Evaluation
- Reconstruction: Frobenius MSE per layer; effective rank; base usage histograms (support sizes, popularity).
- Runtime: tok/s (decode and prefill) at batch=1; cache hit‑rate vs S; VRAM vs baseline MoE.
- Quality: perplexity; small zero‑shot set.
- Ablations: m in {64,128,256}, r in {p, p/2}, target support in {8,12,16}.

---

## Milestones
M0: I/O + normalization + init on a handful of experts (verify shapes/MSE).  
M1: Full weight‑only fit; entmax anneal hits target support.  
M2: Precompose & LRU cache; HF patch smoke‑test.  
M3: End‑to‑end eval (PPL, tok/s, VRAM) vs MoE baseline; ablations.  
M4: Paper‑ready plots, README, and scripts.

---

## Unit tests (minimum)
- Deterministic compose with/without epsilon‑prune; support tracking under temp schedule.  
- Shapes/dtypes; HF patch round‑trip; cache LRU invariants.  
- Recon MSE sanity vs direct least squares on a tiny synthetic set.

