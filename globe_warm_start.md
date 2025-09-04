# GloBE — Warm‑Start & Bank Training Guidelines

Practical, MPS‑friendly ways to initialize a **single global bank** for routed experts, then train it with alternating minimization. Aimed at Qwen1.5‑MoE‑A2.7B scale on a 24‑GB Mac, but general.

---

## 0) Scope & notation
- Per routed expert up/gate: weight \(W_i \in \mathbb{R}^{p\times d}\); GloBE uses \(W_i \approx A_i\,B_i\), with \(A_i\in\mathbb{R}^{p\times r}\) and \(B_i \approx \sum_j \alpha_{i,j} B_j\), \(B_j\in\mathbb{R}^{r\times d}\).
- Typical routed dims for Qwen1.5‑MoE‑A2.7B: \(d\approx 2048\), \(p_r\approx 1408\). Choose \(r\in\{p, 0.75p, 0.5p\}\); bank size \(m\) relative to routed experts \(E_r\): \(m = \rho E_r,\ \rho\in\{1/8,1/6,1/4\}\).
- You will create **proxies** \(B_i^{(0)}\in\mathbb{R}^{r\times d}\), then learn a bank \(\{B_j\}_{j=1}^m\) and codes \(\alpha_i\), and finally refit \(A_i\).

---

## A) Row‑sampled Tall‑Skinny PCA (TS‑PCA) — **fastest on Mac (no SVD per expert)**
**Idea.** Build cheap right‑factor proxies by selecting \(r\) rows of each \(W_i\), then run PCA across experts using a tiny \(N\times N\) Gram matrix.

**Steps.**
1. **Row selector \(S\in\mathbb{R}^{r\times p}\)**
   - Simple: uniform stride or random without replacement.
   - Better: rank rows by energy \(e_k=\sum_i \|W_i[k,:]\|_2^2\) on a subset; pick top‑\(r\).
2. **Proxies**: \(B_i^{(0)} \leftarrow S\,W_i\in\mathbb{R}^{r\times d}\). (Pure gathers; MPS‑friendly.)
3. **Vectorize & normalize**: \(x_i=\mathrm{vec}(B_i^{(0)})/\|\cdot\|_2\). Stack columns into \(X\in\mathbb{R}^{(rd)\times N}\).
4. **Gram PCA**: compute \(C=X^\top X \in \mathbb{R}^{N\times N}\) (stream if needed); eigendecompose \(C=V\Lambda V^\top\).
5. **Atoms**: \(U = X V_{:,1:m}\,\Lambda_{1:m}^{-1/2}\). Reshape each \(u_j\) to \(r\times d\), then **unit‑Frobenius normalize** \(B_j\).
6. **Codes (init)**: NNLS (optionally simplex) per expert to fit \(B_i^{(0)} \approx \sum_j \alpha_{i,j} B_j\).
7. **Left factors (init)**: OLS per expert: \(A_i \leftarrow W_i\,\hat B_i^{+}\) with \(\hat B_i = \sum_j \alpha_{i,j} B_j\).

**When to use.** First pass, small memory, excellent on CPU+MPS; scales to all experts by streaming \(C\) and reconstructing \(U\) without materializing \(X\).

**Complexity.** Forming \(C\): \(O(N\,rd\,N_{blk})\) streamed; eig(\(C\)) is tiny; back‑projection \(X V\) is GEMM‑only.

---

## B) Left‑Gram PCA projector + TS‑PCA — **higher‑quality proxies, still MPS‑friendly**
**Idea.** Find a shared **left** subspace capturing most energy, project all experts into it: \(A_0\in\mathbb{R}^{p\times r}\), then do TS‑PCA over \(B_i^{(0)}=A_0^\top W_i\).

**Steps.**
1. **Left Gram**: \(G=\sum_i W_i W_i^\top\in\mathbb{R}^{p\times p}\) streamed; eigendecompose top‑\(r\) eigenvectors to get \(A_0\) (orthonormal columns). CPU‑OK for \(p\approx 1408\).
2. **Proxies**: \(B_i^{(0)}=A_0^\top W_i\) (single GEMM per expert; MPS‑accelerated).
3. **TS‑PCA** on the vectorized \(B_i^{(0)}\) as in A.5 to get atoms; NNLS codes; OLS lefts.

**When to use.** If row‑sampling quality is marginal; you still avoid per‑expert SVD and keep GEMMs.

**Note.** This **does not** fix \(A_i\) in the model; \(A_0\) is used only to build proxies.

---

## C) Randomized PCA (RPCA) on proxies — **best quality without per‑expert SVD**
**Idea.** Same proxies as B (\(B_i^{(0)}=A_0^\top W_i\)), but replace Gram PCA with **randomized range finder** for tall‑skinny \(X\).

**Steps.** Draw \(\Omega\in\mathbb{R}^{N\times (m+q)}\) (\(q\approx 8\)), form \(Y=X\Omega\) (stream), orthonormalize \(Q\), compute \(B=Q^\top X\) (small), SVD \(B=\tilde U\Sigma V^\top\), set \(U=Q\tilde U\) → atoms.

**When to use.** On a rented GPU or when \(N\) is large and you want top quality; still no per‑expert SVD.

---

## D) Spherical k‑means++ / Farthest‑first — **simplest, very fast**
**Idea.** Skip PCA entirely: cluster the normalized proxies \(x_i\) on the unit sphere.

**Steps.** k‑means++ (cosine distance) with \(k=m\) or farthest‑first (k‑center). Centers → atoms; NNLS codes; OLS lefts.

**When to use.** Quick prototypes; alternations (MOD + entmax codes) will refine the bank rapidly.

---

## E) Residual‑driven greedy (K‑SVD‑style seeding) — **lowest initial MSE**
**Idea.** Add atoms one‑by‑one targeting what current atoms fail to explain.

**Steps.** For \(j=1..m\): (i) code each target by NNLS (or lasso) using current atoms; (ii) form residuals \(R_i\); (iii) stack residuals, compute top singular vector (randomized SVD) → atom \(B_j\); normalize.

**When to use.** If you want the best warm‑start and can afford a bit more compute; still far cheaper than per‑expert SVD.

---

## F) Hybrid seeds — **balanced diversity & variance**
- Take first \(m/2\) atoms from PCA (A or B) and \(m/2\) from spherical k‑means++ or farthest‑first.
- Or run PCA, then do a short residual‑greedy pass to replace the worst‑explaining atoms.

---

## G) Codes & left factors (initialization)
- **Codes \(\alpha_i\)**: NNLS per expert; optionally project to the simplex to start; record support sizes.
- **Left factors \(A_i\)**: OLS per expert with pseudoinverse of \(\hat B_i\); add ridge if ill‑conditioned.

---

## H) Alternating training loop (weight‑only)
1. **Coding step**: maintain logits \(z_i\); \(\alpha_i = \text{entmax}(z_i/T)\) (or sparsemax); tiny L1 if desired; **ε‑prune** tiny entries.
2. **Dictionary step (MOD)**: vectorize targets and update bank
   \[ B \leftarrow X A^\top\,(A A^\top + \lambda I)^{-1}, \]
   reshape to \(r\times d\), **unit‑Frobenius‑normalize** atoms; rescale codes to keep product.
3. **Left refit (OLS)**: \(A_i \leftarrow W_i\,\hat B_i^{+}\).
4. **Temperature control**: every \(K\) steps, update \(T\) to hit a median support (e.g., 12):
   \[ T \leftarrow T\,\exp\!\big(\eta(\mathrm{median}(|\mathrm{supp}(\alpha_i)|)/s_{\text{tgt}}-1)\big). \]
5. **Early‑stop** when validation MSE plateaus and support distribution stabilizes.

---

## I) Numerical guards & stability
- **Atom normalization:** after each MOD step, \(B_j \leftarrow B_j/\|B_j\|_F\); rescale corresponding codes.
- **Coherence penalty (light):** discourage duplicates via \(\|\tilde B^\top \tilde B - I\|_F^2\) on unit‑norm atoms.
- **Ridge in MOD:** \(\lambda\approx 1e{-}4\) stabilizes \((AA^\top)^{-1}\).
- **Epsilon‑prune:** zero \(\alpha<\varepsilon\) before composing \(\tilde B_i\) to save time.
- **Dtype:** bf16/fp16 compute is fine; keep accumulations in fp32 for Gram/eig steps.

---

## J) Streaming recipes (large N, small RAM)
- **Gram PCA (TS‑PCA):** accumulate \(C\leftarrow C + X_{blk}^\top X_{blk}\) over blocks; eigendecompose \(C\); reconstruct atoms via a second pass: \(u_j=\sum_i V_{ij} x_i / \sqrt{\lambda_j}\).
- **Proxy formation:** compute \(B_i^{(0)}\) on the fly (row‑sample or \(A_0^\top W_i\)), normalize, stream into \(C\) and discard.
- **Codes (init):** solve NNLS in mini‑batches; cache only supports+values.

---

## K) What to log during warm‑start
- **Explained variance** of PCA seeds; init **reconstruction MSE** on proxies and on full \(W_i\) (via OLS lefts).
- **Support sizes** of NNLS codes; histogram of base popularity.
- **Atom norms** (post‑normalization); **coherence** (\(\|\tilde B^\top \tilde B - I\|_F\)).
- **Timing** per stage; peak RAM.

---

## L) When to switch approaches
- If TS‑PCA seeds yield high MSE → try **Left‑Gram PCA** (B).
- If you need the best init before long runs → try **Residual‑greedy** (E) or **RPCA** (C).
- If you rent a GPU → C (RPCA) is ideal; otherwise stay with A or B on Mac.

---

## M) Quick defaults
- Row‑energy selection for \(S\); TS‑PCA with \(m\in\{64,128\}\).
- NNLS (+simplex) codes; OLS lefts with small ridge.
- Alternations: 3–6 outer loops with MOD (ridge), entmax codes (target median support 12), atom normalization after each update.
- Precompose `tilde_B_i` with ε‑prune and cache in compute dtype.

