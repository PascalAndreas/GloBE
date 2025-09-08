# GloBE Training Focus - Clean Implementation

## Overview
This document describes the cleaned up, focused implementation for GloBE training. All legacy code has been removed to focus purely on getting the training loop working properly.

## Key Changes Made

### ✅ Cleaned Up Legacy Code
- **Removed from `fit_banks.py`:** SparseMixer imports, GloBEFFN creation functions, extract_expert_weights import
- **Focus:** Pure bank training via alternating minimization
- **No more:** Integration code, inference modules, entmax dependencies

### ✅ Enhanced Expert Weight Extraction
Updated `extract_expert_weights()` in `globe/data/naming_qwen15.py`:
```python
def extract_expert_weights(
    model_name_or_path: str,
    include_shared: bool = False,
    force_download: bool = False,
    max_experts: Optional[int] = None,
    single_layer: Optional[int] = None,          # NEW: Extract from single layer
    sample_method: str = "first_n"               # NEW: How to sample experts
) -> Dict[str, List[torch.Tensor]]:
```

**New Parameters:**
- `single_layer`: Extract experts only from specified layer (better structure exploitation)
- `sample_method`: 
  - `"first_n"`: Take first n experts (faster, good for debugging)
  - `"evenly_spaced"`: Take evenly distributed experts (more representative)

### ✅ Refactored AM Step
The `am_step()` function in `init_bank.py` is now minimal:
- **Removed:** All hyperparameter scheduling logic
- **Added:** Explicit parameters passed from `fit_banks.py`
- **Fixed:** Proper Jacobian blending for homotopy
- **Fixed:** Targets updated every step (proper dictionary learning)

### ✅ Minimal Test Script
New `test_minimal_training.py` focused purely on training:
- **No bank building** - just training metrics
- **No integration code** - pure AM loop testing
- **Family selection:** `--families up|gate|both`
- **Layer selection:** `--single-layer N`
- **Sampling control:** `--sample-method first_n|evenly_spaced`

## Usage Examples

### Single Layer UP Training (Recommended)
```bash
python3 test_minimal_training.py \
    --single-layer 12 \
    --families up \
    --steps 20 \
    --max-experts 60 \
    --no-wandb
```

### Both Families with Sampling
```bash
python3 test_minimal_training.py \
    --single-layer 12 \
    --families both \
    --steps 10 \
    --max-experts 20 \
    --sample-method evenly_spaced \
    --wandb-project my-training-test
```

### All Layers (Original Behavior)
```bash
python3 test_minimal_training.py \
    --families up \
    --steps 15 \
    --max-experts 128 \
    --sample-method first_n
```

## Training Loop Improvements

### Hyperparameter Scheduling
Now properly handled in `fit_banks.py`:
- **λ (lambda)**: Activation homotopy - linear (0) to SiLU (1)
- **β (beta)**: Target blending - warm-start (0) to weight-aware (1)
- **Jacobian**: Properly blended during homotopy transition

### Target Updates
- **Every step:** Targets are ALWAYS updated based on current parameters
- **No more:** Stale targets causing poor convergence
- **Better:** Proper dictionary learning behavior

### Jacobian Blending
For smooth homotopy transition:
```python
if activation_lambda == 0.0:
    J = I  # Identity for linear
elif activation_lambda < 1.0:
    J = (1-λ) * I + λ * φ'(x)  # Blended Jacobian
else:
    J = φ'(x)  # Full SiLU gradient
```

## What's Next

1. **Focus on Training Metrics**: Watch wandb logs for:
   - RelFrob convergence behavior
   - Sparsity patterns (should be ~40-60%)
   - Support size evolution (Seff)
   - Loss stability

2. **Hyperparameter Tuning**: Once training is stable:
   - Adjust λ/β scheduling
   - Tune regularization (λ_A, λ_B, λ_T)
   - Experiment with different seeding methods

3. **Single Layer Experiments**: 
   - Try different layers (early: 4-8, middle: 12-16, late: 20-24)
   - Compare structure learning across layers
   - Validate shared pattern hypothesis

4. **Integration Later**: Only after training loop is solid:
   - Build inference modules
   - Create complete GloBEFFN
   - Add evaluation metrics

## Key Insights

### Why Single Layer?
Experts within a single layer share more structure than experts across layers. This should lead to:
- Better dictionary learning
- More meaningful shared bases  
- Cleaner convergence behavior

### Why UP Only by Default?
UP projections are typically the most important for MoE performance:
- Larger parameter count
- More diverse activation patterns
- Easier to debug and validate

### Why No Bank Building Yet?
Focus on getting the core AM loop right first:
- Stable convergence
- Proper hyperparameter scheduling
- Good sparsity patterns
- Meaningful reconstruction quality

Once these are achieved, integration becomes straightforward.

## Files Structure
```
/Users/pascalandreas/Documents/repositories/GloBE/
├── test_minimal_training.py          # NEW: Focused training script
├── globe/
│   ├── data/naming_qwen15.py         # UPDATED: Enhanced extraction
│   ├── train/fit_banks.py            # CLEANED: Removed legacy code
│   └── init/init_bank.py             # REFACTORED: Minimal am_step
└── TRAINING_FOCUS.md                 # This file
```

The codebase is now clean, focused, and ready for serious training loop development.
