# GloBE AM Loop Refactoring Summary

## Overview
This document summarizes the refactoring changes made to improve the AM (Alternating Minimization) loop implementation and simplify the training process.

## Key Changes

### 1. Minimal `am_step()` Function
**File:** `globe/init/init_bank.py`

The `am_step()` function has been refactored to be minimal and focused only on the core AM logic:
- **Removed:** Hyperparameter scheduling logic
- **Added:** Explicit hyperparameter arguments passed from `fit_banks.py`
- **New parameters:**
  - `activation_lambda`: Current λ value for activation homotopy
  - `route_beta`: Current β value for target blending  
  - `use_weight_aware_targets`: Boolean flag for Route-W vs Route-L

### 2. Centralized Hyperparameter Scheduling
**File:** `globe/train/fit_banks.py`

All hyperparameter scheduling is now handled in `fit_banks.py`:
- Determines current λ and β values based on step and adaptive scheduling
- Decides when to switch to weight-aware targets (Route-W)
- Passes these values explicitly to `am_step()`

### 3. Proper Jacobian Blending
**File:** `globe/init/init_bank.py`

The Jacobian is now properly blended for the homotopy transition:
```python
if activation_lambda == 0.0:
    J = I  # Identity for linear
elif activation_lambda < 1.0:
    J = (1-λ) * I + λ * φ'(x)  # Blended Jacobian
else:
    J = φ'(x)  # Full SiLU gradient
```

This ensures smooth gradient flow during the transition from linear to nonlinear activation.

### 4. Target Updates Every Step
Dictionary learning targets are now **always** updated every step based on current parameters:
- Warm-start targets (S_L) computed from initialization
- Weight-aware targets (S_W) computed when β > 0 or use_weight_aware=True
- Proper blending: S = (1-β)*S_L + β*S_W

### 5. Single-Layer Expert Extraction
**File:** `test_qwen_training.py`

New function `extract_single_layer_experts()`:
- Extracts experts from a single layer (default: layer 12)
- Exploits shared structure within a layer
- More realistic for understanding dictionary learning

### 6. UP-Only Training
**File:** `test_qwen_training.py`

Simplified to train only the UP family:
- Removed gate family training
- Cleaner debugging and validation
- Focus on core algorithm behavior

## Understanding λ and β

### λ (Lambda) - Activation Homotopy
Controls the interpolation between linear and nonlinear activation:
- **λ = 0**: Linear activation φ(x) = x
- **0 < λ < 1**: Blended activation φ_λ(x) = (1-λ)·x + λ·SiLU(x)
- **λ = 1**: Full SiLU activation φ(x) = SiLU(x)

### β (Beta) - Target Blending
Controls the interpolation between warm-start and weight-aware targets:
- **β = 0**: Use warm-start targets from initialization
- **0 < β < 1**: Blend targets S = (1-β)·S_warm + β·S_weight_aware
- **β = 1**: Use pure weight-aware targets

### Why Separate Parameters?
These parameters control different aspects of the optimization:

1. **λ controls the output space** - the shape of the activation function
2. **β controls the optimization objective** - what targets we're fitting to

Separating them allows for sophisticated optimization strategies:
- Start with λ=0, β=0 for stable linear dictionary learning
- After initial steps, use λ=0, β>0 to learn better linear dictionaries
- Gradually increase λ to transition to nonlinear activation
- Adjust β independently to balance stability vs optimization quality

## Usage Examples

### Single-Layer Training
```bash
python test_qwen_training.py \
    --use-single-layer \
    --layer-idx 12 \
    --steps 20 \
    --rank-ratio 0.75 \
    --basis-ratio 0.5
```

### All-Layer Training (Original)
```bash
python test_qwen_training.py \
    --steps 20 \
    --rank-ratio 0.75 \
    --basis-ratio 0.5 \
    --max-experts 128
```

## Benefits of Refactoring

1. **Cleaner Code Structure**: Separation of concerns between AM logic and scheduling
2. **Better Testability**: Can test AM step independently of scheduling
3. **Proper Mathematical Foundation**: Correctly blended Jacobian for homotopy
4. **Improved Convergence**: Targets updated every step, proper dictionary learning
5. **Exploits Structure**: Single-layer extraction finds shared patterns better
6. **Simplified Debugging**: UP-only training reduces complexity

## Next Steps

1. Fine-tune the adaptive scheduling parameters
2. Experiment with different layer choices for single-layer extraction
3. Add visualization of the optimization trajectory
4. Consider alternative target blending strategies
5. Profile performance with the refactored code

## Testing

Run the test script to see the improvements:
```bash
./test_single_layer.py
```

This will demonstrate:
- Single-layer expert extraction
- UP-only training
- Proper hyperparameter scheduling
- Improved convergence behavior
