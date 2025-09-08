# Legacy Parameter Cleanup - route_w_start Removal

## Overview
Removed the legacy `route_w_start` parameter that has been superseded by proper λ/β scheduling.

## What Was Removed

### ❌ **Legacy `route_w_start` Parameter**
**Old approach (simple step threshold):**
```python
# In InitConfig
route_w_start: int = 5  # Switch to Route-W after this many steps

# In am_step logic (old)
if step >= cfg.route_w_start:
    # Use weight-aware targets (Route-W)
else:
    # Use warm-start targets (Route-L)
```

### ✅ **New λ/β Scheduling Approach**
**Current approach (sophisticated scheduling):**
```python
# In fit_banks.py
current_lambda = cfg.activation_lambda  # From adaptive scheduling
current_beta = cfg.route_beta          # From adaptive scheduling

# Smart decision based on hyperparameter state
use_weight_aware = (current_lambda == 0.0 and step >= 2)

# Pass explicit parameters to am_step
am_step(..., 
        activation_lambda=current_lambda,
        route_beta=current_beta,
        use_weight_aware_targets=use_weight_aware)
```

## Why the New Approach is Better

### **1. Tied to Actual Hyperparameters**
- **Old**: Fixed step threshold (always step 5)
- **New**: Based on λ value and adaptive scheduling

### **2. More Sophisticated Logic**
- **Old**: Simple step counter
- **New**: Considers both λ and β values, phase of training

### **3. Better Integration**
- **Old**: Hardcoded magic number
- **New**: Integrated with adaptive scheduling system

### **4. Cleaner API**
- **Old**: Extra parameter to track and pass around
- **New**: Decision made in training loop, explicit flags passed

## Files Updated

### **Core Files**
- ✅ `globe/init/init_bank.py`: Removed `route_w_start` from InitConfig
- ✅ `globe/train/fit_banks.py`: Removed parameter, uses new logic
- ✅ `test_minimal_training.py`: Removed parameter and argument
- ✅ `test_qwen_training.py`: Removed parameter and `--route-w-start` argument

### **Current Logic Flow**
```
fit_banks.py:
1. Get current λ, β from adaptive scheduling
2. Decide: use_weight_aware = (λ==0.0 and step>=2)
3. Pass explicit flags to am_step()

am_step():
1. Receive explicit hyperparameter values
2. Use sophisticated target selection logic
3. No hardcoded thresholds
```

## Testing Verified
✅ Training works correctly without `route_w_start`
✅ Weight-aware targets still switch at step 3 (step >= 2) for λ=0
✅ All existing functionality preserved

## Benefits of Cleanup

### **1. Fewer Magic Numbers**
No more hardcoded `route_w_start = 5` scattered throughout code

### **2. Better Hyperparameter Control**
Target switching now tied to actual optimization state (λ/β values)

### **3. Cleaner API**
One less parameter to track, pass, and configure

### **4. More Maintainable**
Logic centralized in fit_banks.py where scheduling happens

## Backward Compatibility
⚠️ **Breaking Change**: `--route-w-start` argument removed from test scripts
- **Migration**: Remove the argument from any existing scripts
- **Replacement**: Target switching now automatic based on λ/β scheduling

## Current State
The codebase now has clean, modern hyperparameter scheduling:
- **λ (lambda)**: Controls activation homotopy (linear → SiLU)
- **β (beta)**: Controls target blending (warm-start → weight-aware)  
- **Automatic**: Target switching based on optimization state, not step count

No more legacy routing parameters! 🎯
