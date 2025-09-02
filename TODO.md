# GloBE Implementation TODO List

## Critical Issues to Fix

### 1. Device Configuration Propagation ✅
- Device from Hydra config now flows into evaluation utilities.
- `benchmark_globe_model` and `ReconstructionEvaluator` accept a `device` parameter and perform automatic CUDA/MPS/CPU selection.

### 2. Precision/Dtype Mismatch ✅
- Default precision switched to "bf16" in benchmarking utilities.
- Added automatic dtype detection from model parameters.

### 3. MPS Device Compatibility ✅
- Replaced CUDA-only checks with device-agnostic helpers.
- Added MPS-specific cache clearing, memory tracking, and synchronization.

## Missing Implementation Components

### 4. Main Training Entry Point
- **File to create**: `globe/train/fit_banks.py`
- **Requirements**:
  - Hydra CLI integration
  - Load model weights from HuggingFace
  - Extract expert weights by family
  - Initialize and train global banks
  - Save checkpoints and final weights
  - Wandb integration

### 5. CLI Interfaces
- **Files needing CLI**:
  - `globe/infer/export_hf.py` - Add `__main__` block with argparse/Hydra
  - `globe/infer/patch_hf.py` - Add CLI for standalone usage
- **Requirements**:
  - Integrate with Hydra configuration system
  - Add proper argument parsing
  - Error handling and validation

### 6. Model Detection and Support
- **Issue**: Only Qwen2MoeSparseMoeBlock is detected in `patch_hf.py`
- **Fix needed**:
  - Add support for Qwen1.5-MoE-A2.7B specific architecture
  - Create model registry/detection system
  - Handle different MoE architectures gracefully

### 7. Shared Expert Handling
- **Issue**: Current implementation assumes all experts are routed experts, but Qwen1.5-MoE-A2.7B has shared experts
- **Requirements**: 
  - Shared experts are always used and should be left as-is (no decomposition)
  - Only routed experts should use the GloBE basis reconstruction method
  - Need to distinguish between shared and routed experts during weight extraction
- **Files affected**:
  - `globe/data/naming_qwen15.py` - Expert weight extraction logic
  - `globe/modules/ffn_globe.py` - Forward pass must handle shared experts separately
  - `globe/infer/patch_hf.py` - Model patching must preserve shared expert behavior
- **Implementation details**:
  - Shared experts bypass the A_i @ φ(Σ_j α_{i,j} B_j) reconstruction entirely
  - Shared expert weights remain dense and unmodified
  - Forward pass combines shared expert output with routed expert output
  - Cache and precomposition logic should skip shared experts

### 8. Bank Initialization Overhaul
- **Issue**: Current `init_bank.py` is placeholder code that doesn't follow the refined training workflow
- **Reference**: Refined plan in `globe_bank_initialization_training_workflow.md` developed with GPT-5
- **Critical missing components**:
  - **Z-score normalization**: Per-family standardization across experts with mean/scale recording
  - **Proper warm start procedure**: Truncated SVD per expert with centroids-based bank initialization
  - **Alternating minimization**: Coding step (entmax/sparsemax) + Dictionary step (MOD) + Adapter refit (OLS)
  - **Temperature annealing**: Dynamic temperature control to target median support size
  - **Comprehensive metrics**: Energy captured, effective rank, support sizes, atom coherence, etc.
- **Files requiring complete overhaul**:
  - `globe/init/init_bank.py` - Replace placeholder with proper alternating minimization
  - `globe/init/zscore.py` - Integrate with bank initialization workflow
- **Key workflow steps missing**:
  1. Per-family Z-score normalization with scale recording
  2. Truncated SVD warm start: W_i ≈ (U_r Σ_r)(V_r^T) → A_i^(0), B_i^(0)
  3. Initial bank from centroids + NNLS for initial codes α_i^(0)
  4. Alternating minimization loop with proper MOD updates
  5. Temperature control with median support targeting
  6. Normalization constant folding and export preparation

## Integration Issues

### 9. Config to Module Flow
- **Issue**: Hydra configs not connected to actual module initialization
- **Fix needed**:
  - Create config dataclasses matching YAML structure
  - Add config loading utilities
  - Ensure all modules can be initialized from config

### 10. Weight Loading and Tensor Naming
- **Issue**: Tensor naming patterns may not match actual Qwen1.5-MoE-A2.7B
- **Files affected**: `globe/data/naming_qwen15.py`
- **Fix needed**:
  - Validate against actual model weights
  - Add error handling for missing/mismatched tensors
  - Support multiple model versions

### 11. Cache-Model Integration
- **Issue**: Cache integration with actual forward pass incomplete
- **Files affected**: `globe/modules/ffn_globe.py`
- **Fix needed**:
  - Proper integration with HuggingFace forward hooks
  - Cache warmup implementation
  - Performance validation

## Testing and Validation

### 12. Unit Tests
- **Files to create**:
  - `tests/test_globe_bank.py`
  - `tests/test_sparse_mixer.py`
  - `tests/test_reconstruction.py`
  - `tests/test_cache.py`
- **Coverage needed**:
  - Deterministic composition
  - Support tracking
  - Shape/dtype validation
  - Cache LRU invariants

### 13. Integration Tests
- **Create**: `tests/test_integration.py`
- **Test scenarios**:
  - Load small model → extract weights → train → reconstruct
  - Export → import → inference
  - Cache hit/miss behavior
  - Device compatibility (CPU, MPS, CUDA)

## Performance and Optimization

### 14. Memory Management
- **Issue**: No memory profiling for MPS devices
- **Fix needed**:
  - Add MPS memory tracking
  - Implement gradient checkpointing option
  - Add memory-efficient training mode

### 15. Mixed Precision Training
- **Not implemented**: Automatic mixed precision (AMP) support
- **Add**:
  - GradScaler integration for fp16
  - Native bf16 training support
  - MPS-compatible mixed precision

### 16. Stats Accumulation Performance Issue
- **Issue**: Stats accumulation for loop in GloBEFFN forward method creates performance bottleneck in critical path
- **Files affected**: `globe/modules/ffn_globe.py` - Lines 160-165
- **Details**: 
  - Dictionary updates and string formatting in tight loop during inference
  - Per-expert stats collection blocks parallel processing
  - Stats should be optional or moved out of critical path
- **Fix needed**:
  - Add `collect_stats` flag to disable stats collection during inference
  - Batch stats collection outside the expert processing loop
  - Consider async stats collection for training mode
  - Optimize string operations and dictionary access patterns

## Documentation

### 17. Usage Examples
- **Create**: `examples/` directory with:
  - `train_qwen_moe.py` - End-to-end training example
  - `benchmark_model.py` - Performance comparison
  - `export_and_deploy.py` - Export workflow

### 18. API Documentation
- **Add**: Docstrings for all public methods
- **Create**: `docs/API.md` with module documentation
- **Include**: Architecture diagrams

## Immediate Priority Order

1. **Fix device/dtype issues** (Critical for macOS development)
2. **Overhaul bank initialization** (Core algorithm implementation - current code is placeholder)
3. **Implement shared expert handling** (Required for Qwen1.5-MoE-A2.7B prototype)
4. **Create `fit_banks.py`** (Core functionality)
5. **Validate tensor naming** (Required for model loading)
6. **Add basic integration test** (Verify end-to-end flow)
7. **Fix MPS compatibility** (For local development)

## Notes

- Consider using `torch.backends.mps.is_available()` for MPS detection
- May need to handle dtype limitations on MPS (no float64 support)
- Entmax package may have compatibility issues with MPS - needs testing
- SafeTensors loading on MPS may require special handling

## Dependencies to Verify

- `entmax` package MPS compatibility
- `scikit-learn` Apple Silicon optimization
- `wandb` offline mode for development
- HuggingFace `accelerate` library for device management
