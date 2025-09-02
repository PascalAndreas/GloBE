# GloBE Implementation TODO List

## Critical Issues to Fix

### 1. Device Configuration Propagation
- **Issue**: Device configuration from Hydra config (`default.yaml`) not properly propagated to evaluation functions
- **Files affected**: 
  - `globe/eval/latency_mem.py` - Line 484: `benchmark_globe_model()` doesn't pass device parameter
  - `globe/eval/recon_metrics.py` - Line 27: Hardcoded device fallback without config integration
- **Fix needed**: 
  - Pass device from Hydra config through to evaluation classes
  - Update convenience functions to accept device parameter
  - Consider MPS (Metal Performance Shaders) compatibility for macOS

### 2. Precision/Dtype Mismatch
- **Issue**: Default precision is "fp16" but Qwen1.5-MoE-A2.7B uses bfloat16
- **Files affected**:
  - `globe/eval/latency_mem.py` - Line 453: Default precision="fp16"
  - `globe/config/cache/lru.yaml` - Correctly uses "bf16"
  - `globe/config/export/hf.yaml` - Correctly uses "bf16"
- **Fix needed**: 
  - Change default precision to "bf16" in benchmark functions
  - Ensure consistency across all modules
  - Add model-specific dtype detection

### 3. MPS Device Compatibility
- **Issue**: Code uses `torch.cuda` checks which won't work on macOS with MPS
- **Files affected**:
  - `globe/eval/latency_mem.py` - Lines 187, 258, 268-270, 284-285: CUDA-specific calls
  - Multiple modules default to CUDA checks
- **Fix needed**:
  - Replace `torch.cuda.is_available()` with device-agnostic checks
  - Handle MPS-specific memory management
  - Add MPS synchronization where needed

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

### 8. Bank Initialization Overhaul ✅
Implemented the full initialization workflow in `globe/init/init_bank.py`
and `globe/init/zscore.py` following
`globe_bank_initialization_training_workflow.md`. The new pipeline adds
per-family Z-score normalization, truncated SVD warm starts, centroids-
based bank seeding with NNLS codes, an alternating minimization loop,
temperature annealing, and metric tracking.

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
