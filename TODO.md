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

## Integration Issues

### 7. Config to Module Flow
- **Issue**: Hydra configs not connected to actual module initialization
- **Fix needed**:
  - Create config dataclasses matching YAML structure
  - Add config loading utilities
  - Ensure all modules can be initialized from config

### 8. Weight Loading and Tensor Naming
- **Issue**: Tensor naming patterns may not match actual Qwen1.5-MoE-A2.7B
- **Files affected**: `globe/data/naming_qwen15.py`
- **Fix needed**:
  - Validate against actual model weights
  - Add error handling for missing/mismatched tensors
  - Support multiple model versions

### 9. Cache-Model Integration
- **Issue**: Cache integration with actual forward pass incomplete
- **Files affected**: `globe/modules/ffn_globe.py`
- **Fix needed**:
  - Proper integration with HuggingFace forward hooks
  - Cache warmup implementation
  - Performance validation

## Testing and Validation

### 10. Unit Tests
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

### 11. Integration Tests
- **Create**: `tests/test_integration.py`
- **Test scenarios**:
  - Load small model → extract weights → train → reconstruct
  - Export → import → inference
  - Cache hit/miss behavior
  - Device compatibility (CPU, MPS, CUDA)

## Performance and Optimization

### 12. Memory Management
- **Issue**: No memory profiling for MPS devices
- **Fix needed**:
  - Add MPS memory tracking
  - Implement gradient checkpointing option
  - Add memory-efficient training mode

### 13. Mixed Precision Training
- **Not implemented**: Automatic mixed precision (AMP) support
- **Add**:
  - GradScaler integration for fp16
  - Native bf16 training support
  - MPS-compatible mixed precision

## Documentation

### 14. Usage Examples
- **Create**: `examples/` directory with:
  - `train_qwen_moe.py` - End-to-end training example
  - `benchmark_model.py` - Performance comparison
  - `export_and_deploy.py` - Export workflow

### 15. API Documentation
- **Add**: Docstrings for all public methods
- **Create**: `docs/API.md` with module documentation
- **Include**: Architecture diagrams

## Immediate Priority Order

1. **Fix device/dtype issues** (Critical for macOS development)
2. **Create `fit_banks.py`** (Core functionality)
3. **Validate tensor naming** (Required for model loading)
4. **Add basic integration test** (Verify end-to-end flow)
5. **Fix MPS compatibility** (For local development)

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
