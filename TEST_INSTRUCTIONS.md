# GloBE Bank Training Test Instructions

This guide helps you test the GloBE bank training workflow on Qwen1.5-MoE-A2.7B.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements_test.txt
   ```

2. **Run the basic test:**
   ```bash
   python test_qwen_training.py
   ```

3. **Run with custom parameters:**
   ```bash
   python test_qwen_training.py --steps 20 --rank 128 --num-bases 64 --device cuda
   ```

## What the Test Does

1. **Downloads Qwen1.5-MoE-A2.7B** - Automatically downloads the model from HuggingFace
2. **Analyzes Model Structure** - Counts routed vs shared experts, checks dimensions
3. **Extracts Routed Experts Only** - Correctly filters out shared experts (as per TODO #7)
4. **Trains Global Basis Banks** - Uses alternating minimization with z-score normalization
5. **Tests Reconstruction Quality** - Validates that W_i ≈ A_i @ (Σ_j α_{i,j} B_j)
6. **Creates DualGloBEBank** - Tests integration with the module architecture
7. **Saves All Results** - Outputs ready for inference or further analysis

## Key Features Tested

✅ **Shared Expert Handling** - Only routed experts are decomposed  
✅ **Z-score Normalization** - Per-family standardization as per workflow  
✅ **Alternating Minimization** - Complete AM loop with temperature scheduling  
✅ **Architecture Integration** - Creates DualGloBEBank compatible with GloBEFFN  
✅ **Device Compatibility** - Auto-detects CUDA/MPS/CPU  

## Expected Output

```
🌍 GloBE Bank Training Test
==================================================
🔍 Analyzing model: Qwen/Qwen1.5-MoE-A2.7B
✅ Model loaded successfully!
📊 Model Analysis:
   - Total routed experts: X
   - Layers with shared experts: Y
   - Expert counts per layer: {...}
   
🧪 Testing expert weight extraction...
✅ Extracted N routed Up experts
✅ Extracted N routed Gate experts
🔒 Shared experts excluded from training

🚀 Starting minimal bank training test...
🖥️  Using [device] device
🔄 Training banks...
✅ Training completed!

🔍 Testing reconstruction quality...
📊 UP reconstruction:
   - MSE: 0.00XXXX
   - Relative error: 0.0XXX
   
🔧 Testing DualGloBEBank creation...
✅ DualGloBEBank created successfully!

💾 Saving results to test_output/...
✅ Saved up_bank.pt
✅ Saved gate_bank.pt
✅ Saved globe_banks_combined.pt
✅ Saved dual_globe_bank.pt

🎉 All tests completed successfully!
```

## Command Line Options

- `--model`: Model name/path (default: "Qwen/Qwen1.5-MoE-A2.7B")
- `--steps`: Training steps (default: 10, try 25-50 for better quality)  
- `--rank`: SVD rank (default: 64, try 128-256 for larger models)
- `--num-bases`: Number of basis vectors (default: 32, try 64-128)
- `--device`: Device (auto/cpu/cuda/mps, default: auto)
- `--output-dir`: Output directory (default: test_output)
- `--skip-download`: Skip model analysis (for repeated runs)

## Troubleshooting

**Memory Issues**: Try smaller `--rank` and `--num-bases`, or use `--device cpu`

**Download Issues**: Ensure you have HuggingFace access and sufficient disk space

**MPS Issues**: Some operations may fall back to CPU on Apple Silicon

**Import Errors**: Install missing dependencies from `requirements_test.txt`

## What's Next

After successful testing:

1. Use the saved `dual_globe_bank.pt` for inference
2. Integrate with `GloBEFFN` for full model replacement
3. Run performance benchmarks vs original MoE
4. Export to HuggingFace format for deployment

## File Outputs

- `up_bank.pt` / `gate_bank.pt` - Individual family banks
- `globe_banks_combined.pt` - Complete training results + normalizer
- `dual_globe_bank.pt` - Ready-to-use DualGloBEBank state dict  
- `config.json` - Training configuration for reproducibility
