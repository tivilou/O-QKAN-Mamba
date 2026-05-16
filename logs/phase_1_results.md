# Phase 1 Results: Basic Spectral-QKAN-Mamba

**Date:** 2026-05-15
**Status:** PASS

---

## Test Results

```
tests/experimental/test_spectral_qkan_gate.py
  test_gate_shape                      PASSED
  test_gate_values_in_range            PASSED
  test_gate_backward                   PASSED
  test_qkan_params_receive_gradient    PASSED

tests/experimental/test_qmamba_block.py
  test_block_shape                     PASSED
  test_block_with_fallback_mixer       PASSED
  test_block_with_mamba                PASSED
  test_block_backward                  PASSED

tests/experimental/test_tiny_qmamba_lm.py
  test_forward_shape                   PASSED
  test_forward_with_targets            PASSED
  test_one_training_step               PASSED

Total: 11 passed, 0 failed
```

## Training Run (Tiny Shakespeare, 200 steps)

| Metric | Value |
|--------|-------|
| Initial loss | 4.3322 |
| Final loss | 1.8035 |
| Converged | Yes |
| Steps | 200 |
| Avg step time | 47.3 ms |
| Peak GPU memory | 265 MB |
| Parameters | 151,289 |
| Mamba used | Yes |
| Device | Tesla V100S-PCIE-32GB |

## Model Configuration

```
vocab_size: 65 (char-level)
d_model: 64
n_layers: 4
latent_dim: 16
reps: 2
solver: exact
batch_size: 16
block_size: 64
lr: 3e-3
```

## Loss Curve (sampled)

```
step   0: 4.3322
step  20: 2.7022
step  40: 2.3888
step  60: 2.2015
step  80: 2.2013
step 100: 1.9050
step 120: 1.8978
step 140: 1.9062
step 160: 1.8579
step 180: 1.8062
step 199: 1.8035
```

## Architecture Notes

- SpectralQKANGate uses HQKAN bottleneck: Linear(64→16) → QKAN([16,16]) → Linear(16→64) → sigmoid
- QMambaBlock: y = x + gate(x) * mamba(x) with LayerNorm pre-norm
- QKAN config: group=3, preact_trainable=True, postact_weight/bias_trainable=True, ba_trainable=True
- First step is slow (1021ms) due to CUDA kernel compilation, subsequent steps ~40ms
