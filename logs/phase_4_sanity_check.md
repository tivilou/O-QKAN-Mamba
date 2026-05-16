# Phase 4 Results: Ontology-Modulated DARUAN

**Date:** 2026-05-15
**Status:** PASS

---

## Test Results

```
tests/experimental/test_ontology_modulated_daruan.py
  test_shape                              PASSED
  test_modulation_changes_output          PASSED
  test_zero_concept_recovers_base         PASSED
  test_gradient_flows_through_W_KG        PASSED
  test_gradient_flows_through_theta       PASSED
  test_gradient_flows_through_w_base      PASSED

tests/experimental/test_o_qkan_mamba_block.py
  test_block_forward_shape                PASSED
  test_block_with_seq_concepts            PASSED
  test_different_concepts_different_outputs PASSED
  test_one_training_step                  PASSED
  test_gradient_flows_to_W_KG             PASSED

Total: 11 passed, 0 failed
```

## Critical Soundness Checks

| Check | Status | Detail |
|-------|--------|--------|
| Zero concept recovers base | PASS | Identical outputs with c=0 |
| Different concepts → different outputs | PASS | diff=0.013 (>> 1e-4 threshold) |
| W_KG receives non-zero gradients | PASS | grad.abs().sum() > 0 |

## Modulation Verification

```
w_base: [1.0, 2.0, 4.0, 8.0]  (geometric series 2^ℓ)

Modulated weights per layer:
Layer    Zero         Healthy      Septic
  L0     0.6931      0.6901      0.6866
  L1     1.3863      1.4389      1.3777
  L2     2.7726      2.6878      2.8461
  L3     5.5452      5.6110      5.4048

Output differences:
  |healthy - septic|: 0.013180
  |healthy - zero|:   0.005426
  |septic - zero|:    0.008243
```

Different clinical contexts produce measurably different frequency
modulations and outputs, confirming Hypothesis H1.

## Sanity Check Training (Synthetic MIMIC-IV)

| Model | Final Loss | Val AUROC |
|-------|-----------|-----------|
| With ontology modulation | 0.8520 | 0.33 |
| Without ontology modulation | 0.7943 | 0.62 |

Note: AUROC values are not meaningful on synthetic random data.
The key result is that both models train without errors and loss decreases.
Real clinical evaluation requires MIMIC-IV access.

## Architecture Summary

```
OntologyModulatedDARUAN:
  - theta: (dim, reps+1, 2) — variational circuit parameters
  - w_base: (reps,) — geometric frequency initialization
  - W_KG: (reps, d_ontology) — ontology modulation matrix
  - b: (reps,) — modulation bias
  - postact_weight/bias: (dim,) — output scaling

Modulation formula:
  w_ℓ = w_base[ℓ] * softplus(W_KG[ℓ] @ φ(c_t) + b[ℓ])

Key design: uses DARUAN's StateVector/TorchGates directly for
per-sample modulated data re-uploading. No existing QKAN APIs modified.
```
