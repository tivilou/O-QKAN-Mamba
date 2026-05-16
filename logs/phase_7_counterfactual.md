# Phase 7: Counterfactual Engine Results

## Speed Comparison

| Method | Time | Forward Passes |
|--------|------|----------------|
| Ours | 4.8ms | 1 |
| SHAP | 319.5ms | 100 |
| IG | 758.0ms | 50 |

## Clinical Queries

- **What if lactate were normal?**: delta=+0.0000 (no_effect)
- **What if no infection signs?**: delta=+0.0001 (no_effect)
- **What if hemodynamics stable?**: delta=+0.0000 (no_effect)
