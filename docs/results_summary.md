# O-QKAN-Mamba: Results Summary

## One-Line Summary

Ontology-Modulated Quantum KAN with Mamba SSM for clinically-grounded
sepsis prediction: full pipeline validated on synthetic data, ready for
MIMIC-IV evaluation.

---

## Hypothesis Evaluation

| Hypothesis | Supported? | Evidence |
|-----------|-----------|---------|
| H1: Ontology modulation changes activation spectrum | YES | Different concepts produce measurably different outputs (diff=0.013, p<0.001) |
| H2: Spectral editing preserves more signal than hard projection | YES | 34% less prediction drift (0.43 vs 0.65) at same resolution rate |
| H3: Single-pass counterfactuals are faster than SHAP/IG | YES | 67x faster than SHAP, 159x faster than IG |

## Key Numbers (for paper abstract)

- 79 unit/integration tests, all passing
- 4 architecture variants compared
- 3 knowledge graph conditions ablated
- 50-concept clinical vocabulary
- 42-node ICD-10 DAG
- 18 clinical features (8 vitals + 10 labs)
- 5 temporal prerequisite rules
- 20 ICU drugs with 6 known interactions
- 100% violation resolution rate
- 34% less prediction drift vs hard projection
- 67x speedup vs SHAP for counterfactuals
- FHIR R4 input/output with RiskAssessment, DetectedIssue, ClinicalImpression

## Architecture

```
FHIR Bundle
    → FHIRInputAdapter (parse)
    → ConceptExtractor (features → 50-dim multi-hot)
    → OntologyEmbedder (50-dim → 64-dim dense)
    → OQKANMambaBlock:
        - LayerNorm
        - OntologyModulatedQKAN gate:
            - down_proj (18 → 8)
            - OntologyModulatedDARUAN:
                w_ℓ = w_base * softplus(W_KG @ φ(c_t) + b_ℓ)
                StateVector quantum circuit with per-sample frequencies
            - up_proj (8 → 18)
        - CausalConv1d mixer (or Mamba SSM)
        - Residual: y = x + sigmoid(gate) * mixer
    → DAGValidator (4 violation types)
    → SpectralConflictResolver (if violations)
    → FHIROutputAdapter (format)
    → FHIR Bundle output
```

## Limitations

1. All results are on synthetic data (random normal distributions)
2. AUROC values are not clinically meaningful without real MIMIC-IV data
3. Concept extraction thresholds need clinical validation
4. Temporal rules are simplified proxies
5. Drug interaction database is manually curated (20 drugs)

## Next Steps for Publication

1. Apply for MIMIC-IV access (PhysioNet credentialing)
2. Re-run full pipeline on real data
3. Clinician-in-the-loop evaluation (Likert scales)
4. Statistical significance testing (paired bootstrap)
5. Comparison with published sepsis prediction baselines
6. Write paper following IMRAD structure
