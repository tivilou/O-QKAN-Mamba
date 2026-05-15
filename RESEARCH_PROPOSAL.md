# Research Proposal: O-QKAN-Mamba for Clinical Reasoning Agents

## 1. Problem Statement

Current clinical AI faces three gaps:
1. Black-box deep models lack ontology-grounded reasoning
2. Symbolic systems lack adaptive learning from EHR signals
3. Existing neuro-symbolic methods do not exploit spectral interpretability

## 2. Proposed Method

### 2.1 Ontology-Modulated DARUAN

Standard DARUAN uses static re-uploading weights w_ℓ.
We propose:

  w_ℓ(t, x_t, c_t) = w_ℓ^base · σ(W_KG · φ(c_t) + b_ℓ)

This makes the activation **spectrum** dependent on clinical context.

**Hypothesis H1**: Different clinical contexts (e.g., sepsis suspicion vs.
chronic disease management) require different activation frequencies for
the same physiological signal.

### 2.2 Spectral Conflict Resolver

When the predicted output y_pred violates an ontology rule r:
1. Compute spectral attribution: which DARUAN frequencies contributed most to y_pred?
2. Identify offending frequencies via gradient-based attribution
3. Apply frequency-domain minimal-edit: dampen offending frequencies until r is satisfied

**Hypothesis H2**: Frequency-domain editing preserves more useful neural signal
than hard projection onto the rule-compliant subspace.

### 2.3 Counterfactual via Ontology Perturbation

For query "what if drug X is removed":
1. Perturb c_t by zeroing out the SNOMED concept for drug X
2. Recompute w_ℓ → new spectrum → new prediction
3. Output Δy as counterfactual estimate

**Hypothesis H3**: This is orders of magnitude faster than retraining or
sampling-based counterfactuals (e.g., CF-SHAP).

## 3. Tasks and Datasets

| Task                  | Dataset           | Metric                  |
|-----------------------|-------------------|-------------------------|
| Sepsis early warning  | MIMIC-IV          | AUROC, AUPRC, lead time |
| ICU mortality         | MIMIC-IV          | AUROC, calibration      |
| 30-day readmission    | MIMIC-IV          | AUROC, AUPRC            |

## 4. Knowledge Sources

| Source           | Use                                  | Access |
|------------------|--------------------------------------|--------|
| ICD-10           | Diagnosis hierarchy DAG              | Public |
| Sepsis-3 rules   | DAG validator rules                  | Public |
| SNOMED-CT subset | Concept embeddings (if available)    | License|
| DrugBank Open    | Drug-disease-interaction graph       | Public |

Fallback if SNOMED-CT is unavailable: use ICD-10 + Sepsis-3 only.

## 5. Experimental Matrix

### 5.1 Architecture ablation
- MLP baseline
- KAN baseline
- QKAN baseline
- HQKAN baseline
- Mamba baseline
- Mamba + MLP gate
- Mamba + HQKAN gate (our O-QKAN-Mamba without KG)
- Mamba + Ontology-Modulated HQKAN (full)

### 5.2 Knowledge ablation
- No KG
- Random embedding (same dim as φ(c_t))
- Shuffled ontology
- True ontology

### 5.3 Conflict resolution ablation
- No resolver
- Hard projection
- Spectral resolver (ours)

### 5.4 Counterfactual evaluation
- Compare to SHAP, Integrated Gradients
- Metrics: speed, faithfulness, monotonicity

## 6. Violation Categories

| Type             | Detection                              |
|------------------|----------------------------------------|
| Type violation   | Sepsis-3 rule engine                   |
| Hierarchy violation | SNOMED/ICD mutual exclusion         |
| Temporal causal  | Time-DAG check                         |
| Treatment conflict | DrugBank + patient allergy list      |

## 7. Clinical Evaluation Plan

- Recruit 3-5 medical students or residents
- Sample 50 cases (balanced: easy, hard, conflict-triggering)
- Score on:
  - Prediction reasonableness (1-5 Likert)
  - Explanation clarity (1-5)
  - Conflict resolution quality (1-5)
- Compute inter-rater agreement (Cohen's kappa)

## 8. Timeline (12 months)

| Month | Milestone                                    |
|-------|----------------------------------------------|
| 1     | Env, MVP Spectral-QKAN-Mamba (Phase 0-1)     |
| 2     | MIMIC-IV access + data pipeline (Phase 2)    |
| 3     | Ontology infrastructure (Phase 3)            |
| 4     | Ontology-Modulated DARUAN (Phase 4)          |
| 5     | DAG Validator (Phase 5)                      |
| 6     | Spectral Conflict Resolver (Phase 6)         |
| 7     | Counterfactual Engine (Phase 7)              |
| 8     | FHIR integration (Phase 8)                   |
| 9-10  | Full experiments + clinician evaluation      |
| 11    | Writing                                      |
| 12    | Revision + submission                        |