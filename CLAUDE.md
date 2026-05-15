# O-QKAN-Mamba: Ontology-Modulated Spectral Reasoning for Clinical Agents
## Research Goal
Build a clinical reasoning core for healthcare agents that combines:
1. **Mamba** for long EHR time-series encoding
2. **HQKAN with ontology-modulated DARUAN** as the spectral activation core
3. **Spectral Conflict Resolver** for neuro-symbolic conflict handling
4. **Counterfactual reasoning** via ontology embedding perturbation
5. **FHIR-compliant I/O** for clinical deployment
Target submission: IEEE J-BHI Special Issue on "Knowledge-Guided Agentic AI"
Deadline: 31 August 2026
## Key Documents
- `RESEARCH_PROPOSAL.md` — Full research proposal including hypotheses (H1-H3), experimental matrix (architecture/knowledge/conflict/counterfactual ablations), clinical evaluation plan, dataset details, and 12-month timeline. Refer to it for methodology rationale and experiment design decisions.
- `papers/Quantum.Variational.Activation.Functions.Empower.Kolmogorov-Arnold.Networks.pdf` — QKAN 原论文 (arXiv:2509.14026)，本项目基于该工作扩展，是理解 QKAN 架构、DARUAN 激活函数和量子变分电路设计的核心参考。

## Hard Constraints
### Code organization
- All new code under:
  - `src/qkan/experimental/` — research modules
  - `src/clinical/` — clinical-specific code (data, KG, validator)
  - `examples/o_qkan_mamba/` — training/eval scripts
  - `tests/` — unit and integration tests
  - `logs/` — experiment logs (markdown + csv)
  - `outputs/` — model checkpoints and results
  - `docs/` — design documents
- **Do NOT modify existing QKAN public APIs** unless absolutely necessary
- **Do NOT modify CUDA kernels**
### Methodology constraints
- Use `solver="exact"` for all initial development
- Switch to `flash` / `cutile` only after correctness is verified
- Always keep a CPU/fallback path for unit tests
- All experiments must be reproducible: fix seeds, log configs, save versions
### Reporting constraints
- Never claim performance superiority without supporting metrics in logs
- Always report: parameter count, training loss, val metrics, tokens/sec, GPU memory
- For clinical tasks, always report: AUROC, AUPRC, **ontology violation rate**, conflict resolution success rate
### Data handling
- MIMIC-IV requires PhysioNet credentialed access
- **Never commit raw patient data** to git
- Use `.gitignore` for `data/raw/`, `data/processed/`
- All paths to data should be config-driven, never hardcoded
## Architecture Overview

FHIR Input 

↓ 

KG Retriever (SNOMED-CT / ICD-10 / Sepsis-3 rules) 

↓ 

Ontology-Modulated QKAN-Mamba Encoder

- Mamba blocks for sequence
- HQKAN with knowledge-conditioned DARUAN frequencies ↓ Prediction Head + Spectral Attribution ↓ DAG Validator (4 violation types) ↓ Spectral Conflict Resolver (if violations) ↓ Counterfactual Engine (on demand) ↓ FHIR Output Bundle

## Key Mathematical Definition
The knowledge-conditioned data re-uploading weight for layer ℓ is:
  $w_ℓ(t, x_t, c_t) = w_ℓ^base * σ(W_KG · φ(c_t) + b_ℓ)$
where:

- $c_t$: multi-hot vector of active SNOMED/ICD concepts at time t
- $φ(c_t)$: pretrained ontology embedding
- $w_ℓ^base$: base frequency (initialized as 2^(ℓ-1) for geometric expansion)
- $σ$: softplus activation
## Development Phases
| Phase | Goal | Status |
|-------|------|--------|
| 0 | Environment & repo setup | TODO |
| 1 | Baseline Spectral-QKAN-Mamba (no clinical) | TODO |
| 2 | Clinical data pipeline (MIMIC-IV) | TODO |
| 3 | Ontology infrastructure | TODO |
| 4 | Ontology-Modulated DARUAN | TODO |
| 5 | DAG Validator | TODO |
| 6 | Spectral Conflict Resolver | TODO |
| 7 | Counterfactual Engine | TODO |
| 8 | FHIR I/O integration | TODO |
| 9 | Full agent integration + evaluation | TODO |
## Required Checks Before Claiming Phase Completion
1. All unit tests pass
2. Smoke test on small data runs
3. Logs saved under `logs/phase_N_*.md`
4. Code committed with descriptive message
5. Design doc updated under `docs/`
## Forbidden Actions
- Running large experiments (>1000 steps) without explicit user approval
- Modifying `pyproject.toml` dependencies without explanation
- Deleting any files under `src/qkan/` outside `experimental/`
- Touching files under `docs/api.rst`, `docs/intro/`, `docs/examples/`
- Pushing to remote branches
## Coding Style
- Prefer simple, readable PyTorch first
- Type hints required for public functions
- Docstrings for all classes and public methods
- Keep commits small and focused
- One concept per file when possible