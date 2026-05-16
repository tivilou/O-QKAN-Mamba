# O-QKAN-Mamba Design Document

## Section A: Component Map

### A.1 SpectralQKANGate (Phase 1)

**Purpose**: Replace the MLP gate in Mamba blocks with an HQKAN-based spectral gate.

**QKAN dependency**: Uses `QKANLayer` with `preact_trainable=True` in a bottleneck
pattern (Linear → QKANLayer → Linear), following the HQKAN pattern from `gqkan_gpt.ipynb`.

**Key design**: The gate projects Mamba's inner dimension down to a small spectral
dimension (`d_spectral = ceil(log2(d_inner))`), applies QKAN activation, then projects
back. This gives spectral interpretability while keeping parameter count low.

### A.2 OntologyModulatedDARUAN (Phase 4)

**Purpose**: Dynamically condition DARUAN re-uploading frequencies on clinical context.

**QKAN dependency**: Extends the `preacts_weight` mechanism in `QKANLayer`. The existing
`preacts_weight` has shape `(*group, reps)` and controls data re-uploading frequency.
We modulate it via: `w_ℓ(t, x_t, c_t) = w_ℓ^base · σ(W_KG · φ(c_t) + b_ℓ)`

**Critical constraint**: We do NOT modify `QKANLayer` itself. Instead, we create a
wrapper that computes modulated weights and passes them to the solver via the existing
`preact_trainable=True` pathway. The solver already supports dynamic `preacts_weight`
through `encoded_x = einsum("oir,bi->boir", preacts_weight, x) + preacts_bias`.

### A.3 MambaWithQKANGate (Phase 1)

**Purpose**: Full Mamba block using SpectralQKANGate instead of the default SiLU gate.

**QKAN dependency**: Composes `mamba_ssm.Mamba` with `SpectralQKANGate`. Does not
modify Mamba internals — wraps the output gating mechanism.

### A.4 ClinicalDataPipeline (Phase 2)

**Purpose**: Load MIMIC-IV data, extract time-series features, produce tensors for
the model. Handles irregular sampling, missing values, and variable-length stays.

**QKAN dependency**: None. Pure data engineering.

### A.5 OntologyEmbedder (Phase 3)

**Purpose**: Convert multi-hot clinical concept vectors (ICD-10, SNOMED-CT) into
dense embeddings φ(c_t) that condition the DARUAN frequencies.

**QKAN dependency**: None directly. Produces the `c_t` vector consumed by
OntologyModulatedDARUAN.

### A.6 DAGValidator (Phase 5)

**Purpose**: Check model predictions against ontology-derived rules. Detects four
violation types: type violations (Sepsis-3), hierarchy violations (ICD mutual
exclusion), temporal causal violations (time-DAG), treatment conflicts (DrugBank).

**QKAN dependency**: None. Operates on model outputs, not internals.

### A.7 SpectralConflictResolver (Phase 6)

**Purpose**: When DAGValidator detects a violation, resolve it by editing the
spectral domain (DARUAN frequencies) rather than hard-projecting the output.

**QKAN dependency**: Accesses `QKANLayer.postact_weights` and the solver's internal
`encoded_x` to compute spectral attribution. Uses gradient-based attribution through
the DARUAN circuit to identify offending frequencies, then dampens them.

### A.8 CounterfactualEngine (Phase 7)

**Purpose**: Answer "what-if" queries by perturbing the ontology embedding φ(c_t)
and observing the change in prediction through the modulated DARUAN.

**QKAN dependency**: Reuses OntologyModulatedDARUAN's forward pass with modified c_t.
No new QKAN APIs needed.

### A.9 FHIRAdapter (Phase 8)

**Purpose**: Parse FHIR Bundle inputs into model-ready tensors, and format model
outputs (predictions, explanations, counterfactuals) as FHIR resources.

**QKAN dependency**: None.

### A.10 ClinicalAgent (Phase 9)

**Purpose**: Orchestrator that ties all components together. Receives a patient case,
runs inference, validates, resolves conflicts, and returns structured output.

**QKAN dependency**: Indirect — composes all above components.

---

## Section B: File Layout

```
src/
├── qkan/experimental/
│   ├── __init__.py
│   ├── spectral_gate.py          # SpectralQKANGate
│   ├── mamba_qkan.py             # MambaWithQKANGate
│   ├── ontology_daruan.py        # OntologyModulatedDARUAN
│   ├── conflict_resolver.py      # SpectralConflictResolver
│   └── counterfactual.py         # CounterfactualEngine
├── clinical/
│   ├── __init__.py
│   ├── data_pipeline.py          # ClinicalDataPipeline
│   ├── ontology_embedder.py      # OntologyEmbedder
│   ├── dag_validator.py          # DAGValidator
│   ├── fhir_adapter.py           # FHIRAdapter
│   └── agent.py                  # ClinicalAgent

examples/o_qkan_mamba/
├── train_baseline.py             # Phase 1 training script
├── train_clinical.py             # Phase 4+ training script
├── evaluate.py                   # Evaluation with all metrics
└── configs/
    ├── baseline.yaml
    └── clinical.yaml

tests/
├── experimental/
│   ├── __init__.py
│   ├── test_spectral_gate.py
│   ├── test_mamba_qkan.py
│   ├── test_ontology_daruan.py
│   ├── test_conflict_resolver.py
│   └── test_counterfactual.py
├── clinical/
│   ├── __init__.py
│   ├── test_data_pipeline.py
│   ├── test_ontology_embedder.py
│   ├── test_dag_validator.py
│   ├── test_fhir_adapter.py
│   └── test_agent.py
└── integration/
    ├── test_end_to_end.py
    └── test_mamba_qkan_smoke.py
```

---

## Section C: API Sketches

### C.1 SpectralQKANGate

```python
class SpectralQKANGate(nn.Module):
    """HQKAN-based gating mechanism for Mamba blocks.
    
    Uses bottleneck pattern: Linear(d_in, d_spectral) → QKANLayer → Linear(d_spectral, d_out)
    """
    def __init__(
        self,
        d_model: int,
        d_spectral: Optional[int] = None,  # default: ceil(log2(d_model))
        reps: int = 3,
        group: int = 3,
        solver: str = "exact",
        device: str = "cuda",
    ): ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model) -> (batch, seq_len, d_model)"""
        ...

    def get_spectral_attribution(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-frequency attribution scores for interpretability."""
        ...
```

### C.2 OntologyModulatedDARUAN

```python
class OntologyModulatedDARUAN(nn.Module):
    """DARUAN with knowledge-conditioned re-uploading frequencies.
    
    w_ℓ(t, x_t, c_t) = w_ℓ^base · σ(W_KG · φ(c_t) + b_ℓ)
    
    Does NOT modify QKANLayer. Wraps it by computing modulated preacts_weight
    externally and injecting via the existing preact_trainable pathway.
    """
    def __init__(
        self,
        d_model: int,
        d_ontology: int,       # dimension of φ(c_t)
        reps: int = 3,
        group: int = 3,
        base_freq_init: str = "geometric",  # 2^(ℓ-1)
        solver: str = "exact",
        device: str = "cuda",
    ): ...

    def forward(
        self, x: torch.Tensor, c_t: torch.Tensor
    ) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model) — input features
        c_t: (batch, seq_len, d_ontology) — ontology embedding
        Returns: (batch, seq_len, d_model)
        """
        ...

    def compute_modulated_weights(
        self, c_t: torch.Tensor
    ) -> torch.Tensor:
        """Compute w_ℓ given ontology context. Shape: (*group, reps)"""
        ...
```

### C.3 MambaWithQKANGate

```python
class MambaWithQKANGate(nn.Module):
    """Mamba block with SpectralQKANGate replacing the default SiLU gate."""
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        qkan_reps: int = 3,
        qkan_group: int = 3,
        solver: str = "exact",
        device: str = "cuda",
    ): ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model) -> (batch, seq_len, d_model)"""
        ...
```

### C.4 ClinicalDataPipeline

```python
class ClinicalDataPipeline:
    """MIMIC-IV data loading and preprocessing."""
    def __init__(self, data_dir: str, config: dict): ...

    def load_cohort(self, task: str) -> pd.DataFrame:
        """Load patient cohort for a specific task (sepsis/mortality/readmission)."""
        ...

    def extract_features(
        self, stays: pd.DataFrame, max_len: int = 48
    ) -> dict[str, torch.Tensor]:
        """Extract time-series features, concept vectors, and labels."""
        ...

    def get_dataloaders(
        self, batch_size: int, split_ratio: tuple = (0.7, 0.15, 0.15)
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        ...
```

### C.5 OntologyEmbedder

```python
class OntologyEmbedder(nn.Module):
    """Convert multi-hot concept vectors to dense ontology embeddings."""
    def __init__(
        self,
        vocab_size: int,       # total number of concepts
        d_ontology: int = 64,  # embedding dimension
        pretrained: Optional[torch.Tensor] = None,
    ): ...

    def forward(self, concepts: torch.Tensor) -> torch.Tensor:
        """concepts: (batch, seq_len, vocab_size) multi-hot -> (batch, seq_len, d_ontology)"""
        ...
```

### C.6 DAGValidator

```python
class DAGValidator:
    """Check predictions against ontology-derived clinical rules."""
    def __init__(self, rules_path: str): ...

    def validate(
        self, predictions: dict, patient_context: dict
    ) -> list[Violation]:
        """Return list of violations (may be empty)."""
        ...

@dataclass
class Violation:
    type: Literal["type", "hierarchy", "temporal", "treatment"]
    rule_id: str
    description: str
    severity: float
    offending_output: str
```

### C.7 SpectralConflictResolver

```python
class SpectralConflictResolver(nn.Module):
    """Resolve DAG violations by editing DARUAN spectral domain."""
    def __init__(self, max_iterations: int = 10, damping: float = 0.1): ...

    def resolve(
        self,
        model: MambaWithQKANGate,
        x: torch.Tensor,
        violations: list[Violation],
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns: (corrected_output, resolution_metadata)
        resolution_metadata includes: frequencies_dampened, iterations_used, success
        """
        ...

    def spectral_attribution(
        self, model: MambaWithQKANGate, x: torch.Tensor, target_output: int
    ) -> torch.Tensor:
        """Gradient-based attribution over DARUAN frequencies."""
        ...
```

### C.8 CounterfactualEngine

```python
class CounterfactualEngine:
    """Generate counterfactual predictions via ontology perturbation."""
    def __init__(self, model: nn.Module, embedder: OntologyEmbedder): ...

    def query(
        self,
        x: torch.Tensor,
        c_t: torch.Tensor,
        perturbation: dict,  # e.g. {"remove": ["SNOMED:12345"], "add": ["ICD10:E11"]}
    ) -> CounterfactualResult:
        ...

@dataclass
class CounterfactualResult:
    original_prediction: torch.Tensor
    counterfactual_prediction: torch.Tensor
    delta: torch.Tensor
    affected_frequencies: list[int]
```

### C.9 FHIRAdapter

```python
class FHIRAdapter:
    """FHIR Bundle ↔ model tensor conversion."""
    def __init__(self, concept_vocab: dict): ...

    def parse_bundle(self, bundle: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """FHIR Bundle -> (features, concepts)"""
        ...

    def format_output(
        self, prediction: torch.Tensor, explanation: dict
    ) -> dict:
        """Model output -> FHIR RiskAssessment resource"""
        ...
```

### C.10 ClinicalAgent

```python
class ClinicalAgent:
    """Top-level orchestrator for clinical reasoning."""
    def __init__(self, config: dict): ...

    def predict(self, fhir_bundle: dict) -> dict:
        """Full pipeline: parse → infer → validate → resolve → format."""
        ...

    def explain(self, fhir_bundle: dict) -> dict:
        """Prediction + spectral attribution explanation."""
        ...

    def counterfactual(self, fhir_bundle: dict, query: dict) -> dict:
        """Counterfactual reasoning on a patient case."""
        ...
```

---

## Section D: Data Flow Diagram

```
                         FHIR Bundle (Patient Case)
                                    │
                                    ▼
                        ┌───────────────────────┐
                        │     FHIRAdapter       │
                        │  parse_bundle()       │
                        └───────────┬───────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼                               ▼
        ┌──────────────────┐           ┌──────────────────┐
        │  features (x_t)  │           │  concepts (c_t)  │
        │  (B, T, D_feat)  │           │  (B, T, V_vocab) │
        └────────┬─────────┘           └────────┬─────────┘
                 │                               │
                 │                               ▼
                 │                   ┌──────────────────────┐
                 │                   │  OntologyEmbedder    │
                 │                   │  φ(c_t)             │
                 │                   └────────┬─────────────┘
                 │                            │
                 │                            ▼
                 │                   ┌──────────────────────┐
                 │                   │  Modulated weights   │
                 │                   │  w_ℓ = w^base·σ(Wφ) │
                 │                   └────────┬─────────────┘
                 │                            │
                 ▼                            ▼
        ┌─────────────────────────────────────────────┐
        │          MambaWithQKANGate                   │
        │  ┌─────────────────────────────────────┐    │
        │  │  Mamba SSM (selective scan)         │    │
        │  │  x_t → hidden states                │    │
        │  └──────────────┬──────────────────────┘    │
        │                 │                            │
        │                 ▼                            │
        │  ┌─────────────────────────────────────┐    │
        │  │  SpectralQKANGate                   │    │
        │  │  (OntologyModulatedDARUAN inside)   │    │
        │  │  Linear → QKAN(w_ℓ) → Linear       │    │
        │  └──────────────┬──────────────────────┘    │
        │                 │                            │
        └─────────────────┼────────────────────────────┘
                          │
                          ▼
                ┌──────────────────┐
                │  y_pred          │
                │  (B, num_tasks)  │
                └────────┬─────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   DAGValidator      │
              │   validate(y_pred)  │
              └────────┬────────────┘
                       │
            ┌──────────┼──────────┐
            │ no violations       │ violations found
            ▼                     ▼
    ┌──────────────┐   ┌─────────────────────────┐
    │  Output as-is│   │ SpectralConflictResolver │
    └──────┬───────┘   │ dampen offending freqs   │
           │           └────────────┬──────────────┘
           │                        │
           └────────────┬───────────┘
                        │
                        ▼
              ┌──────────────────────┐
              │  CounterfactualEngine│ (optional, on query)
              │  perturb φ(c_t)     │
              └────────┬─────────────┘
                       │
                       ▼
              ┌──────────────────────┐
              │  FHIRAdapter         │
              │  format_output()     │
              └────────┬─────────────┘
                       │
                       ▼
              FHIR RiskAssessment + Explanation
```

---

## Section E: Test Plan

### E.1 SpectralQKANGate

| Type | Test |
|------|------|
| Unit | Forward pass shape correctness (batch, seq, d_model) → (batch, seq, d_model) |
| Unit | Gradient flows through QKAN layer (no dead gradients) |
| Unit | Spectral attribution returns non-zero values |
| Smoke | Train 100 steps on random data, loss decreases |

### E.2 MambaWithQKANGate

| Type | Test |
|------|------|
| Unit | Output shape matches standard Mamba |
| Unit | Causal property: output[t] independent of input[t+1:] |
| Smoke | Forward + backward on (B=4, L=128, D=64), no OOM on V100 |
| Smoke | Compare parameter count vs vanilla Mamba+MLP |

### E.3 OntologyModulatedDARUAN

| Type | Test |
|------|------|
| Unit | Modulated weights differ from base when c_t ≠ 0 |
| Unit | Modulated weights equal base when c_t = 0 (identity modulation) |
| Unit | Gradient flows through both x and c_t paths |
| Smoke | Different c_t vectors produce different outputs for same x |

### E.4 ClinicalDataPipeline

| Type | Test |
|------|------|
| Unit | Synthetic MIMIC-like CSV → correct tensor shapes |
| Unit | Missing value handling (NaN → imputed) |
| Unit | Variable-length stays padded correctly |
| Integration | Full pipeline on 100-patient synthetic cohort |

### E.5 OntologyEmbedder

| Type | Test |
|------|------|
| Unit | Multi-hot (B, T, V) → dense (B, T, d_ontology) |
| Unit | Zero input → zero embedding (or learned bias) |
| Unit | Pretrained weights loaded correctly |

### E.6 DAGValidator

| Type | Test |
|------|------|
| Unit | Known violation detected (e.g., mutually exclusive ICD codes) |
| Unit | Valid prediction passes without violations |
| Unit | All four violation types detected on crafted inputs |

### E.7 SpectralConflictResolver

| Type | Test |
|------|------|
| Unit | After resolution, DAGValidator returns no violations |
| Unit | Resolution changes fewer than 50% of frequencies |
| Smoke | Convergence within max_iterations on synthetic violation |

### E.8 CounterfactualEngine

| Type | Test |
|------|------|
| Unit | Removing a concept changes prediction |
| Unit | Adding a concept changes prediction |
| Unit | No perturbation → delta ≈ 0 |
| Smoke | Speed: < 100ms per counterfactual query (single forward pass) |

### E.9 FHIRAdapter

| Type | Test |
|------|------|
| Unit | Valid FHIR Bundle parses without error |
| Unit | Output conforms to FHIR RiskAssessment schema |
| Unit | Round-trip: parse → predict → format → validate schema |

### E.10 ClinicalAgent

| Type | Test |
|------|------|
| Integration | End-to-end on synthetic patient: FHIR in → FHIR out |
| Integration | Conflict case: prediction violates rule → resolved output |
| Integration | Counterfactual query returns valid result |

---

## Section F: Risk Register

### F.1 Components Most Likely to Fail

| Component | Risk | Likelihood | Impact |
|-----------|------|-----------|--------|
| mamba-ssm + QKANLayer integration | Shape mismatch between Mamba hidden states and QKAN group dimensions | High | Blocks Phase 1 |
| OntologyModulatedDARUAN | Modulation collapses (σ saturates → all weights ≈ 1) | Medium | Reduces to vanilla QKAN (no ontology benefit) |
| SpectralConflictResolver | Frequency dampening destroys useful signal | Medium | Predictions degrade after resolution |
| ClinicalDataPipeline | MIMIC-IV access delayed (PhysioNet credentialing) | High | Blocks Phase 2+ |

### F.2 Fallback Strategies

| Risk | Fallback |
|------|----------|
| Mamba shape mismatch | Use flatten/reshape adapters; worst case, use Mamba output directly without gating |
| Modulation collapse | Add auxiliary loss to encourage weight diversity; use temperature scaling in σ |
| Resolver destroys signal | Cap maximum dampening at 30%; add reconstruction loss |
| MIMIC-IV access delayed | Use synthetic EHR data (generated via rules) for development; switch to real data later |
| V100 dropped by future PyTorch | Pin PyTorch 2.4; or migrate to cloud A100 |

### F.3 Components Blocked by Data Access

| Component | Blocked by | Workaround |
|-----------|-----------|------------|
| ClinicalDataPipeline | MIMIC-IV credentials | Synthetic data generator |
| OntologyEmbedder (SNOMED) | SNOMED-CT license | Use ICD-10 (public) only |
| DAGValidator (full rules) | Clinical expert review | Start with Sepsis-3 rules (published) |
| Clinical evaluation | IRB + clinician recruitment | Defer to Month 9-10 |

### F.4 QKAN API Boundary Constraints

These are things we MUST NOT do (per CLAUDE.md):

- Do NOT modify `QKANLayer.__init__` signature
- Do NOT modify `QKANLayer.forward` logic
- Do NOT change the solver interface contract
- Do NOT alter `DARUAN` class behavior

Instead, we extend by:
- Creating new classes that COMPOSE `QKANLayer`
- Using `preact_trainable=True` to inject dynamic weights externally
- Using the `callable` solver pathway for custom circuits if needed
- Subclassing only our own experimental modules
