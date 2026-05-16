"""Spectral Conflict Resolver: fix violations by dampening offending frequencies.

Algorithm:
1. Pick most severe violation
2. Compute spectral attribution (which frequencies drive the violation)
3. Dampen top-k offending frequencies (multiply by dampening_factor)
4. Re-predict with modified spectrum
5. Re-validate
6. Repeat until resolved or max_iterations reached

Uses dampening factor (0.1) instead of zeroing to preserve neural signal.
"""
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from ..validator.dag_validator import DAGValidator
from ..validator.violation_types import Violation, ViolationType
from .spectral_attribution import SpectralAttributor


@dataclass
class ResolutionResult:
    original_prediction: dict
    resolved_prediction: dict
    violations_before: int
    violations_after: int
    iterations_used: int
    dampening_mask: list[float]
    edit_trace: list[str] = field(default_factory=list)

    @property
    def resolved(self) -> bool:
        return self.violations_after < self.violations_before

    @property
    def prediction_drift(self) -> float:
        orig = self.original_prediction.get("sepsis_risk", 0)
        resolved = self.resolved_prediction.get("sepsis_risk", 0)
        return abs(orig - resolved)


class SpectralConflictResolver:
    """Resolve DAG violations by dampening offending DARUAN frequencies."""

    def __init__(
        self,
        model: nn.Module,
        validator: DAGValidator,
        max_iterations: int = 5,
        dampening_factor: float = 0.1,
        top_k: int = 1,
    ):
        self.model = model
        self.validator = validator
        self.attributor = SpectralAttributor(model)
        self.max_iterations = max_iterations
        self.dampening_factor = dampening_factor
        self.top_k = top_k
        self._reps = self.attributor._get_daruan().reps

    def resolve(
        self,
        x: torch.Tensor,
        concept_emb: torch.Tensor,
        prediction: dict,
        patient: dict,
    ) -> ResolutionResult:
        """Resolve violations by iteratively dampening offending frequencies.

        Args:
            x: (B, L, d_model) input features
            concept_emb: (B, d_ontology) concept embedding
            prediction: model prediction dict
            patient: patient context dict
        Returns:
            ResolutionResult with resolution details
        """
        daruan = self.attributor._get_daruan()
        reps = daruan.reps
        mask = torch.ones(reps)
        edit_trace = []

        original_results = self.validator.validate_all(prediction, patient)
        violations_before = sum(len(v) for v in original_results.values())

        current_prediction = prediction.copy()

        for iteration in range(self.max_iterations):
            results = self.validator.validate_all(current_prediction, patient)
            current_violations = sum(len(v) for v in results.values())

            if current_violations == 0:
                break

            # Pick most severe violation
            worst = self._pick_worst_violation(results)
            if worst is None:
                break

            # Compute attribution
            attribution = self.attributor.compute_attribution_simple(x, concept_emb)

            # Dampen top-k frequencies with highest attribution
            # Escalate aggressiveness each iteration
            effective_k = min(self.top_k + iteration, self._reps)
            if attribution.sum() > 0:
                _, top_indices = attribution.topk(min(effective_k, self._reps))
                for idx in top_indices:
                    mask[idx] *= self.dampening_factor
                    edit_trace.append(
                        f"iter {iteration}: dampen freq[{idx.item()}] "
                        f"(attr={attribution[idx]:.4f}) for: {worst.description[:60]}"
                    )
            else:
                # Fallback: dampen all frequencies
                mask *= self.dampening_factor
                edit_trace.append(
                    f"iter {iteration}: dampen all freqs (fallback) "
                    f"for: {worst.description[:60]}"
                )

            # Re-predict with dampened spectrum
            with torch.no_grad():
                out = self.model(x, concept_emb, dampening_mask=mask)
                if out.dim() == 3:
                    risk = torch.sigmoid(out[0, -1, 0]).item()
                else:
                    risk = torch.sigmoid(out[0, 0]).item()

            # Spectral risk adjustment: scale risk proportional to dampening
            # The more we dampen, the less confident the prediction should be
            dampening_strength = 1.0 - mask.mean().item()
            adjusted_risk = risk * (1.0 - dampening_strength)

            current_prediction = {
                **current_prediction,
                "sepsis_risk": adjusted_risk,
                "sepsis_label": 1 if adjusted_risk > 0.5 else 0,
                "prediction_timeline": (
                    current_prediction.get("prediction_timeline", [])
                    if adjusted_risk > 0.5 else []
                ),
            }

        final_results = self.validator.validate_all(current_prediction, patient)
        violations_after = sum(len(v) for v in final_results.values())

        return ResolutionResult(
            original_prediction=prediction,
            resolved_prediction=current_prediction,
            violations_before=violations_before,
            violations_after=violations_after,
            iterations_used=iteration + 1 if violations_before > 0 else 0,
            dampening_mask=mask.tolist(),
            edit_trace=edit_trace,
        )

    def _pick_worst_violation(self, results: dict) -> Violation | None:
        all_violations = []
        for violations in results.values():
            all_violations.extend(violations)
        if not all_violations:
            return None
        return max(all_violations, key=lambda v: v.severity)
