"""Sepsis-3 clinical rules as Python predicates.

Reference: Singer et al., "The Third International Consensus Definitions
for Sepsis and Septic Shock (Sepsis-3)", JAMA 2016;315(8):801-810.

Sepsis-3 definition:
- Sepsis = suspected infection + SOFA increase >= 2
- Septic shock = sepsis + vasopressors required to maintain MAP >= 65
  AND lactate > 2 mmol/L despite adequate fluid resuscitation
"""
from dataclasses import dataclass
from typing import Optional

from ..data.sofa import (
    compute_sofa_cardiovascular,
    compute_sofa_coagulation,
    compute_sofa_liver,
    compute_sofa_renal,
    compute_sofa_respiratory,
)


@dataclass
class RuleResult:
    name: str
    fired: bool
    score: Optional[float] = None
    detail: str = ""


def sofa_score(features: dict) -> int:
    """Compute total SOFA score from feature dict.

    Args:
        features: dict with keys matching clinical measurements
    """
    score = 0
    pf_ratio = features.get("pao2_fio2_ratio")
    if pf_ratio is None and features.get("pao2") and features.get("fio2"):
        fio2 = features["fio2"]
        if fio2 > 1:
            fio2 = fio2 / 100
        if fio2 > 0:
            pf_ratio = features["pao2"] / fio2
    score += compute_sofa_respiratory(pf_ratio)
    score += compute_sofa_coagulation(features.get("platelet"))
    score += compute_sofa_liver(features.get("bilirubin"))
    score += compute_sofa_cardiovascular(features.get("map"))
    score += compute_sofa_renal(features.get("creatinine"))
    return score


def suspected_infection(features: dict) -> bool:
    """Heuristic for suspected infection.

    In real MIMIC-IV: culture order + antibiotics within ±48h.
    Simplified proxy: elevated WBC + temperature + tachycardia.
    """
    wbc = features.get("wbc")
    temp = features.get("temperature")
    hr = features.get("heart_rate")
    if wbc is not None and wbc > 12.0:
        return True
    if temp is not None and temp > 38.3:
        return True
    if hr is not None and hr > 100 and wbc is not None and wbc > 10.0:
        return True
    return False


def sepsis3_positive(features_sequence: list[dict], time_window: int = 24) -> RuleResult:
    """Check Sepsis-3 criteria over a time window.

    Sepsis = suspected infection + SOFA increase >= 2 within time_window hours.
    """
    if len(features_sequence) < 2:
        return RuleResult(name="sepsis3", fired=False, detail="Insufficient data")

    baseline_sofa = sofa_score(features_sequence[0])
    has_infection = False

    for t, feat in enumerate(features_sequence[:time_window]):
        if suspected_infection(feat):
            has_infection = True
        current_sofa = sofa_score(feat)
        if has_infection and (current_sofa - baseline_sofa) >= 2:
            return RuleResult(
                name="sepsis3", fired=True, score=float(current_sofa),
                detail=f"SOFA {baseline_sofa}->{current_sofa} at t={t}h with infection"
            )
    return RuleResult(name="sepsis3", fired=False, score=float(baseline_sofa))


def septic_shock_positive(features: dict) -> RuleResult:
    """Check septic shock criteria.

    Septic shock = sepsis + vasopressors for MAP >= 65 + lactate > 2 mmol/L.
    Simplified: MAP < 65 + lactate > 2.
    """
    map_val = features.get("map")
    lactate = features.get("lactate")
    if map_val is not None and lactate is not None:
        if map_val < 65 and lactate > 2.0:
            return RuleResult(
                name="septic_shock", fired=True,
                detail=f"MAP={map_val:.0f}, lactate={lactate:.1f}"
            )
    return RuleResult(name="septic_shock", fired=False)


class SepsisRuleEngine:
    """Aggregates all sepsis-related clinical rules."""

    def __init__(self):
        self.rules = [
            ("sofa", lambda f: RuleResult("sofa", True, score=float(sofa_score(f)))),
            ("suspected_infection", lambda f: RuleResult("suspected_infection", suspected_infection(f))),
            ("septic_shock", septic_shock_positive),
        ]

    def evaluate_all(self, features: dict) -> list[RuleResult]:
        return [rule_fn(features) for _, rule_fn in self.rules]

    def evaluate_sequence(self, features_seq: list[dict]) -> list[RuleResult]:
        results = self.evaluate_all(features_seq[-1]) if features_seq else []
        results.append(sepsis3_positive(features_seq))
        return results
