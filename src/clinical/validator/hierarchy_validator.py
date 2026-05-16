"""Hierarchy validator: checks ICD-10 mutual exclusion constraints."""
from ..knowledge.icd_dag import ICD10DAG
from .violation_types import Violation, ViolationType


class HierarchyValidator:
    """Validates that predicted diagnosis codes respect ICD-10 hierarchy."""

    def __init__(self):
        self.dag = ICD10DAG()

    def validate(self, predicted_codes: list[str]) -> list[Violation]:
        """
        Args:
            predicted_codes: list of ICD-10 codes predicted/assigned
        Returns:
            List of hierarchy violations
        """
        violations = []

        # Check all pairs for mutual exclusion
        for i, code1 in enumerate(predicted_codes):
            for code2 in predicted_codes[i + 1:]:
                if self.dag.check_mutex(code1, code2):
                    desc1 = self.dag.get_description(code1)
                    desc2 = self.dag.get_description(code2)
                    violations.append(Violation(
                        type=ViolationType.HIERARCHY,
                        description=(
                            f"Mutually exclusive codes: {code1} ({desc1}) "
                            f"and {code2} ({desc2})"
                        ),
                        severity=0.9,
                        source_rule="ICD-10-CM mutual exclusion constraint",
                        evidence={"code1": code1, "code2": code2},
                    ))

        # Check parent-child redundancy
        for code in predicted_codes:
            ancestors = self.dag.get_ancestors(code)
            for ancestor in ancestors:
                if ancestor in predicted_codes and ancestor != code:
                    violations.append(Violation(
                        type=ViolationType.HIERARCHY,
                        description=(
                            f"Redundant: {ancestor} is ancestor of {code}, "
                            f"only the most specific code should be used"
                        ),
                        severity=0.3,
                        source_rule="ICD-10 coding: use most specific code available",
                        evidence={"ancestor": ancestor, "descendant": code},
                    ))

        return violations
