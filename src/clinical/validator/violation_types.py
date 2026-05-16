"""Violation types for the DAG validator."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ViolationType(Enum):
    TYPE = "type"
    HIERARCHY = "hierarchy"
    TEMPORAL = "temporal"
    TREATMENT = "treatment"


@dataclass
class Violation:
    type: ViolationType
    description: str
    severity: float  # 0.0 to 1.0
    source_rule: str
    evidence: dict = field(default_factory=dict)

    def __str__(self):
        return f"[{self.type.value}|sev={self.severity:.1f}] {self.description}"
