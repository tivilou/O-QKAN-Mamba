"""Minimal ICU drug interaction database.

Manually curated subset of ~20 ICU-relevant drugs with known
contraindications and interactions. Expandable later via DrugBank Open API.
"""
from dataclasses import dataclass


@dataclass
class Interaction:
    drug1: str
    drug2: str
    severity: str  # "major", "moderate", "minor"
    description: str


# ICU-relevant drugs
ICU_DRUGS = {
    "norepinephrine": {"class": "vasopressor", "route": "IV"},
    "vasopressin": {"class": "vasopressor", "route": "IV"},
    "epinephrine": {"class": "vasopressor", "route": "IV"},
    "dobutamine": {"class": "inotrope", "route": "IV"},
    "vancomycin": {"class": "antibiotic", "route": "IV"},
    "piperacillin_tazobactam": {"class": "antibiotic", "route": "IV"},
    "meropenem": {"class": "antibiotic", "route": "IV"},
    "ceftriaxone": {"class": "antibiotic", "route": "IV"},
    "metronidazole": {"class": "antibiotic", "route": "IV"},
    "heparin": {"class": "anticoagulant", "route": "IV"},
    "enoxaparin": {"class": "anticoagulant", "route": "SC"},
    "insulin": {"class": "antidiabetic", "route": "IV/SC"},
    "propofol": {"class": "sedative", "route": "IV"},
    "midazolam": {"class": "sedative", "route": "IV"},
    "fentanyl": {"class": "analgesic", "route": "IV"},
    "morphine": {"class": "analgesic", "route": "IV"},
    "furosemide": {"class": "diuretic", "route": "IV"},
    "hydrocortisone": {"class": "corticosteroid", "route": "IV"},
    "amiodarone": {"class": "antiarrhythmic", "route": "IV"},
    "pantoprazole": {"class": "PPI", "route": "IV"},
}

# Known drug-drug interactions (major severity)
_INTERACTIONS = [
    Interaction("heparin", "enoxaparin", "major", "Dual anticoagulation: bleeding risk"),
    Interaction("amiodarone", "midazolam", "major", "QT prolongation + CNS depression"),
    Interaction("vancomycin", "furosemide", "moderate", "Additive nephrotoxicity"),
    Interaction("meropenem", "heparin", "moderate", "Increased bleeding risk"),
    Interaction("propofol", "fentanyl", "moderate", "Additive respiratory depression"),
    Interaction("norepinephrine", "dobutamine", "minor", "Opposing hemodynamic effects"),
]

# Common drug allergies in ICU
_ALLERGY_CROSS_REACTIVITY = {
    "penicillin": ["piperacillin_tazobactam"],
    "cephalosporin": ["ceftriaxone"],
    "sulfa": [],
    "vancomycin": ["vancomycin"],
}


class DrugInteractionChecker:
    """Check drug-drug and drug-allergy interactions."""

    def __init__(self):
        self.drugs = ICU_DRUGS
        self.interactions = _INTERACTIONS
        self.allergy_map = _ALLERGY_CROSS_REACTIVITY

    def check_contraindication(
        self, patient_meds: list[str], patient_allergies: list[str]
    ) -> list[str]:
        """Check for contraindications given current meds and allergies.

        Returns list of warning strings.
        """
        warnings = []
        for allergy in patient_allergies:
            cross_reactive = self.allergy_map.get(allergy, [])
            for med in patient_meds:
                if med in cross_reactive:
                    warnings.append(
                        f"ALLERGY: {med} contraindicated (patient allergic to {allergy})"
                    )

        for interaction in self.interactions:
            if interaction.drug1 in patient_meds and interaction.drug2 in patient_meds:
                warnings.append(
                    f"INTERACTION [{interaction.severity}]: "
                    f"{interaction.drug1} + {interaction.drug2}: {interaction.description}"
                )
        return warnings

    def get_drug_class(self, drug: str) -> str:
        info = self.drugs.get(drug, {})
        return info.get("class", "unknown")
