"""ICD-10-CM hierarchy DAG for ICU/sepsis-relevant codes.

Uses a hardcoded subset of ~80 ICU-relevant codes rather than the full
CMS hierarchy (20k+ codes). Expandable via YAML config later.
"""
import networkx as nx


# ICU/sepsis-relevant ICD-10-CM subset with parent-child relationships
_ICU_ICD10_HIERARCHY = {
    # Infectious diseases (A00-B99)
    "A00-B99": {"parent": None, "desc": "Infectious and parasitic diseases"},
    "A40": {"parent": "A00-B99", "desc": "Streptococcal sepsis"},
    "A41": {"parent": "A00-B99", "desc": "Other sepsis"},
    "A41.9": {"parent": "A41", "desc": "Sepsis, unspecified organism"},
    "B37.7": {"parent": "A00-B99", "desc": "Candidal sepsis"},
    # Endocrine (E00-E89)
    "E00-E89": {"parent": None, "desc": "Endocrine, nutritional and metabolic"},
    "E11": {"parent": "E00-E89", "desc": "Type 2 diabetes mellitus"},
    "E87": {"parent": "E00-E89", "desc": "Fluid, electrolyte and acid-base disorders"},
    "E87.2": {"parent": "E87", "desc": "Acidosis"},
    "E87.4": {"parent": "E87", "desc": "Mixed acid-base disorders"},
    # Circulatory (I00-I99)
    "I00-I99": {"parent": None, "desc": "Diseases of the circulatory system"},
    "I10": {"parent": "I00-I99", "desc": "Essential hypertension"},
    "I21": {"parent": "I00-I99", "desc": "Acute myocardial infarction"},
    "I48": {"parent": "I00-I99", "desc": "Atrial fibrillation and flutter"},
    "I50": {"parent": "I00-I99", "desc": "Heart failure"},
    "I50.9": {"parent": "I50", "desc": "Heart failure, unspecified"},
    # Respiratory (J00-J99)
    "J00-J99": {"parent": None, "desc": "Diseases of the respiratory system"},
    "J18": {"parent": "J00-J99", "desc": "Pneumonia, unspecified organism"},
    "J44": {"parent": "J00-J99", "desc": "COPD"},
    "J80": {"parent": "J00-J99", "desc": "ARDS"},
    "J96": {"parent": "J00-J99", "desc": "Respiratory failure"},
    "J96.0": {"parent": "J96", "desc": "Acute respiratory failure"},
    "J96.1": {"parent": "J96", "desc": "Chronic respiratory failure"},
    # Genitourinary (N00-N99)
    "N00-N99": {"parent": None, "desc": "Diseases of the genitourinary system"},
    "N17": {"parent": "N00-N99", "desc": "Acute kidney failure"},
    "N17.9": {"parent": "N17", "desc": "Acute kidney failure, unspecified"},
    "N18": {"parent": "N00-N99", "desc": "Chronic kidney disease"},
    # Injury/poisoning (S00-T88)
    "S00-T88": {"parent": None, "desc": "Injury, poisoning"},
    "T81.4": {"parent": "S00-T88", "desc": "Infection following a procedure"},
    # Symptoms (R00-R99)
    "R00-R99": {"parent": None, "desc": "Symptoms and signs"},
    "R00.0": {"parent": "R00-R99", "desc": "Tachycardia, unspecified"},
    "R03.0": {"parent": "R00-R99", "desc": "Elevated blood pressure"},
    "R06.0": {"parent": "R00-R99", "desc": "Dyspnea"},
    "R09.2": {"parent": "R00-R99", "desc": "Respiratory arrest"},
    "R40": {"parent": "R00-R99", "desc": "Altered consciousness"},
    "R57": {"parent": "R00-R99", "desc": "Shock"},
    "R57.0": {"parent": "R57", "desc": "Cardiogenic shock"},
    "R57.1": {"parent": "R57", "desc": "Hypovolemic shock"},
    "R57.2": {"parent": "R57", "desc": "Septic shock"},
    "R65": {"parent": "R00-R99", "desc": "SIRS/Sepsis symptoms"},
    "R65.2": {"parent": "R65", "desc": "Severe sepsis"},
    "R65.21": {"parent": "R65.2", "desc": "Severe sepsis with septic shock"},
}

# Known mutually exclusive pairs
_MUTEX_PAIRS = [
    ("J96.0", "J96.1"),   # acute vs chronic resp failure
    ("R57.0", "R57.2"),   # cardiogenic vs septic shock
    ("N17", "N18"),       # acute vs chronic kidney disease
]


class ICD10DAG:
    """ICD-10-CM hierarchy as a directed acyclic graph."""

    def __init__(self):
        self.graph = nx.DiGraph()
        self._build_graph()
        self._mutex_pairs = _MUTEX_PAIRS

    def _build_graph(self):
        for code, info in _ICU_ICD10_HIERARCHY.items():
            self.graph.add_node(code, desc=info["desc"])
            if info["parent"]:
                self.graph.add_edge(info["parent"], code)

    @property
    def num_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self.graph.number_of_edges()

    def get_parents(self, code: str) -> list[str]:
        if code not in self.graph:
            return []
        return list(self.graph.predecessors(code))

    def get_children(self, code: str) -> list[str]:
        if code not in self.graph:
            return []
        return list(self.graph.successors(code))

    def get_ancestors(self, code: str) -> list[str]:
        if code not in self.graph:
            return []
        return list(nx.ancestors(self.graph, code))

    def check_mutex(self, code1: str, code2: str) -> bool:
        """Check if two codes are mutually exclusive."""
        for a, b in self._mutex_pairs:
            if (code1 == a and code2 == b) or (code1 == b and code2 == a):
                return True
        return False

    def get_all_codes(self) -> list[str]:
        return list(self.graph.nodes())

    def get_description(self, code: str) -> str:
        if code in self.graph:
            return self.graph.nodes[code].get("desc", "")
        return ""
