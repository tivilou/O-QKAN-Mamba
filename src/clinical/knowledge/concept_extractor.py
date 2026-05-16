"""Extract clinical concepts from patient features at each timestep.

Maps continuous vital/lab values to discrete clinical concepts using
threshold rules based on standard clinical definitions.

Concept vocabulary is intentionally small (~50) for tractability.
"""
import numpy as np
import torch


# Concept vocabulary: (concept_id, name, condition_description)
CONCEPT_VOCABULARY = [
    # Cardiovascular (0-9)
    (0, "tachycardia", "HR > 100"),
    (1, "bradycardia", "HR < 60"),
    (2, "hypotension", "MAP < 65"),
    (3, "hypertension", "SBP > 180"),
    (4, "shock_index_elevated", "HR/SBP > 0.9"),
    # Respiratory (10-19)
    (10, "tachypnea", "RR > 22"),
    (11, "hypoxemia", "SpO2 < 92"),
    (12, "severe_hypoxemia", "SpO2 < 88"),
    (13, "high_fio2", "FiO2 > 0.4"),
    (14, "respiratory_failure", "PaO2 < 60"),
    # Temperature (20-24)
    (20, "fever", "Temp > 38.3"),
    (21, "hypothermia", "Temp < 36.0"),
    (22, "high_fever", "Temp > 39.0"),
    # Hematologic (25-29)
    (25, "leukocytosis", "WBC > 12"),
    (26, "leukopenia", "WBC < 4"),
    (27, "thrombocytopenia", "Platelets < 100"),
    (28, "severe_thrombocytopenia", "Platelets < 50"),
    (29, "anemia", "Hemoglobin < 7"),
    # Metabolic (30-39)
    (30, "elevated_lactate", "Lactate > 2"),
    (31, "severe_lactate", "Lactate > 4"),
    (32, "acidosis", "pH < 7.35"),
    (33, "alkalosis", "pH > 7.45"),
    (34, "hyperglycemia", "Glucose > 180"),
    (35, "hypoglycemia", "Glucose < 70"),
    # Renal (40-44)
    (40, "elevated_creatinine", "Creatinine > 1.5"),
    (41, "aki_stage2", "Creatinine > 2.0"),
    (42, "aki_stage3", "Creatinine > 3.5"),
    # Hepatic (45-49)
    (45, "elevated_bilirubin", "Bilirubin > 2.0"),
    (46, "severe_liver_dysfunction", "Bilirubin > 6.0"),
]

NUM_CONCEPTS = 50  # Fixed vocabulary size (some slots reserved for future)


class ConceptExtractor:
    """Extract multi-hot concept vectors from continuous features."""

    def __init__(self):
        self.vocab = {entry[0]: entry for entry in CONCEPT_VOCABULARY}
        self.vocab_size = NUM_CONCEPTS
        self.concept_names = {entry[0]: entry[1] for entry in CONCEPT_VOCABULARY}

    def extract(self, features: dict) -> np.ndarray:
        """Extract multi-hot concept vector from a single timestep's features.

        Args:
            features: dict with keys like 'heart_rate', 'map', 'spo2', etc.
        Returns:
            (NUM_CONCEPTS,) binary array
        """
        concepts = np.zeros(self.vocab_size, dtype=np.float32)

        hr = features.get("heart_rate")
        sbp = features.get("sbp")
        map_val = features.get("map")
        rr = features.get("resp_rate")
        spo2 = features.get("spo2")
        fio2 = features.get("fio2")
        temp = features.get("temperature")
        wbc = features.get("wbc")
        hgb = features.get("hemoglobin")
        plt_val = features.get("platelet")
        cr = features.get("creatinine")
        bili = features.get("bilirubin")
        lactate = features.get("lactate")
        pao2 = features.get("pao2")
        ph = features.get("ph")
        glucose = features.get("glucose")

        # Cardiovascular
        if hr is not None and hr > 100:
            concepts[0] = 1
        if hr is not None and hr < 60:
            concepts[1] = 1
        if map_val is not None and map_val < 65:
            concepts[2] = 1
        if sbp is not None and sbp > 180:
            concepts[3] = 1
        if hr is not None and sbp is not None and sbp > 0 and hr / sbp > 0.9:
            concepts[4] = 1

        # Respiratory
        if rr is not None and rr > 22:
            concepts[10] = 1
        if spo2 is not None and spo2 < 92:
            concepts[11] = 1
        if spo2 is not None and spo2 < 88:
            concepts[12] = 1
        if fio2 is not None and fio2 > 0.4:
            concepts[13] = 1
        if pao2 is not None and pao2 < 60:
            concepts[14] = 1

        # Temperature
        if temp is not None and temp > 38.3:
            concepts[20] = 1
        if temp is not None and temp < 36.0:
            concepts[21] = 1
        if temp is not None and temp > 39.0:
            concepts[22] = 1

        # Hematologic
        if wbc is not None and wbc > 12:
            concepts[25] = 1
        if wbc is not None and wbc < 4:
            concepts[26] = 1
        if plt_val is not None and plt_val < 100:
            concepts[27] = 1
        if plt_val is not None and plt_val < 50:
            concepts[28] = 1
        if hgb is not None and hgb < 7:
            concepts[29] = 1

        # Metabolic
        if lactate is not None and lactate > 2:
            concepts[30] = 1
        if lactate is not None and lactate > 4:
            concepts[31] = 1
        if ph is not None and ph < 7.35:
            concepts[32] = 1
        if ph is not None and ph > 7.45:
            concepts[33] = 1
        if glucose is not None and glucose > 180:
            concepts[34] = 1
        if glucose is not None and glucose < 70:
            concepts[35] = 1

        # Renal
        if cr is not None and cr > 1.5:
            concepts[40] = 1
        if cr is not None and cr > 2.0:
            concepts[41] = 1
        if cr is not None and cr > 3.5:
            concepts[42] = 1

        # Hepatic
        if bili is not None and bili > 2.0:
            concepts[45] = 1
        if bili is not None and bili > 6.0:
            concepts[46] = 1

        return concepts

    def extract_sequence(self, features_df, stay_id) -> np.ndarray:
        """Extract concept vectors for all timesteps of a patient.

        Args:
            features_df: DataFrame with feature columns
            stay_id: patient stay identifier
        Returns:
            (T, NUM_CONCEPTS) array
        """
        patient = features_df[features_df["stay_id"] == stay_id]
        feature_names = [
            "heart_rate", "sbp", "dbp", "map", "resp_rate", "spo2",
            "temperature", "fio2", "wbc", "hemoglobin", "platelet",
            "creatinine", "bilirubin", "lactate", "pao2", "pco2", "ph", "glucose",
        ]
        concepts = []
        for _, row in patient.iterrows():
            feat_dict = {k: row.get(k) for k in feature_names if k in row.index}
            feat_dict = {k: v for k, v in feat_dict.items() if not (isinstance(v, float) and np.isnan(v))}
            concepts.append(self.extract(feat_dict))
        if not concepts:
            return np.zeros((0, self.vocab_size), dtype=np.float32)
        return np.stack(concepts)

    def extract_batch_tensor(
        self, features_df, stay_ids, max_hours: int = 48
    ) -> torch.Tensor:
        """Extract concepts for a batch of patients.

        Returns: (N, max_hours, NUM_CONCEPTS) tensor
        """
        N = len(stay_ids)
        result = np.zeros((N, max_hours, self.vocab_size), dtype=np.float32)
        for i, sid in enumerate(stay_ids):
            seq = self.extract_sequence(features_df, sid)
            T = min(len(seq), max_hours)
            if T > 0:
                result[i, :T, :] = seq[:T]
        return torch.from_numpy(result)

    def get_active_concept_names(self, concept_vector: np.ndarray) -> list[str]:
        """Get names of active concepts from a binary vector."""
        active = np.where(concept_vector > 0)[0]
        return [self.concept_names.get(i, f"concept_{i}") for i in active]
