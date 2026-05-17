"""Counterfactual Engine: single-forward-pass counterfactual reasoning.

Key insight: since OntologyModulatedDARUAN conditions on concept_emb,
counterfactuals are just a forward pass with perturbed concept vectors.
No retraining, no SHAP-style N×forward passes needed.
"""
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import numpy as np

from ..knowledge.concept_extractor import ConceptExtractor, NUM_CONCEPTS


@dataclass
class CounterfactualResult:
    original_risk: float
    counterfactual_risk: float
    delta: float
    intervention: dict
    affected_concepts: list[str]
    attribution: Optional[torch.Tensor] = None

    @property
    def direction(self) -> str:
        if self.delta > 0.01:
            return "increases_risk"
        elif self.delta < -0.01:
            return "decreases_risk"
        return "no_effect"


class CounterfactualEngine:
    """Generate counterfactual predictions via concept perturbation."""

    def __init__(self, model: nn.Module, concept_extractor: ConceptExtractor):
        self.model = model
        self.extractor = concept_extractor

    def query(
        self,
        x: torch.Tensor,
        concept_vector: torch.Tensor,
        concept_emb: torch.Tensor,
        intervention: dict,
    ) -> CounterfactualResult:
        """Run counterfactual query.

        Args:
            x: (1, L, d_model) patient features
            concept_vector: (1, L, num_concepts) current concepts
            concept_emb: (1, d_ontology) current embedding
            intervention: dict with keys:
                - "remove_concepts": list of concept IDs to remove
                - "add_concepts": list of concept IDs to add
        Returns:
            CounterfactualResult
        """
        self.model.eval()

        # Original prediction
        with torch.no_grad():
            out_orig = self.model(x, concept_emb)
            if out_orig.dim() == 3:
                orig_risk = torch.sigmoid(out_orig[0, -1, 0]).item()
            else:
                orig_risk = torch.sigmoid(out_orig[0, 0]).item()

        # Perturb concept vector
        cf_concepts = concept_vector.clone()
        affected = []

        remove_ids = intervention.get("remove_concepts", [])
        for cid in remove_ids:
            cf_concepts[:, :, cid] = 0.0
            name = self.extractor.concept_names.get(cid, f"concept_{cid}")
            affected.append(f"-{name}")

        add_ids = intervention.get("add_concepts", [])
        for cid in add_ids:
            cf_concepts[:, :, cid] = 1.0
            name = self.extractor.concept_names.get(cid, f"concept_{cid}")
            affected.append(f"+{name}")

        # Compute counterfactual embedding
        from ..knowledge.ontology_embedder import OntologyEmbedder
        embedder = OntologyEmbedder(
            vocab_size=NUM_CONCEPTS,
            d_ontology=concept_emb.shape[-1],
            device=concept_emb.device,
        )
        # Use mean of perturbed concepts as new embedding
        cf_emb = cf_concepts.mean(dim=1) @ embedder.embedding

        # Counterfactual prediction
        with torch.no_grad():
            out_cf = self.model(x, cf_emb)
            if isinstance(out_cf, tuple):
                out_cf = out_cf[0]
            if out_cf.dim() == 3:
                cf_risk = torch.sigmoid(out_cf[0, -1, 0]).item()
            else:
                cf_risk = torch.sigmoid(out_cf[0, 0]).item()

        delta = cf_risk - orig_risk

        return CounterfactualResult(
            original_risk=orig_risk,
            counterfactual_risk=cf_risk,
            delta=delta,
            intervention=intervention,
            affected_concepts=affected,
        )

    def query_simple(
        self,
        x: torch.Tensor,
        concept_emb: torch.Tensor,
        intervention: dict,
    ) -> CounterfactualResult:
        """Simplified query that directly perturbs concept_emb.

        Args:
            x: (1, L, d_model) patient features
            concept_emb: (1, d_ontology) current embedding
            intervention: dict with "scale_factor" or "perturb_dims"
        Returns:
            CounterfactualResult
        """
        self.model.eval()

        with torch.no_grad():
            out_orig = self.model(x, concept_emb)
            if isinstance(out_orig, tuple):
                out_orig = out_orig[0]
            if out_orig.dim() == 3:
                orig_risk = torch.sigmoid(out_orig[0, -1, 0]).item()
            else:
                orig_risk = torch.sigmoid(out_orig[0, 0]).item()

        # Perturb embedding directly
        cf_emb = concept_emb.clone()
        affected = []

        if "zero_dims" in intervention:
            for dim in intervention["zero_dims"]:
                cf_emb[:, dim] = 0.0
                affected.append(f"zero_dim_{dim}")

        if "scale_factor" in intervention:
            cf_emb = cf_emb * intervention["scale_factor"]
            affected.append(f"scale_{intervention['scale_factor']}")

        if "add_noise" in intervention:
            noise = torch.randn_like(cf_emb) * intervention["add_noise"]
            cf_emb = cf_emb + noise
            affected.append(f"noise_{intervention['add_noise']}")

        with torch.no_grad():
            out_cf = self.model(x, cf_emb)
            if isinstance(out_cf, tuple):
                out_cf = out_cf[0]
            if out_cf.dim() == 3:
                cf_risk = torch.sigmoid(out_cf[0, -1, 0]).item()
            else:
                cf_risk = torch.sigmoid(out_cf[0, 0]).item()

        delta = cf_risk - orig_risk

        return CounterfactualResult(
            original_risk=orig_risk,
            counterfactual_risk=cf_risk,
            delta=delta,
            intervention=intervention,
            affected_concepts=affected,
        )
