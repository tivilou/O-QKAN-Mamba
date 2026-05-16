"""Ontology concept embedder: concept IDs → dense vectors.

Phase 3: random + learnable embeddings.
Future: node2vec on ICD-10 DAG for structure-aware initialization.
"""
import os
from typing import Optional

import torch
import torch.nn as nn
import numpy as np


class OntologyEmbedder(nn.Module):
    """Convert multi-hot concept vectors to dense ontology embeddings."""

    def __init__(
        self,
        vocab_size: int = 50,
        d_ontology: int = 64,
        pretrained: Optional[torch.Tensor] = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_ontology = d_ontology

        if pretrained is not None:
            assert pretrained.shape == (vocab_size, d_ontology)
            self.embedding = nn.Parameter(pretrained.to(device))
        else:
            self.embedding = nn.Parameter(
                torch.randn(vocab_size, d_ontology, device=device) * 0.02
            )

    def forward(self, concepts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            concepts: (B, T, V) multi-hot concept vectors
        Returns:
            (B, T, d_ontology) dense embeddings
        """
        return concepts @ self.embedding

    def get_embedding(self, concept_id: int) -> torch.Tensor:
        """Get embedding for a single concept."""
        return self.embedding[concept_id]

    def get_embeddings(self, concept_ids: list[int]) -> torch.Tensor:
        """Get embeddings for multiple concepts. Shape: (N, d_ontology)."""
        return self.embedding[concept_ids]

    def encode_concepts_at_time(
        self, concept_vector: torch.Tensor
    ) -> torch.Tensor:
        """Encode a single timestep's concept vector.

        Args:
            concept_vector: (V,) or (B, V) multi-hot
        Returns:
            (d_ontology,) or (B, d_ontology)
        """
        return concept_vector @ self.embedding

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location="cpu"))
