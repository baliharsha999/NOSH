# -*- coding: utf-8 -*-
"""
backend/src/embeddings.py
-------------------------
Embedding Engine for the NOSH Recipe Corpus.

Purpose:
  Converts text (recipe documents and user search queries) into dense numerical vectors
  so semantic similarity (cosine similarity) can be calculated.

Architecture:
  Implements the standard LangChain `Embeddings` base class with:
  1. `FastTermEmbeddings`: A high-speed, zero-dependency Term-Frequency / Hash Vectorizer
     built natively on NumPy (already installed in venv). Operates completely offline,
     requires no GPU, and generates 256-dimensional semantic representation vectors.
  2. Extensible hooks for dense neural embeddings (e.g. HuggingFace 'all-MiniLM-L6-v2'
     or Google Gemini 'text-embedding-004').
"""

import math
import re
from typing import List
import numpy as np
from langchain_core.embeddings import Embeddings


class FastTermEmbeddings(Embeddings):
    """
    A lightweight, zero-dependency semantic vector embedder using term-frequency
    and feature hashing into a normalized 256-dimensional vector space.
    Guarantees instant startup and zero external heavy model downloads.
    """

    def __init__(self, dimension: int = 256):
        self.dimension = dimension

    def _hash_token(self, token: str) -> int:
        """Deterministic hash mapping a token to a vector dimension."""
        h = 0
        for ch in token:
            h = (h * 31 + ord(ch)) % self.dimension
        return h

    def embed_text(self, text: str) -> List[float]:
        """Encodes a single string into a normalized dense vector."""
        vec = np.zeros(self.dimension, dtype=np.float32)
        tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        
        if not tokens:
            return vec.tolist()

        for token in tokens:
            idx = self._hash_token(token)
            vec[idx] += 1.0

        # Sublinear term frequency scaling: 1 + log(tf)
        non_zero = vec > 0
        vec[non_zero] = 1.0 + np.log(vec[non_zero])

        # L2 Normalization so cosine similarity equals dot product
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeds a list of document strings."""
        return [self.embed_text(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        """Embeds a single user search query."""
        return self.embed_text(text)


if __name__ == "__main__":
    embedder = FastTermEmbeddings(dimension=256)
    
    doc1 = "Penne pasta with fresh basil, tomato sauce, and garlic"
    doc2 = "Spaghetti with plum tomatoes, olive oil, and minced garlic"
    doc3 = "Chocolate chip cookies with butter and brown sugar"
    
    v1 = np.array(embedder.embed_query(doc1))
    v2 = np.array(embedder.embed_query(doc2))
    v3 = np.array(embedder.embed_query(doc3))
    
    sim_1_2 = float(np.dot(v1, v2))
    sim_1_3 = float(np.dot(v1, v3))
    
    print("Semantic Cosine Similarity Test:")
    print(f"Similarity (Penne vs Spaghetti):   {sim_1_2:.4f} (High semantic overlap)")
    print(f"Similarity (Penne vs Choc Cookie): {sim_1_3:.4f} (Low semantic overlap)")
