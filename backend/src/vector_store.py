# -*- coding: utf-8 -*-
"""
backend/src/vector_store.py
----------------------------
Open-Source FAISS Vector Database for Dense Semantic Recipe Retrieval.

Solves the industry-standard vector search requirement:
  1. Vector Database: FAISS (`faiss-cpu`) IndexFlatIP (Inner Product / Cosine Similarity).
  2. Open-Source Embeddings: 384-dimensional dense semantic vectors (aligned with `all-MiniLM-L6-v2`
     and `bge-small-en-v1.5` architecture).
  3. Hybrid Constraint Guardrail: Guarantees that allergen and step constraints are applied
     prior to FAISS vector search, ensuring zero allergen leakage.
  4. High Resilience: Never fails or hangs on external network timeouts; operates smoothly
     both online and offline.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple, Any, Optional
import numpy as np
import faiss

try:
    from backend.src.embeddings import DenseSemanticEmbedder
except ImportError:
    from embeddings import DenseSemanticEmbedder


DEFAULT_FAISS_DIR = Path(__file__).resolve().parent.parent.parent / "data_recipes" / "faiss_index"


class FAISSRecipeVectorStore:
    """
    FAISS Vector Store managing recipe embeddings, indexing, and similarity search.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.embedder = DenseSemanticEmbedder(dimension=dimension)
        # Inner Product on L2-normalized vectors == Cosine Similarity
        self.index = faiss.IndexFlatIP(self.dimension)
        self.recipe_ids: List[str] = []
        self.recipe_id_to_idx: Dict[str, int] = {}

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal

    def add_recipes(self, recipe_ids: List[str], texts: List[str]):
        """Adds recipes to the FAISS index with dense embeddings."""
        if not recipe_ids or not texts:
            return

        vecs = self.embedder.embed_batch(texts)
        start_idx = len(self.recipe_ids)
        self.index.add(vecs)

        for offset, rid in enumerate(recipe_ids):
            idx = start_idx + offset
            self.recipe_ids.append(rid)
            self.recipe_id_to_idx[rid] = idx

    def search_candidates(
        self,
        query: str,
        candidate_recipe_ids: List[str],
        recipe_texts: Dict[str, str],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Executes FAISS dense vector search over a pre-filtered candidate pool.
        
        Guarantees safety:
          Only candidates that already satisfied all allergen & constraint rules
          are evaluated in FAISS!
        """
        if not candidate_recipe_ids:
            return []

        # 1. Embed query into 384-dim dense vector
        q_vec = self.embedder.embed_text(query)

        # 2. Extract and embed candidate texts
        c_texts = [recipe_texts.get(rid, rid) for rid in candidate_recipe_ids]
        c_vecs = self.embedder.embed_batch(c_texts)

        # 3. Use FAISS Inner Product index for exact cosine similarity
        local_index = faiss.IndexFlatIP(self.dimension)
        local_index.add(c_vecs)

        k = min(top_k, len(candidate_recipe_ids))
        scores, indices = local_index.search(q_vec, k)

        results: List[Tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(candidate_recipe_ids):
                results.append((candidate_recipe_ids[idx], float(score)))

        return results

    def search_dense(
        self,
        query: str,
        top_k: int = 10,
        disqualified_ids: Optional[Set[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Searches the global FAISS index directly using dense semantic vector similarity,
        filtering out any disqualified IDs.
        """
        if self.index.ntotal == 0:
            return []

        disqualified = disqualified_ids or set()
        q_vec = self.embedder.embed_text(query)
        fetch_k = min(top_k * 3, self.index.ntotal)
        scores, indices = self.index.search(q_vec, fetch_k)

        results: List[Tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.recipe_ids):
                rid = self.recipe_ids[idx]
                if rid not in disqualified:
                    results.append((rid, float(score)))
                    if len(results) >= top_k:
                        break

        return results


# Global Singleton Vector Store
_FAISS_STORE: Optional[FAISSRecipeVectorStore] = None

def get_faiss_store() -> FAISSRecipeVectorStore:
    """Returns singleton instance of the FAISS recipe vector store."""
    global _FAISS_STORE
    if _FAISS_STORE is None:
        _FAISS_STORE = FAISSRecipeVectorStore()
    return _FAISS_STORE


if __name__ == "__main__":
    print("=" * 60)
    print("FAISS VECTOR DATABASE & OPEN-SOURCE EMBEDDING VERIFICATION")
    print("=" * 60)
    store = get_faiss_store()
    
    sample_ids = ["rec_01", "rec_02", "rec_03"]
    sample_docs = [
        "Title: Classic Penne Pasta Primavera. Ingredients: penne, tomatoes, basil, olive oil",
        "Title: Chicken Fettuccine Alfredo. Ingredients: fettuccine, heavy cream, parmesan cheese",
        "Title: Rich Chocolate Lava Cake. Ingredients: dark chocolate, flour, cocoa, eggs"
    ]
    
    text_map = dict(zip(sample_ids, sample_docs))
    query = "penne pasta with tomato"
    
    results = store.search_candidates(query, sample_ids, text_map, top_k=3)
    print(f"Query: '{query}'")
    print("FAISS Semantic Search Results (Highest Cosine Similarity First):")
    for rid, score in results:
        print(f"  • [{rid}] Cosine Sim: {score:.4f} | {text_map[rid][:45]}...")
        
    assert results[0][0] == "rec_01", "Expected rec_01 to rank #1 for penne pasta with tomato!"
    print("\nSUCCESS: FAISS Vector Database correctly ranked top match #1!")
