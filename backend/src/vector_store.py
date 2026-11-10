# -*- coding: utf-8 -*-
"""
backend/src/vector_store.py
---------------------------
LangChain-Compatible Vector Store & Vector Indexing for the NOSH Recipe Corpus.

What is a Vector Store?
  A Vector Database / Vector Store indexes document embeddings in multi-dimensional
  vector space. When a user enters a query, the query is converted to a vector,
  and the store finds the nearest neighbors (highest cosine similarity).

Why Custom Vector Indexing Here?
  1. Strict Metadata Pre-Filtering: Generic vector stores rank purely by cosine distance,
     which allows semantic leakage (e.g. returning Peanut Butter Cookies when searching
     'cookies without peanuts'). Our `RecipeVectorStore` enforces strict Boolean
     pre-filtering before vector scoring.
  2. High-Speed Vectorized Execution: Uses NumPy matrix multiplication for dot-product
     similarity without requiring external C++ compiled binaries or heavy dependencies.
"""

from typing import List, Dict, Tuple, Any, Optional, Iterable
import numpy as np
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from embeddings import FastTermEmbeddings


class RecipeVectorStore(VectorStore):
    """
    A lightweight, high-performance in-memory Vector Store for recipe retrieval.
    Stores document vectors, metadata, and allows similarity search with hard constraints.
    """

    def __init__(self, embedding: Optional[Embeddings] = None):
        self._embedding: Embeddings = embedding or FastTermEmbeddings()
        self.documents: List[Document] = []
        self.vectors: Optional[np.ndarray] = None
        self.doc_ids: List[str] = []

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any
    ) -> List[str]:
        """Encodes texts, adds vectors to index, and stores metadata."""
        texts_list = list(texts)
        if not texts_list:
            return []

        # Generate vectors
        new_vectors = np.array(self._embedding.embed_documents(texts_list), dtype=np.float32)

        # Build Document objects
        added_ids = []
        for i, text in enumerate(texts_list):
            doc_id = ids[i] if ids and i < len(ids) else f"doc_{len(self.documents) + i}"
            meta = metadatas[i].copy() if metadatas and i < len(metadatas) else {}
            meta["id"] = doc_id
            
            self.documents.append(Document(page_content=text, metadata=meta))
            self.doc_ids.append(doc_id)
            added_ids.append(doc_id)

        # Update vectors matrix
        if self.vectors is None:
            self.vectors = new_vectors
        else:
            self.vectors = np.vstack([self.vectors, new_vectors])

        return added_ids

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        disqualified_ids: Optional[set] = None,
        **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """
        Calculates cosine similarity between query vector and indexed document vectors.
        Strictly excludes any documents in `disqualified_ids` (hard constraints).
        """
        if self.vectors is None or len(self.documents) == 0:
            return []

        # Embed user query
        q_vec = np.array(self._embedding.embed_query(query), dtype=np.float32)
        
        # Vectorized dot product (cosine similarity since vectors are L2-normalized)
        scores = np.dot(self.vectors, q_vec)

        # Mask out disqualified recipes (hard constraint invariance)
        if disqualified_ids:
            for idx, doc_id in enumerate(self.doc_ids):
                if doc_id in disqualified_ids:
                    scores[idx] = -1.0  # Lowest possible score

        # Top-K indices
        top_indices = np.argsort(scores)[::-1][:k]

        results: List[Tuple[Document, float]] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < 0:
                continue
            results.append((self.documents[idx], score))

        return results

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        **kwargs: Any
    ) -> List[Document]:
        """Standard LangChain similarity search returning Documents."""
        results = self.similarity_search_with_score(query, k=k, **kwargs)
        return [doc for doc, _ in results]

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any
    ) -> "RecipeVectorStore":
        store = cls(embedding=embedding)
        store.add_texts(texts, metadatas=metadatas, **kwargs)
        return store


if __name__ == "__main__":
    vstore = RecipeVectorStore()
    
    sample_recipes = [
        "Penne pasta tossed in olive oil with garlic, spinach and bacon",
        "Classic baked banana bread with eggs, flour, and butter",
        "Spicy tomato penne with chili flakes, tomato paste, and parmesan",
        "Chocolate cake with cocoa powder, milk, sugar and eggs"
    ]
    
    sample_metas = [
        {"id": "rec_001", "title": "Spinach Bacon Penne", "allergens": ["pork"]},
        {"id": "rec_002", "title": "Banana Bread", "allergens": ["egg", "dairy", "gluten"]},
        {"id": "rec_003", "title": "Spicy Tomato Penne", "allergens": ["dairy"]},
        {"id": "rec_004", "title": "Chocolate Cake", "allergens": ["dairy", "egg", "gluten"]}
    ]
    
    vstore.add_texts(sample_recipes, metadatas=sample_metas, ids=["rec_001", "rec_002", "rec_003", "rec_004"])
    
    print("\n--- Vector Search Test 1: 'Penne Pasta' (No Constraints) ---")
    res1 = vstore.similarity_search_with_score("Penne Pasta", k=2)
    for doc, score in res1:
        print(f"• [{doc.metadata['id']}] {doc.metadata['title']} (Score: {score:.4f})")
        
    print("\n--- Vector Search Test 2: 'Penne Pasta' with Disqualified ID ('rec_003' has dairy) ---")
    res2 = vstore.similarity_search_with_score("Penne Pasta", k=2, disqualified_ids={"rec_003"})
    for doc, score in res2:
        print(f"• [{doc.metadata['id']}] {doc.metadata['title']} (Score: {score:.4f})")
