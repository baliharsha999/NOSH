# -*- coding: utf-8 -*-
"""
backend/src/retriever.py
------------------------
LangChain-Powered Recipe Retriever for Task 1 (Natural Language Search & Q&A).

Why LangChain Here?
  LangChain provides the standard `BaseRetriever` and `Document` abstractions.
  By wrapping our high-speed Inverted Index into a LangChain Retriever, we:
  1. Adhere to enterprise standard LLM frameworks.
  2. Emit clean `Document(page_content=..., metadata={...})` objects with stable Recipe IDs.
  3. Seamlessly support hard constraint pre-filtering (e.g. `exclude_allergens=['dairy']`).
  4. Enable transparent pipelining into LangChain Expression Language (LCEL) chains.
"""

import re
from typing import List, Dict, Set, Any, Optional
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field, ConfigDict

from normalization import clean_raw_ingredient
from ingestion import RecipeIndexBundle, get_index_bundle
from vector_store import get_faiss_store

STOPWORDS = {
    "help", "me", "prepare", "make", "cook", "find", "recipe", "recipes",
    "for", "the", "a", "an", "how", "to", "do", "i", "what", "is", "show",
    "give", "please", "can", "you", "tell", "want", "need", "dish", "dishes",
    "using", "with", "about", "suggest", "some", "good"
}


def _get_title_index(bundle: Any) -> Dict[str, Set[str]]:
    """Builds and caches an inverted index over words in all recipe titles."""
    if hasattr(bundle, "_title_index") and bundle._title_index is not None:
        return bundle._title_index

    title_index: Dict[str, Set[str]] = {}
    for rid, rec in bundle.recipe_store.items():
        title = rec.get("title", "").lower()
        words = re.findall(r"\b[a-z0-9]+\b", title)
        for w in words:
            if len(w) > 2:
                if w not in title_index:
                    title_index[w] = set()
                title_index[w].add(rid)
            if w.endswith("s") and len(w) > 3:
                stem = w[:-1]
                if stem not in title_index:
                    title_index[stem] = set()
                title_index[stem].add(rid)

    bundle._title_index = title_index
    return title_index


class RecipeRetriever(BaseRetriever):
    """
    A LangChain-native Retriever that performs fast lexical-token retrieval
    over 121,684 curated recipes with strict constraint enforcement.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    bundle: Any = Field(default_factory=get_index_bundle)
    hard_exclusions: List[str] = Field(default_factory=list)
    top_k: int = 5

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        Executes query retrieval:
          1. Identifies disqualified recipes from hard_exclusions (e.g. dairy).
          2. Scores recipes based on query token matches in title and ingredients.
          3. Emits LangChain Document objects with stable Recipe IDs and metadata.
        """
        # Step 1: Disqualify excluded items / allergens
        disqualified_ids: Set[str] = set()
        for excl in self.hard_exclusions:
            clean_ex = clean_raw_ingredient(excl) or excl.strip().lower()
            if clean_ex in self.bundle.allergen_index:
                disqualified_ids.update(self.bundle.allergen_index[clean_ex])
            if clean_ex in self.bundle.inverted_index:
                disqualified_ids.update(self.bundle.inverted_index[clean_ex])
            for token, ids in self.bundle.inverted_index.items():
                if clean_ex in token.split():
                    disqualified_ids.update(ids)

        # Step 2: Tokenize and extract search intent words
        q_raw = query.strip().lower()
        all_words = re.findall(r"\b[a-z0-9]+\b", q_raw)
        meaningful_words = [w for w in all_words if w not in STOPWORDS and len(w) > 2]
        if not meaningful_words:
            meaningful_words = [w for w in all_words if len(w) > 1] or [q_raw]

        search_phrase = " ".join(meaningful_words)
        title_index = _get_title_index(self.bundle)

        candidate_scores: Dict[str, float] = {}

        # Step 3A: Title Index Matching (high precision for recipe title queries)
        for w in meaningful_words:
            stems = [w]
            if w.endswith("s") and len(w) > 3:
                stems.append(w[:-1])
            else:
                stems.append(w + "s")
            
            matched_rids: Set[str] = set()
            for s in stems:
                matched_rids.update(title_index.get(s, set()))
            for rid in matched_rids:
                if rid not in disqualified_ids:
                    candidate_scores[rid] = candidate_scores.get(rid, 0.0) + 15.0

        # Step 3B: Inverted Index Matching (ingredient matching)
        for w in meaningful_words:
            matched_ids = self.bundle.inverted_index.get(w, set())
            for rid in matched_ids:
                if rid not in disqualified_ids:
                    candidate_scores[rid] = candidate_scores.get(rid, 0.0) + 5.0

        # Step 3C: Title exact phrase and coverage boost
        for rid in list(candidate_scores.keys()):
            rec = self.bundle.recipe_store.get(rid)
            if not rec:
                continue
            t_lower = rec["title"].lower()
            # Direct phrase match in title (e.g. "turkey sausage tacos" in "Turkey Sausage Tacos")
            if search_phrase in t_lower or search_phrase.rstrip("s") in t_lower:
                candidate_scores[rid] += 150.0
            # All meaningful words present in title
            elif all((w in t_lower or w.rstrip("s") in t_lower) for w in meaningful_words):
                candidate_scores[rid] += 80.0

        # FAISS Dense Semantic Vector Reranking on pre-filtered candidate pool
        top_candidate_ids = [r for r, _ in sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:50]]
        recipe_texts = {}
        for rid in top_candidate_ids:
            rec = self.bundle.recipe_store.get(rid)
            if rec:
                recipe_texts[rid] = f"Recipe: {rec['title']}. Ingredients: {', '.join(rec.get('cleaned_ing_names', [])[:8])}"

        faiss_sim_map = {}
        if top_candidate_ids:
            faiss_store = get_faiss_store()
            faiss_res = faiss_store.search_candidates(search_phrase or q_raw, top_candidate_ids, recipe_texts, top_k=self.top_k * 2)
            for f_rid, f_score in faiss_res:
                faiss_sim_map[f_rid] = f_score
                candidate_scores[f_rid] = candidate_scores.get(f_rid, 0.0) + (f_score * 4.0)

        # Sort candidates by combined hybrid score (Lexical + FAISS Dense Cosine Similarity)
        sorted_candidates = sorted(
            candidate_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.top_k]

        # Step 4: Convert to LangChain Document objects with complete recipe details
        documents: List[Document] = []
        for rid, score in sorted_candidates:
            rec = self.bundle.recipe_store.get(rid)
            if not rec:
                continue

            # Format page_content for LLM consumption
            page_content = (
                f"Recipe Title: {rec['title']}\n"
                f"Source: {rec['source']}\n"
                f"Recipe ID: {rec['id']}\n"
                f"Instruction Steps Count: {rec['num_steps']}\n"
                f"Ingredients Count: {rec['num_ingredients']}\n\n"
                f"Ingredients:\n" + "\n".join(f"- {ing}" for ing in rec['ingredients']) + "\n\n"
                f"Instructions:\n{rec['instructions']}"
            )

            metadata = {
                "id": rid,
                "title": rec["title"],
                "source": rec["source"],
                "score": score,
                "faiss_cosine_similarity": faiss_sim_map.get(rid, 0.0),
                "num_steps": rec["num_steps"],
                "num_ingredients": rec["num_ingredients"],
                "picture_link": rec.get("picture_link")
            }

            documents.append(Document(page_content=page_content, metadata=metadata))

        return documents


if __name__ == "__main__":
    retriever = RecipeRetriever(top_k=3)
    
    print("\n" + "="*70)
    print("LANGCHAIN RETRIEVER TEST 1: 'Penne Pasta' (No Exclusions)")
    print("="*70)
    docs1 = retriever.invoke("Penne Pasta")
    for d in docs1:
        print(f"• [{d.metadata['id']}] {d.metadata['title']} (Score: {d.metadata['score']})")
        print(f"  Ingredients Preview: {d.page_content.split('Instructions:')[0][:150]}...\n")

    print("="*70)
    print("LANGCHAIN RETRIEVER TEST 2: 'Penne Pasta' with Hard Exclusion ['dairy']")
    print("="*70)
    dairy_free_retriever = RecipeRetriever(hard_exclusions=["dairy"], top_k=3)
    docs2 = dairy_free_retriever.invoke("Penne Pasta")
    for d in docs2:
        print(f"• [{d.metadata['id']}] {d.metadata['title']} (Score: {d.metadata['score']})")
        print(f"  Ingredients Preview: {d.page_content.split('Instructions:')[0][:150]}...\n")
