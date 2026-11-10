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

from typing import List, Dict, Set, Any, Optional
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field

from normalization import clean_raw_ingredient
from ingestion import RecipeIndexBundle, get_index_bundle


class RecipeRetriever(BaseRetriever):
    """
    A LangChain-native Retriever that performs fast lexical-token retrieval
    over 121,684 curated recipes with strict constraint enforcement.
    """
    bundle: RecipeIndexBundle = Field(default_factory=get_index_bundle)
    hard_exclusions: List[str] = Field(default_factory=list)
    top_k: int = 5

    class Config:
        arbitrary_types_allowed = True

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

        # Step 2: Tokenize query
        q_clean = clean_raw_ingredient(query) or query.strip().lower()
        q_tokens = [t for t in q_clean.split() if len(t) > 2]
        if not q_tokens:
            q_tokens = [query.strip().lower()]

        # Step 3: Candidate matching via Inverted Index (sub-15ms)
        candidate_scores: Dict[str, float] = {}

        for token in q_tokens:
            matched_ids = self.bundle.inverted_index.get(token, set())
            for rid in matched_ids:
                if rid not in disqualified_ids:
                    candidate_scores[rid] = candidate_scores.get(rid, 0.0) + 2.0

        # Title exact match bonus on the candidate pool only (instant)
        for rid in list(candidate_scores.keys()):
            rec = self.bundle.recipe_store.get(rid)
            if not rec:
                continue
            title_lower = rec["title"].lower()
            if q_clean in title_lower:
                candidate_scores[rid] += 5.0
            for token in q_tokens:
                if token in title_lower:
                    candidate_scores[rid] += 1.0

        # Sort candidates by score
        sorted_candidates = sorted(
            candidate_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.top_k]

        # Step 4: Convert to LangChain Document objects
        documents: List[Document] = []
        for rid, score in sorted_candidates:
            rec = self.bundle.recipe_store.get(rid)
            if not rec:
                continue

            # Format page_content for LLM consumption
            page_content = (
                f"Recipe Title: {rec['title']}\n"
                f"Source: {rec['source']}\n"
                f"Ingredients:\n" + "\n".join(f"- {ing}" for ing in rec['ingredients']) + "\n\n"
                f"Instructions:\n{rec['instructions']}"
            )

            metadata = {
                "id": rid,
                "title": rec["title"],
                "source": rec["source"],
                "score": score,
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
