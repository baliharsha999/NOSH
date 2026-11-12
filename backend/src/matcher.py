# -*- coding: utf-8 -*-
"""
backend/src/matcher.py
----------------------
Ingredient-Based Recipe Matching & Constraint Solver Engine for Task 2.

Core System Specifications:
  "Given a list of available ingredients and a set of criteria, retrieve and
   rank the closest existing recipes from the dataset. Do not invent new recipes.
   Never relax a hard constraint silently. If no eligible recipe exists, state
   that no match was found and, separately, offer the closest alternatives only
   if they are clearly marked as violating a constraint."

Key Architecture Features:
  1. Strict Constraint Solver: Disqualifies any recipe containing excluded allergens
     or forbidden ingredients BEFORE ranking.
  2. Explainable Set Decomposition:
     - Matched Ingredients (M): Ingredients the user already has.
     - Missing Ingredients (U): Ingredients the user needs to buy.
     - Unused Pantry Items (E): User ingredients not utilized.
  3. Transparent Scoring Formula:
     Score = (10.0 * |M| / |Recipe_Ingredients|) - (2.0 * |U|) - (0.2 * num_steps)
  4. LangChain / Pydantic Compatible Schemas for clear data transfer and pipeline integration.
"""

from typing import List, Dict, Set, Any, Optional
from pydantic import BaseModel, Field

try:
    from backend.src.normalization import clean_raw_ingredient, extract_normalized_keys, ALLERGEN_TAXONOMY
    from backend.src.ingestion import RecipeIndexBundle, get_index_bundle
except ImportError:
    from normalization import clean_raw_ingredient, extract_normalized_keys, ALLERGEN_TAXONOMY
    from ingestion import RecipeIndexBundle, get_index_bundle


class MatchRequest(BaseModel):
    """Structured request for ingredient-based recipe matching."""
    available_ingredients: List[str] = Field(..., description="Ingredients the user currently has in their pantry.")
    hard_exclusions: List[str] = Field(default_factory=list, description="Allergens or ingredients that MUST NOT be present.")
    max_steps: Optional[int] = Field(default=None, description="Maximum allowed instruction steps (e.g. at most 6 steps).")
    prefer_fewest_extra: bool = Field(default=True, description="Prefer recipes needing the fewest additional ingredients.")
    top_k: int = Field(default=5, description="Number of top recipes to return.")


class RecipeMatchResult(BaseModel):
    """Explainable result item for a matched recipe."""
    id: str
    title: str
    source: str
    score: float
    matched_ingredients: List[str]
    missing_ingredients: List[str]
    unused_pantry_items: List[str]
    num_steps: int
    num_ingredients: int
    explanation: str
    instructions: str
    original_ingredients: List[str]
    is_constraint_violation: bool = False
    violation_reason: Optional[str] = None


class MatchResponse(BaseModel):
    """Complete response conforming to Task 2 specifications."""
    query_available: List[str]
    hard_constraints_enforced: List[str]
    max_steps_limit: Optional[int]
    total_eligible_found: int
    no_match_found: bool
    results: List[RecipeMatchResult]
    closest_violating_alternatives: List[RecipeMatchResult] = Field(default_factory=list)
    quantity_sufficiency_notice: str = (
        "Name-level ingredient matching verified. Specific quantity sufficiency "
        "was not verified against pantry inventory."
    )


def _token_match(query_phrase: str, target_phrase: str) -> bool:
    """
    Checks if query_phrase matches target_phrase using strict word boundaries.
    Prevents false positive substring matches (e.g. 'egg' matching 'eggplant',
    'pea' matching 'peanut', or 'rice' matching 'licorice').
    """
    q_tokens = query_phrase.split()
    t_tokens = target_phrase.split()
    if not q_tokens or not t_tokens:
        return False
    if len(q_tokens) == 1:
        return q_tokens[0] in t_tokens
    # Multi-word phrase matching
    for i in range(len(t_tokens) - len(q_tokens) + 1):
        if t_tokens[i:i + len(q_tokens)] == q_tokens:
            return True
    return False


class RecipeMatcher:
    """
    Deterministic constraint-satisfaction and explainable ranking engine.
    """

    def __init__(self, index_bundle: Optional[RecipeIndexBundle] = None):
        if index_bundle is None:
            self.bundle = get_index_bundle()
        else:
            self.bundle = index_bundle

        self.inverted_index = self.bundle.inverted_index
        self.allergen_index = self.bundle.allergen_index
        self.recipe_store = self.bundle.recipe_store
        self.recipes_df = self.bundle.recipes_df

    def match(self, request: MatchRequest) -> MatchResponse:
        """
        Executes the 4-stage matching pipeline:
          Stage 1: Normalize user inputs (available & exclusions).
          Stage 2: Strict Boolean filtering (eliminate all hard exclusions & step violations).
          Stage 3: Explainable set partitioning & scoring over candidate recipes.
          Stage 4: Return ranked results or safe constraint-violation fallback.
        """
        # ---------------------------------------------------------------------
        # Stage 1: Normalize Inputs
        # ---------------------------------------------------------------------
        norm_avail_map: Dict[str, str] = {}
        for raw in request.available_ingredients:
            clean = clean_raw_ingredient(raw)
            if clean:
                norm_avail_map[clean] = raw
            else:
                norm_avail_map[raw.strip().lower()] = raw

        norm_avail_keys: Set[str] = set(norm_avail_map.keys())

        # Collect normalized exclusion keys & expand allergen categories
        norm_excl_keys: Set[str] = set()
        for raw_excl in request.hard_exclusions:
            clean_ex = clean_raw_ingredient(raw_excl) or raw_excl.strip().lower()
            norm_excl_keys.add(clean_ex)
            # Expand allergen aliases (e.g., peanuts -> groundnut, groundnut oil, peanut butter)
            for cat, keywords in ALLERGEN_TAXONOMY.items():
                if clean_ex == cat or clean_ex in keywords:
                    norm_excl_keys.update(keywords)

        # ---------------------------------------------------------------------
        # Stage 2: Strict Boolean Filtering (Hard Exclusions)
        # ---------------------------------------------------------------------
        # Collect all recipe IDs that MUST be disqualified
        disqualified_ids: Set[str] = set()

        for excl in norm_excl_keys:
            # Check allergen index first (e.g. 'dairy', 'peanuts')
            if excl in self.allergen_index:
                disqualified_ids.update(self.allergen_index[excl])
            
            # Check inverted index directly
            if excl in self.inverted_index:
                disqualified_ids.update(self.inverted_index[excl])
                
            # Check variant keys (e.g. 'peanut butter' if excl is 'peanut')
            for token, ids in self.inverted_index.items():
                if excl in token.split():
                    disqualified_ids.update(ids)

        # Candidate pool: recipes that contain AT LEAST ONE available ingredient
        candidate_ids: Set[str] = set()
        for avail_key in norm_avail_keys:
            if avail_key in self.inverted_index:
                candidate_ids.update(self.inverted_index[avail_key])
            for token, ids in self.inverted_index.items():
                if avail_key in token.split():
                    candidate_ids.update(ids)

        # Apply strict disqualification: candidates - disqualified_ids
        eligible_ids: Set[str] = candidate_ids.difference(disqualified_ids)

        # Step limit constraint
        if request.max_steps is not None and eligible_ids:
            # Filter by num_steps using recipes_df
            step_eligible = set(
                self.recipes_df[
                    (self.recipes_df["id"].isin(eligible_ids)) &
                    (self.recipes_df["num_steps"] <= request.max_steps)
                ]["id"]
            )
            eligible_ids = step_eligible

        # ---------------------------------------------------------------------
        # Stage 3: Explainable Set Partitioning & Scoring
        # ---------------------------------------------------------------------
        scored_eligible: List[RecipeMatchResult] = []

        for rid in eligible_ids:
            rec = self.recipe_store.get(rid)
            if not rec:
                continue

            rec_clean_ings = set(rec.get("cleaned_ing_names", []))
            if not rec_clean_ings:
                continue

            # Set operations using word-boundary matching
            matched = set()
            for r_ing in rec_clean_ings:
                if any(_token_match(u_ing, r_ing) for u_ing in norm_avail_keys):
                    matched.add(r_ing)

            # Missing = ingredients in recipe that user does not have
            missing = rec_clean_ings.difference(matched)

            # Which of the user's distinct pantry ingredients were utilized?
            user_items_utilized = set()
            for u_ing in norm_avail_keys:
                if any(_token_match(u_ing, r_ing) for r_ing in matched):
                    user_items_utilized.add(norm_avail_map.get(u_ing, u_ing))

            # Unused = user available ingredients not used in this recipe
            unused = set(norm_avail_map.values()).difference(user_items_utilized)

            # Explainable scoring formula:
            # We heavily reward using MORE of the user's distinct pantry ingredients (e.g. using all 4 items = 80 pts!),
            # reward high recipe coverage (|M| / |R|), penalize missing items to buy (|U|),
            # and gently penalize excessive instruction steps.
            matched_ratio = len(matched) / len(rec_clean_ings) if rec_clean_ings else 0.0
            
            # Multi-objective score:
            # - 20.0 pts per distinct user pantry ingredient utilized
            # - 10.0 pts * coverage ratio
            # - 3.0 pts penalty per missing ingredient needed
            # - 0.2 pts penalty per instruction step
            score = (
                (20.0 * len(user_items_utilized)) +
                (10.0 * matched_ratio) -
                (3.0 * len(missing)) -
                (0.2 * rec.get("num_steps", 0))
            )

            explanation = (
                f"Utilizes {len(user_items_utilized)}/{len(norm_avail_keys)} pantry ingredients ({', '.join(sorted(list(user_items_utilized)))}). "
                f"Recipe coverage: {matched_ratio*100:.0f}%. "
                f"Needs {len(missing)} additional item(s) to buy. "
                f"Steps: {rec.get('num_steps', 0)}."
            )

            scored_eligible.append(RecipeMatchResult(
                id=rid,
                title=rec["title"],
                source=rec.get("source", ""),
                score=round(score, 2),
                matched_ingredients=sorted(list(matched)),
                missing_ingredients=sorted(list(missing)),
                unused_pantry_items=sorted(list(unused)),
                num_steps=rec.get("num_steps", 0),
                num_ingredients=rec.get("num_ingredients", len(rec_clean_ings)),
                explanation=explanation,
                instructions=rec.get("instructions", ""),
                original_ingredients=rec.get("ingredients", []),
                is_constraint_violation=False
            ))

        # Sort eligible matches by highest score (and lowest missing ingredients)
        scored_eligible.sort(
            key=lambda x: (x.score, -len(x.missing_ingredients)),
            reverse=True
        )

        top_results = scored_eligible[:request.top_k]
        no_match = (len(top_results) == 0)

        # ---------------------------------------------------------------------
        # Stage 4: Strict No-Match & Soft Violation Fallback Protocol
        # ---------------------------------------------------------------------
        # If no eligible recipe matches, NEVER relax constraints silently.
        # Separately provide the closest alternatives clearly flagged.
        violating_alternatives: List[RecipeMatchResult] = []

        if no_match and candidate_ids:
            # Inspect candidate recipes that were excluded only by the hard constraint
            violating_pool = candidate_ids.intersection(disqualified_ids)
            for rid in list(violating_pool)[:5]:
                rec = self.recipe_store.get(rid)
                if not rec:
                    continue
                rec_clean_ings = set(rec.get("cleaned_ing_names", []))
                matched = rec_clean_ings.intersection(norm_avail_keys)
                missing = rec_clean_ings.difference(matched)
                
                # Determine which constraint was violated
                violated_reasons = []
                for ex in norm_excl_keys:
                    if ex in rec.get("allergens", []) or any(ex in k for k in rec.get("normalized_keys", [])):
                        violated_reasons.append(f"Contains excluded '{ex}'")

                violating_alternatives.append(RecipeMatchResult(
                    id=rid,
                    title=rec["title"],
                    source=rec.get("source", ""),
                    score=0.0,
                    matched_ingredients=sorted(list(matched)),
                    missing_ingredients=sorted(list(missing)),
                    unused_pantry_items=[],
                    num_steps=rec.get("num_steps", 0),
                    num_ingredients=len(rec_clean_ings),
                    explanation=f"[WARNING] Violates hard constraint: {', '.join(violated_reasons)}",
                    instructions=rec.get("instructions", ""),
                    original_ingredients=rec.get("ingredients", []),
                    is_constraint_violation=True,
                    violation_reason=", ".join(violated_reasons)
                ))

        return MatchResponse(
            query_available=request.available_ingredients,
            hard_constraints_enforced=request.hard_exclusions,
            max_steps_limit=request.max_steps,
            total_eligible_found=len(scored_eligible),
            no_match_found=no_match,
            results=top_results,
            closest_violating_alternatives=violating_alternatives
        )


if __name__ == "__main__":
    matcher = RecipeMatcher()
    
    print("\n" + "="*70)
    print("TASK 2 VERIFICATION TEST: EGGS, ONIONS, TOMATOES, RICE (MAX 6 STEPS, NO PEANUTS)")
    print("="*70)
    
    req = MatchRequest(
        available_ingredients=["eggs", "onions", "tomatoes", "rice"],
        hard_exclusions=["peanuts"],
        max_steps=6,
        prefer_fewest_extra=True,
        top_k=5
    )
    
    res = matcher.match(req)
    print(f"Total Eligible Matches Found: {res.total_eligible_found:,}")
    print(f"Hard Constraints Enforced: {res.hard_constraints_enforced}")
    print(f"Max Steps Limit: {res.max_steps_limit}")
    print("-" * 70)
    
    for idx, r in enumerate(res.results, 1):
        print(f"\n#{idx}: [{r.id}] {r.title} (Source: {r.source})")
        print(f"  • Score: {r.score} | Steps: {r.num_steps} | Total Ings: {r.num_ingredients}")
        print(f"  • Matched Pantry: {r.matched_ingredients}")
        print(f"  • Missing to Buy: {r.missing_ingredients}")
        print(f"  • Unused Pantry:  {r.unused_pantry_items}")
        print(f"  • Rationale: {r.explanation}")
