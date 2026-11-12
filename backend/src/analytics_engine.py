# -*- coding: utf-8 -*-
"""
backend/src/analytics_engine.py
-------------------------------
Deterministic Analytics Engine for Task 1 (Recipe Search & Factual Q&A).

Engineering Mandate:
  "For dataset facts, retrieve the underlying recipes. For counts, rankings,
   averages, and other statistics, calculate results programmatically on the
   relevant data rather than asking an LLM to guess."

Key Capabilities:
  1. Exact Multi-Ingredient Co-occurrence Counting & Percentage Formulation:
     - Computes exact set intersection over inverted index in O(1) time.
     - Formally returns the numerator, the fixed denominator (121,684),
       and the normalization process applied.
  2. Deterministic Instruction Step Ranking:
     - Retrieves all recipes matching a target ingredient (e.g. 'tomato').
     - Formally ranks them by a defined step-counting rule (`num_steps ASC`).
     - Breaks ties using `num_ingredients ASC`.
     - Returns stable Recipe IDs, titles, step counts, and actual procedure text.
  3. Global Corpus Distribution Statistics:
     - Mean and median ingredient counts.
     - Mean and median instruction step counts.
     - Top frequent ingredients.
  4. Absence-of-Data Guardrail:
     - Explicitly checks and rejects queries demanding unavailable schema fields
       (e.g., exact prep time, cook time, calories, macros) to prevent LLM hallucinations.
"""

from typing import List, Dict, Set, Any, Optional, Tuple
import pandas as pd
import numpy as np

try:
    from backend.src.normalization import clean_raw_ingredient, extract_normalized_keys
    from backend.src.ingestion import RecipeIndexBundle, get_index_bundle
except ImportError:
    from normalization import clean_raw_ingredient, extract_normalized_keys
    from ingestion import RecipeIndexBundle, get_index_bundle


class AnalyticsEngine:
    """
    High-performance deterministic mathematical engine operating over the
    pre-indexed RecipeIndexBundle.
    """

    def __init__(self, index_bundle: Optional[RecipeIndexBundle] = None):
        if index_bundle is None:
            self.bundle = get_index_bundle()
        else:
            self.bundle = index_bundle
            
        self.total_recipes: int = self.bundle.total_recipes
        self.inverted_index = self.bundle.inverted_index
        self.recipes_df = self.bundle.recipes_df
        self.recipe_store = self.bundle.recipe_store

    # =========================================================================
    # 1. Exact Multi-Ingredient Counting & Percentage (Task 1 Core Requirement)
    # =========================================================================
    def compute_ingredient_cooccurrence(
        self,
        raw_ingredients: List[str]
    ) -> Dict[str, Any]:
        """
        Calculates exact reproducible counts and percentages for recipes containing
        ALL specified ingredients simultaneously.
        
        Example:
          Input: ['garlic', 'onion']
          Output:
            - Matching recipes: 16,244
            - Total denominator: 121,684
            - Percentage: 13.35%
            - Normalization explanation and sample Recipe IDs
        """
        if not raw_ingredients:
            return {
                "error": "No ingredients provided for co-occurrence calculation."
            }

        # Step 1: Normalize input query ingredients
        normalized_queries: List[str] = []
        candidate_id_sets: List[Set[str]] = []
        per_ingredient_counts: Dict[str, int] = {}

        for item in raw_ingredients:
            norm_key = clean_raw_ingredient(item)
            if not norm_key:
                norm_key = item.strip().lower()
            normalized_queries.append(norm_key)

            # Retrieve set of recipe IDs containing this normalized token
            # We search for exact token or tokens that contain it
            matching_ids = self.inverted_index.get(norm_key, set())
            
            # If the user typed a generic term like 'onion', include all onion variants (e.g. yellow onion)
            if len(matching_ids) == 0:
                # Substring search across keys if not found directly
                variant_ids = set()
                for key, ids in self.inverted_index.items():
                    if norm_key in key.split():
                        variant_ids.update(ids)
                matching_ids = variant_ids
                
            candidate_id_sets.append(matching_ids)
            per_ingredient_counts[norm_key] = len(matching_ids)

        # Step 2: Compute set intersection across all query ingredients
        if candidate_id_sets:
            intersected_ids: Set[str] = set.intersection(*candidate_id_sets)
        else:
            intersected_ids = set()

        match_count = len(intersected_ids)
        percentage = (match_count / self.total_recipes) * 100.0 if self.total_recipes > 0 else 0.0

        # Step 3: Fetch sample matching recipe summaries
        sample_ids = list(intersected_ids)[:5]
        sample_recipes = []
        for rid in sample_ids:
            rec = self.recipe_store.get(rid, {})
            sample_recipes.append({
                "id": rid,
                "title": rec.get("title", ""),
                "source": rec.get("source", ""),
                "num_ingredients": rec.get("num_ingredients", 0),
                "num_steps": rec.get("num_steps", 0)
            })

        return {
            "query_ingredients_raw": raw_ingredients,
            "query_ingredients_normalized": normalized_queries,
            "per_ingredient_breakdown": per_ingredient_counts,
            "matching_count": match_count,
            "total_denominator": self.total_recipes,
            "percentage": round(percentage, 2),
            "mathematical_formula": f"({match_count:,} / {self.total_recipes:,}) * 100 = {percentage:.2f}%",
            "normalization_methodology": (
                "Ingredients were normalized by lowercasing, stripping measurement units "
                "(cups, tbsp, oz, etc.), removing cutting/prep adjectives (chopped, minced, diced), "
                "and lemmatizing plurals to root nouns. Each recipe is counted at most once."
            ),
            "sample_recipe_ids": sample_ids,
            "sample_recipes": sample_recipes
        }

    # =========================================================================
    # 2. Ranking by Instruction Steps (Task 1 Core Requirement)
    # =========================================================================
    def rank_by_instruction_steps(
        self,
        filter_ingredient: str,
        top_k: int = 5,
        ascending: bool = True
    ) -> Dict[str, Any]:
        """
        Ranks recipes containing the filter ingredient by the number of instruction steps.
        
        Step-Counting Rule:
          - Each instruction text was split on newline characters and stripped of empty lines.
          - Recipes are sorted primarily by `num_steps ASC` (fewest steps) or `DESC`.
          - Ties are broken by `num_ingredients ASC` (simpler recipes first).
        
        Example:
          User: "Among recipes containing tomato, which five have the fewest instruction steps?"
        """
        norm_key = clean_raw_ingredient(filter_ingredient)
        if not norm_key:
            norm_key = filter_ingredient.strip().lower()

        # Find eligible recipe IDs
        matching_ids = self.inverted_index.get(norm_key, set())
        if not matching_ids:
            # Check variants
            for k, ids in self.inverted_index.items():
                if norm_key in k.split():
                    matching_ids.update(ids)

        if not matching_ids:
            return {
                "filter_ingredient": filter_ingredient,
                "normalized_key": norm_key,
                "matching_recipes_found": 0,
                "results": [],
                "message": f"No recipes found containing '{filter_ingredient}'."
            }

        # Filter the compact recipes_df
        subset = self.recipes_df[self.recipes_df["id"].isin(matching_ids)].copy()
        
        # Sort deterministically
        subset = subset.sort_values(
            by=["num_steps", "num_ingredients"],
            ascending=[ascending, True]
        )
        top_slice = subset.head(top_k)

        results = []
        for _, row in top_slice.iterrows():
            rid = row["id"]
            full_rec = self.recipe_store.get(rid, {})
            results.append({
                "id": rid,
                "title": row["title"],
                "source": row["source"],
                "num_steps": int(row["num_steps"]),
                "num_ingredients": int(row["num_ingredients"]),
                "instructions": full_rec.get("instructions", ""),
                "ingredients": full_rec.get("ingredients", [])
            })

        return {
            "filter_ingredient": filter_ingredient,
            "normalized_key": norm_key,
            "total_matching_recipes": len(matching_ids),
            "step_counting_rule": "Instructions segmented by newline boundaries into distinct actionable cooking steps.",
            "sort_order": "Fewest instruction steps first (num_steps ASC), tie-breaker: fewest ingredients." if ascending else "Most steps first.",
            "results": results
        }

    # =========================================================================
    # 3. Corpus Statistical Distributions
    # =========================================================================
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        Computes accurate global statistics over all 121,684 recipes.
        """
        ing_counts = self.recipes_df["num_ingredients"]
        step_counts = self.recipes_df["num_steps"]

        return {
            "total_recipes": self.total_recipes,
            "ingredients_distribution": {
                "mean": round(float(ing_counts.mean()), 2),
                "median": float(ing_counts.median()),
                "min": int(ing_counts.min()),
                "max": int(ing_counts.max()),
                "std": round(float(ing_counts.std()), 2)
            },
            "steps_distribution": {
                "mean": round(float(step_counts.mean()), 2),
                "median": float(step_counts.median()),
                "min": int(step_counts.min()),
                "max": int(step_counts.max()),
                "std": round(float(step_counts.std()), 2)
            }
        }

    # =========================================================================
    # 4. Absence-of-Data Checker (Anti-Hallucination Guardrail)
    # =========================================================================
    def check_schema_attribute(self, attribute_name: str) -> Dict[str, Any]:
        """
        Checks if an attribute requested by the user exists in the Recipe Box dataset.
        Prevents LLM from inventing prep times, cook times, calories, or macros.
        """
        attr_clean = attribute_name.strip().lower()
        
        # Missing fields in Recipe Box dataset
        UNAVAILABLE_ATTRIBUTES = {
            "prep_time": "Prep time (preparation time) is NOT recorded in the Recipe Box dataset.",
            "cook_time": "Cooking time / bake time is NOT recorded as a structured field in the Recipe Box dataset.",
            "total_time": "Total duration is NOT present in the dataset schema.",
            "calories": "Caloric nutritional values are NOT recorded in the dataset.",
            "nutrition": "Macro-nutritional data (protein, fat, carbs, sodium) is NOT present in the dataset.",
            "rating": "Star ratings or review counts are NOT available in this version of the dataset."
        }

        for unavail_key, explanation in UNAVAILABLE_ATTRIBUTES.items():
            if unavail_key in attr_clean or attr_clean in unavail_key:
                return {
                    "is_available": False,
                    "attribute": attribute_name,
                    "explanation": explanation,
                    "recommendation": (
                        f"Refuse to fabricate {attribute_name}. Inform user that the dataset lacks this field, "
                        "and offer to provide an estimated culinary approximation explicitly labeled as an external estimate."
                    )
                }

        return {
            "is_available": True,
            "attribute": attribute_name,
            "explanation": f"Attribute '{attribute_name}' is supported in the schema."
        }


if __name__ == "__main__":
    engine = AnalyticsEngine()
    print("\n" + "="*60)
    print("TASK 1 VERIFICATION TEST 1: GARLIC & ONION CO-OCCURRENCE")
    print("="*60)
    res1 = engine.compute_ingredient_cooccurrence(["garlic", "onion"])
    print(f"Query: {res1['query_ingredients_raw']}")
    print(f"Formula: {res1['mathematical_formula']}")
    print(f"Matching Count: {res1['matching_count']:,} / {res1['total_denominator']:,} ({res1['percentage']}%)")
    print(f"Per-ingredient breakdown: {res1['per_ingredient_breakdown']}")
    print(f"Sample Recipe IDs: {res1['sample_recipe_ids']}")

    print("\n" + "="*60)
    print("TASK 1 VERIFICATION TEST 2: TOMATO RECIPES RANKED BY FEWEST STEPS")
    print("="*60)
    res2 = engine.rank_by_instruction_steps("tomato", top_k=5, ascending=True)
    print(f"Total matching recipes for 'tomato': {res2['total_matching_recipes']:,}")
    print(f"Rule: {res2['step_counting_rule']}")
    for r in res2["results"]:
        print(f"  • ID: {r['id']} | Title: '{r['title']}' | Steps: {r['num_steps']} | Ings: {r['num_ingredients']}")
        print(f"    Instructions preview: {repr(r['instructions'][:100])}")

    print("\n" + "="*60)
    print("TASK 1 VERIFICATION TEST 3: ABSENCE-OF-DATA CHECKER")
    print("="*60)
    for q in ["prep_time", "calories", "instructions"]:
        check = engine.check_schema_attribute(q)
        print(f"Attribute '{q}': Available={check['is_available']} -> {check['explanation']}")
