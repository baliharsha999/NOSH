# -*- coding: utf-8 -*-
"""
backend/src/ingestion.py
------------------------
Data Ingestion, Indexing, and Fast Retrieval Storage Engine.

Purpose:
  1. Ingests the cleaned recipe corpus (`data_recipes/recipes_cleaned.json`).
  2. Applies the Normalization Engine (`normalization.py`) across all 121,684 recipes.
  3. Constructs high-performance indexing data structures:
     - Inverted Ingredient Index: Mapping `normalized_key -> Set[recipe_id]` for O(1) set intersections.
     - Inverted Allergen Index: Mapping `allergen_group -> Set[recipe_id]` for instant constraint filtering.
     - Primary Metadata Store: In-memory table / dictionary for sub-millisecond recipe detail lookup.
  4. Caches the compiled index bundle to disk (`data_recipes/recipe_index.pkl`) for instant cold starts.
"""

import os
import json
import time
import pickle
from typing import Dict, Set, List, Any, Optional
from dataclasses import dataclass
import pandas as pd
try:
    from backend.src.normalization import extract_normalized_keys, detect_allergens, clean_raw_ingredient
except ImportError:
    from normalization import extract_normalized_keys, detect_allergens, clean_raw_ingredient


@dataclass
class RecipeIndexBundle:
    """
    Container holding all pre-indexed representations of the recipe corpus.
    Allows sub-10ms queries for search, deterministic counting, and constraint checking.
    """
    total_recipes: int
    inverted_index: Dict[str, Set[str]]     # ingredient_token -> set of recipe IDs
    allergen_index: Dict[str, Set[str]]     # allergen_name -> set of recipe IDs
    recipe_store: Dict[str, Dict[str, Any]] # recipe_id -> full recipe dict
    recipes_df: pd.DataFrame                # DataFrame for vectorized sorting / filtering


DEFAULT_JSON_PATH = os.path.join("data_recipes", "recipes_cleaned.json")
DEFAULT_CACHE_PATH = os.path.join("data_recipes", "recipe_index.pkl")


def build_and_cache_index(
    json_path: str = DEFAULT_JSON_PATH,
    cache_path: str = DEFAULT_CACHE_PATH,
    force_rebuild: bool = False
) -> RecipeIndexBundle:
    """
    Parses cleaned recipes, computes normalized ingredient tags and allergens,
    constructs the inverted indexes, and serializes the bundle to disk.
    """
    # Check if a valid pre-computed index cache exists
    if not force_rebuild and os.path.exists(cache_path):
        print(f"[Index] Loading pre-computed recipe index from '{cache_path}'...")
        t0 = time.time()
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
            
        if isinstance(data, dict):
            bundle = RecipeIndexBundle(
                total_recipes=data["total_recipes"],
                inverted_index=data["inverted_index"],
                allergen_index=data["allergen_index"],
                recipe_store=data["recipe_store"],
                recipes_df=data["recipes_df"]
            )
        else:
            bundle = data
            
        print(f"[Index] Successfully loaded {bundle.total_recipes:,} recipes in {time.time() - t0:.2f}s!")
        return bundle

    print(f"[Index] Building new index from '{json_path}'...")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Cleaned recipes file not found at: {json_path}")

    t_start = time.time()
    with open(json_path, "r", encoding="utf-8") as f:
        raw_recipes: List[Dict[str, Any]] = json.load(f)

    total_count = len(raw_recipes)
    print(f"[Index] Loaded {total_count:,} recipes. Building inverted index and allergen tags...")

    inverted_index: Dict[str, Set[str]] = {}
    allergen_index: Dict[str, Set[str]] = {}
    recipe_store: Dict[str, Dict[str, Any]] = {}
    
    rows_for_df = []

    for idx, r in enumerate(raw_recipes):
        rid = r["id"]
        title = r["title"]
        ingredients = r["ingredients"]
        instructions = r["instructions"]
        num_steps = r.get("num_steps", len(instructions.splitlines()) if instructions else 0)
        num_ingredients = r.get("num_ingredients", len(ingredients))
        source = r.get("source", "Unknown")
        picture_link = r.get("picture_link")

        # Extract normalized keys and allergens for every ingredient in this recipe
        recipe_keys: Set[str] = set()
        recipe_allergens: Set[str] = set()
        cleaned_ing_names: List[str] = []

        for raw_ing in ingredients:
            primary_clean = clean_raw_ingredient(raw_ing)
            if primary_clean:
                cleaned_ing_names.append(primary_clean)
                
            keys = extract_normalized_keys(raw_ing)
            recipe_keys.update(keys)
            recipe_allergens.update(detect_allergens(keys))

        # Register recipe into inverted index
        for k in recipe_keys:
            if k not in inverted_index:
                inverted_index[k] = set()
            inverted_index[k].add(rid)

        # Register recipe into allergen index
        for a in recipe_allergens:
            if a not in allergen_index:
                allergen_index[a] = set()
            allergen_index[a].add(rid)

        # Build full recipe record (preserving 100% of the original text)
        recipe_record = {
            "id": rid,
            "title": title,
            "ingredients": ingredients,
            "instructions": instructions,
            "source": source,
            "picture_link": picture_link,
            "num_steps": num_steps,
            "num_ingredients": num_ingredients,
            "normalized_keys": list(recipe_keys),
            "allergens": list(recipe_allergens),
            "cleaned_ing_names": cleaned_ing_names
        }
        recipe_store[rid] = recipe_record

        # Collect compact row for DataFrame
        rows_for_df.append({
            "id": rid,
            "title": title,
            "source": source,
            "num_steps": num_steps,
            "num_ingredients": num_ingredients,
            "picture_link": picture_link
        })

        if (idx + 1) % 25000 == 0 or (idx + 1) == total_count:
            print(f"  Processed {idx + 1:,} / {total_count:,} recipes ({(idx + 1)/total_count*100:.1f}%)...")

    # Construct DataFrame
    recipes_df = pd.DataFrame(rows_for_df)

    bundle = RecipeIndexBundle(
        total_recipes=total_count,
        inverted_index=inverted_index,
        allergen_index=allergen_index,
        recipe_store=recipe_store,
        recipes_df=recipes_df
    )

    bundle_dict = {
        "total_recipes": total_count,
        "inverted_index": inverted_index,
        "allergen_index": allergen_index,
        "recipe_store": recipe_store,
        "recipes_df": recipes_df
    }

    # Save to disk cache
    print(f"[Index] Serializing index bundle to '{cache_path}'...")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(bundle_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size_mb = os.path.getsize(cache_path) / (1024 * 1024)
    print(f"[Index] Completed index generation in {time.time() - t_start:.2f}s! (Cache size: {file_size_mb:.2f} MB)")
    print(f"[Index] Total unique indexed ingredient tokens: {len(inverted_index):,}")
    print(f"[Index] Tracked allergen categories: {list(allergen_index.keys())}")

    return bundle


def get_index_bundle() -> RecipeIndexBundle:
    """Helper to retrieve the global index bundle, loading from cache if ready."""
    return build_and_cache_index()


if __name__ == "__main__":
    bundle = build_and_cache_index(force_rebuild=True)
    print("\n--- Sanity Checks ---")
    garlic_count = len(bundle.inverted_index.get("garlic", set()))
    onion_count = len(bundle.inverted_index.get("onion", set()))
    both_count = len(bundle.inverted_index.get("garlic", set()).intersection(bundle.inverted_index.get("onion", set())))
    print(f"Recipes containing garlic: {garlic_count:,}")
    print(f"Recipes containing onion:  {onion_count:,}")
    print(f"Recipes containing both:   {both_count:,} ({(both_count / bundle.total_recipes) * 100:.2f}%)")