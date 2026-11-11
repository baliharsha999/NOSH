# -*- coding: utf-8 -*-
"""
datacleaning.py
---------------
Batch Data Cleaning & Ingestion Pipeline for NOSH Recipe Corpus.

Extracts, cleans, standardizes, deduplicates, and exports the 121,684 recipe corpus
from raw Epicurious, FoodNetwork, and AllRecipes datasets to 'data_recipes/recipes_cleaned.json'.
"""

import os
import glob
import json
import re
import html
import pandas as pd


def clean_ingredient_text(text: str) -> str:
    """Removes advertisement artifacts, unescapes HTML entities, and trims whitespace."""
    if not isinstance(text, str):
        return ""
    # Strip standalone or appended ADVERTISEMENT text
    cleaned = re.sub(r"\bADVERTISEMENT\b", "", text, flags=re.IGNORECASE)
    # Strip HTML tags
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    # Unescape HTML entities (&amp; -> &, &quot; -> ", etc.)
    cleaned = html.unescape(cleaned)
    # Normalize multiple whitespace characters
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def run_data_cleaning(data_dir: str = "data_recipes", output_path: str = "data_recipes/recipes_cleaned.json"):
    print("=" * 60)
    print("NOSH Recipe Corpus Cleaning & Consolidation")
    print("=" * 60)

    raw_frames = []
    pattern = os.path.join(data_dir, "recipes_raw_nosource_*.json")
    raw_files = sorted(glob.glob(pattern))

    if not raw_files:
        print(f"Error: No raw recipe files found matching '{pattern}'.")
        return

    for file_path in raw_files:
        print(f"Loading '{file_path}'...")
        with open(file_path, "r", encoding="utf-8") as f:
            raw_dict = json.load(f)

        frame = pd.DataFrame.from_dict(raw_dict, orient="index")
        frame.index.name = "id"
        frame = frame.reset_index()
        frame["source_file"] = file_path

        if "ar" in file_path:
            frame["source"] = "AllRecipes"
        elif "epi" in file_path:
            frame["source"] = "Epicurious"
        elif "fn" in file_path:
            frame["source"] = "FoodNetwork"
        else:
            frame["source"] = "Unknown"

        raw_frames.append(frame)
        print(f"  -> Ingested {len(frame):,} raw entries from {frame['source'].iloc[0]}")

    df_raw = pd.concat(raw_frames, ignore_index=True)
    df_raw["id"] = df_raw["id"].astype(str)
    print(f"\nTotal raw consolidated entries: {len(df_raw):,}")

    # Cleaning ingredients
    print("Cleaning ingredient lines and removing scraping artifacts...")
    def clean_ing_list(ing_list):
        if not isinstance(ing_list, list):
            return []
        cleaned = []
        for item in ing_list:
            c = clean_ingredient_text(item)
            if c and len(c) > 1 and c.lower() != "advertisement":
                cleaned.append(c)
        return cleaned

    df_raw["ingredients"] = df_raw["ingredients"].apply(clean_ing_list)

    # Cleaning instructions and titles
    print("Standardizing instructions and titles...")
    df_raw["title"] = df_raw["title"].astype(str).apply(lambda x: html.unescape(x).strip())
    df_raw["instructions"] = df_raw["instructions"].astype(str).apply(lambda x: html.unescape(x).strip())

    # Metadata metrics
    df_raw["num_ingredients"] = df_raw["ingredients"].apply(len)
    df_raw["instruction_len"] = df_raw["instructions"].apply(len)
    df_raw["num_steps"] = df_raw["instructions"].apply(lambda s: len([line for line in s.splitlines() if line.strip()]))

    # Quality filtering: non-empty title, at least 1 ingredient, instruction >= 10 chars
    valid_mask = (
        (df_raw["title"].str.strip() != "") &
        (df_raw["title"] != "nan") &
        (df_raw["num_ingredients"] > 0) &
        (df_raw["instruction_len"] >= 10) &
        (df_raw["instructions"].str.contains(r"[a-zA-Z]", regex=True))
    )
    df_clean = df_raw[valid_mask].copy()

    # Deduplication preferring entries with pictures
    df_clean["has_pic"] = df_clean["picture_link"].notna()
    df_clean = df_clean.sort_values(by=["has_pic"], ascending=False)
    initial_len = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=["title", "instructions"], keep="first")
    print(f"Deduplicated {initial_len - len(df_clean):,} duplicate recipes.")
    df_clean = df_clean.drop(columns=["has_pic"]).reset_index(drop=True)

    export_cols = [
        "id",
        "title",
        "ingredients",
        "instructions",
        "picture_link",
        "source",
        "num_ingredients",
        "num_steps",
        "instruction_len"
    ]
    df_export = df_clean[export_cols]

    print(f"Exporting {len(df_export):,} verified recipes to '{output_path}'...")
    df_export.to_json(output_path, orient="records", indent=2)
    mb_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Export Complete! Final corpus size: {mb_size:.2f} MB across {len(df_export):,} recipes.")


if __name__ == "__main__":
    run_data_cleaning()
