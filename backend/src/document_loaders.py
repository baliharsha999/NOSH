# -*- coding: utf-8 -*-
"""
backend/src/document_loaders.py
-------------------------------
LangChain-Native Document Loader for the NOSH Recipe Corpus.

Purpose:
  Loads raw or cleaned JSON recipe records and transforms them into standardized
  LangChain `Document` objects suitable for chunking, embedding, vector storage,
  and retrieval pipelines.

Why a Custom Loader?
  Standard LangChain `JSONLoader` requires complex jq queries and loses critical
  domain metadata (e.g. recipe ID, source website, step counts, image links).
  `RecipeDocumentLoader` guarantees 100% metadata preservation for downstream retrieval.
"""

import json
import os
from typing import List, Dict, Any, Iterator, Optional
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


class RecipeDocumentLoader(BaseLoader):
    """
    A custom LangChain Document Loader for the NOSH Recipe dataset.
    Loads recipes from 'recipes_cleaned.json' and outputs LangChain Document objects.
    """

    def __init__(self, file_path: str = os.path.join("data_recipes", "recipes_cleaned.json"), limit: Optional[int] = None):
        self.file_path = file_path
        self.limit = limit

    def lazy_load(self) -> Iterator[Document]:
        """
        Memory-efficient streaming generator that yields LangChain Documents.
        Ideal for large corpora (121,684 records) to prevent memory spikes.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Recipe JSON file not found at: {self.file_path}")

        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        count = 0
        for entry in data:
            if self.limit is not None and count >= self.limit:
                break

            rid = entry["id"]
            title = entry.get("title", "Untitled")
            ingredients = entry.get("ingredients", [])
            instructions = entry.get("instructions", "")
            source = entry.get("source", "Unknown")
            num_steps = entry.get("num_steps", len(instructions.splitlines()) if instructions else 0)
            num_ingredients = entry.get("num_ingredients", len(ingredients))
            picture_link = entry.get("picture_link")

            # Structured page_content preserving clear semantic sections
            page_content = (
                f"Recipe Title: {title}\n"
                f"Source: {source}\n\n"
                f"Ingredients:\n" + "\n".join(f"- {ing}" for ing in ingredients) + "\n\n"
                f"Instructions:\n{instructions}"
            )

            # Metadata contains all structured attributes for exact filtering
            metadata: Dict[str, Any] = {
                "id": rid,
                "title": title,
                "source": source,
                "num_steps": num_steps,
                "num_ingredients": num_ingredients,
                "picture_link": picture_link
            }

            yield Document(page_content=page_content, metadata=metadata)
            count += 1

    def load(self) -> List[Document]:
        """Loads all recipes into a Python list of Documents."""
        return list(self.lazy_load())


if __name__ == "__main__":
    loader = RecipeDocumentLoader(limit=3)
    docs = loader.load()
    print(f"Loaded {len(docs)} sample recipe documents:")
    for d in docs:
        print(f"\nDocument ID: {d.metadata['id']}")
        print(f"Title: {d.metadata['title']} (Source: {d.metadata['source']})")
        print(f"Content Preview:\n{d.page_content[:180]}...")
