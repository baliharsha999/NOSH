"""
backend/src/chunking.py
-----------------------
Semantic Chunking Strategies for Culinary Documents in LangChain.

The Problem with Traditional Chunking:
  Generic text splitters (e.g. RecursiveCharacterTextSplitter with chunk_size=500)
  blindly slice text at character boundaries. In a recipe dataset, this causes
  "Culinary Context Fragmentation":
  1. Splitting an ingredient list in half separates allergens from the recipe title.
  2. A chunk containing only baking instructions lacks the ingredient measurements.
  3. If a vector search retrieves only the instructions chunk, an LLM will hallucinate
     missing quantities.

The Solution: Domain-Aware Semantic Chunking
  1. `AtomicRecipeSplitter`: Keeps short/medium recipes as unified atomic documents.
     Since the median recipe in our dataset has only 8 ingredients and 4 steps (~750 chars),
     keeping the whole recipe intact guarantees 100% data integrity with zero information loss.
  2. `SectionAwareRecipeSplitter`: For long, complex recipes, splits into distinct
     semantic sections ('ingredients' vs 'instructions') while preserving the parent
     Recipe ID, title, and metadata on every child chunk.
"""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document


class RecipeSemanticSplitter:
    """
    Culinary-aware document splitter that avoids fragmenting ingredient-instruction dependencies.
    """

    def __init__(self, mode: str = "atomic", max_char_limit: int = 1500):
        """
        Args:
            mode: 'atomic' (whole recipe per chunk) or 'sectional' (separate ingredients & instructions)
            max_char_limit: Maximum characters before forcing a sectional split.
        """
        self.mode = mode
        self.max_char_limit = max_char_limit

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Splits a list of recipe documents according to the chosen strategy."""
        split_docs: List[Document] = []
        for doc in documents:
            split_docs.extend(self.split_document(doc))
        return split_docs

    def split_document(self, document: Document) -> List[Document]:
        """Splits an individual recipe document into semantic chunks."""
        content = document.page_content
        meta = document.metadata.copy()

        # Strategy 1: Atomic (Default) - Keep recipe intact if within character limits
        if self.mode == "atomic" or len(content) <= self.max_char_limit:
            meta["chunk_type"] = "atomic_recipe"
            return [Document(page_content=content, metadata=meta)]

        # Strategy 2: Sectional - Split into Ingredients Section and Instructions Section
        chunks: List[Document] = []
        
        # Partition content into sections
        parts = content.split("Instructions:\n")
        header_and_ingredients = parts[0].strip()
        instructions = parts[1].strip() if len(parts) > 1 else ""

        # Chunk 1: Ingredients & Dish Summary Chunk
        ing_meta = meta.copy()
        ing_meta["chunk_type"] = "ingredients_section"
        ing_chunk = Document(
            page_content=f"{header_and_ingredients}",
            metadata=ing_meta
        )
        chunks.append(ing_chunk)

        # Chunk 2: Procedure & Steps Chunk (inheriting title & ID)
        if instructions:
            inst_meta = meta.copy()
            inst_meta["chunk_type"] = "instructions_section"
            inst_chunk = Document(
                page_content=(
                    f"Recipe: {meta.get('title', 'Untitled')}\n"
                    f"Recipe ID: {meta.get('id', '')}\n\n"
                    f"Instructions:\n{instructions}"
                ),
                metadata=inst_meta
            )
            chunks.append(inst_chunk)

        return chunks


if __name__ == "__main__":
    from document_loaders import RecipeDocumentLoader
    
    loader = RecipeDocumentLoader(limit=2)
    docs = loader.load()
    
    splitter = RecipeSemanticSplitter(mode="atomic")
    atomic_chunks = splitter.split_documents(docs)
    print(f"Atomic Splitter produced {len(atomic_chunks)} chunks for {len(docs)} recipes.")
    print(f"Sample chunk type: {atomic_chunks[0].metadata['chunk_type']}")
    print(f"Sample length: {len(atomic_chunks[0].page_content)} characters")
