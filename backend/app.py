# -*- coding: utf-8 -*-
"""
backend/app.py
--------------
Production-Grade FastAPI Backend for NOSH Conversational Recipe Intelligence.

Provides REST and Conversational API endpoints:
  - GET  /api/health                       : Index and system health check
  - POST /api/chat                         : Multi-turn conversational endpoint
  - POST /api/match                        : Direct ingredient inventory matching
  - POST /api/analytics                    : Deterministic co-occurrence & step rankings
  - POST /api/substitute                   : Evidence-based recipe adaptation
  - GET  /api/recipe/{recipe_id}           : Retrieve full recipe details
  - DELETE /api/session/{session_id}/restrictions : Reset active session constraints
  - Static Mount /                         : Serves the interactive web UI from /frontend
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure backend and backend/src are on sys.path
BACKEND_DIR = Path(__file__).resolve().parent
SRC_DIR = BACKEND_DIR / "src"
ROOT_DIR = BACKEND_DIR.parent

for _p in [str(SRC_DIR), str(BACKEND_DIR), str(ROOT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from src.ingestion import get_index_bundle
    from src.pipeline import NoshPipeline
    from src.matcher import MatchRequest
    from src.substitution_engine import SubstitutionRequest
except ImportError:
    from backend.src.ingestion import get_index_bundle
    from backend.src.pipeline import NoshPipeline
    from backend.src.matcher import MatchRequest
    from backend.src.substitution_engine import SubstitutionRequest

# Initialize FastAPI App
app = FastAPI(
    title="NOSH Conversational Recipe Intelligence API",
    description="Deterministic culinary analytics, semantic recipe matching, and evidence-based substitution over 121,684 recipes.",
    version="1.0.0"
)

# Enable CORS for cross-origin local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Pipeline Singleton (loads indexed 121,684 recipes on startup)
print("[FastAPI] Initializing NOSH Conversational Pipeline...")
bundle = get_index_bundle()
pipeline = NoshPipeline(bundle=bundle)
print(f"[FastAPI] Pipeline ready with {len(bundle.recipe_store):,} recipes loaded.")


# =============================================================================
# Request / Response Schemas
# =============================================================================
class ChatRequest(BaseModel):
    message: str = Field(..., description="User chat query")
    session_id: str = Field(default="default", description="Conversation session ID for multi-turn state")
    thread_id: Optional[str] = Field(default=None, description="Conversation Thread ID (alias for session_id)")
    use_llm: bool = Field(default=True, description="Whether to synthesize response using Groq LLM")


class AnalyticsQueryRequest(BaseModel):
    query_type: str = Field(..., description="'cooccurrence' or 'step_ranking'")
    ingredients: List[str] = Field(default_factory=list, description="Ingredients for cooccurrence (e.g. ['garlic', 'onion']) or target for ranking (['tomato'])")
    top_k: int = Field(default=5, description="Number of results to return")


class RecipeDetailResponse(BaseModel):
    id: str
    title: str
    source: str
    ingredients: List[str]
    instructions: str
    num_steps: int
    num_ingredients: int


# =============================================================================
# API Endpoints
# =============================================================================
@app.get("/api/health", tags=["System"])
def health_check():
    """Returns system status, recipe count, LLM readiness, LangSmith tracing, and SQLite memory store."""
    return {
        "status": "healthy",
        "dataset_name": "Recipe Box (Epicurious, FoodNetwork, AllRecipes)",
        "curated_recipe_count": len(bundle.recipe_store),
        "inverted_index_token_count": len(bundle.inverted_index),
        "llm_connected": pipeline.llm.is_available(),
        "llm_model": pipeline.llm.model_name,
        "langsmith_tracing": bool(os.environ.get("LANGCHAIN_API_KEY")),
        "memory_store": "SQLite (data_recipes/nosh_sessions.db)"
    }


@app.post("/api/chat", tags=["Conversational Assistant"])
def chat_endpoint(req: ChatRequest):
    """
    Main multi-turn conversational endpoint.
    Maintains session history, accumulates dietary restrictions, and handles:
      - Statistical queries (Task 1)
      - Factual Q&A and Search (Task 1)
      - Absence-of-data guardrails (Task 1)
      - Inventory-based matching (Task 2)
      - Evidence-based substitution with local citations (Task 3)
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    effective_thread_id = req.thread_id or req.session_id or "default"
    result = pipeline.process_query(
        user_query=req.message,
        session_id=effective_thread_id,
        thread_id=effective_thread_id,
        use_llm=req.use_llm
    )
    
    session = pipeline.get_session(thread_id=effective_thread_id)
    return {
        "thread_id": effective_thread_id,
        "session_id": effective_thread_id,
        "intent": result.get("intent"),
        "response": result.get("response"),
        "guardrail_triggered": result.get("guardrail_triggered"),
        "active_restrictions": list(session.accumulated_restrictions),
        "data": result.get("data")
    }


@app.get("/api/pipeline/graph", tags=["Pipeline Architecture"])
def get_pipeline_graph():
    """Returns the live LangGraph StateGraph topology in Mermaid format."""
    return {
        "framework": "LangGraph StateGraph",
        "nodes": [
            "guardrail_node",
            "intent_router_node",
            "analytics_node",
            "matching_node",
            "substitution_node",
            "refinement_node",
            "search_node",
            "synthesis_node",
            "output_guardrail_node"
        ],
        "mermaid_chart": pipeline.get_graph_mermaid()
    }


@app.get("/api/threads", tags=["Session State"])
def list_threads(limit: int = 50):
    """Lists persistent conversation threads and turns stored in SQLite."""
    return pipeline.session_store.list_threads(limit=limit)


@app.get("/api/thread/{thread_id}/history", tags=["Session State"])
def get_thread_history(thread_id: str):
    """Fetches complete turn history and memory state for a given thread_id."""
    session = pipeline.get_session(thread_id=thread_id)
    history = pipeline.session_store.load_turn_history(thread_id)
    return {
        "thread_id": thread_id,
        "active_search_query": session.active_search_query,
        "active_restrictions": list(session.accumulated_restrictions),
        "history": history
    }


@app.delete("/api/thread/{thread_id}", tags=["Session State"])
def delete_thread(thread_id: str):
    """Deletes a conversation thread and its turns from SQLite memory."""
    pipeline.session_store.delete_thread(thread_id)
    if thread_id in pipeline.sessions:
        del pipeline.sessions[thread_id]
    return {"status": "deleted", "thread_id": thread_id}


@app.post("/api/match", tags=["Task 2: Inventory Matching"])
def match_inventory(req: MatchRequest):
    """
    Direct endpoint for Task 2: Constraint-based pantry matching.
    """
    response = pipeline.matcher.match(req)
    return response.model_dump()


@app.post("/api/analytics", tags=["Task 1: Deterministic Analytics"])
def analytics_endpoint(req: AnalyticsQueryRequest):
    """
    Direct endpoint for Task 1: Co-occurrence computation and step rankings.
    """
    if req.query_type == "cooccurrence":
        if not req.ingredients:
            raise HTTPException(status_code=400, detail="Must provide at least one ingredient.")
        return pipeline.analytics.compute_ingredient_cooccurrence(req.ingredients)
    elif req.query_type == "step_ranking":
        target = req.ingredients[0] if req.ingredients else "tomato"
        return pipeline.analytics.rank_by_instruction_steps(target, top_k=req.top_k)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown query_type: '{req.query_type}'. Use 'cooccurrence' or 'step_ranking'.")


@app.post("/api/substitute", tags=["Task 3: Recipe Substitution"])
def substitute_endpoint(req: SubstitutionRequest):
    """
    Direct endpoint for Task 3: Evidence-based recipe adaptation citing local Recipe IDs.
    """
    adapted = pipeline.substitution.adapt_recipe(req)
    return adapted.model_dump()


@app.get("/api/recipe/{recipe_id}", response_model=RecipeDetailResponse, tags=["Dataset Access"])
def get_recipe_by_id(recipe_id: str):
    """
    Fetches full metadata and text for a given recipe ID from the 121,684 database.
    """
    rec = bundle.recipe_store.get(recipe_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Recipe ID '{recipe_id}' not found in database.")

    inst = rec.get("instructions", "")
    steps = [s.strip() for s in inst.split("\n") if s.strip()] or [s.strip() for s in inst.split(". ") if s.strip()]

    return RecipeDetailResponse(
        id=rec["id"],
        title=rec["title"],
        source=rec.get("source", "Recipe Box"),
        ingredients=rec.get("ingredients", []),
        instructions=inst,
        num_steps=len(steps),
        num_ingredients=len(rec.get("ingredients", []))
    )


@app.delete("/api/session/{session_id}/restrictions", tags=["Session State"])
def clear_session_restrictions(session_id: str):
    """Resets accumulated dietary restrictions for a specific conversation session."""
    session = pipeline.get_session(session_id)
    session.clear_restrictions()
    return {"status": "cleared", "session_id": session_id, "active_restrictions": []}


# =============================================================================
# Root & System Information
# =============================================================================
@app.get("/", tags=["System"])
def root_info():
    """Root info page pointing to interactive docs and Streamlit UI."""
    return {
        "title": "NOSH Conversational Recipe Intelligence API",
        "status": "online",
        "curated_recipes": len(bundle.recipe_store),
        "docs_url": "http://localhost:8000/docs",
        "health_check": "http://localhost:8000/api/health",
        "streamlit_ui": "Run 'streamlit run frontend/streamlit_app.py' to launch interactive UI at http://localhost:8501"
    }


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("🚀 STARTING NOSH CONVERSATIONAL RECIPE ASSISTANT SERVER")
    print("📍 URL: http://localhost:8000")
    print("📄 API Docs: http://localhost:8000/docs")
    print("="*70 + "\n")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
