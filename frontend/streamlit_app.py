# -*- coding: utf-8 -*-
"""
frontend/streamlit_app.py
-------------------------
Interactive Streamlit Frontend for NOSH Conversational Recipe Intelligence.

Connects to the FastAPI backend at http://localhost:8000/api
Features:
  - Multi-turn Conversational Chat Interface with persistent Thread ID tracking
  - Thread Management: Select prior conversations, start new threads, switch threads
  - SQLite Conversation Memory State persistence (`data_recipes/nosh_sessions.db`)
  - LangSmith Observability & Tracing indicators (`@traceable` chain tracking)
  - Task 1: Deterministic Analytics & Absence Guardrail
  - Task 2: Inventory Matching with Hard Constraints & Step Filters
  - Task 3: Evidence-Based Recipe Substitution with Local Dataset Citations
  - Active Dietary Restriction State Tracker
  - Direct Recipe Inspector Modal / Expander
"""

import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import requests
import streamlit as st

# Setup Path for local fallback if FastAPI server is not started yet
SRC_DIR = Path(__file__).resolve().parent.parent / "backend" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Backend FastAPI URL
API_BASE_URL = os.environ.get("NOSH_API_URL", "http://127.0.0.1:8000/api")

# Page Config
st.set_page_config(
    page_title="NOSH — Recipe Intelligence",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #f97316;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-bottom: 20px;
    }
    .badge-id {
        background-color: #1e293b;
        color: #f97316;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.85rem;
    }
    .matched-tag {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10b981;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .missing-tag {
        background-color: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .disclaimer-box {
        background-color: rgba(239, 68, 68, 0.1);
        border-left: 4px solid #ef4444;
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #fca5a5;
        margin-top: 10px;
    }
    .thread-card {
        background-color: #0f172a;
        padding: 10px;
        border-radius: 6px;
        border: 1px solid #334155;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

WELCOME_MESSAGE = {
    "role": "assistant",
    "content": (
        "👋 **Welcome to NOSH Conversational Recipe Intelligence!**\n\n"
        "I am your grounded culinary AI assistant, operating over **121,684 curated recipes** "
        "from Epicurious, FoodNetwork, and AllRecipes.\n\n"
        "**Try testing these core capabilities:**\n"
        "- **Task 1 Analytics**: Exact co-occurrence of Garlic & Onion (13.35%) or tomato recipes ranked by fewest steps.\n"
        "- **Task 2 Inventory**: What to cook with *eggs, onions, tomatoes, rice (max 6 steps, no peanuts)*.\n"
        "- **Task 3 Substitution**: Remove eggs from banana bread with *empirical local recipe citations*.\n"
        "- **Absence-of-Data Guardrail**: Ask for prep times or calories to verify zero hallucinations.\n\n"
        "Click any quick test in the sidebar or type a message below!"
    )
}

# Initialize Session State (ephemeral active session)
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{uuid.uuid4().hex[:8]}"

if "thread_id" not in st.session_state:
    st.session_state.thread_id = st.session_state.session_id

if "messages" not in st.session_state:
    st.session_state.messages = [WELCOME_MESSAGE]

if "active_restrictions" not in st.session_state:
    st.session_state.active_restrictions = []

if "local_pipeline" not in st.session_state:
    st.session_state.local_pipeline = None


def check_fastapi_backend() -> bool:
    """Checks if the FastAPI backend is running and reachable."""
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=1.5)
        return r.status_code == 200
    except Exception:
        return False


def get_local_pipeline():
    """Fallback: Lazy-loads local NoshPipeline if FastAPI is not running."""
    if st.session_state.local_pipeline is None:
        with st.spinner("Initializing local pipeline & loading 121,684 indexed recipes..."):
            from pipeline import NoshPipeline
            from ingestion import get_index_bundle
            bundle = get_index_bundle()
            st.session_state.local_pipeline = NoshPipeline(bundle=bundle)
    return st.session_state.local_pipeline


def send_chat_message(query: str) -> Dict[str, Any]:
    """Sends chat message to FastAPI backend, with seamless local fallback."""
    is_backend_up = check_fastapi_backend()
    current_tid = st.session_state.thread_id
    
    if is_backend_up:
        try:
            payload = {
                "message": query,
                "session_id": current_tid,
                "thread_id": current_tid,
                "use_llm": True
            }
            res = requests.post(f"{API_BASE_URL}/chat", json=payload, timeout=30.0)
            if res.status_code == 200:
                data = res.json()
                st.session_state.active_restrictions = data.get("active_restrictions", [])
                return data
        except Exception as e:
            st.warning(f"FastAPI call timed out or failed ({e}). Using local pipeline fallback.")

    # Local Pipeline fallback
    pipe = get_local_pipeline()
    result = pipe.process_query(query, session_id=current_tid, thread_id=current_tid, use_llm=True)
    sess = pipe.get_session(thread_id=current_tid)
    st.session_state.active_restrictions = list(sess.accumulated_restrictions)
    return {
        "thread_id": current_tid,
        "session_id": current_tid,
        "intent": result.get("intent"),
        "response": result.get("response"),
        "guardrail_triggered": result.get("guardrail_triggered"),
        "active_restrictions": list(sess.accumulated_restrictions),
        "data": result.get("data")
    }


def clear_restrictions():
    """Resets dietary restrictions in backend and session state."""
    current_tid = st.session_state.thread_id
    try:
        requests.delete(f"{API_BASE_URL}/session/{current_tid}/restrictions", timeout=1.5)
    except Exception:
        pass
    if st.session_state.local_pipeline:
        sess = st.session_state.local_pipeline.get_session(thread_id=current_tid)
        sess.clear_restrictions()
    try:
        from session_store import SQLiteSessionStore
        SQLiteSessionStore().clear_restrictions(current_tid)
    except Exception:
        pass
    st.session_state.active_restrictions = []
    st.toast("Active dietary restrictions cleared!", icon="🧹")


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("### 🍳 NOSH Recipe Intelligence")
    st.caption("AI Culinary Assistant • 121,684 Curated Recipes")

    # 1. Primary Action: Start a Fresh New Chat
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        old_id = st.session_state.get("session_id")
        # Remove old session from local SQLite store so previous data is not lingering
        try:
            from session_store import SQLiteSessionStore
            store = SQLiteSessionStore()
            if old_id:
                store.delete_thread(old_id)
        except Exception:
            pass

        # Generate fresh session for current user
        new_id = f"session_{uuid.uuid4().hex[:8]}"
        st.session_state.session_id = new_id
        st.session_state.thread_id = new_id
        st.session_state.messages = [WELCOME_MESSAGE]
        st.session_state.active_restrictions = []
        if st.session_state.local_pipeline:
            sess = st.session_state.local_pipeline.get_session(thread_id=new_id)
            sess.clear_restrictions()
        st.rerun()

    st.markdown("---")

    # 2. Active Dietary Constraints (User-facing culinary preferences)
    st.markdown("#### 🛡️ Active Dietary Constraints")
    if st.session_state.active_restrictions:
        for r in st.session_state.active_restrictions:
            st.error(f"🚫 Exclusion Enforced: **{r.upper()}**")
        if st.button("🧹 Clear Restrictions", use_container_width=True):
            clear_restrictions()
            st.rerun()
    else:
        st.caption("No active restrictions (Pantry neutral).")

    st.markdown("---")

    # 3. Suggested Questions (Interactive quick test prompts)
    st.markdown("### 📋 Suggested Questions")
    presets = [
        ("🧅 Garlic & Onion Co-occurrence %", "How many recipes contain both garlic and onion? What percentage is that?"),
        ("🍅 Tomato Fewest Steps Ranking", "Show me recipes containing tomato ranked by fewest instruction steps"),
        ("🛡️ Negative Guardrail (Prep/Calories)", "What is the prep time and calorie count for recipe 78qGO.ejAdsFEJLRl08hVUjiji9zyG6?"),
        ("🧺 Eggs, Onions, Tomatoes, Rice (≤6 Steps)", "I have eggs, onions, tomatoes, and rice. What can I cook with at most 6 steps? Exclude peanuts."),
        ("🍌 Banana Bread without Eggs (Citations)", "How can I make banana bread without eggs? Cite empirical recipe IDs from the dataset."),
        ("🌐 Panna Cotta without Gelatin (Web Search)", "How can I adapt panna cotta without gelatin? If local evidence is inadequate, search the web with structured output."),
        ("🍝 Turn 1: Find Penne Pasta", "Find a penne pasta recipe"),
        ("🥛 Turn 2: I cannot have dairy", "I cannot have dairy"),
    ]

    selected_prompt = None
    for label, query_text in presets:
        if st.button(label, use_container_width=True):
            selected_prompt = query_text

    st.markdown("---")
    st.caption("Curated: 121,684 Recipes | Sources: Epicurious, FoodNetwork, AllRecipes")


# =============================================================================
# MAIN INTERACTION VIEW
# =============================================================================
st.markdown('<div class="main-title">🍳 NOSH Conversational Recipe Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Deterministic Statistical Analytics • Constraint-Based Inventory Matching • Evidence-Based Local Substitutions</div>', unsafe_allow_html=True)

# Display Message History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("guardrail_triggered"):
            st.warning(f"🛡️ **System Guardrail Triggered:** `{msg['guardrail_triggered']}`")
        st.markdown(msg["content"])
        
        # Display rich structured widgets if data exists
        if "data" in msg and msg["data"]:
            d = msg["data"]

            # Task 1: Co-occurrence Metric Display
            if "percentage" in d and "matching_count" in d:
                c1, c2, c3 = st.columns(3)
                c1.metric("Matching Recipes", f"{d['matching_count']:,}")
                c2.metric("Curated Denominator", f"{d['total_denominator']:,}")
                c3.metric("Dataset Co-occurrence", f"{d['percentage']}%")
                st.caption(f"**Reproducible Formula:** `{d.get('mathematical_formula')}`")

            # Task 1: Step Ranking Table
            elif "results" in d and isinstance(d["results"], list) and len(d["results"]) > 0 and "num_steps" in d["results"][0]:
                st.markdown("##### 📊 Step-Ranked Recipe Results")
                table_data = []
                for idx, r in enumerate(d["results"], 1):
                    table_data.append({
                        "Rank": idx,
                        "Recipe Title": r.get("title"),
                        "Recipe ID": r.get("id"),
                        "Steps": r.get("num_steps"),
                        "Ingredients": r.get("num_ingredients"),
                        "Source": r.get("source")
                    })
                st.dataframe(table_data, use_container_width=True)

            # Task 2: Inventory Matching Cards
            elif "results" in d and isinstance(d["results"], list) and len(d["results"]) > 0 and "matched_ingredients" in d["results"][0]:
                st.markdown("##### 🍽️ Top Candidate Recipe Matches")
                for r in d["results"]:
                    with st.expander(f"⭐ **{r['title']}** (Score: {r['score']} | {r['num_steps']} Steps)"):
                        st.markdown(f"**Recipe ID:** <span class='badge-id'>{r['id']}</span> | **Source:** `{r['source']}`", unsafe_allow_html=True)
                        st.markdown(f"**Why this matches:** {r['explanation']}")
                        
                        col_m, col_u = st.columns(2)
                        with col_m:
                            st.markdown("**Matched Pantry Items:**")
                            st.write(", ".join([f"`{item}`" for item in r["matched_ingredients"]]))
                        with col_u:
                            st.markdown("**Missing Items to Buy:**")
                            missing = r.get("missing_ingredients", [])
                            if missing:
                                st.write(", ".join([f"`{item}`" for item in missing]))
                            else:
                                st.success("None! 100% pantry satisfied.")

            # Task 3: Evidence Substitution Card
            elif "local_empirical_evidence" in d or "external_web_citations" in d:
                if d.get("is_external_web_search"):
                    st.markdown("##### 🌐 External Web Search Culinary Authority Evidence")
                    if d.get("uncertainty_statement"):
                        st.caption(f"ℹ️ {d['uncertainty_statement']}")
                    for cit in d.get("external_web_citations", []):
                        st.info(
                            f"**Authority:** `{cit.get('authority_name', 'External Authority')}` — **Article:** [{cit.get('title', 'Reference Article')}]({cit.get('url', '#')})\n\n"
                            f"**Verified Excerpt:** *\"{cit.get('snippet', '')}\"*"
                        )
                else:
                    st.markdown("##### 🧬 Local Empirical Evidence from Curated Dataset")
                    for ev in d.get("local_empirical_evidence", []):
                        st.info(f"**Recipe ID:** `{ev['id']}` — **Title:** *{ev['title']}* ({ev['source']})\n\n"
                                f"**Substitute Identified:** `{ev['key_substitute_used']}`\n\n"
                                f"**Sample Ingredients:** {', '.join(ev.get('ingredients_preview', [])[:3])}")

                # Sensory Trade-offs
                st.markdown("##### 👅 Sensory & Culinary Trade-Offs")
                trade_offs = d.get("sensory_trade_offs", {})
                t_cols = st.columns(3)
                t_cols[0].markdown(f"**Taste:**\n{trade_offs.get('taste', 'N/A')}")
                t_cols[1].markdown(f"**Texture:**\n{trade_offs.get('texture', 'N/A')}")
                t_cols[2].markdown(f"**Preparation:**\n{trade_offs.get('preparation', 'N/A')}")

                # Medical Disclaimer
                st.markdown(f'<div class="disclaimer-box">⚠️ <strong>Safety Disclaimer:</strong> {d.get("safety_disclaimer")}</div>', unsafe_allow_html=True)


# Handle Input (Either from chat input or quick prompt button)
user_prompt = st.chat_input("Ask a question about recipes, enter pantry ingredients, or request substitutions...")
if selected_prompt:
    user_prompt = selected_prompt

if user_prompt:
    # 1. Append user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # 2. Query backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking & querying 121,684 recipe index..."):
            backend_res = send_chat_message(user_prompt)
            resp_content = backend_res.get("response", "No response generated.")
            data_payload = backend_res.get("data")
            guard_name = backend_res.get("guardrail_triggered")
            
            st.markdown(resp_content)

            # Store in session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": resp_content,
                "guardrail_triggered": guard_name,
                "data": data_payload
            })
            st.rerun()
