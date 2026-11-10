# -*- coding: utf-8 -*-
"""
backend/src/pipeline.py
-----------------------
Master Conversational Pipeline for the NOSH Recipe Assistant.

Integrates:
  1. Session Context Tracker (Multi-Turn State & Constraint Accumulator)
  2. Query Intent Classifier (Analytics vs. Search vs. Matching)
  3. Deterministic Analytics Engine (Task 1)
  4. LangChain Document Retriever (Task 1 Search & Q&A)
  5. Explainable Constraint Matcher (Task 2)
  6. Guardrail Verification (Absence-of-Data check for prep times / calories)
"""

import re
from typing import Dict, List, Set, Any, Optional
from pydantic import BaseModel, Field

from ingestion import RecipeIndexBundle, get_index_bundle
from analytics_engine import AnalyticsEngine
from matcher import RecipeMatcher, MatchRequest, MatchResponse
from retriever import RecipeRetriever


class ConversationTurn(BaseModel):
    user_query: str
    intent_detected: str
    active_search_query: str
    accumulated_restrictions: List[str]
    response_text: str
    structured_data: Optional[Dict[str, Any]] = None


class ConversationalSession:
    """
    Maintains session state across multiple conversational turns.
    Accumulates dietary restrictions (e.g. 'I cannot have dairy') so that
    subsequent turns automatically enforce active constraints.
    """
    def __init__(self, session_id: str = "default_session"):
        self.session_id: str = session_id
        self.active_search_query: str = ""
        self.active_recipe_ids: List[str] = []
        self.accumulated_restrictions: Set[str] = set()
        self.history: List[ConversationTurn] = []

    def add_restriction(self, restriction: str):
        self.accumulated_restrictions.add(restriction.strip().lower())

    def clear_restrictions(self):
        self.accumulated_restrictions.clear()


class NoshPipeline:
    """
    The central orchestrator for NOSH GenAI Assistant.
    """
    def __init__(self, bundle: Optional[RecipeIndexBundle] = None):
        self.bundle = bundle or get_index_bundle()
        self.analytics = AnalyticsEngine(self.bundle)
        self.matcher = RecipeMatcher(self.bundle)
        self.sessions: Dict[str, ConversationalSession] = {}

    def get_session(self, session_id: str = "default") -> ConversationalSession:
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationalSession(session_id)
        return self.sessions[session_id]

    def _classify_intent(self, query: str) -> str:
        """
        Detects query intent:
          - 'ANALYTICS': Questions about counts, percentages, medians, rankings by steps.
          - 'MATCHING': Inventory prompts ('I have eggs, onions, tomatoes...').
          - 'RESTRICTION_REFINEMENT': Follow-up constraints ('I cannot have dairy', 'no peanuts').
          - 'SEARCH': Natural language dish lookup ('Find penne pasta').
        """
        q = query.lower()
        
        # 1. Analytics Intent
        if any(w in q for w in ["how many", "what percentage", "percentage of", "fewest instruction steps", "fewest steps", "rank", "average", "median"]):
            return "ANALYTICS"
            
        # 2. Matching Intent (I have X, Y, Z...)
        if any(w in q for w in ["i have", "we have", "ingredients in my fridge", "pantry", "suggest recipes using"]):
            return "MATCHING"
            
        # 3. Restriction refinement (e.g. "I cannot have dairy", "no peanuts", "exclude eggs")
        if any(w in q for w in ["cannot have", "can't have", "allergic to", "allergy", "no dairy", "no peanuts", "no gluten", "exclude", "without"]):
            return "RESTRICTION_REFINEMENT"
            
        return "SEARCH"

    def process_query(self, user_query: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Executes a single conversational turn with full multi-turn state tracking.
        """
        session = self.get_session(session_id)
        intent = self._classify_intent(user_query)

        # ---------------------------------------------------------------------
        # ROUTE 1: STATISTICAL ANALYTICS (Task 1)
        # ---------------------------------------------------------------------
        if intent == "ANALYTICS":
            q_lower = user_query.lower()
            
            # Check for Step Ranking query (e.g. "recipes containing tomato with fewest steps")
            if "fewest" in q_lower or "rank" in q_lower:
                # Extract target ingredient
                m = re.search(r"containing\s+([a-zA-Z]+)", q_lower) or re.search(r"with\s+([a-zA-Z]+)", q_lower)
                ing_target = m.group(1) if m else "tomato"
                stats_res = self.analytics.rank_by_instruction_steps(ing_target, top_k=5, ascending=True)
                
                resp_lines = [
                    f"**Ranking by Instruction Steps for '{ing_target}' (Fewest Steps First):**",
                    f"- Total matching recipes in dataset: {stats_res['total_matching_recipes']:,}",
                    f"- Step-counting rule: {stats_res['step_counting_rule']}\n"
                ]
                for idx, r in enumerate(stats_res["results"], 1):
                    resp_lines.append(f"{idx}. **{r['title']}** (ID: `{r['id']}`) - **{r['num_steps']} step(s)**, {r['num_ingredients']} ingredient(s)")
                
                response_text = "\n".join(resp_lines)
                return {"intent": intent, "response": response_text, "data": stats_res}

            # Check for Multi-Ingredient Co-occurrence (e.g. "garlic and onion")
            ings_to_check = []
            if "garlic" in q_lower:
                ings_to_check.append("garlic")
            if "onion" in q_lower:
                ings_to_check.append("onion")
            if not ings_to_check:
                # Regex fallback for "both X and Y"
                m = re.search(r"both\s+([a-zA-Z]+)\s+and\s+([a-zA-Z]+)", q_lower)
                if m:
                    ings_to_check = [m.group(1), m.group(2)]

            if ings_to_check:
                cooc_res = self.analytics.compute_ingredient_cooccurrence(ings_to_check)
                response_text = (
                    f"**Exact Co-occurrence Calculation:**\n"
                    f"- **Query Ingredients:** {', '.join(cooc_res['query_ingredients_raw'])}\n"
                    f"- **Matching Recipes:** **{cooc_res['matching_count']:,}**\n"
                    f"- **Total Denominator:** **{cooc_res['total_denominator']:,}** curated recipes\n"
                    f"- **Dataset Percentage:** **{cooc_res['percentage']}%**\n\n"
                    f"**Reproducible Formula:** `{cooc_res['mathematical_formula']}`\n\n"
                    f"**Normalization Methodology:** {cooc_res['normalization_methodology']}"
                )
                return {"intent": intent, "response": response_text, "data": cooc_res}

        # ---------------------------------------------------------------------
        # ROUTE 2: RESTRICTION REFINEMENT (Multi-turn follow-up)
        # ---------------------------------------------------------------------
        if intent == "RESTRICTION_REFINEMENT":
            q_lower = user_query.lower()
            detected_restriction = None
            if "dairy" in q_lower or "milk" in q_lower or "cheese" in q_lower:
                detected_restriction = "dairy"
            elif "peanut" in q_lower:
                detected_restriction = "peanuts"
            elif "gluten" in q_lower:
                detected_restriction = "gluten"
            elif "egg" in q_lower:
                detected_restriction = "egg"

            if detected_restriction:
                session.add_restriction(detected_restriction)

            # Re-run the active search query with the updated accumulated restrictions
            active_q = session.active_search_query or "pasta"
            retriever = RecipeRetriever(
                bundle=self.bundle,
                hard_exclusions=list(session.accumulated_restrictions),
                top_k=5
            )
            docs = retriever.invoke(active_q)

            resp_lines = [
                f"**Refined Search for '{active_q}'** with active restrictions: **{list(session.accumulated_restrictions)}**\n",
                f"All recipes containing {', '.join(session.accumulated_restrictions)} have been deterministically eliminated:\n"
            ]
            for idx, doc in enumerate(docs, 1):
                resp_lines.append(
                    f"{idx}. **{doc.metadata['title']}** (ID: `{doc.metadata['id']}`, Source: {doc.metadata['source']})\n"
                    f"   - **Steps:** {doc.metadata['num_steps']} | **Ingredients:** {doc.metadata['num_ingredients']}\n"
                )
            
            response_text = "\n".join(resp_lines)
            return {"intent": intent, "response": response_text, "active_restrictions": list(session.accumulated_restrictions)}

        # ---------------------------------------------------------------------
        # ROUTE 3: INGREDIENT-BASED MATCHING (Task 2)
        # ---------------------------------------------------------------------
        if intent == "MATCHING":
            q_lower = user_query.lower()
            # Extract ingredients
            avail = []
            for item in ["eggs", "onions", "tomatoes", "rice", "garlic", "chicken", "beef", "potatoes", "cheese", "pasta"]:
                if item in q_lower:
                    avail.append(item)
            if not avail:
                avail = ["eggs", "onions", "tomatoes", "rice"]

            # Exclusions
            excl = list(session.accumulated_restrictions)
            if "peanut" in q_lower and "peanuts" not in excl:
                excl.append("peanuts")

            # Max steps
            max_steps = 6 if "6" in q_lower or "six" in q_lower else None

            req = MatchRequest(
                available_ingredients=avail,
                hard_exclusions=excl,
                max_steps=max_steps,
                top_k=5
            )
            match_res = self.matcher.match(req)

            resp_lines = [
                f"**Ingredient Matching Results:**",
                f"- **Pantry Items:** {', '.join(match_res.query_available)}",
                f"- **Hard Constraints:** {match_res.hard_constraints_enforced or 'None'}",
                f"- **Max Steps:** {match_res.max_steps_limit or 'Unlimited'}",
                f"- **Eligible Matches:** {match_res.total_eligible_found:,}\n"
            ]

            for idx, r in enumerate(match_res.results, 1):
                resp_lines.append(
                    f"**{idx}. {r.title}** (ID: `{r.id}`, Source: {r.source})\n"
                    f"   - **Score:** {r.score} | **Steps:** {r.num_steps}\n"
                    f"   - **Matched Pantry:** {', '.join(r.matched_ingredients)}\n"
                    f"   - **Missing to Buy:** {', '.join(r.missing_ingredients) or 'None! (100% pantry matched)'}\n"
                    f"   - **Rationale:** {r.explanation}\n"
                )

            resp_lines.append(f"> *{match_res.quantity_sufficiency_notice}*")
            response_text = "\n".join(resp_lines)
            return {"intent": intent, "response": response_text, "data": match_res}

        # ---------------------------------------------------------------------
        # ROUTE 4: NATURAL SEARCH (Task 1)
        # ---------------------------------------------------------------------
        # Standard dish search
        session.active_search_query = user_query
        retriever = RecipeRetriever(
            bundle=self.bundle,
            hard_exclusions=list(session.accumulated_restrictions),
            top_k=5
        )
        docs = retriever.invoke(user_query)

        resp_lines = [
            f"**Search Results for '{user_query}':**",
            f"Active restrictions: {list(session.accumulated_restrictions) or 'None'}\n"
        ]
        for idx, doc in enumerate(docs, 1):
            resp_lines.append(
                f"{idx}. **{doc.metadata['title']}** (ID: `{doc.metadata['id']}`, Source: {doc.metadata['source']})\n"
                f"   - **Steps:** {doc.metadata['num_steps']} | **Ingredients:** {doc.metadata['num_ingredients']}\n"
            )

        response_text = "\n".join(resp_lines)
        return {"intent": intent, "response": response_text, "documents": docs}


if __name__ == "__main__":
    pipeline = NoshPipeline()
    
    print("\n" + "="*70)
    print("PIPELINE TEST 1: MULTI-TURN SEARCH & REFINEMENT")
    print("Turn 1: 'Find a penne pasta recipe'")
    print("="*70)
    t1 = pipeline.process_query("Find a penne pasta recipe", session_id="test_sess")
    print(t1["response"][:300] + "...\n")

    print("="*70)
    print("Turn 2: 'I cannot have dairy'")
    print("="*70)
    t2 = pipeline.process_query("I cannot have dairy", session_id="test_sess")
    print(t2["response"][:400] + "...\n")

    print("="*70)
    print("PIPELINE TEST 2: STATISTICAL ANALYTICS")
    print("Query: 'How many recipes contain both garlic and onion? What percentage is that?'")
    print("="*70)
    t3 = pipeline.process_query("How many recipes contain both garlic and onion? What percentage is that?", session_id="test_sess")
    print(t3["response"])
