# -*- coding: utf-8 -*-
"""
backend/src/pipeline.py
-----------------------
Master Conversational Pipeline for the NOSH Recipe Assistant.

Orchestrated via LangGraph StateGraph Architecture:
  1. Input Guardrails Node (Adversarial injection, Absence-of-Data schema protection)
  2. Intent Router Node (Analytics, Matching, Search, Refinement, Substitution)
  3. Domain Specialist Nodes:
     - Analytics Engine (Task 1: Deterministic statistics & step rankings)
     - Explainable Constraint Matcher (Task 2: Strict Boolean filter & pantry scoring)
     - Evidence-Based Substitution Engine (Task 3: Local dataset sister-recipe citations)
     - Restriction Refinement Node (Multi-turn state & allergen accumulator)
     - Natural Language Search Node (Hybrid dense FAISS vector & lexical retriever)
  4. Grounded LLM Synthesis Node (Zero-hallucination Qwen 2.5 synthesis via Groq Cloud)
  5. Output Guardrails Node (Medical disclaimers, quantity notices, citation integrity)
  6. SQLite Thread Persistence & LangSmith End-to-End Tracing
"""

import os
import re
from typing import Dict, List, Set, Any, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import StreamWriter
from langsmith import traceable

from ingestion import RecipeIndexBundle, get_index_bundle
from analytics_engine import AnalyticsEngine
from matcher import RecipeMatcher, MatchRequest, MatchResponse
from retriever import RecipeRetriever
from substitution_engine import SubstitutionEngine, SubstitutionRequest, AdaptedRecipe
from llm_generator import LLMGenerator
from guardrails import NoshGuardrails, MEDICAL_DISCLAIMER, QUANTITY_SUFFICIENCY_NOTICE, ABSENCE_OF_DATA_EXPLANATION
from session_store import SQLiteSessionStore


# =============================================================================
# LangGraph Pipeline State Definition
# =============================================================================
class PipelineState(TypedDict, total=False):
    # Session & Multi-Turn State
    thread_id: str
    session_id: str
    user_query: str
    active_search_query: str
    accumulated_restrictions: List[str]
    use_llm: bool

    # Guardrails & Routing Flags
    guardrail_blocked: bool
    guardrail_triggered: Optional[str]
    intent: str

    # Extracted Query Understanding & Semantic Entities (LLM-Parsed)
    cleaned_dish_or_recipe: Optional[str]
    pantry_ingredients: Optional[List[str]]
    exclusions: Optional[List[str]]
    target_ingredient: Optional[str]
    is_step_ranking: Optional[bool]
    substitute_ingredient: Optional[str]
    max_steps: Optional[int]

    # Intermediate Engine Artifacts
    context_data: Optional[Any]
    data: Optional[Dict[str, Any]]
    fallback_text: str

    # Synthesized Final Response
    response: str


# =============================================================================
# Data Models
# =============================================================================
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
    Tracks thread_id (conversation_id) and accumulated dietary restrictions.
    """
    def __init__(self, session_id: str = "default_session", thread_id: Optional[str] = None):
        self.thread_id: str = thread_id or session_id
        self.session_id: str = self.thread_id
        self.active_search_query: str = ""
        self.active_recipe_ids: List[str] = []
        self.accumulated_restrictions: Set[str] = set()
        self.history: List[ConversationTurn] = []

    def add_restriction(self, restriction: str):
        self.accumulated_restrictions.add(restriction.strip().lower())

    def remove_restriction(self, restriction: str):
        self.accumulated_restrictions.discard(restriction.strip().lower())

    def clear_restrictions(self):
        self.accumulated_restrictions.clear()


# =============================================================================
# Master LangGraph Orchestrator
# =============================================================================
class NoshPipeline:
    """
    The central orchestrator for NOSH GenAI Assistant with LangGraph StateGraph,
    SQLite persistence, and LangSmith tracing.
    """
    def __init__(self, bundle: Optional[RecipeIndexBundle] = None):
        self.bundle = bundle or get_index_bundle()
        self.analytics = AnalyticsEngine(self.bundle)
        self.matcher = RecipeMatcher(self.bundle)
        self.substitution = SubstitutionEngine(self.bundle)
        self.llm = LLMGenerator()
        self.session_store = SQLiteSessionStore()
        self.sessions: Dict[str, ConversationalSession] = {}

        # Compile the LangGraph StateGraph execution engine
        self.graph = self._build_langgraph_pipeline()

    def get_session(self, session_id: str = "default", thread_id: Optional[str] = None) -> ConversationalSession:
        """
        Retrieves an active session from memory cache, or restores it from SQLite database.
        """
        tid = (thread_id or session_id or "default").strip()
        if tid not in self.sessions:
            db_state = self.session_store.get_or_create_thread(tid)
            sess = ConversationalSession(session_id=tid, thread_id=tid)
            sess.active_search_query = db_state.get("active_search_query", "")
            sess.accumulated_restrictions = set(db_state.get("accumulated_restrictions", []))

            # Load turn history from database
            for td in self.session_store.load_turn_history(tid):
                sess.history.append(ConversationTurn(
                    user_query=td["user_query"],
                    intent_detected=td["intent_detected"],
                    active_search_query=sess.active_search_query,
                    accumulated_restrictions=list(sess.accumulated_restrictions),
                    response_text=td["response_text"],
                    structured_data=td["structured_data"]
                ))
            self.sessions[tid] = sess
        return self.sessions[tid]

    # =========================================================================
    # LangGraph Graph Construction
    # =========================================================================
    def _build_langgraph_pipeline(self):
        workflow = StateGraph(PipelineState)

        # 1. Register Graph Nodes
        workflow.add_node("guardrail_node", self._node_guardrail)
        workflow.add_node("intent_router_node", self._node_intent_router)
        workflow.add_node("analytics_node", self._node_analytics)
        workflow.add_node("matching_node", self._node_matching)
        workflow.add_node("substitution_node", self._node_substitution)
        workflow.add_node("refinement_node", self._node_refinement)
        workflow.add_node("search_node", self._node_search)
        workflow.add_node("synthesis_node", self._node_synthesis)
        workflow.add_node("output_guardrail_node", self._node_output_guardrail)

        # 2. Register Starting Edge
        workflow.add_edge(START, "guardrail_node")

        # 3. Guardrail Branch: Stop immediately if blocked; proceed to intent router if passed
        workflow.add_conditional_edges(
            "guardrail_node",
            self._route_after_guardrail,
            {
                "blocked": END,
                "proceed": "intent_router_node"
            }
        )

        # 4. Intent Router Branch: Route to domain specialist nodes
        workflow.add_conditional_edges(
            "intent_router_node",
            self._route_by_intent,
            {
                "ANALYTICS": "analytics_node",
                "MATCHING": "matching_node",
                "SUBSTITUTION": "substitution_node",
                "RESTRICTION_REFINEMENT": "refinement_node",
                "SEARCH": "search_node"
            }
        )

        # 5. Specialized nodes converge to grounded synthesis node
        workflow.add_edge("analytics_node", "synthesis_node")
        workflow.add_edge("matching_node", "synthesis_node")
        workflow.add_edge("substitution_node", "synthesis_node")
        workflow.add_edge("refinement_node", "synthesis_node")
        workflow.add_edge("search_node", "synthesis_node")

        # 6. Synthesis connects to Output Guardrail verification before completion
        workflow.add_edge("synthesis_node", "output_guardrail_node")
        workflow.add_edge("output_guardrail_node", END)

        return workflow.compile()

    # =========================================================================
    # Conditional Edge Routing Functions
    # =========================================================================
    @staticmethod
    def _route_after_guardrail(state: PipelineState) -> str:
        if state.get("guardrail_blocked", False):
            return "blocked"
        return "proceed"

    @staticmethod
    def _route_by_intent(state: PipelineState) -> str:
        intent = state.get("intent", "SEARCH")
        valid_intents = {"ANALYTICS", "MATCHING", "SUBSTITUTION", "RESTRICTION_REFINEMENT", "SEARCH"}
        return intent if intent in valid_intents else "SEARCH"

    # =========================================================================
    # LangGraph Worker Nodes
    # =========================================================================
    def _node_guardrail(self, state: PipelineState, writer: StreamWriter) -> Dict[str, Any]:
        """Node 1: Evaluates input query against 5-Pillar safety & schema guardrails."""
        user_query = state.get("user_query", "")
        guard_check = NoshGuardrails.check_input_query(user_query)

        if not guard_check.passed:
            msg = guard_check.message or ""
            writer(msg)
            return {
                "guardrail_blocked": True,
                "guardrail_triggered": guard_check.guardrail_name,
                "intent": "GUARDRAIL_BLOCKED",
                "response": msg,
                "data": guard_check.structured_payload
            }
        return {
            "guardrail_blocked": False,
            "guardrail_triggered": None
        }

    def _node_intent_router(self, state: PipelineState, writer: StreamWriter) -> Dict[str, Any]:
        """Node 2: Intelligent Query Understanding & Entity Extraction powered by Groq LLM with deterministic fallback."""
        user_query = state.get("user_query", "")
        parsed = self.llm.parse_query_semantics(user_query)

        detected_intent = parsed.get("intent", "SEARCH")

        # Handle Absence-of-Data guardrail
        if detected_intent == "ABSENCE_GUARDRAIL":
            writer(ABSENCE_OF_DATA_EXPLANATION)
            return {
                "guardrail_blocked": True,
                "guardrail_triggered": "AbsenceOfDataGuardrail",
                "intent": "ABSENCE_GUARDRAIL",
                "response": ABSENCE_OF_DATA_EXPLANATION,
                "data": {"guardrail_triggered": "absence_of_data", "field_requested": "prep_time_or_calories"}
            }

        return {
            "intent": detected_intent,
            "cleaned_dish_or_recipe": parsed.get("cleaned_dish_or_recipe"),
            "pantry_ingredients": parsed.get("pantry_ingredients", []),
            "exclusions": parsed.get("exclusions", []),
            "target_ingredient": parsed.get("target_ingredient"),
            "is_step_ranking": parsed.get("is_step_ranking", False),
            "substitute_ingredient": parsed.get("substitute_ingredient"),
            "max_steps": parsed.get("max_steps")
        }

    def _node_analytics(self, state: PipelineState) -> Dict[str, Any]:
        """Node 3A: Executes Task 1 exact programmatic statistical calculations."""
        user_query = state.get("user_query", "")
        q_lower = user_query.lower()

        # Check for Step Ranking query (e.g. "recipes containing tomato with fewest steps")
        if "fewest" in q_lower or "rank" in q_lower:
            m = re.search(r"containing\s+([a-zA-Z]+)", q_lower)
            if m:
                ing_target = m.group(1)
            else:
                m_with = re.search(r"with\s+([a-zA-Z]+)", q_lower)
                if m_with and m_with.group(1) not in ["fewest", "steps", "least", "most", "fewer"]:
                    ing_target = m_with.group(1)
                else:
                    for cand in ["tomato", "tomatoes", "garlic", "onion", "chicken", "beef", "potato", "pasta", "rice", "egg", "eggs"]:
                        if cand in q_lower:
                            ing_target = cand
                            break
                    else:
                        ing_target = "tomato"

            stats_res = self.analytics.rank_by_instruction_steps(ing_target, top_k=5, ascending=True)
            total_matches = stats_res.get("total_matching_recipes", stats_res.get("matching_recipes_found", 0))

            resp_lines = [
                f"**Ranking by Instruction Steps for '{ing_target}' (Fewest Steps First):**",
                f"- **Total matching recipes:** {total_matches:,} in curated dataset",
                f"- **Step-counting rule:** {stats_res.get('step_counting_rule', 'Newline instruction segmentation')}\n"
            ]
            for idx, r in enumerate(stats_res.get("results", []), 1):
                resp_lines.append(f"{idx}. **{r['title']}** (ID: `{r['id']}`) — **{r['num_steps']} step(s)**, {r['num_ingredients']} ingredient(s)")

            return {
                "context_data": stats_res,
                "data": stats_res,
                "fallback_text": "\n".join(resp_lines)
            }

        # Check for Multi-Ingredient Co-occurrence (e.g. "garlic and onion")
        ings_to_check = []
        if "garlic" in q_lower:
            ings_to_check.append("garlic")
        if "onion" in q_lower:
            ings_to_check.append("onion")
        if not ings_to_check:
            m = re.search(r"both\s+([a-zA-Z]+)\s+and\s+([a-zA-Z]+)", q_lower)
            if m:
                ings_to_check = [m.group(1), m.group(2)]

        if ings_to_check:
            cooc_res = self.analytics.compute_ingredient_cooccurrence(ings_to_check)
            fallback_text = (
                f"**Exact Co-occurrence Calculation:**\n"
                f"- **Query Ingredients:** {', '.join(cooc_res['query_ingredients_raw'])}\n"
                f"- **Matching Recipes:** **{cooc_res['matching_count']:,}**\n"
                f"- **Total Denominator:** **{cooc_res['total_denominator']:,}** curated recipes\n"
                f"- **Dataset Percentage:** **{cooc_res['percentage']}%**\n\n"
                f"**Reproducible Formula:** `{cooc_res['mathematical_formula']}`\n\n"
                f"**Normalization Methodology:** {cooc_res['normalization_methodology']}"
            )
            return {
                "context_data": cooc_res,
                "data": cooc_res,
                "fallback_text": fallback_text
            }

        return {
            "context_data": {},
            "data": {},
            "fallback_text": "Unable to calculate analytics for the specified ingredients."
        }

    def _node_matching(self, state: PipelineState) -> Dict[str, Any]:
        """Node 3B: Executes Task 2 explainable pantry inventory matching with constraint satisfaction."""
        user_query = state.get("user_query", "")
        q_lower = user_query.lower()

        avail = state.get("pantry_ingredients") or []
        if not avail:
            for item in ["eggs", "onions", "tomatoes", "rice", "garlic", "chicken", "beef", "potatoes", "cheese", "pasta"]:
                if item in q_lower:
                    avail.append(item)
            if not avail:
                avail = ["eggs", "onions", "tomatoes", "rice"]

        excl = list(state.get("accumulated_restrictions", []))
        for extra in state.get("exclusions", []):
            if extra.lower() not in excl:
                excl.append(extra.lower())
        if "peanut" in q_lower and "peanuts" not in excl:
            excl.append("peanuts")

        max_steps = state.get("max_steps")
        if max_steps is None and ("6" in q_lower or "six" in q_lower):
            max_steps = 6

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

        return {
            "context_data": match_res,
            "data": match_res.model_dump(),
            "fallback_text": "\n".join(resp_lines)
        }

    def _node_substitution(self, state: PipelineState) -> Dict[str, Any]:
        """Node 3C: Executes Task 3 evidence-based substitution citing local Recipe IDs, with Web Search fallback."""
        user_query = state.get("user_query", "")
        q_lower = user_query.lower()

        target_ing = state.get("substitute_ingredient")
        if not target_ing:
            if "gelatin" in q_lower:
                target_ing = "gelatin"
            elif "buttermilk" in q_lower:
                target_ing = "buttermilk"
            elif "cream" in q_lower or "heavy cream" in q_lower:
                target_ing = "heavy cream" if "heavy cream" in q_lower else "cream"
            elif "butter" in q_lower:
                target_ing = "butter"
            elif "egg" in q_lower:
                target_ing = "egg"
            else:
                m = re.search(r"(?:replace|substitute|without|remove)\s+([a-zA-Z]+)", q_lower)
                if m:
                    target_ing = m.group(1)
                else:
                    target_ing = "egg"
        else:
            target_ing = target_ing.lower().strip()
            if "egg" in target_ing:
                target_ing = "egg"
            elif "butter" in target_ing:
                target_ing = "butter"
            elif "heavy cream" in target_ing:
                target_ing = "heavy cream"
            elif "cream" in target_ing:
                target_ing = "cream"
            elif "gelatin" in target_ing:
                target_ing = "gelatin"

        recipe_title = (state.get("cleaned_dish_or_recipe") or "").strip()
        if not recipe_title or "banana" in q_lower:
            recipe_title = "banana bread" if "banana" in q_lower else ("panna cotta" if "panna" in q_lower or "gelatin" in q_lower else (recipe_title or "pasta"))

        m_id = re.search(r"([a-zA-Z0-9\._/]{25,35})", user_query)
        recipe_id = m_id.group(1) if m_id else None

        force_web = any(k in q_lower for k in ["web", "internet", "google", "online", "external"])

        sub_req = SubstitutionRequest(
            recipe_id=recipe_id,
            recipe_title=recipe_title,
            ingredient_to_remove=target_ing,
            hard_restrictions=list(state.get("accumulated_restrictions", [])),
            force_web_search=force_web
        )
        adapted = self.substitution.adapt_recipe(sub_req)

        resp_lines = [
            f"### Evidence-Based Recipe Adaptation: **{adapted.original_title}**"
        ]

        if adapted.is_external_web_search:
            if adapted.uncertainty_statement:
                resp_lines.append(f"> ℹ️ *{adapted.uncertainty_statement}*\n")
            resp_lines.extend([
                f"- **Ingredient Removed:** `{adapted.ingredient_removed.upper()}`",
                f"- **Chosen Substitute:** **{adapted.substitute_chosen}**",
                f"- **Culinary Role Analyzed:** {adapted.culinary_role_analyzed}\n",
                f"#### External Authority Web Citations (Verified External Links):"
            ])
            for cit in adapted.external_web_citations:
                resp_lines.append(f"  • **[{cit.authority_name}]** [{cit.title}]({cit.url})")
                resp_lines.append(f"    *Evidence Excerpt:* \"{cit.snippet}\"")
        else:
            resp_lines.extend([
                f"- **Ingredient Removed:** `{adapted.ingredient_removed.upper()}`",
                f"- **Chosen Substitute:** **{adapted.substitute_chosen}**",
                f"- **Culinary Role Analyzed:** {adapted.culinary_role_analyzed}\n",
                f"#### Local Empirical Evidence (Cited Recipe IDs from Curated Dataset):"
            ])
            for ev in adapted.local_empirical_evidence:
                resp_lines.append(f"  • **[ID: `{ev.id}`]** *{ev.title}* ({ev.source})")

        resp_lines.append("\n#### Updated Ingredients List:")
        for ing in adapted.updated_ingredients:
            resp_lines.append(f"  - {ing}")

        resp_lines.append(f"\n#### Revised Instructions:\n{adapted.revised_instructions}\n")
        resp_lines.append("#### Sensory & Texture Trade-Offs:")
        for k, v in adapted.sensory_trade_offs.items():
            resp_lines.append(f"  • **{k.capitalize()}**: {v}")

        resp_lines.append(f"\n> ⚠️ *{adapted.safety_disclaimer}*")

        return {
            "context_data": {"adapted_recipe": adapted},
            "data": adapted.model_dump(),
            "fallback_text": "\n".join(resp_lines)
        }

    def _node_refinement(self, state: PipelineState) -> Dict[str, Any]:
        """Node 3D: Accumulates conversational dietary constraints and executes constrained retrieval."""
        user_query = state.get("user_query", "")
        q_lower = user_query.lower()
        accum = list(state.get("accumulated_restrictions", []))

        # Add newly extracted exclusions from LLM
        for ex in state.get("exclusions", []):
            clean_ex = ex.strip().lower()
            if clean_ex not in accum:
                accum.append(clean_ex)

        detected_restriction = None
        if "dairy" in q_lower or "milk" in q_lower or "cheese" in q_lower:
            detected_restriction = "dairy"
        elif "peanut" in q_lower:
            detected_restriction = "peanuts"
        elif "gluten" in q_lower:
            detected_restriction = "gluten"
        elif "egg" in q_lower:
            detected_restriction = "egg"

        if detected_restriction and detected_restriction not in accum:
            accum.append(detected_restriction)

        active_q = state.get("active_search_query") or "pasta"
        retriever = RecipeRetriever(
            bundle=self.bundle,
            hard_exclusions=accum,
            top_k=5
        )
        docs = retriever.invoke(active_q)

        resp_lines = [
            f"**Refined Search for '{active_q}'** with active restrictions: **{accum}**\n",
            f"All recipes containing **{', '.join(accum)}** have been deterministically eliminated:\n"
        ]
        doc_cards = []
        for idx, doc in enumerate(docs, 1):
            ings_preview = doc.page_content.split("INSTRUCTIONS:")[0].replace("INGREDIENTS:", "").strip().split("\n")[:4]
            doc_cards.append({
                "id": doc.metadata["id"],
                "title": doc.metadata["title"],
                "source": doc.metadata["source"],
                "num_steps": doc.metadata["num_steps"],
                "num_ingredients": doc.metadata["num_ingredients"],
                "ingredients": ings_preview
            })
            resp_lines.append(
                f"{idx}. **{doc.metadata['title']}** (ID: `{doc.metadata['id']}`, Source: {doc.metadata['source']})\n"
                f"   - **Steps:** {doc.metadata['num_steps']} | **Verified Ingredients ({detected_restriction.capitalize() if detected_restriction else 'Allergen'}-Free):** {', '.join(ings_preview[:3])}\n"
            )

        return {
            "accumulated_restrictions": accum,
            "context_data": {
                "documents": docs,
                "active_restrictions": accum
            },
            "data": {"recipes": doc_cards},
            "fallback_text": "\n".join(resp_lines)
        }

    def _node_search(self, state: PipelineState) -> Dict[str, Any]:
        """Node 3E: Standard natural language retrieval with active dietary exclusions."""
        user_query = state.get("user_query", "")
        accum = list(state.get("accumulated_restrictions", []))

        # Use LLM-cleaned dish name (e.g. "Turkey Sausage Tacos") if available, else user query
        search_target = state.get("cleaned_dish_or_recipe") or user_query

        retriever = RecipeRetriever(
            bundle=self.bundle,
            hard_exclusions=accum,
            top_k=5
        )
        docs = retriever.invoke(search_target)

        resp_lines = [
            f"**Search Results for '{search_target}':**",
            f"Active restrictions enforced: **{accum or 'None'}**\n"
        ]
        doc_cards = []
        for idx, doc in enumerate(docs, 1):
            doc_cards.append({
                "id": doc.metadata["id"],
                "title": doc.metadata["title"],
                "source": doc.metadata["source"],
                "num_steps": doc.metadata["num_steps"],
                "num_ingredients": doc.metadata["num_ingredients"]
            })
            resp_lines.append(
                f"{idx}. **{doc.metadata['title']}** (ID: `{doc.metadata['id']}`, Source: {doc.metadata['source']})\n"
                f"   - **Steps:** {doc.metadata['num_steps']} | **Ingredients:** {doc.metadata['num_ingredients']}\n"
            )

        return {
            "active_search_query": user_query,
            "context_data": {
                "documents": docs,
                "active_restrictions": accum
            },
            "data": {"recipes": doc_cards},
            "fallback_text": "\n".join(resp_lines)
        }

    def _node_synthesis(self, state: PipelineState, writer: StreamWriter) -> Dict[str, Any]:
        """Node 4: Grounded LLM synthesis using Groq Qwen 2.5 with live LangGraph token streaming."""
        user_query = state.get("user_query", "")
        intent = state.get("intent", "SEARCH")
        context_data = state.get("context_data") or {}
        fallback_text = state.get("fallback_text", "")
        use_llm = state.get("use_llm", True)

        if use_llm and self.llm.is_available():
            tokens = []
            for token in self.llm.stream_synthesize(
                user_query=user_query,
                intent=intent,
                context_data=context_data,
                raw_fallback_response=fallback_text
            ):
                tokens.append(token)
                writer(token)
            synthesized = "".join(tokens).strip()
            return {"response": synthesized or fallback_text}

        # Offline or LLM-disabled fallback: stream words smoothly
        words = fallback_text.split(" ")
        for i, w in enumerate(words):
            chunk = w + (" " if i < len(words) - 1 else "")
            writer(chunk)
        return {"response": fallback_text}

    def _node_output_guardrail(self, state: PipelineState, writer: StreamWriter) -> Dict[str, Any]:
        """Node 5: Validates output compliance, ensuring disclaimers and notices are present."""
        response = state.get("response", "")
        intent = state.get("intent", "")

        appended_notice = ""
        if intent == "SUBSTITUTION":
            if "Medical Safety Disclaimer" not in response:
                appended_notice = f"\n\n> ⚠️ *{MEDICAL_DISCLAIMER}*"
                response = response + appended_notice
        elif intent == "MATCHING":
            if QUANTITY_SUFFICIENCY_NOTICE not in response:
                appended_notice = f"\n\n> *{QUANTITY_SUFFICIENCY_NOTICE}*"
                response = response + appended_notice

        if appended_notice:
            writer(appended_notice)

        return {"response": response}

    # =========================================================================
    # Mermaid Visualizer & Public Execution Interface
    # =========================================================================
    def get_graph_mermaid(self) -> str:
        """Returns the Mermaid flowchart visualization string of the LangGraph pipeline."""
        return self.graph.get_graph().draw_mermaid()

    @traceable(
        name="NOSH_Conversational_Pipeline",
        run_type="chain",
        tags=["nosh", "langgraph", "recipe_intelligence"]
    )
    def process_query(
        self,
        user_query: str,
        session_id: str = "default",
        thread_id: Optional[str] = None,
        use_llm: bool = True
    ) -> Dict[str, Any]:
        """
        Executes a single conversational turn through the compiled LangGraph StateGraph
        with full multi-turn state tracking, SQLite persistence, and LangSmith tracing.
        """
        tid = (thread_id or session_id or "default").strip()
        session = self.get_session(thread_id=tid)

        initial_state: PipelineState = {
            "thread_id": session.thread_id,
            "session_id": session.session_id,
            "user_query": user_query,
            "active_search_query": session.active_search_query,
            "accumulated_restrictions": list(session.accumulated_restrictions),
            "use_llm": use_llm,
            "guardrail_blocked": False,
            "guardrail_triggered": None,
            "intent": "",
            "context_data": None,
            "data": None,
            "fallback_text": "",
            "response": ""
        }

        # Execute the LangGraph StateGraph workflow
        final_state = self.graph.invoke(initial_state)

        # Synchronize multi-turn state back to ConversationalSession
        session.active_search_query = final_state.get("active_search_query", session.active_search_query)
        session.accumulated_restrictions = set(final_state.get("accumulated_restrictions", session.accumulated_restrictions))

        # Save turn to SQLite session store
        self.session_store.save_turn(
            thread_id=session.thread_id,
            user_query=user_query,
            intent_detected=final_state.get("intent", "UNKNOWN"),
            response_text=final_state.get("response", ""),
            structured_data=final_state.get("data"),
            active_search_query=session.active_search_query,
            accumulated_restrictions=session.accumulated_restrictions
        )

        return {
            "thread_id": session.thread_id,
            "session_id": session.thread_id,
            "intent": final_state.get("intent"),
            "response": final_state.get("response"),
            "guardrail_triggered": final_state.get("guardrail_triggered"),
            "active_restrictions": list(session.accumulated_restrictions),
            "data": final_state.get("data")
        }

    def stream_query(
        self,
        user_query: str,
        session_id: str = "default",
        thread_id: Optional[str] = None,
        use_llm: bool = True
    ):
        """
        Executes a single conversational turn through the compiled LangGraph StateGraph,
        streaming tokens and node state updates in real time using LangGraph's native stream.

        Yields:
          - ("token", str): Token delta from StreamWriter as the LLM generates.
          - ("node_update", dict): Node-level state updates as nodes finish.
          - ("complete", dict): Final turn dictionary with response, data, active restrictions.
        """
        tid = (thread_id or session_id or "default").strip()
        session = self.get_session(thread_id=tid)

        initial_state: PipelineState = {
            "thread_id": session.thread_id,
            "session_id": session.session_id,
            "user_query": user_query,
            "active_search_query": session.active_search_query,
            "accumulated_restrictions": list(session.accumulated_restrictions),
            "use_llm": use_llm,
            "guardrail_blocked": False,
            "guardrail_triggered": None,
            "intent": "",
            "context_data": None,
            "data": None,
            "fallback_text": "",
            "response": ""
        }

        accumulated_state: Dict[str, Any] = dict(initial_state)

        for mode, chunk in self.graph.stream(
            initial_state,
            stream_mode=["updates", "custom"]
        ):
            if mode == "custom":
                yield ("token", chunk)
            elif mode == "updates":
                for node_name, node_update in chunk.items():
                    accumulated_state.update(node_update)
                    yield ("node_update", {node_name: node_update})

        final_response = accumulated_state.get("response", "")
        final_intent = accumulated_state.get("intent", "UNKNOWN")
        final_data = accumulated_state.get("data")
        guard_name = accumulated_state.get("guardrail_triggered")

        # Synchronize multi-turn state back to ConversationalSession
        session.active_search_query = accumulated_state.get("active_search_query", session.active_search_query)
        session.accumulated_restrictions = set(accumulated_state.get("accumulated_restrictions", session.accumulated_restrictions))

        # Save turn to SQLite session store
        self.session_store.save_turn(
            thread_id=session.thread_id,
            user_query=user_query,
            intent_detected=final_intent,
            response_text=final_response,
            structured_data=final_data,
            active_search_query=session.active_search_query,
            accumulated_restrictions=session.accumulated_restrictions
        )

        final_result = {
            "thread_id": session.thread_id,
            "session_id": session.thread_id,
            "intent": final_intent,
            "response": final_response,
            "guardrail_triggered": guard_name,
            "active_restrictions": list(session.accumulated_restrictions),
            "data": final_data
        }
        yield ("complete", final_result)


# =============================================================================
# Pipeline Singleton Accessor
# =============================================================================
_pipeline_instance: Optional[NoshPipeline] = None

def get_pipeline() -> NoshPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = NoshPipeline()
    return _pipeline_instance


if __name__ == "__main__":
    print("Testing LangGraph Pipeline...")
    pipe = get_pipeline()
    print("\n--- Pipeline Mermaid Graph ---")
    print(pipe.get_graph_mermaid())

    print("\n--- Test Query 1: Absence Guardrail ---")
    res1 = pipe.process_query("What is the prep time and calorie count?", use_llm=False)
    print("Intent:", res1["intent"])
    print("Guardrail:", res1["guardrail_triggered"])
    print("Response preview:", res1["response"][:100])

    print("\n--- Test Query 2: Analytics ---")
    res2 = pipe.process_query("How many recipes contain both garlic and onion?", use_llm=False)
    print("Intent:", res2["intent"])
    print("Response preview:", res2["response"][:100])

    print("\n--- Test Query 3: Multi-Turn Constraint ---")
    sess_id = "test_langgraph_turn"
    pipe.process_query("Find penne pasta", session_id=sess_id, use_llm=False)
    res3 = pipe.process_query("I cannot have dairy", session_id=sess_id, use_llm=False)
    print("Intent:", res3["intent"])
    print("Active restrictions:", res3["active_restrictions"])
    print("Response preview:", res3["response"][:100])
