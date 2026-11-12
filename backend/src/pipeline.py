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
     - Natural Language Search Node (Title Inverted Index & lexical retriever)
  4. Grounded LLM Synthesis Node (Zero-hallucination Qwen synthesis via Groq Cloud with live StreamWriter)
  5. Output Guardrails Node (Medical disclaimers, quantity notices, citation integrity)
  6. SQLite Thread Persistence & LangSmith End-to-End Tracing

All execution nodes and conditional edge routers are encapsulated in `graph_nodes.py`.
"""

import os
from typing import Dict, List, Set, Any, Optional
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langsmith import traceable

try:
    from backend.src.ingestion import RecipeIndexBundle, get_index_bundle
    from backend.src.analytics_engine import AnalyticsEngine
    from backend.src.matcher import RecipeMatcher
    from backend.src.substitution_engine import SubstitutionEngine
    from backend.src.llm_generator import LLMGenerator
    from backend.src.session_store import SQLiteSessionStore
    from backend.src.graph_nodes import (
        PipelineState,
        GraphNodes,
        route_after_guardrail,
        route_by_intent,
    )
except ImportError:
    from ingestion import RecipeIndexBundle, get_index_bundle
    from analytics_engine import AnalyticsEngine
    from matcher import RecipeMatcher
    from substitution_engine import SubstitutionEngine
    from llm_generator import LLMGenerator
    from session_store import SQLiteSessionStore
    from graph_nodes import (
        PipelineState,
        GraphNodes,
        route_after_guardrail,
        route_by_intent,
    )


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

        # Instantiate graph execution nodes from graph_nodes.py
        self.nodes = GraphNodes(
            bundle=self.bundle,
            analytics=self.analytics,
            matcher=self.matcher,
            substitution=self.substitution,
            llm=self.llm,
            session_store=self.session_store
        )

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

        # 1. Register Graph Nodes from graph_nodes.py
        workflow.add_node("guardrail_node", self.nodes.node_guardrail)
        workflow.add_node("intent_router_node", self.nodes.node_intent_router)
        workflow.add_node("analytics_node", self.nodes.node_analytics)
        workflow.add_node("matching_node", self.nodes.node_matching)
        workflow.add_node("substitution_node", self.nodes.node_substitution)
        workflow.add_node("refinement_node", self.nodes.node_refinement)
        workflow.add_node("search_node", self.nodes.node_search)
        workflow.add_node("synthesis_node", self.nodes.node_synthesis)
        workflow.add_node("output_guardrail_node", self.nodes.node_output_guardrail)

        # 2. Register Starting Edge
        workflow.add_edge(START, "guardrail_node")

        # 3. Guardrail Branch: Stop immediately if blocked; proceed to intent router if passed
        workflow.add_conditional_edges(
            "guardrail_node",
            route_after_guardrail,
            {
                "blocked": END,
                "proceed": "intent_router_node"
            }
        )

        # 4. Intent Router Branch: Route to domain specialist nodes
        workflow.add_conditional_edges(
            "intent_router_node",
            route_by_intent,
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
    print("Testing LangGraph Pipeline with graph_nodes.py...")
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
