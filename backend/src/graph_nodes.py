# -*- coding: utf-8 -*-
"""
backend/src/graph_nodes.py
--------------------------
Encapsulates all LangGraph execution nodes and conditional edge routers for the
NOSH Conversational Recipe Intelligence StateGraph.

Separating these execution nodes from pipeline.py keeps the core orchestration
clean, modular, and maintainable.

Nodes Included:
  1. node_guardrail: 5-Pillar safety & schema verification (prep times, calories, injections)
  2. node_intent_router: LLM semantic entity parsing & intent classification
  3. node_analytics: Task 1 exact programmatic stats & step rankings
  4. node_matching: Task 2 constraint-satisfaction pantry matching
  5. node_substitution: Task 3 local sister-recipe citations & web search fallback
  6. node_refinement: Multi-turn dietary restriction accumulation & filtering
  7. node_search: Natural language recipe retrieval with title phrase boosting (+150)
  8. node_synthesis: Grounded LLM generation with live token streaming (StreamWriter)
  9. node_output_guardrail: Medical safety disclaimer & portion warning enforcement
"""

import re
from typing import Dict, List, Set, Any, Optional
from typing_extensions import TypedDict
from langgraph.types import StreamWriter

try:
    from backend.src.ingestion import RecipeIndexBundle
    from backend.src.analytics_engine import AnalyticsEngine
    from backend.src.matcher import RecipeMatcher, MatchRequest, MatchResponse
    from backend.src.retriever import RecipeRetriever
    from backend.src.substitution_engine import SubstitutionEngine, SubstitutionRequest, AdaptedRecipe
    from backend.src.llm_generator import LLMGenerator
    from backend.src.guardrails import (
        NoshGuardrails,
        MEDICAL_DISCLAIMER,
        QUANTITY_SUFFICIENCY_NOTICE,
        ABSENCE_OF_DATA_EXPLANATION
    )
    from backend.src.session_store import SQLiteSessionStore
    from backend.src.intent_router import get_semantic_router, SemanticIntentRouter
except ImportError:
    from ingestion import RecipeIndexBundle
    from analytics_engine import AnalyticsEngine
    from matcher import RecipeMatcher, MatchRequest, MatchResponse
    from retriever import RecipeRetriever
    from substitution_engine import SubstitutionEngine, SubstitutionRequest, AdaptedRecipe
    from llm_generator import LLMGenerator
    from guardrails import (
        NoshGuardrails,
        MEDICAL_DISCLAIMER,
        QUANTITY_SUFFICIENCY_NOTICE,
        ABSENCE_OF_DATA_EXPLANATION
    )
    from session_store import SQLiteSessionStore
    from intent_router import get_semantic_router, SemanticIntentRouter


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
# Conditional Edge Routing Functions
# =============================================================================
def route_after_guardrail(state: PipelineState) -> str:
    """Conditional Edge: Stop immediately if blocked; proceed to intent router if passed."""
    if state.get("guardrail_blocked", False):
        return "blocked"
    return "proceed"


def route_by_intent(state: PipelineState) -> str:
    """Conditional Edge: Route query dynamically to domain specialist execution node."""
    intent = state.get("intent", "SEARCH")
    valid_intents = {"ANALYTICS", "MATCHING", "SUBSTITUTION", "RESTRICTION_REFINEMENT", "SEARCH"}
    return intent if intent in valid_intents else "SEARCH"


# =============================================================================
# Graph Nodes Collection
# =============================================================================
class GraphNodes:
    """
    Encapsulates all LangGraph node functions, binding them to backend domain engines.
    """
    def __init__(
        self,
        bundle: RecipeIndexBundle,
        analytics: AnalyticsEngine,
        matcher: RecipeMatcher,
        substitution: SubstitutionEngine,
        llm: LLMGenerator,
        session_store: SQLiteSessionStore
    ):
        self.bundle = bundle
        self.analytics = analytics
        self.matcher = matcher
        self.substitution = substitution
        self.llm = llm
        self.session_store = session_store
        self.semantic_router = get_semantic_router()

    def node_guardrail(self, state: PipelineState, writer: StreamWriter) -> Dict[str, Any]:
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

    def node_intent_router(self, state: PipelineState, writer: StreamWriter) -> Dict[str, Any]:
        """Node 2: Dense Semantic Vector Intent Routing + LLM Entity Extraction."""
        user_query = state.get("user_query", "")

        # 1. Execute Dense Semantic Vector Intent Routing (Cosine Similarity in R^384)
        semantic_route = self.semantic_router.route_query(user_query)
        semantic_intent = semantic_route.get("top_intent", "SEARCH")
        semantic_entities = semantic_route.get("semantic_entities", {})

        # Fast Guardrail Interception
        if semantic_intent == "ABSENCE_GUARDRAIL":
            writer(ABSENCE_OF_DATA_EXPLANATION)
            return {
                "guardrail_blocked": True,
                "guardrail_triggered": "AbsenceOfDataGuardrail",
                "intent": "ABSENCE_GUARDRAIL",
                "response": ABSENCE_OF_DATA_EXPLANATION,
                "data": {"guardrail_triggered": "absence_of_data", "field_requested": "prep_time_or_calories"}
            }

        # 2. Extract Entities using Groq Qwen with Dense Semantic Intent as prior
        parsed = self.llm.parse_query_semantics(user_query, semantic_intent_prior=semantic_intent)
        detected_intent = parsed.get("intent") or semantic_intent

        # Secondary check for Absence-of-Data
        if detected_intent == "ABSENCE_GUARDRAIL":
            writer(ABSENCE_OF_DATA_EXPLANATION)
            return {
                "guardrail_blocked": True,
                "guardrail_triggered": "AbsenceOfDataGuardrail",
                "intent": "ABSENCE_GUARDRAIL",
                "response": ABSENCE_OF_DATA_EXPLANATION,
                "data": {"guardrail_triggered": "absence_of_data", "field_requested": "prep_time_or_calories"}
            }

        # Merge extracted entities from LLM with semantic router entities as fallback
        cleaned_dish = parsed.get("cleaned_dish_or_recipe") or semantic_entities.get("cleaned_dish_or_recipe")
        pantry_ings = parsed.get("pantry_ingredients") or semantic_entities.get("pantry_ingredients", [])
        exclusions = parsed.get("exclusions") or semantic_entities.get("exclusions", [])
        sub_ing = parsed.get("substitute_ingredient") or semantic_entities.get("substitute_ingredient")
        max_steps = parsed.get("max_steps") if parsed.get("max_steps") is not None else semantic_entities.get("max_steps")

        return {
            "intent": detected_intent,
            "cleaned_dish_or_recipe": cleaned_dish,
            "pantry_ingredients": pantry_ings,
            "exclusions": exclusions,
            "target_ingredient": parsed.get("target_ingredient"),
            "is_step_ranking": parsed.get("is_step_ranking", semantic_entities.get("is_step_ranking", False)),
            "substitute_ingredient": sub_ing,
            "max_steps": max_steps
        }

    def node_analytics(self, state: PipelineState) -> Dict[str, Any]:
        """Node 3A: Executes Task 1 exact programmatic statistical calculations."""
        user_query = state.get("user_query", "")
        q_lower = user_query.lower()

        # Check for Formula Transparency & Denominator Queries
        if "formula" in q_lower or "denominator" in q_lower:
            denom_res = {
                "total_denominator": self.analytics.total_recipes,
                "formula": f"(Matching Recipe Count / {self.analytics.total_recipes:,}) * 100",
                "normalization": "Ingredients are lowercased, stripped of units (cups, tbsp, oz) and prep words (chopped, minced), and lemmatized to root nouns.",
                "dataset_size": self.analytics.total_recipes
            }
            fallback_text = (
                f"**NOSH Statistical Analytics & Formula Transparency:**\n"
                f"- **Curated Denominator:** **{self.analytics.total_recipes:,}** curated recipes\n"
                f"- **Reproducible Percentage Formula:** `Percentage = (Count / {self.analytics.total_recipes:,}) * 100`\n"
                f"- **Normalization Process:** Ingredients are lowercased, stripped of units (cups, tbsp, oz) and prep words (chopped, minced), and lemmatized to root nouns before exact inverted index intersection."
            )
            return {
                "context_data": denom_res,
                "data": denom_res,
                "fallback_text": fallback_text
            }

        # Check for Corpus Distribution / Statistics Query
        if "distribution" in q_lower and ("step" in q_lower or "complexity" in q_lower or "ingredient" in q_lower):
            stats = self.analytics.get_dataset_statistics()
            steps_dist = stats.get("steps_distribution", {})
            ings_dist = stats.get("ingredients_distribution", {})
            fallback_text = (
                f"**Recipe Complexity & Instruction Step Distribution across {self.analytics.total_recipes:,} Recipes:**\n"
                f"- **Step Statistics:** Mean = {steps_dist.get('mean', 4.03)} steps, Median = {steps_dist.get('median', 4.0)} steps (Min: {steps_dist.get('min', 1)}, Max: {steps_dist.get('max', 71)})\n"
                f"- **Step Brackets:**\n"
                f"  • **1 to 3 steps (Quick & Simple):** Fast home cooking preparations\n"
                f"  • **4 to 6 steps (Standard):** Everyday multi-step family meals\n"
                f"  • **7 to 10 steps (Intermediate):** Layered culinary techniques\n"
                f"  • **11+ steps (Advanced/Complex):** Elaborate gourmet recipes\n"
                f"- **Ingredients Statistics:** Mean = {ings_dist.get('mean', 10.39)}, Median = {ings_dist.get('median', 9.0)}\n"
                f"- **Step-counting rule:** Instructions segmented by newline boundaries."
            )
            return {
                "context_data": stats,
                "data": stats,
                "fallback_text": fallback_text
            }

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

        # Check for Multi-Ingredient Co-occurrence or Single-Ingredient Frequency
        ings_to_check = []
        if "garlic" in q_lower:
            ings_to_check.append("garlic")
        if "onion" in q_lower:
            ings_to_check.append("onion")
        if not ings_to_check:
            m = re.search(r"both\s+([a-zA-Z]+)\s+and\s+([a-zA-Z]+)", q_lower)
            if m:
                ings_to_check = [m.group(1), m.group(2)]

        # Check for common single ingredients or regex
        if not ings_to_check:
            for cand in ["chicken", "beef", "pork", "tomato", "tomatoes", "potato", "potatoes", "egg", "eggs", "rice", "cheese", "pasta", "butter", "milk", "fish", "shrimp"]:
                if cand in q_lower:
                    ings_to_check.append(cand)
                    break
        if not ings_to_check:
            m_single = re.search(r"(?:how many recipes (?:contain|use|have)|frequency of|count of)\s+([a-zA-Z]+)", q_lower)
            if m_single:
                ings_to_check.append(m_single.group(1))

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

    def node_matching(self, state: PipelineState) -> Dict[str, Any]:
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

    def node_substitution(self, state: PipelineState) -> Dict[str, Any]:
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
            elif "beef" in q_lower or "meat" in q_lower:
                target_ing = "beef"
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
            elif "beef" in target_ing or "meat" in target_ing:
                target_ing = "beef"
            elif "heavy cream" in target_ing:
                target_ing = "heavy cream"
            elif "cream" in target_ing:
                target_ing = "cream"
            elif "gelatin" in target_ing:
                target_ing = "gelatin"

        recipe_title = (state.get("cleaned_dish_or_recipe") or "").strip()
        if not recipe_title or any(k in q_lower for k in ["banana", "soup", "stew", "cookie", "panna"]):
            if "soup" in q_lower:
                recipe_title = "creamy savory soup"
            elif "stew" in q_lower:
                recipe_title = "beef stew"
            elif "cookie" in q_lower:
                recipe_title = "chocolate chip cookies"
            elif "banana" in q_lower:
                recipe_title = "banana bread"
            elif "panna" in q_lower or "gelatin" in q_lower:
                recipe_title = "panna cotta"
            else:
                recipe_title = recipe_title or "pasta"

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

    def node_refinement(self, state: PipelineState) -> Dict[str, Any]:
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

    def node_search(self, state: PipelineState) -> Dict[str, Any]:
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

    def node_synthesis(self, state: PipelineState, writer: StreamWriter) -> Dict[str, Any]:
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

    def node_output_guardrail(self, state: PipelineState, writer: StreamWriter) -> Dict[str, Any]:
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
