# -*- coding: utf-8 -*-
"""
backend/src/llm_generator.py
----------------------------
Grounded LLM Synthesis Layer for NOSH Conversational Recipe Intelligence.

Features:
  1. Strictly grounded in deterministic backend outputs (analytics, retrieved docs, substitutions).
  2. Zero-Hallucination Guardrails:
     - Rejects fabricating missing metadata (prep times, calories, cook times).
     - Respects allergen/dietary hard constraints.
  3. Powered by Groq Cloud (`qwen/qwen3.8-27b`).
  4. Robust Offline Fallback: If API key is missing or network fails, gracefully returns
     cleanly formatted markdown responses without crashing.
"""

import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """You are the NOSH Conversational Recipe Intelligence Assistant, an expert culinary AI pair-programmer and chef.
You are assisting users with a curated collection of 121,684 recipes from Epicurious, FoodNetwork, and AllRecipes.

STRICT BEHAVIORAL GUARDRAILS:
1. STRICT DATA GROUNDING: You must answer strictly based on the facts, recipe details, counts, and percentages provided in the context. Never fabricate recipe counts, percentages, or ingredients.
2. ABSENCE-OF-DATA INTEGRITY: The underlying Recipe Box dataset does NOT contain preparation times, cooking times, or nutritional/caloric values. If the user asks for prep time, cooking time, or calories, you MUST explicitly state that this data does not exist in the dataset and you will not guess or hallucinate it.
3. ALLERGEN SAFETY: Never claim a recipe substitution is "medically safe". Always include a disclaimer that substitutions are based on empirical culinary data, not medical advice.
4. RECIPE CITATION: Always reference the official Recipe ID (e.g. `[ID: abc123xyz]`) and recipe title when presenting or referencing recipes.
5. CONCISE, PROFESSIONAL CHEF TONE: Be helpful, structured, clear, and culinary-minded. Use markdown formatting with bullet points and bold headers.
"""

class LLMGenerator:
    def __init__(self, model_name: str = "qwen/qwen3.8-27b"):
        self.model_name = model_name
        self.api_key = os.environ.get("GROQ_API_KEY", "").strip()
        self.client = None
        if self.api_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key, timeout=25.0)
            except Exception as e:
                print(f"[LLMGenerator] Warning: Could not initialize Groq client: {e}")

    def is_available(self) -> bool:
        return self.client is not None

    def synthesize(self, user_query: str, intent: str, context_data: Dict[str, Any], raw_fallback_response: str) -> str:
        """
        Synthesizes a natural, grounded culinary response using the LLM with context data.
        Falls back to raw_fallback_response if LLM fails or is unavailable.
        """
        if not self.is_available():
            return raw_fallback_response

        # Build prompt depending on intent
        user_prompt = f"""User Request: "{user_query}"
Detected Intent: {intent}

Deterministic Context & Ground Truth from Local Dataset:
{self._format_context_for_llm(intent, context_data, raw_fallback_response)}

Instructions:
Provide a polished, helpful, and natural response directly answering the user query.
Adhere strictly to the ground truth provided above. Do not hallucinate numbers or unverified steps.
"""
        try:
            chat_completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,  # Low temperature for factual grounding
                max_tokens=850
            )
            response = chat_completion.choices[0].message.content
            return response.strip() if response else raw_fallback_response
        except Exception as e:
            print(f"[LLMGenerator] Error during LLM generation: {e}. Falling back to deterministic response.")
            return raw_fallback_response

    def _format_context_for_llm(self, intent: str, context_data: Dict[str, Any], fallback_text: str) -> str:
        """
        Structures the context dictionary cleanly into text for the LLM.
        """
        if intent == "ANALYTICS":
            if "percentage" in context_data:
                return (
                    f"- Exact Co-occurrence Count: {context_data.get('matching_count')}\n"
                    f"- Total Dataset Denominator: {context_data.get('total_denominator')} recipes\n"
                    f"- Exact Percentage: {context_data.get('percentage')}%\n"
                    f"- Formula Used: {context_data.get('mathematical_formula')}\n"
                    f"- Methodology: {context_data.get('normalization_methodology')}"
                )
            elif "results" in context_data:
                res_lines = []
                for idx, r in enumerate(context_data.get("results", [])[:5], 1):
                    res_lines.append(f"{idx}. {r.get('title')} (ID: {r.get('id')}) - Steps: {r.get('num_steps')}, Ingredients: {r.get('num_ingredients')}")
                return (
                    f"- Target: {context_data.get('target_ingredient')}\n"
                    f"- Total Matching Recipes: {context_data.get('total_matching_recipes')}\n"
                    f"- Ranked Recipes:\n" + "\n".join(res_lines)
                )

        elif intent == "MATCHING":
            # MatchResponse object or dict
            if hasattr(context_data, "results"):
                res_lines = []
                for idx, r in enumerate(context_data.results[:5], 1):
                    res_lines.append(
                        f"{idx}. {r.title} (ID: {r.id}, Source: {r.source})\n"
                        f"   - Step Count: {r.num_steps}\n"
                        f"   - Matched Ingredients: {', '.join(r.matched_ingredients)}\n"
                        f"   - Missing to Buy: {', '.join(r.missing_ingredients) or 'None (100% matched)'}\n"
                        f"   - Score: {r.score}\n"
                        f"   - Rationale: {r.explanation}"
                    )
                return (
                    f"- Pantry Ingredients: {', '.join(context_data.query_available)}\n"
                    f"- Enforced Restrictions: {context_data.hard_constraints_enforced}\n"
                    f"- Max Steps Limit: {context_data.max_steps_limit}\n"
                    f"- Total Eligible Found: {context_data.total_eligible_found}\n"
                    f"- Quantity Notice: {context_data.quantity_sufficiency_notice}\n"
                    f"- Top Matches:\n" + "\n".join(res_lines)
                )

        elif intent == "SUBSTITUTION":
            # AdaptedRecipe object or dict
            adapted = context_data.get("adapted_recipe")
            if adapted:
                ev_lines = []
                for ev in adapted.local_empirical_evidence:
                    ev_lines.append(f"  • ID: {ev.id} | Title: '{ev.title}' (Source: {ev.source})")
                
                return (
                    f"- Original Dish: {adapted.original_title} (ID: {adapted.original_id})\n"
                    f"- Removed Ingredient: {adapted.ingredient_removed}\n"
                    f"- Chosen Substitute: {adapted.substitute_chosen}\n"
                    f"- Functional Culinary Role: {adapted.culinary_role_analyzed}\n"
                    f"- Empirical Evidence Recipes in Local Dataset:\n" + "\n".join(ev_lines) + "\n"
                    f"- Sensory Trade-Offs:\n"
                    f"    Taste: {adapted.sensory_trade_offs.get('taste')}\n"
                    f"    Texture: {adapted.sensory_trade_offs.get('texture')}\n"
                    f"    Preparation: {adapted.sensory_trade_offs.get('preparation')}\n"
                    f"- Medical Safety Disclaimer: {adapted.safety_disclaimer}\n"
                    f"- Updated Ingredients:\n" + "\n".join([f"    {ing}" for ing in adapted.updated_ingredients]) + "\n"
                    f"- Revised Procedural Steps:\n{adapted.revised_instructions}"
                )

        return fallback_text
