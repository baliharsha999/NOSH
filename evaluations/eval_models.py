# -*- coding: utf-8 -*-
"""
evaluations/eval_models.py
--------------------------
Custom LLM Model Adapter for DeepEval using Groq Cloud API.

Wraps Groq's `qwen/qwen3.8-27b` into DeepEval's `DeepEvalBaseLLM` interface,
allowing full G-Eval and LLM-as-a-Judge evaluations without requiring OpenAI keys.
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

# Ensure robust project root path and .env loading
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

from groq import Groq
from deepeval.models.base_model import DeepEvalBaseLLM

class GroqDeepEvalModel(DeepEvalBaseLLM):
    """
    Adapter enabling DeepEval metrics to execute using Groq Cloud LLMs with auto-retry & fresh socket pool.
    """
    def __init__(self, model_name: str = "qwen/qwen3.8-27b"):
        self.model_name = model_name
        self.api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is missing from .env")
        self.client = Groq(api_key=self.api_key, max_retries=5, timeout=45.0)

    def load_model(self) -> Groq:
        return self.client

    def get_model_name(self) -> str:
        return self.model_name

    def generate(self, prompt: str, schema: Optional[Any] = None) -> str:
        """
        Synchronous prompt generation for DeepEval metric evaluation with auto-reconnect backoff.
        """
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=1000
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                # Re-establish fresh client connection on socket drop
                try:
                    self.client = Groq(api_key=self.api_key, max_retries=5, timeout=45.0)
                except Exception:
                    pass
                if attempt < max_retries - 1:
                    time.sleep(3.0 * (attempt + 1))
                else:
                    print(f"[GroqDeepEvalModel] Generation error after {max_retries} attempts: {e}")
                    raise e

    async def a_generate(self, prompt: str, schema: Optional[Any] = None) -> str:
        """
        Asynchronous generation for concurrent DeepEval runs.
        """
        return self.generate(prompt, schema)
