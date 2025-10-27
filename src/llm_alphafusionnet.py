"""
OpenAI_LLM: LLM Wrapper for AlphaFusionNet Policy Decisions
======================================================

Overview
--------
This module provides the OpenAI_LLM class, which serves as a wrapper around 
OpenAI's GPT-4 (or similar) models to produce qualitative, context-aware 
portfolio fusion policies for AlphaFusionNet. It converts NeuralFusionCore and NetWeaver 
outputs into structured JSON recommendations, which guide the quantitative 
fusion engine (QuantAlphaFusionNet).

Key Features
------------
1. Structured policy generation:
   - Generates JSON policies containing alpha, weighting method, gross exposure, top-k, 
     overrides, sector multipliers, and reasoning.
   - Ensures policies are interpretable and auditable.

2. JSON Function Schema Support:
   - Uses OpenAI ChatCompletion functions (as of API version 1.0+) to request structured JSON.

3. Fallback and Robustness:
   - Includes error handling for LLM failures (network, API errors, invalid JSON).
   - Provides default policy values if LLM response is missing or malformed.

Inputs
------
- JSON string prompt containing:
    - NeuralFusionCore weights (dict[str,float])
    - NetWeaver predicted returns (dict[str,float])
    - Optional notes (str) describing market context or user bias

Outputs
-------
- policy : dict[str, Any]
    Structured JSON policy containing:
        - alpha : float in [0,1] — fusion coefficient
        - method : str — weighting method ('rank', 'softmax', 'proportional')
        - gross_net : float — NetWeaver gross exposure fraction
        - topk_net : int | None — optional top-k for NetWeaver weights
        - overrides : dict[str,float] — optional ticker-specific weight overrides
        - sector_multipliers : dict[str,float] — optional sector/ticker multipliers
        - reasoning : str — LLM-provided textual justification

Author: Elham Esmaeilnia(elham.e.shirvani@gmail.com)
Date: 2025 Oct 22
Version: 1.1.0 
"""
import os
import json
from typing import Dict, Any
import openai
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in .env")
openai.api_key = OPENAI_API_KEY

class OpenAI_LLM:
    def __init__(self, model_name: str = "gpt-4o-mini", timeout: int = 20):
        self.model = model_name
        self.timeout = timeout

    def request_policy(self, prompt_json: str) -> Dict[str, Any]:
        function_spec = {
            "name": "set_AlphaFusionNet_policy",
            "description": "Return a JSON policy for AlphaFusionNet fusion parameters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "alpha": {"type": "number"},
                    "method": {"type": "string", "enum": ["rank","softmax","proportional"]},
                    "gross_net": {"type": "number"},
                    "topk_net": {"type": ["integer","null"]},
                    "topk_final": {"type": ["integer","null"]},
                    "overrides": {"type": "object","additionalProperties":{"type":"number"}},
                    "sector_multipliers": {"type": "object","additionalProperties":{"type":"number"}},
                    "reasoning": {"type": "string"}
                }
            }
        }

        system_msg = (
            "You are AlphaFusionNet policy advisor. Based on NeuralFusionCore and NetWeaver outputs, "
            "return a fusion policy JSON with alpha, method, gross_net, topk_net, topk_final, overrides, sector_multipliers, reasoning."
        )

        user_msg = "Inputs:\n" + prompt_json + "\nReturn JSON only. Defaults: alpha=0.7, method='rank', gross_net=0.5"

        try:
            resp = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role":"system","content":system_msg},
                    {"role":"user","content":user_msg},
                ],
                functions=[function_spec],
                function_call={"name":"set_AlphaFusionNet_policy"},
                temperature=0.0,
                request_timeout=self.timeout,
            )
        except Exception as e:
            raise RuntimeError(f"LLM API call failed: {e}")

        try:
            args_text = resp["choices"][0]["message"]["function_call"]["arguments"]
            return json.loads(args_text)
        except Exception:
            return {}
