"""
TradingAgent: Generates human-readable reasoning for AlphaFusionNet portfolio decisions
======================================================================================

This module provides reasoning and explanation for final portfolio weights, stop-loss
and take-profit levels, using AlphaFusionNet outputs and LLM analysis.

It produces:
- A textual reasoning per portfolio decision
- Stores it in MongoDB collection 'TradingAgent_reasons' with the same timestamp

Author: Elham Esmaeilnia(elham.e.shirvani@gmail.com)
Date: 2025 Nov 08
Version: 1.1.0 
"""

import os
import json
from datetime import datetime
from typing import Dict, Any
from pymongo import MongoClient
from dotenv import load_dotenv
from src.llm_alphafusionnet import OpenAI_LLM
from pathlib import Path 

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in .env")

# -----------------------------
# MongoDB Setup
# -----------------------------
mongo_user = os.getenv("NOVO_MONGO_USER")
mongo_pass = os.getenv("NOVO_MONGO_PASS")
mongo_host = os.getenv("NOVO_MONGO_HOST")
mongo_port = os.getenv("NOVO_MONGO_PORT")
mongo_auth_db = os.getenv("NOVO_MONGO_AUTH_DB")
mongo_db_name = os.getenv("NOVO_MONGO_DB")

mongo_uri = (
    f"mongodb://{mongo_user}:{mongo_pass}@"
    f"{mongo_host}:{mongo_port}/"
    f"{mongo_db_name}?authSource={mongo_auth_db}"
)
client = MongoClient(mongo_uri)
db = client[mongo_db_name]

# -----------------------------
# TradingAgent Class
# -----------------------------
class TradingAgent:
    def __init__(self, llm_model: str = "gpt-4o-mini", timeout: int = 30):
        """
        Initialize TradingAgent with an OpenAI LLM client.
        """
        self.llm = OpenAI_LLM(model_name=llm_model, timeout=timeout)
    
    def generate_reasoning(
        self,
        final_weights: Dict[str, float],
        risk_controls: Dict[str, Dict[str, float]],
        notes: str = "",
        timestamp: datetime = None
    ) -> str:
        """
        Generate human-readable, trader-friendly reasoning for portfolio weights.
        Each asset with weight > 2% receives a short paragraph explaining its allocation.
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        timestamp_str = timestamp.isoformat()

        # Prepare prompt with real values
        prompt_text = f"""
You are an expert financial analyst writing reasoning for a trading portfolio.
Your audience is a trader or investor who does not know the internal models used to generate the portfolio.
You must explain why each asset was assigned its weight, using only market-relevant information such as:

- Recent price trends and volatility
- Macro conditions (e.g., interest rates, inflation, global events)
- News sentiment or recent announcements
- Sector performance or rotations
- Index correlations or overall market context
- Fundamentals (e.g., earnings, growth, P/E ratios)

Instructions:

1. Only reference observable market data. Do NOT mention internal model names, alpha coefficients, or weighting formulas.
2. Give a short paragraph for each major asset (weight > 2%) explaining why it has its current weight in the portfolio.
3. Keep reasoning concise, readable, and trader-friendly.
4. Highlight risks or factors to watch for each asset if relevant.
5. Provide a summary paragraph at the end describing the overall portfolio outlook in plain language.

Input Data:

final_weights: {final_weights}
risk_controls: {risk_controls}
notes: {notes}

Output Format:

- A JSON object with a single field "reasoning" containing the user-facing textual explanation.
-Important note:
    e.g: for IG:EURUSD in output should have : 
        EURUSD (-3.11%): A negative position indicates a bearish outlook on the Euro against the USD, driven by potential interest rate differentials. Keep an eye on economic indicators from both regions.
-Becareful to not insert "IG:" , just use the name of assets in final_weights.



"""

        reasoning_text = ""
        try:
            # Request reasoning from the LLM
            reasoning_dict = self.llm.request_policy(prompt_text)
            reasoning_text = reasoning_dict.get("reasoning", "")
        except Exception as e:
            reasoning_text = f"LLM reasoning failed: {e}"

        # Save reasoning to MongoDB
        try:
            collection = db["TradingAgent_reasons"]
            doc = {
                "timestamp": timestamp_str,
                "reasoning": reasoning_text
            }
            collection.insert_one(doc)
        except Exception as e:
            print("Failed to save reasoning to MongoDB:", e)

        return reasoning_text
  
