"""
AlphaFusionNet Service: Hybrid Portfolio Decision-Making Engine (Production Ready)
============================================================================

Overview
--------
This module provides the main service for AlphaFusionNet, a hybrid portfolio decision-making engine 
that fuses outputs from two intelligent portfolio modules — NeuralFusionCore and NetWeaver — 
and combines both quantitative and qualitative reasoning to produce final portfolio weights.

Key Features
------------
1. Quantitative Fusion:
   - Combines NeuralFusionCore (risk-aware weights) and NetWeaver (return predictions) outputs.
   - Applies a fusion coefficient alpha to balance risk vs return signals.
   - Normalizes weights to ensure total gross exposure equals 1.
   - Supports both long and short positions.

2. Top-K Filtering:
   - Top-K filtering for NetWeaver outputs before fusion (`topk_net`).
   - Top-K filtering for final portfolio after fusion (`topk_final`).

3. LLM Integration:
   - Uses OpenAI GPT-4 (via `OpenAI_LLM` wrapper) to provide qualitative guidance.
   - Generates policy JSON with alpha, weighting method, gross exposure, and reasoning.
   - Graceful fallback to default parameters if LLM fails.

4. Config-driven:
   - All parameters (alpha, gross, min/max weights, top-k values, LLM model) are configurable 
     via `config/AFN_config.yaml`.
   - Supports production deployment with logging and audit trails.

Inputs
------
- NeuralFusionCore weights (`w_neural`): Dict[str, float]
  Example: {"AAPL": 0.12, "MSFT": 0.08, "NVDA": 0.15, "TSLA": -0.05}

- NetWeaver scores (`s_net`): Dict[str, float]
  Example: {"NVDA": 0.07, "TSLA": -0.06, "AAPL": 0.03, "GME": 0.11}

- Optional notes string to bias LLM decision-making (`notes`).

Outputs
-------
- Policy JSON: Dict with fusion parameters, overrides, top-k info, and reasoning.
- NetWeaver converted weights: pd.Series after normalization and top-k filtering.
- Final AlphaFusionNet weights: pd.Series after fusion and optional final top-k filtering.

Author: Elham Esmaeilnia(elham.e.shirvani@gmail.com)
Date: 2025 Oct 22
Version: 1.1.0 
"""
# AlphaFusionNet: full quantitative + qualitative hybrid module
# (Quant core + OpenAI GPT-4 LLM supervisor)
# Loads OPENAI_API_KEY from .env (KEY name: OPENAI_API_KEY)
# - Safety/validation enforced before applying policy

import os
import json
import logging
import yaml
from pymongo import MongoClient
from datetime import datetime, timezone
import pandas as pd
from dotenv import load_dotenv
from src.quant_alphafusionnet import QuantAlphaFusionNet
from src.llm_alphafusionnet import OpenAI_LLM
from src.controller import LLMAlphaFusionNetController

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in .env")

#--------------------------------
# Mongo Client
#--------------------------------
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
# -------------------------------
# Define paths relative to project root
# -------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))             # /AlphaFusionNet/scripts
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))          # /AlphaFusionNet

# -------------------------------
# Logging setup
# -------------------------------
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "alphafusionnet_service.log")
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------------------
# Load config
# -------------------------------
# Config file
CONFIG_FILE = os.path.join(BASE_DIR, "config", "AFN_config.yaml")

try:
    with open(CONFIG_FILE, "r") as f:
        cfg = yaml.safe_load(f)
except FileNotFoundError:
    raise RuntimeError(f"Config file not found: {CONFIG_FILE}")

quant_cfg = cfg.get("quant", {})
llm_cfg = cfg.get("llm", {})
topk_cfg = cfg.get("topk", {})

# -------------------------------
# Initialize AlphaFusionNet cores
# -------------------------------
quant = QuantAlphaFusionNet(
    alpha=quant_cfg.get("alpha", 0.7),
    gross=quant_cfg.get("gross", 1.0),
    w_min=quant_cfg.get("w_min", -0.4),
    w_max=quant_cfg.get("w_max", 0.4),
)

llm = OpenAI_LLM(
    model_name=llm_cfg.get("model_name", "gpt-4o-mini"),
    timeout=llm_cfg.get("timeout", 30)
)

controller = LLMAlphaFusionNetController(quant_core=quant, llm_client=llm)

# -------------------------------
# inputs 
# -------------------------------
# NeuralFusionCore prediction file
NFC_PATH = os.path.join(BASE_DIR, "apps", "NeuralFusionCore", "scripts", "NeuralFusionCore_prediction.json")

try:
    with open(NFC_PATH, "r") as f:
        nf_data = json.load(f)
    nf_weights = dict(zip(nf_data["stocks"], nf_data["weights"]))
    nf_ts = nf_data.get("ts")
    nf_ts_parsed = datetime.fromisoformat(nf_ts.replace("Z", "+00:00")) if nf_ts else datetime.now(timezone.utc)
    logger.info(f"Loaded NeuralFusionCore weights from {NFC_PATH}")
except Exception as e:
    logger.error(f"Failed to load NeuralFusionCore_prediction: {e}")
    raise RuntimeError(f"Error reading {NFC_PATH}: {e}")

print("✅ Loaded nf weights:")
print(nf_weights)

# NetWeaver prediction CSV
NW_PATH = os.path.join(BASE_DIR, "apps", "NetWeaver", "results", "predict", "selected_prediction.csv")

# Load the CSV into a DataFrame
df = pd.read_csv(NW_PATH)

# Make sure columns exist
if not {"symbol", "predicted_return"}.issubset(df.columns):
    raise ValueError(f"CSV file {NW_PATH} must contain 'symbol' and 'predicted_return' columns")

# Convert to dictionary: {'AAPL': 0.03, 'NVDA': 0.07, ...}
nw_scores = dict(zip(df["symbol"], df["predicted_return"]))

print("✅ Loaded scores:")
print(nw_scores)
try:
    collection = db["NetWeaver_predictions"]

    doc = {
        "timestamp": nf_ts_parsed,
        "predicted_return": nw_scores.fillna(0.0).to_dict()
    }

    collection.insert_one(doc)
    logger.info("NetWeaver predictions saved to MongoDB successfully.")
    print("NetWeaver predictions saved to MongoDB successfully.")
except Exception as e:
    logger.error("Failed to save NetWeaver predictions to MongoDB: %s", e)
    print("Failed to save NetWeaver Predictions to MongoDB:", e)

# -------------------------------
# Run AlphaFusionNet decision
# -------------------------------
out = controller.decide(
    w_neural=nf_weights,
    s_net=nw_scores,
    notes="""You are the decision-making policy analyst for AlphaFusionNet — a hybrid AI portfolio optimizer 
that fuses quantitative and relational intelligence. You are provided with two model outputs:

1. NeuralFusion module:
   - Produces signed portfolio weights for all assets (longs and shorts).
   - These weights are derived from a fusion of TimesNet (temporal OHLCV dynamics) 
     and BigBird (news and sentiment embeddings).
   - The optimization objective is a differentiable Sharpe ratio loss, 
     so weights reflect *risk-adjusted confidence* in long/short exposure.
   - Positive weights → expected outperformance with controlled volatility.
   - Negative weights → expected underperformance or hedging candidates.

2. NetWeaver module:
   - Produces a graph-based understanding of inter-stock relationships using a GAT (Graph Attention Network).
   - Each score represents predicted *relative return ratio* and direction (+ for upward, - for downward).
   - Top-k assets represent highest predicted alpha opportunities 
     within their sectoral or structural relationships.

Your task:
----------
Fuse these two model perspectives into a coherent *policy JSON* that defines how AlphaFusionNet 
should combine risk-adjusted weights and relational alpha signals.

Be explicit about:
  - The **fusion coefficient α** (balance between risk control and return-seeking behavior).
  - The **weighting method** ("rank", "softmax", or "proportional").
  - The **gross exposure fraction** allocated to the NetWeaver signals (`gross_net`).
  - Whether to apply **top-k selection** for the NetWeaver or final fusion stage.
  - Any **ticker-specific overrides** or **sector multipliers** (e.g., overweight tech if market sentiment is strong).
  - A concise but detailed **reasoning** paragraph, grounded in market logic.

Market context (recent regime example):
---------------------------------------
- Volatility regime: Moderate-to-high (VIX ~ 22)
- Sector rotation: Technology showing resilience, consumer discretionary slowing.
- Macro backdrop: Inflation cooling but rates remain elevated.
- Market breadth: Narrow leadership in large caps; small caps lagging.
- News sentiment: Positive tone in earnings but mixed on policy outlook.

Your output must be a structured JSON object:
{
  "alpha": float, 
  "method": str, 
  "gross_net": float, 
  "topk_net": int | null, 
  "topk_final": int | null, 
  "overrides": dict, 
  "sector_multipliers": dict, 
  "reasoning": str
}
Ensure your reasoning connects model insights to market context and clearly explains 
the policy choices."""
)

# -------------------------------
# Apply top-k final portfolio
# -------------------------------
topk_final = topk_cfg.get("default_final", None)
if topk_final is not None:
    abs_sorted = out["final_weights"].abs().sort_values(ascending=False)
    topk_idxs = abs_sorted.head(topk_final).index
    out["final_weights"] = out["final_weights"].reindex(topk_idxs).copy()
    out["final_weights"] /= out["final_weights"].abs().sum()  # normalize gross exposure

# -------------------------------
# Logging results
# -------------------------------
logger.info("AlphaFusionNet Policy: %s", json.dumps(out["policy"]))
logger.info("NetWeaver converted weights: %s", out["w_net_converted"].to_dict())
logger.info("Final AlphaFusionNet weights: %s", out["final_weights"].to_dict())

# -------------------------------
# Console output
# -------------------------------
print("Policy:", out["policy"])
print("NetWeaver converted weights:", out["w_net_converted"])
print("Final AlphaFusionNet weights:", out["final_weights"])
final_weights = out["final_weights"]
try:
    collection = db["AlphaFusionNet_predictions"]

    doc = {
        "timestamp": nf_ts_parsed,
        "policy": out["policy"],
        "final_weights": final_weights.fillna(0.0).to_dict()
    }

    collection.insert_one(doc)
    logger.info("AlphaFusionNet decision saved to MongoDB successfully.")
    print("AlphaFusionNet decision saved to MongoDB successfully.")
except Exception as e:
    logger.error("Failed to save AlphaFusionNet decision to MongoDB: %s", e)
    print("Failed to save AlphaFusionNet decision to MongoDB:", e)
