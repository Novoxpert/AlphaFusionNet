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
   - Generates policy JSON with alpha, weighting method, gross exposure, and a tradingagent for reasoning.
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
import random
from pymongo import MongoClient
from datetime import datetime, timezone
import pandas as pd
from dotenv import load_dotenv
from src.quant_alphafusionnet import QuantAlphaFusionNet
from src.llm_alphafusionnet import OpenAI_LLM
from src.controller import LLMAlphaFusionNetController
from src.TradingAgent import TradingAgent

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
    #gross=quant_cfg.get("gross", 1.0),
    gross= 1.0,
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
        "predicted_return": nw_scores
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
    notes="""You are the decision-making policy analyst for **AlphaFusionNet** — a hybrid AI portfolio optimizer 
that fuses quantitative and relational intelligence to generate coherent, risk-adjusted trading policies.  
You are provided with two model outputs: **NeuralFusion** and **NetWeaver**.

---

### 1. Model Inputs and Their Roles

#### **NeuralFusion module**
- Produces **signed portfolio weights** for all assets (longs and shorts).
- These weights are derived from a fusion of **TimesNet** (temporal OHLCV dynamics) 
  and **BigBird** (news and sentiment embeddings).
- The optimization objective is a **differentiable Sharpe ratio loss**, so weights reflect 
  *risk-adjusted confidence* in long/short exposure.
- **Positive weights →** expected outperformance with controlled volatility.  
  **Negative weights →** expected underperformance or hedging candidates.

#### **NetWeaver module**
- Produces a **graph-based understanding** of inter-stock relationships using a **GAT (Graph Attention Network)**.
- Each score represents a **predicted relative return ratio** and direction  
  (+ for expected upward performance, − for expected downward).
- **Top-k assets** represent the highest relational alpha opportunities 
  within their sectoral or structural graph neighborhoods.

---

### 2. Task: Construct a Unified Policy

Your objective is to **fuse both model perspectives** into a single *policy JSON* that governs how AlphaFusionNet 
combines risk-adjusted portfolio weights (NeuralFusion) and relational alpha signals (NetWeaver).

Be explicit about:
- **Fusion coefficient (α):** numeric balance between risk control and return-seeking behavior.  
  - α ≈ 1 → prioritize NeuralFusion (risk discipline).  
  - α ≈ 0 → prioritize NetWeaver (aggressive alpha-hunting).
- **Weighting method:** one of `"rank"`, `"softmax"`, or `"proportional"`.
- **Gross exposure fraction (`gross_net`):**  
  Always set `gross_net = 1.0` — NetWeaver operates under the same normalized gross exposure as NeuralFusion.
- **Top-k selections:** whether to apply top-k filtering to NetWeaver (`topk_net`) or to final fusion (`topk_final`).
- **Ticker-specific overrides:** manual adjustments for individual tickers (if justified).
- **Sector multipliers:** optional bias by sector (e.g., overweight tech, underweight energy).

---

### 3. Market Context (for reasoning)

Assume the following recent market regime:
- **Volatility:** Moderate-to-high (VIX ≈ 22)  
- **Sector rotation:** Technology showing resilience; consumer discretionary slowing  
- **Macro backdrop:** Inflation cooling but interest rates remain elevated  
- **Market breadth:** Narrow leadership in large caps; small caps lagging  
- **News sentiment:** Positive earnings tone but mixed policy outlook  

---

### 4. Risk Management (Mandatory Section)

You must design per-asset **Stop Loss (SL)** and **Take Profit (TP)** levels, assuming the **total portfolio 
market capitalization = 100,000 USD**.

Rules:
- For each asset *i*, compute its **position size** = `weight_i × 100,000`.
- Then define:
  - **SL**: expected downside threshold (negative % return or equivalent USD loss)
  - **TP**: expected upside threshold (positive % return or equivalent USD gain)
- SL/TP can be expressed either as **percentages** or **absolute USD** values, 
  but always provide **both** in your JSON.
- Example:  
  `AAPL weight = 0.10 → position = $10,000 → SL = -0.05 (−5% = −$500), TP = +0.10 (+10% = +$1,000)`

---
### 4.1 Mandatory JSON Schema Enforcement

MANDATORY: For every asset in the final portfolio, you MUST include a "risk_controls" entry with
SL and TP, expressed both as percentages and in USD (SL_usd, TP_usd).  
Do NOT omit any asset. Every ticker must appear in "risk_controls".  

The output JSON must strictly follow this structure (no extra fields, no missing tickers):

{
  "alpha": float,
  "method": str,
  "gross_net": 1.0,
  "topk_net": int | null,
  "topk_final": int | null,
  "overrides": dict,
  "sector_multipliers": dict,
  "reasoning": str,
  "risk_controls": {
      "TICKER1": {"SL": -0.05, "TP": 0.10, "SL_usd": float, "TP_usd": float},
      "TICKER2": {"SL": -0.05, "TP": 0.10, "SL_usd": float, "TP_usd": float},
      ...
  }
}
---
### 4.2 Explicit Calculation Rule

For every asset in the final portfolio:

1. Compute position size: position_i = weight_i × 100,000
2. Stop Loss (SL): -5% of position size → SL_usd = position_i × (-0.05)
3. Take Profit (TP): +10% of position size → TP_usd = position_i × 0.10
4. Fill both percentage and USD fields in "risk_controls" for each ticker.
5. Ensure every ticker in final_weights has an entry in "risk_controls".
---
### 5. Output Requirements

Return a single **structured JSON object** strictly in this format:

```json
{
  "alpha": 0.65,
  "method": "softmax",
  "gross_net": 1.0,
  "topk_net": 10,
  "topk_final": 15,
  "overrides": {
    "TSLA": 0.05
  },
  "sector_multipliers": {
    "Technology": 1.1,
    "Energy": 0.9
  },
  "reasoning": "Concise but analytically grounded explanation linking NeuralFusion risk signals, NetWeaver alpha maps, and macro context. Describe why α, method, and exposure splits were chosen.",
  "risk_controls": {
    "AAPL": {"SL": -0.05, "TP": 0.10, "SL_usd": -500, "TP_usd": 1000},
    "MSFT": {"SL": -0.04, "TP": 0.08, "SL_usd": -400, "TP_usd": 800},
    "NVDA": {"SL": -0.06, "TP": 0.12, "SL_usd": -600, "TP_usd": 1200}
  }
}"""
)

# -------------------------------
# Apply top-k final portfolio
# -------------------------------
topk_final = out["policy"].get("topk_final") or topk_cfg.get("default_final", None)

if topk_final is not None and isinstance(topk_final, int):
    abs_sorted = out["final_weights"].abs().sort_values(ascending=False)
    topk_idxs = abs_sorted.head(topk_final).index
    out["final_weights"] = out["final_weights"].reindex(topk_idxs).copy()

# ✅ Always normalize final weights to gross = 1
out["final_weights"] = out["final_weights"] / out["final_weights"].abs().sum()

# -------------------------------
# Logging results
# -------------------------------
logger.info("AlphaFusionNet Policy: %s", json.dumps(out["policy"], indent=2))
logger.info("NetWeaver converted weights: %s", out["w_net_converted"].to_dict())
logger.info("Final AlphaFusionNet weights: %s", out["final_weights"].to_dict())

# If risk controls (SL/TP) exist in the policy, log them separately
if "risk_controls" in out["policy"]:
    logger.info("Risk Controls (SL/TP): %s", json.dumps(out["policy"]["risk_controls"], indent=2))

# -------------------------------
# Ensure risk_controls exist (mandatory)
# -------------------------------
total_portfolio_value = 100_000  # USD

risk_controls = out["policy"].get("risk_controls", {})

for ticker, weight in out["final_weights"].items():

    # Default SL/TP percentages (can be adjusted or made dynamic)
    sl_pct = -0.01  # -5% loss
    tp_pct = 0.15   # +10% gain

    # random confidence
    confidence = round(random.uniform(0.55, 0.85), 2)

    # Only add if not already present from LLM
    if ticker not in risk_controls:
        risk_controls[ticker] = {
            "SL": sl_pct,
            "TP": tp_pct,
            "CONF": confidence
        }

# Update policy
out["policy"]["risk_controls"] = risk_controls

# -------------------------------
# Console output
# -------------------------------
print("\n✅ --- AlphaFusionNet Decision Summary ---")
print("Policy JSON:")
print(json.dumps(out["policy"], indent=2))
print("\nFinal AlphaFusionNet Weights:")
print(out["final_weights"])
print("Gross exposure check:", round(out["final_weights"].abs().sum(), 4))
print(out["policy"].get("risk_controls", {}))

print("------------------------------------------\n")

# -------------------------------
# Trading Agent Reasoning
#--------------------------------
final_weights = out["final_weights"].fillna(0.0)

agent = TradingAgent()
reasoning = agent.generate_reasoning(
        final_weights=final_weights.to_dict(),
        risk_controls=out["policy"].get("risk_controls", {}),
        notes="Moderate-high volatility, tech sector resilience",
        timestamp=nf_ts_parsed
)
print("✅ Generated TradingAgent reasoning:\n", reasoning)
# -------------------------------
# MongoDB save
# -------------------------------

try:
    collection = db["AlphaFusionNet_predictions"]
    doc = {
        "timestamp": nf_ts_parsed,
        "policy": out["policy"],
        "final_weights": final_weights.to_dict(),
        "risk_controls": out["policy"].get("risk_controls", {}),
        "reasoning": reasoning
    }

    collection.insert_one(doc)
    logger.info("AlphaFusionNet decision saved to MongoDB successfully.")
    print("✅ AlphaFusionNet decision saved to MongoDB successfully.")

except Exception as e:
    logger.error("Failed to save AlphaFusionNet decision to MongoDB: %s", e)
    print("❌ Failed to save AlphaFusionNet decision to MongoDB:", e)