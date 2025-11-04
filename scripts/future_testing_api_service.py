#!/usr/bin/env python3
"""
future_testing_api_service.py
AlphaFusionNet Future Testing API Service

This FastAPI service exposes endpoints to retrieve results produced by the
future-testing engine of AlphaFusionNet. Future testing reconstructs model
inputs after predictions are made, enabling real-time validation, drift
detection, and auditability of model behavior in production environments.

Key Features
-----------
• Serves the latest future-testing record stored in MongoDB  
• Returns reconstructed model features, predicted weights, and timestamps  
• Supports model monitoring, compliance, and post-prediction evaluation workflows  

Endpoints
---------
GET /future-testing/latest  
    Returns the most recent reconstructed feature snapshot and model weights.

Author: Elham Esmaeilnia(elham.e.shirvani@gmail.com)
Date: 2025 Nov 03
Version: 1.0.0
"""

from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
import pandas as pd
import uvicorn, os
from dotenv import load_dotenv
# --------------------------- MongoDB setup ---------------------------
load_dotenv()
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
mongo_db = client[mongo_db_name]
future_testing_col = mongo_db["AlphaFusionNet_future_testing"]

app = FastAPI(title="AlphaFusionNet Future Testing API")


@app.get("/future-testing/latest")
def get_latest_future_testing():
    """
    Fetch the latest saved future testing results from MongoDB.
    Returns:
        {
            "timestamp": <prediction timestamp>,
            "features": [...],
            "weights": {...},
            "created_at": <timestamp>
        }
    """
    doc = future_testing_col.find_one(sort=[("timestamp", -1)])
    if doc is None:
        raise HTTPException(status_code=404, detail="No future testing data found.")

    # Optional: convert features list back to DataFrame for API consumption
    features = pd.DataFrame(doc["features"]).to_dict(orient="records")

    return {
        "timestamp": str(doc["timestamp"]),
        "features": features,
        "weights": doc["weights"],
        "created_at": str(doc["created_at"])
    }
# -------------------- Run --------------------
if __name__ == "__main__":
    uvicorn.run("scripts.future_testing_api_service:app", host="0.0.0.0", port=8005, reload=True)