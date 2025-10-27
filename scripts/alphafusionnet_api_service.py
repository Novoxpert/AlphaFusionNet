#!/usr/bin/env python3
"""
alphafusionnet_api_service.py
--------------
FastAPI service to serve latest AlphaFusionNet portfolio predictions from MongoDB.
Author: Elham Esmaeilnia(elham.e.shirvani@gmail.com)
Date: 2025 Oct 25
Version: 1.0.0
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from bson import json_util
from datetime import datetime, timezone
import json
import uvicorn

# -------------------- App & CORS --------------------
app = FastAPI(title="AlphaFusionNet API")

origins = [
    "http://localhost:3000",  # React dev server
    # Add production frontend origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- MongoDB --------------------
client = MongoClient("mongodb://127.0.0.1:27017/")
db = client["db_portfolio"]
collection = db["AlphaFusionNet_predictions"]

# -------------------- Helpers --------------------
def serialize_doc(doc):
    """Convert MongoDB BSON doc to JSON-serializable dict"""
    if not doc:
        return None
    safe_json = json.loads(json_util.dumps(doc))
    
    # Flatten $date and $oid
    def flatten(obj):
        if isinstance(obj, dict):
            if "$oid" in obj:
                return obj["$oid"]
            if "$date" in obj:
                return obj["$date"]
            return {k: flatten(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [flatten(i) for i in obj]
        return obj
    
    return flatten(safe_json)

# -------------------- Endpoints --------------------
@app.get("/latest_alphafusionnet")
def latest_alphafusionnet():
    """Return the most recent AlphaFusionNet prediction"""
    doc = collection.find_one(sort=[("timestamp", -1)])
    if not doc:
        return JSONResponse(content={"error": "No prediction found"}, status_code=404)
    return JSONResponse(content=serialize_doc(doc))

@app.get("/alphafusionnet_history")
def alphafusionnet_history(
    start: str = Query(None, description="Start ISO timestamp, e.g. 2025-10-01T00:00:00"),
    end: str = Query(None, description="End ISO timestamp"),
    limit: int = 100
):
    """Return historical AlphaFusionNet predictions within a time range"""
    query = {}
    if start:
        query["timestamp"] = {"$gte": datetime.fromisoformat(start)}
    if end:
        query.setdefault("timestamp", {})["$lte"] = datetime.fromisoformat(end)
    
    cursor = collection.find(query).sort("timestamp", 1).limit(limit)
    docs = [serialize_doc(d) for d in cursor]
    return JSONResponse(content=docs)

@app.get("/health")
def health_check():
    """Simple health check endpoint"""
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}

# -------------------- Run --------------------
if __name__ == "__main__":
    uvicorn.run("scripts.alphafusionnet_api_service:app", host="0.0.0.0", port=8003, reload=True)
