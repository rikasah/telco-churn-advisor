"""FastAPI backend: /predict (model) and /chat (agent)."""
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import agent
import explain
import logging_store
import model as model_module
import system_status

app = FastAPI(title="Telco Churn Advisor API")


class PredictRequest(BaseModel):
    customer_id: str


class PredictResponse(BaseModel):
    churn_probability: float
    risk_level: str


class ChatRequest(BaseModel):
    customer_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    sources: list[str]
    tool_calls: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    start = time.perf_counter()
    try:
        result = model_module.predict_churn(req.customer_id)
    except model_module.CustomerNotFound:
        raise HTTPException(status_code=404, detail=f"customer {req.customer_id} not found")
    duration_ms = (time.perf_counter() - start) * 1000

    try:
        logging_store.log_predict(
            req.customer_id, result["churn_probability"], result["risk_level"], duration_ms
        )
    except Exception:
        pass  # logging is best-effort, never break the actual response

    return PredictResponse(
        churn_probability=result["churn_probability"], risk_level=result["risk_level"]
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    start = time.perf_counter()
    result = agent.run_chat(req.customer_id, req.message)
    duration_ms = (time.perf_counter() - start) * 1000

    try:
        logging_store.log_chat(req.customer_id, result["tool_calls"], duration_ms)
    except Exception:
        pass

    return ChatResponse(**result)


@app.get("/stats")
def stats():
    return logging_store.get_stats()


@app.get("/system-status")
def system_status_endpoint():
    return {
        "data": system_status.get_data_status(),
        "champion": system_status.get_champion_info(),
    }


@app.get("/model-trend")
def model_trend():
    return system_status.get_model_trend()


@app.get("/top-risk")
def top_risk(n: int = 10):
    return system_status.get_top_risk_customers(n)


@app.get("/risk-summary")
def risk_summary():
    return system_status.get_risk_summary()


@app.post("/explain")
def explain_endpoint(req: PredictRequest):
    try:
        return explain.explain_churn(req.customer_id)
    except model_module.CustomerNotFound:
        raise HTTPException(status_code=404, detail=f"customer {req.customer_id} not found")
