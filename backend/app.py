"""FastAPI backend: /predict (model) and /chat (agent)."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import agent
import model as model_module

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
    try:
        result = model_module.predict_churn(req.customer_id)
    except model_module.CustomerNotFound:
        raise HTTPException(status_code=404, detail=f"customer {req.customer_id} not found")
    return PredictResponse(
        churn_probability=result["churn_probability"], risk_level=result["risk_level"]
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    result = agent.run_chat(req.customer_id, req.message)
    return ChatResponse(**result)
