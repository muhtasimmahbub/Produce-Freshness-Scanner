from contextlib import asynccontextmanager
from typing import Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.predict import load_model, predict_freshness

# Dict state initialized using built-in dict[str, Any] (PEP 585)
state: dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager managing server lifecycle.
    Loads ML artifacts on startup and cleans state on shutdown.
    """
    try:
        state["model"] = load_model()
        print("🟢 Model loaded successfully into memory.")
    except Exception as e:
        print(f"🔴 Failed to load model artifact: {e}")
        state["model"] = None
    
    yield  # Application handles incoming HTTP traffic
    
    state.clear()
    print("🧹 App state cleared on shutdown.")

app = FastAPI(
    title="Produce Freshness Scanner API",
    description="REST API for predicting produce freshness based on color hue and texture smoothness.",
    version="0.1.0",
    lifespan=lifespan,
)

class FreshnessRequest(BaseModel):
    color: float = Field(..., json_schema_extra={"example": 8.2}, description="Color hue value (e.g., 8.0 for fresh, 2.0 for rotten)")
    texture: float = Field(..., json_schema_extra={"example": 7.1}, description="Texture smoothness value (e.g., 7.5 for smooth, 1.8 for bruised)")

class FreshnessResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "FRESH"})
    confidence: float = Field(..., json_schema_extra={"example": 100.0})

@app.get("/")
def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "service": "Produce Freshness Scanner API"}

@app.post("/predict", response_model=FreshnessResponse)
def predict(payload: FreshnessRequest):
    """
    Accept color and texture metrics, run model inference, and return status + confidence score.
    """
    model = state.get("model")
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded or failed to initialize.")
    
    status, confidence = predict_freshness(payload.color, payload.texture, model)
    return FreshnessResponse(status=status, confidence=round(confidence * 100.0, 2))