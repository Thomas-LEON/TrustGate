"""
TrustGate — Enterprise LLM Gateway & Explainable AI Middleware.
"""
import os
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from api.routes import router

app = FastAPI(
    title="TrustGate — Enterprise LLM Gateway",
    description=(
        "Resilient, observable, and cost-optimized Enterprise LLM Gateway. "
        "Features multi-provider fallback, deterministic semantic caching, token bucket rate limiting, "
        "and explainable risk scoring with citation grounding."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
