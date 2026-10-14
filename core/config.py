"""
TrustGate — Central Configuration & Enterprise Settings.

Handles:
- Logical model role mappings (PRIMARY, SECONDARY, TERTIARY).
- Corporate SSL proxy handling.
- Tier definitions for Rate Limiting (Token Bucket).
- Semantic Cache & Observability parameters.
"""
from __future__ import annotations

import os
import ssl
import warnings
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Ensure .env is loaded
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# ---------------------------------------------------------------------------
# Corporate SSL proxy & LiteLLM environment optimizations
# ---------------------------------------------------------------------------
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

# SSL verification bypass — ONLY active when SSL_VERIFY=false in environment.
# Required for corporate proxy environments with self-signed certificates.
# WARNING: Do NOT set SSL_VERIFY=false in production unless behind a
# trusted corporate proxy — this disables certificate validation globally.
_ssl_verify = os.environ.get("SSL_VERIFY", "true").lower() not in ("false", "0", "no")

if not _ssl_verify:
    try:
        import httpx
        _orig_client_init = httpx.Client.__init__
        def _patched_client_init(self, *args, **kwargs):
            kwargs["verify"] = False
            _orig_client_init(self, *args, **kwargs)
        httpx.Client.__init__ = _patched_client_init

        _orig_async_init = httpx.AsyncClient.__init__
        def _patched_async_init(self, *args, **kwargs):
            kwargs["verify"] = False
            _orig_async_init(self, *args, **kwargs)
        httpx.AsyncClient.__init__ = _patched_async_init

        import aiohttp
        _orig_tcp_init = aiohttp.TCPConnector.__init__
        def _patched_tcp_init(self, *args, **kwargs):
            kwargs["ssl"] = False
            _orig_tcp_init(self, *args, **kwargs)
        aiohttp.TCPConnector.__init__ = _patched_tcp_init

        import litellm
        litellm.ssl_verify = False

        ssl._create_default_https_context = ssl._create_unverified_context
        warnings.filterwarnings("ignore")
    except Exception:
        pass

# Synchronize GEMINI_API_KEY with GOOGLE_API_KEY for LiteLLM compatibility
gemini_key = os.environ.get("GEMINI_API_KEY", "")
if gemini_key:
    os.environ.setdefault("GOOGLE_API_KEY", gemini_key)


class TierConfig(BaseModel):
    """Rate limit configuration for a client priority tier."""
    name: str
    rpm: int = Field(..., description="Requests per minute (bucket capacity)")
    tpm: int = Field(..., description="Tokens per minute (bucket capacity)")
    burst_multiplier: float = Field(default=1.5, description="Allowed burst capacity multiplier")


class Settings(BaseModel):
    """Central configuration for TrustGate Enterprise Gateway."""

    # 1. Logical Model Tiers (Decoupled from specific vendors)
    # Default cascade using the user's Gemini 2026 models:
    model_primary: str = Field(
        default=os.environ.get("MODEL_PRIMARY", "gemini/gemini-3.8-flash"),
        description="Logical primary model (tier 1).",
    )
    model_secondary: str = Field(
        default=os.environ.get("MODEL_SECONDARY", "gemini/gemini-3.5-flash-lite"),
        description="Logical secondary model (fallback 1).",
    )
    model_tertiary: str = Field(
        default=os.environ.get("MODEL_TERTIARY", "gemini/gemini-3.1-flash-lite"),
        description="Logical tertiary model (fallback 2 / safety net).",
    )

    # 2. Semantic Cache Settings
    cache_enabled: bool = Field(
        default=os.environ.get("CACHE_ENABLED", "true").lower() in ("true", "1", "yes"),
        description="Enable or disable semantic in-memory cache.",
    )
    cache_similarity_threshold: float = Field(
        default=float(os.environ.get("CACHE_SIMILARITY_THRESHOLD", "0.90")),
        ge=0.5,
        le=1.0,
        description="Cosine / fuzzy similarity threshold for semantic cache HIT.",
    )
    cache_max_entries: int = Field(
        default=int(os.environ.get("CACHE_MAX_ENTRIES", "1000")),
        description="Maximum entries stored in memory before LRU eviction.",
    )
    cache_ttl_seconds: int = Field(
        default=int(os.environ.get("CACHE_TTL_SECONDS", "86400")),
        description="Cache entry time-to-live in seconds (default: 24h).",
    )

    # 3. Rate Limiting Priority Tiers (Reusable across any company)
    tiers: dict[str, TierConfig] = Field(
        default={
            "CRITICAL": TierConfig(name="CRITICAL", rpm=120, tpm=300000, burst_multiplier=1.5),
            "STANDARD": TierConfig(name="STANDARD", rpm=60, tpm=100000, burst_multiplier=1.25),
            "BATCH": TierConfig(name="BATCH", rpm=20, tpm=30000, burst_multiplier=1.0),
        }
    )

    # Mapping of API keys / Client tokens to Tiers
    key_tier_mapping: dict[str, str] = Field(
        default={
            "key-critical-prod": "CRITICAL",
            "key-standard-prod": "STANDARD",
            "key-batch-dev": "BATCH",
            "key-ctfr": "CRITICAL",
            "key-risk": "STANDARD",
            "key-front": "BATCH",
        }
    )

    # 4. Observability & FinOps
    langfuse_public_key: str = Field(default=os.environ.get("LANGFUSE_PUBLIC_KEY", ""))
    langfuse_secret_key: str = Field(default=os.environ.get("LANGFUSE_SECRET_KEY", ""))
    langfuse_host: str = Field(default=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"))
    metrics_log_path: Path = Field(default=BASE_DIR / "data" / "gateway_metrics.jsonl")


# Singleton instance
settings = Settings()
