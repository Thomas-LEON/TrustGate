# TrustGate 🛡️

> **Enterprise LLM Gateway & Explainable AI Middleware**  
> *Turn opaque LLM completions into resilient, auditable, mathematically decomposed, source-verified decisions.*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## What is TrustGate?

TrustGate is a high-performance **Enterprise LLM Gateway and Explainable AI Middleware** designed for production environments requiring strict auditability, low latency, and zero vendor lock-in.

It provides two core capabilities:
1. **Explainable AI Middleware (`/v1/evaluate`)**: Evaluates unstructured context against deterministic rubrics, verifies textual citations against source data using exact and fuzzy matching, and recalculates risk scores deterministically server-side.
2. **Resilient LLM Gateway (`/v1/chat/completions`)**: An OpenAI-compatible universal proxy featuring automated cascade fallback (Primary → Secondary → Tertiary), an in-memory semantic cache with cost tracking, and token bucket rate limiting across priority tiers.

```
                                    POST /v1/chat/completions
                                    POST /v1/evaluate
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │  Token Bucket Rate Limiter│
                               │  (CRITICAL/STANDARD/BATCH)│
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │   In-Memory Semantic Cache│ ──HIT──► [Zero-Cost Response]
                               │   (RapidFuzz + SHA-256)   │
                               └─────────────┬─────────────┘
                                             │ MISS
                                             ▼
                               ┌───────────────────────────┐
                               │   Multi-Provider Gateway  │
                               │   (LiteLLM Cascade Router)│
                               └─────────────┬─────────────┘
                                             │
                       ┌─────────────────────┼─────────────────────┐
                       ▼                     ▼                     ▼
               Tier 1: PRIMARY       Tier 2: SECONDARY     Tier 3: TERTIARY
             (gemini-3.8-flash)    (gemini-3.5-flash-lite) (gemini-3.1-flash-lite)
                       │                     │                     │
                       └─────────────────────┼─────────────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │ Grounding & Rubric Engine │ (for /v1/evaluate)
                               │  - RapidFuzz Citation Ver.│
                               │  - Server-Side Math Sum   │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │ Telemetry & SRE Cockpit   │
                               │  - Streamlit Dashboard    │
                               │  - FinOps Cost & P95/P99  │
                               │  - Optional Langfuse sync │
                               └───────────────────────────┘
```

---

## Core Capabilities

### 1. Resilient Cascade Routing
Abstracts model providers into logical tiers (`PRIMARY`, `SECONDARY`, `TERTIARY`). If the primary provider experiences a 429 rate limit, 503 overload, or transient network failure, TrustGate automatically falls back with exponential backoff and jitter.

### 2. Semantic In-Memory Cache (FinOps)
Intercepts requests and checks for identical or semantically similar prompts using RapidFuzz Levenshtein similarity. Cache hits return instantly with zero upstream API cost and a detailed FinOps audit header.

### 3. Tiered Token Bucket Rate Limiter
Enforces requests-per-minute (RPM) and tokens-per-minute (TPM) across priority tiers (`CRITICAL`, `STANDARD`, `BATCH`), preventing cascading failures under load.

### 4. Deterministic Rubric & Grounding Engine
The LLM evaluates atomic sub-criteria but **never computes the global score**. The final score is computed deterministically server-side. Every citation (`source_quote`) is verified against the input context. Unverified quotes are flagged `UNVERIFIED_CITATION`.

### 5. SRE & FinOps Cockpit
A technical Streamlit dashboard providing real-time P50/P95/P99 latency tracking, token economics, fallback health, and an interactive test console.

---

## API Reference

### 1. `POST /v1/evaluate` — Audited Risk Evaluation

**Request:**
```json
{
  "context": "A critical vulnerability CVE-2026-1104 in OpenSSL allows remote code execution without authentication. Exploitation scripts are actively circulating on public repositories. The impacted service runs inside an isolated VLAN with no direct internet access.",
  "task_type": "THREAT_SEVERITY",
  "config": {
    "sampling_runs": 1,
    "grounding_threshold": 0.85
  }
}
```

**Response:**
```json
{
  "decision": {
    "computed_score": 75,
    "scale": [0, 100],
    "severity_label": "HIGH",
    "confidence": {
      "level": "NOT_EVALUATED",
      "variance": null,
      "runs_evaluated": 1
    }
  },
  "rubric_breakdown": [
    {
      "criteria": "Technical Impact",
      "score": 35,
      "max_points": 40,
      "weight": 0.4,
      "justification": "Remote code execution allows complete compromise.",
      "source_quote": "remote code execution without authentication",
      "grounding": {
        "verified": true,
        "match_ratio": 1.0,
        "method": "EXACT"
      }
    },
    {
      "criteria": "Exploit Availability",
      "score": 30,
      "max_points": 30,
      "weight": 0.3,
      "justification": "Exploit code is public and actively circulating.",
      "source_quote": "Exploitation scripts are actively circulating on public repositories",
      "grounding": {
        "verified": true,
        "match_ratio": 1.0,
        "method": "EXACT"
      }
    },
    {
      "criteria": "Exposure & Scope",
      "score": 10,
      "max_points": 30,
      "weight": 0.3,
      "justification": "Isolated VLAN without direct internet access limits blast radius.",
      "source_quote": "isolated VLAN with no direct internet access",
      "grounding": {
        "verified": true,
        "match_ratio": 1.0,
        "method": "EXACT"
      }
    }
  ],
  "audit_metadata": {
    "model": "gemini/gemini-3.8-flash",
    "logical_tier": "PRIMARY",
    "fallback_occurred": false,
    "cached": false,
    "hallucination_flags": 0,
    "unverified_citations": 0,
    "latency_ms": 680,
    "total_tokens": 420,
    "cost_usd": 0.00018,
    "timestamp": "2026-10-14T18:00:00Z"
  }
}
```

---

### 2. `POST /v1/chat/completions` — OpenAI Compatible

Drop-in replacement for OpenAI SDK:

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="key-ctfr",  # Maps to CRITICAL priority tier
)

response = client.chat.completions.create(
    model="gemini/gemini-3.8-flash",
    messages=[{"role": "user", "content": "Explain zero-trust architecture in one sentence."}],
)
print(response.choices[0].message.content)
```

---

### 3. `GET /v1/gateway/stats` — Live FinOps Telemetry

Returns real-time P50/P95/P99 latencies, cache hit rates, net USD savings, and tier distributions.

---

## Repository Structure

```
TrustGate/
├── api/
│   ├── __init__.py
│   └── routes.py               # REST API endpoints & middleware routing
├── core/
│   ├── __init__.py
│   ├── cache.py                # In-memory semantic cache (RapidFuzz + LRU)
│   ├── config.py               # Central configuration, logical tiers, SSL handling
│   ├── gateway.py              # Resilient LiteLLM router with fallback cascade
│   ├── grounding.py            # Citation verification engine (exact + fuzzy)
│   ├── limiter.py              # Priority Token Bucket rate limiter
│   ├── observability.py        # Local JSONL telemetry & FinOps rolling buffer
│   └── rubric.py               # Scoring rubrics & deterministic computation
├── dashboard/
│   └── app.py                  # Streamlit SRE & FinOps Cockpit
├── providers/
│   ├── __init__.py
│   ├── base.py                 # Abstract LLM provider interface
│   └── gemini.py               # Direct Gemini provider (legacy / direct SDK)
├── schemas/
│   ├── __init__.py
│   ├── chat.py                 # OpenAI-compatible request/response schemas
│   ├── request.py              # Evaluation input schemas
│   └── response.py             # Audited output schemas & decision metadata
├── tests/
│   ├── test_api.py             # End-to-end API integration & rate limiting tests
│   ├── test_cache.py           # Exact & semantic cache tests
│   ├── test_core.py            # Rubric & grounding engine unit tests
│   ├── test_gateway.py         # Multi-provider cascade & fallback tests
│   └── test_limiter.py         # Token bucket priority tests
├── data/                       # Local telemetry storage (gitignored)
├── Dockerfile                  # Container definition
├── main.py                     # FastAPI application entrypoint
├── pytest.ini                  # Pytest configuration
├── requirements.txt            # Python dependencies
├── .env.example                # Example configuration template
├── .gitignore                  # Git ignore rules
├── LICENSE                     # MIT License
└── README.md
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **API Framework** | FastAPI + Uvicorn | High-throughput async ASGI server |
| **LLM Gateway** | LiteLLM | Multi-provider async routing & cost mapping |
| **Validation** | Pydantic v2 | Strict schema enforcement & serialization |
| **Matching Engine** | RapidFuzz | C++ Levenshtein string matching (< 1ms) |
| **Cockpit** | Streamlit + Pandas | Real-time quantitative SRE & FinOps telemetry |
| **Tracing** | Langfuse (optional) | Cloud-native LLM observability & tracing |

---

## Quickstart

### Prerequisites
- Python 3.10+
- A Google Gemini API key ([get one here](https://aistudio.google.com/apikey))

### Installation

```bash
git clone https://github.com/Thomas-LEON/TrustGate.git
cd TrustGate
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
# Edit .env and set your GEMINI_API_KEY
```

### Run the API Server

```bash
uvicorn main:app --reload --port 8000
```
Interactive OpenAPI documentation: [http://localhost:8000/docs](http://localhost:8000/docs)

### Run the Streamlit SRE Cockpit

```bash
streamlit run dashboard/app.py
```
Dashboard interface: [http://localhost:8501](http://localhost:8501)

### Run Tests

```bash
pytest -v
```

---

## License

This project is licensed under the [MIT License](LICENSE).
