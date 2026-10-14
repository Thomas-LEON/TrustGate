"""
TrustGate — Technical SRE & FinOps Cockpit (Streamlit Dashboard).

Features:
- Dual-process synchronization: reads live telemetry from FastAPI server & JSONL log file.
- Clean, responsive metrics layout (no truncation of model names).
- Persistent session state: updates top KPIs, transaction logs, and SRE charts instantly.
"""
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import httpx

from core.config import settings
from core.observability import observability
from core.cache import semantic_cache

st.set_page_config(
    page_title="TrustGate — SRE & FinOps Cockpit",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for compact, high-density technical cards
st.markdown("""
<style>
    .metric-card {
        background-color: #0e1117;
        border: 1px solid #262730;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .metric-val {
        font-size: 1.5rem;
        font-weight: 700;
        color: #00bcd4;
    }
    .metric-lbl {
        font-size: 0.8rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .pill-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .pill-cache { background-color: #238636; color: white; }
    .pill-live { background-color: #1f6feb; color: white; }
    .pill-warn { background-color: #d29922; color: black; }
    .pill-crit { background-color: #da3633; color: white; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Telemetry Loader: Reads from Server + Local JSONL log (Zero Loss)
# ---------------------------------------------------------------------------
def load_all_events() -> list[dict]:
    """Load all transaction events from JSONL log file + in-memory buffer."""
    events = []
    log_file = settings.metrics_log_path
    if log_file.exists():
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            events.append(json.loads(line))
                        except Exception:
                            pass
        except Exception:
            pass

    # Merge with memory buffer if needed
    mem_events = observability.get_recent_transactions(limit=1000)
    seen_ids = {e.get("request_id") for e in events if "request_id" in e}
    for me in mem_events:
        if me.get("request_id") not in seen_ids:
            events.append(me)

    return events


def compute_live_stats(events: list[dict]) -> dict:
    """Compute aggregated SRE & FinOps statistics from events."""
    if not events:
        return {
            "total_requests": 0,
            "cache_hit_rate": 0.0,
            "total_tokens_spent": 0,
            "total_tokens_saved": 0,
            "total_cost_spent_usd": 0.0,
            "total_cost_saved_usd": 0.0,
            "p50_latency_ms": 0,
            "p95_latency_ms": 0,
            "p99_latency_ms": 0,
            "fallback_count": 0,
            "hallucination_rate": 0.0,
            "tier_distribution": {},
            "model_distribution": {},
        }

    total = len(events)
    cached_count = sum(1 for e in events if e.get("cached", False))
    fallbacks = sum(1 for e in events if e.get("fallback_occurred", False))
    hallucinations = sum(1 for e in events if e.get("hallucination_detected", False))

    tokens_spent = sum(e.get("total_tokens", 0) for e in events if not e.get("cached", False))
    tokens_saved = sum(e.get("total_tokens", 0) for e in events if e.get("cached", False))

    cost_spent = sum(e.get("cost_usd", 0.0) for e in events if not e.get("cached", False))
    cost_saved = sum(e.get("cost_saved_usd", 0.0) for e in events if e.get("cached", False))

    latencies = sorted(e.get("latency_ms", 0) for e in events)
    p50 = latencies[int(total * 0.50)] if latencies else 0
    p95 = latencies[min(total - 1, int(total * 0.95))] if latencies else 0
    p99 = latencies[min(total - 1, int(total * 0.99))] if latencies else 0

    tier_dist = {}
    model_dist = {}
    for e in events:
        t = e.get("client_tier", "STANDARD")
        tier_dist[t] = tier_dist.get(t, 0) + 1
        m = e.get("model_used", "none")
        if m != "none":
            model_dist[m] = model_dist.get(m, 0) + 1

    return {
        "total_requests": total,
        "cache_hit_rate": round((cached_count / total) * 100, 2),
        "total_tokens_spent": tokens_spent,
        "total_tokens_saved": tokens_saved,
        "total_cost_spent_usd": round(cost_spent, 6),
        "total_cost_saved_usd": round(cost_saved, 6),
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "p99_latency_ms": p99,
        "fallback_count": fallbacks,
        "hallucination_rate": round((hallucinations / total) * 100, 2),
        "tier_distribution": tier_dist,
        "model_distribution": model_dist,
    }


# Load live data
all_events = load_all_events()
stats = compute_live_stats(all_events)

# ---------------------------------------------------------------------------
# Sidebar: System Health & Configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🛡️ TrustGate Core")
    st.caption("Enterprise LLM Gateway v1.0.0")

    st.markdown("### 🔌 Active Model Cascade")
    st.markdown(f"**Primary (P0):** `{settings.model_primary}`")
    st.markdown(f"**Secondary (P1):** `{settings.model_secondary}`")
    st.markdown(f"**Tertiary (P2):** `{settings.model_tertiary}`")

    st.markdown("---")
    st.markdown("### ⚡ Semantic Cache Engine")
    st.markdown(f"**Status:** {'🟢 Active' if settings.cache_enabled else '🔴 Disabled'}")
    st.markdown(f"**Threshold:** `{settings.cache_similarity_threshold * 100:.0f}%` similarity")
    st.markdown(f"**In-Memory Entries:** `{len(semantic_cache._store)}`")

    st.markdown("---")
    st.markdown("### ⚖️ Priority Tiers (Token Bucket)")
    for name, tier in settings.tiers.items():
        st.markdown(f"• **{name}**: `{tier.rpm}` RPM | `{tier.tpm:,}` TPM")

    st.markdown("---")
    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
    with c_btn2:
        if st.button("🧹 Clear Cache", use_container_width=True):
            semantic_cache.clear()
            st.success("Cache cleared!")
            st.rerun()

# ---------------------------------------------------------------------------
# Main Dashboard Header & Aggregated KPIs (Always in Sync)
# ---------------------------------------------------------------------------
st.title("📊 LLM Gateway Telemetry & FinOps Cockpit")
st.caption("Quantitative real-time telemetry: execution latency, token economics, fallback health, and grounding audit.")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-lbl">Total Queries</div>
        <div class="metric-val">{:,}</div>
    </div>
    """.format(stats["total_requests"]), unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-lbl">Cache Hit Rate</div>
        <div class="metric-val" style="color: #4caf50;">{:.1f}%</div>
    </div>
    """.format(stats["cache_hit_rate"]), unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-lbl">Net USD Saved</div>
        <div class="metric-val" style="color: #4caf50;">${:.5f}</div>
    </div>
    """.format(stats["total_cost_saved_usd"]), unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-lbl">Tokens Saved (Free)</div>
        <div class="metric-val">{:,}</div>
    </div>
    """.format(stats["total_tokens_saved"]), unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-lbl">Latency P95 / P99</div>
        <div class="metric-val" style="font-size: 1.25rem;">{}ms / {}ms</div>
    </div>
    """.format(stats["p95_latency_ms"], stats["p99_latency_ms"]), unsafe_allow_html=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Tabs: 1. Interactive Console, 2. Transactions Log, 3. Deep SRE Analytics
# ---------------------------------------------------------------------------
tab_test, tab_logs, tab_analytics = st.tabs([
    "🧪 Live Gateway Tester",
    "📋 Real-Time Transaction Logs",
    "📈 SRE & FinOps Analytics",
])

# ---------------------------------------------------------------------------
# TAB 1: Live Gateway Tester (Refined, perfectly readable layout)
# ---------------------------------------------------------------------------
with tab_test:
    st.subheader("Test Live Inference, Fallback & Cache Interception")
    st.write("Submit any security report or risk description to test the gateway pipeline in real time.")

    c1, c2 = st.columns([3, 1])
    with c1:
        test_context = st.text_area(
            "Incident / Context to evaluate:",
            value="A critical vulnerability CVE-2026-1104 in OpenSSL allows remote code execution without authentication. Exploitation scripts are actively circulating on public repositories. The impacted service runs inside an isolated VLAN with no direct internet access.",
            height=100,
        )
    with c2:
        test_tier_key = st.selectbox(
            "Client Tier / Key:",
            options=["key-ctfr (CRITICAL)", "key-risk (STANDARD)", "key-front (BATCH)"],
            index=0,
        )
        task_type_choice = st.selectbox(
            "Task Type Rubric:",
            options=["THREAT_SEVERITY", "CREDIT_RISK", "COMPLIANCE_CHECK"],
            index=0,
        )

    if st.button("🚀 Send to Gateway (/v1/evaluate)", type="primary", use_container_width=True):
        api_key = test_tier_key.split()[0]
        payload = {
            "context": test_context,
            "task_type": task_type_choice,
            "config": {"sampling_runs": 1, "grounding_threshold": 0.85},
        }

        with st.spinner("Processing through TrustGate Gateway..."):
            t_start = time.perf_counter()
            resp = None
            try:
                # 1. Try calling live server via HTTP
                try:
                    with httpx.Client(base_url="http://127.0.0.1:8000", timeout=45.0) as http_client:
                        resp = http_client.post(
                            "/v1/evaluate",
                            json=payload,
                            headers={"X-API-Key": api_key},
                        )
                except Exception:
                    # 2. Fallback to in-process TestClient if standalone server is offline
                    from fastapi.testclient import TestClient
                    from main import app
                    client = TestClient(app)
                    resp = client.post(
                        "/v1/evaluate",
                        json=payload,
                        headers={"X-API-Key": api_key},
                    )

                if resp and resp.status_code == 200:
                    data = resp.json()
                    st.session_state["latest_result"] = data
                    # Auto-rerun so top KPIs and logs update immediately!
                    st.rerun()

                elif resp and resp.status_code == 429:
                    st.error(f"🛑 **HTTP 429 Rate Limit Exceeded (Token Bucket)**: {resp.json().get('detail')}")
                    st.info(f"Retry-After: {resp.headers.get('Retry-After')} seconds.")
                else:
                    err_msg = resp.text if resp else "No response"
                    st.error(f"Execution Error: {err_msg}")

            except Exception as e:
                st.error(f"Execution failed: {e}")

    # Display Latest Result if present in session state
    if "latest_result" in st.session_state:
        data = st.session_state["latest_result"]
        audit = data.get("audit_metadata", {})
        decision = data.get("decision", {})
        breakdown = data.get("rubric_breakdown", [])

        is_cached = audit.get("cached", False)
        fallback = audit.get("fallback_occurred", False)

        st.markdown("---")
        st.markdown("### 🎯 Latest Evaluation Results")

        # Row 1: High-Level Badges & Status
        b1, b2, b3 = st.columns([1.5, 2, 1.5])
        with b1:
            sev = decision.get("severity_label", "UNKNOWN")
            score = decision.get("computed_score", 0)
            st.metric("Deterministic Risk Score", f"{score} / 100", delta=sev, delta_color="inverse")

        with b2:
            st.markdown("**Engine & Model Route:**")
            model_name = audit.get("model", "unknown")
            tier_name = audit.get("logical_tier", "PRIMARY")

            if is_cached:
                st.markdown(f"<span class='pill-badge pill-cache'>⚡ CACHED (< 5ms)</span> Model Origin: ` {model_name} `", unsafe_allow_html=True)
            elif fallback:
                st.markdown(f"<span class='pill-badge pill-warn'>⚠️ FALLBACK ACTIVATED</span> Route: ` {model_name} ` ({tier_name})", unsafe_allow_html=True)
            else:
                st.markdown(f"<span class='pill-badge pill-live'>🌐 PRIMARY ROUTE</span> Route: ` {model_name} ` ({tier_name})", unsafe_allow_html=True)

        with b3:
            st.markdown("**Execution Cost & Latency:**")
            lat = audit.get("latency_ms", 0)
            cost = audit.get("cost_usd", 0.0)
            if is_cached:
                st.markdown(f"**Cost:** `$0.000000` (Free)\n\n**Latency:** `{lat} ms`")
            else:
                st.markdown(f"**Cost:** `${cost:.6f}`\n\n**Latency:** `{lat} ms`")

        # Row 2: Token Economics
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Total Tokens", audit.get("total_tokens", 0))
        t2.metric("Hallucination Flags", audit.get("hallucination_flags", 0))
        t3.metric("Client Key", api_key if "api_key" in locals() else "Active")
        t4.metric("Audited Citations", len(breakdown))

        # Rubric & Grounding Table
        st.markdown("#### 🔬 Decomposed Rubric Breakdown & Anti-Hallucination Citations")
        b_rows = []
        for b in breakdown:
            g = b.get("grounding", {})
            b_rows.append({
                "Criterion": b.get("criteria", "Unknown"),
                "Score": f"{b.get('score', 0)} / {b.get('max_points', 0)}",
                "Weight": f"{b.get('weight', 0) * 100:.0f}%",
                "Justification": b.get("justification", ""),
                "Source Quote Cited": b.get("source_quote", ""),
                "Verified?": "✅ TRUE" if g.get("verified", False) else "❌ HALLUCINATED",
                "Match Ratio": f"{g.get('match_ratio', 0) * 100:.1f}%",
                "Method": g.get("method", "NONE"),
            })
        st.dataframe(pd.DataFrame(b_rows), use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 2: Real-Time Transaction Logs
# ---------------------------------------------------------------------------
with tab_logs:
    st.subheader(f"All Gateway Transaction Events ({len(all_events)} recorded)")
    st.caption("Full audit trail containing every single metric: latency, model, tokens, costs, and citations.")

    if all_events:
        df = pd.DataFrame(list(reversed(all_events)))
        desired_cols = [
            "timestamp", "endpoint", "client_tier", "cached", "cache_type",
            "model_used", "logical_tier", "fallback_occurred",
            "prompt_tokens", "completion_tokens", "total_tokens",
            "cost_usd", "cost_saved_usd", "latency_ms", "status_code", "unverified_citations"
        ]
        available_cols = [c for c in desired_cols if c in df.columns]
        st.dataframe(df[available_cols], use_container_width=True)
    else:
        st.info("No transaction events recorded yet. Submit a test request in Tab 1 to populate.")

# ---------------------------------------------------------------------------
# TAB 3: Deep SRE Analytics
# ---------------------------------------------------------------------------
with tab_analytics:
    st.subheader("Deep SRE & FinOps Telemetry")

    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown("#### ⏱️ Latency Percentiles (ms)")
        st.table(pd.DataFrame({
            "Metric": ["P50 (Median)", "P95", "P99", "Mean Latency"],
            "Value": [
                f"{stats['p50_latency_ms']} ms",
                f"{stats['p95_latency_ms']} ms",
                f"{stats['p99_latency_ms']} ms",
                f"{int(pd.DataFrame(all_events)['latency_ms'].mean()) if all_events else 0} ms",
            ]
        }))

        st.markdown("#### 🏢 Consumption by Priority Tier")
        if stats["tier_distribution"]:
            st.bar_chart(pd.Series(stats["tier_distribution"]))
        else:
            st.write("Awaiting tier traffic...")

    with c_right:
        st.markdown("#### 🤖 Model Invocations & Fallbacks")
        st.write(f"**Total Fallbacks Triggered:** `{stats['fallback_count']}`")
        st.write(f"**Hallucination Rate:** `{stats['hallucination_rate']}%`")

        if stats["model_distribution"]:
            st.bar_chart(pd.Series(stats["model_distribution"]))
        else:
            st.write("Awaiting model invocations...")
