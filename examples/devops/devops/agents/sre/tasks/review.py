"""Ops review — weekly service health analysis with a single LLM call.

Demonstrates the core Switchplane thesis: use deterministic code for data
processing and statistical analysis, reserve LLM calls for the single step
that genuinely requires judgment — interpreting the results.

Graph:

    fetch_metrics ──→ analyze ──→ summarize ──→ compile_report
    (deterministic)   (deterministic)  (LLM)      (deterministic)

LLM calls: 1
Deterministic nodes: 3
"""

import datetime
import json
import re
from typing import Any, TypedDict

import numpy as np
import pandas as pd
from langgraph.graph import END, START, StateGraph

from switchplane import Task
from switchplane.agent_runtime import AgentContext
from switchplane.llm import build_llm

# -- Mock data generation ------------------------------------------------------
#
# In production this would be a NewRelic API call. Here we generate realistic
# synthetic data so the example runs without credentials. The mock includes
# injected anomalies so the analysis has something meaningful to find.

ENDPOINTS = ["/api/users", "/api/orders", "/api/payments"]

_BASELINE_RPM = {
    "/api/users": 1200,
    "/api/orders": 450,
    "/api/payments": 180,
}

_LATENCY_MS = {
    "/api/users": {"p50": 45, "p95": 120, "p99": 250},
    "/api/orders": {"p50": 85, "p95": 200, "p99": 450},
    "/api/payments": {"p50": 150, "p95": 350, "p99": 800},
}

_STATUS_WEIGHTS = {
    "/api/users": {200: 0.920, 201: 0.040, 400: 0.025, 404: 0.010, 500: 0.003, 503: 0.002},
    "/api/orders": {200: 0.880, 201: 0.060, 400: 0.035, 404: 0.015, 500: 0.005, 503: 0.005},
    "/api/payments": {200: 0.900, 201: 0.050, 400: 0.030, 404: 0.005, 500: 0.008, 503: 0.007},
}


def _diurnal_factor(hour: int) -> float:
    """Simulate real traffic: low overnight, peak during business hours."""
    if 2 <= hour < 6:
        return 0.15
    if 6 <= hour < 9:
        return 0.5 + 0.15 * (hour - 6)
    if 9 <= hour < 17:
        return 1.0
    if 17 <= hour < 21:
        return 0.7 - 0.1 * (hour - 17)
    return 0.25


def _generate_week(
    start: datetime.datetime,
    rng: np.random.Generator,
    anomalies: dict[tuple[int, int], dict] | None = None,
) -> pd.DataFrame:
    """Generate one week of synthetic metrics at 5-minute granularity."""
    rows: list[dict[str, Any]] = []
    interval = datetime.timedelta(minutes=5)

    for i in range(7 * 24 * 12):
        ts = start + i * interval
        hour = ts.hour
        dow = ts.weekday()
        factor = _diurnal_factor(hour)

        for endpoint in ENDPOINTS:
            base = int(_BASELINE_RPM[endpoint] * factor)
            volume = max(1, int(rng.normal(base, base * 0.08)))

            weights = _STATUS_WEIGHTS[endpoint].copy()
            lat_mult = 1.0

            if anomalies and (dow, hour) in anomalies:
                anom = anomalies[(dow, hour)]
                if anom.get("endpoint") in (endpoint, None):
                    if "5xx_mult" in anom:
                        weights[500] *= anom["5xx_mult"]
                        weights[503] *= anom["5xx_mult"]
                    total = sum(weights.values())
                    weights = {k: v / total for k, v in weights.items()}
                    lat_mult = anom.get("lat_mult", 1.0)

            for code, weight in weights.items():
                count = max(0, int(rng.poisson(volume * weight)))
                if count == 0:
                    continue

                lat = _LATENCY_MS[endpoint]
                rows.append(
                    {
                        "timestamp": ts,
                        "endpoint": endpoint,
                        "status_code": code,
                        "count": count,
                        "p50_ms": round(max(1.0, rng.normal(lat["p50"] * lat_mult, lat["p50"] * 0.10)), 1),
                        "p95_ms": round(max(1.0, rng.normal(lat["p95"] * lat_mult, lat["p95"] * 0.12)), 1),
                        "p99_ms": round(max(1.0, rng.normal(lat["p99"] * lat_mult, lat["p99"] * 0.15)), 1),
                    }
                )

    return pd.DataFrame(rows)


def generate_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate current + previous week. Current week has injected anomalies."""
    rng = np.random.default_rng(42)

    now = datetime.datetime.now(tz=datetime.UTC)
    monday = now - datetime.timedelta(days=now.weekday(), hours=now.hour, minutes=now.minute, seconds=now.second)
    prev_monday = monday - datetime.timedelta(weeks=1)

    previous = _generate_week(prev_monday, rng)

    anomalies: dict[tuple[int, int], dict] = {}
    for h in range(14, 17):  # Wednesday 14:00–16:59 UTC
        anomalies[(2, h)] = {"endpoint": "/api/payments", "5xx_mult": 15, "lat_mult": 2.5}
    for h in range(10, 12):  # Thursday 10:00–11:59 UTC
        anomalies[(3, h)] = {"endpoint": "/api/orders", "lat_mult": 4.0}

    current = _generate_week(monday, rng, anomalies=anomalies)
    return current, previous


# -- Analysis (deterministic — pandas, zero LLM calls) ------------------------


def _spike_windows(df: pd.DataFrame, group_col: str, z_threshold: float = 2.5) -> list[dict]:
    """Detect hourly windows that deviate >z_threshold stddevs from the weekly mean."""
    df = df.copy()
    df["hour"] = df["timestamp"].dt.floor("h")
    hourly = df.groupby([group_col, "hour"])["count"].sum().reset_index()

    spikes: list[dict] = []
    for group_val, group_df in hourly.groupby(group_col):
        mu = group_df["count"].mean()
        sigma = group_df["count"].std()
        if sigma == 0:
            continue
        z = (group_df["count"] - mu) / sigma
        for _, row in group_df[z > z_threshold].iterrows():
            spikes.append(
                {
                    "group": group_val,
                    "timestamp": row["hour"].isoformat(),
                    "value": int(row["count"]),
                    "z_score": round(float((row["count"] - mu) / sigma), 1),
                }
            )

    return sorted(spikes, key=lambda x: -x["z_score"])


def analyze(current: pd.DataFrame, previous: pd.DataFrame) -> dict:
    """Compute week-over-week statistics. Pure pandas — deterministic, zero cost."""
    analysis: dict[str, Any] = {}

    # -- Volume by endpoint --
    cur_vol = current.groupby("endpoint")["count"].sum()
    prev_vol = previous.groupby("endpoint")["count"].sum()
    analysis["volume_by_endpoint"] = {
        ep: {
            "current": int(cur_vol.get(ep, 0)),
            "previous": int(prev_vol.get(ep, 0)),
            "wow_pct": round((cur_vol.get(ep, 0) - prev_vol.get(ep, 0)) / max(prev_vol.get(ep, 1), 1) * 100, 1),
        }
        for ep in ENDPOINTS
    }

    # -- Error rates by endpoint --
    cur_total = current.groupby("endpoint")["count"].sum()
    prev_total = previous.groupby("endpoint")["count"].sum()
    cur_4xx = current[current["status_code"].between(400, 499)].groupby("endpoint")["count"].sum()
    cur_5xx = current[current["status_code"].between(500, 599)].groupby("endpoint")["count"].sum()
    prev_4xx = previous[previous["status_code"].between(400, 499)].groupby("endpoint")["count"].sum()
    prev_5xx = previous[previous["status_code"].between(500, 599)].groupby("endpoint")["count"].sum()

    analysis["error_rates"] = {
        ep: {
            "4xx_current_pct": round(cur_4xx.get(ep, 0) / max(cur_total.get(ep, 1), 1) * 100, 3),
            "4xx_previous_pct": round(prev_4xx.get(ep, 0) / max(prev_total.get(ep, 1), 1) * 100, 3),
            "5xx_current_pct": round(cur_5xx.get(ep, 0) / max(cur_total.get(ep, 1), 1) * 100, 3),
            "5xx_previous_pct": round(prev_5xx.get(ep, 0) / max(prev_total.get(ep, 1), 1) * 100, 3),
        }
        for ep in ENDPOINTS
    }

    # -- Latency by endpoint (request-count weighted mean + peak) --
    analysis["latency"] = {}
    for ep in ENDPOINTS:
        analysis["latency"][ep] = {}
        for pct in ("p50", "p95", "p99"):
            col = f"{pct}_ms"
            for label, df in [("current", current), ("previous", previous)]:
                ep_df = df[df["endpoint"] == ep]
                weighted = (ep_df[col] * ep_df["count"]).sum() / max(ep_df["count"].sum(), 1)
                analysis["latency"][ep][f"{pct}_mean_{label}_ms"] = round(float(weighted), 1)
                analysis["latency"][ep][f"{pct}_peak_{label}_ms"] = round(float(ep_df[col].max()), 1)

    # -- Status code breakdown (global) --
    cur_by_code = current.groupby("status_code")["count"].sum()
    prev_by_code = previous.groupby("status_code")["count"].sum()
    grand_total = cur_by_code.sum()
    codes = sorted(set(cur_by_code.index) | set(prev_by_code.index))
    analysis["status_codes"] = sorted(
        [
            {
                "code": int(c),
                "current": int(cur_by_code.get(c, 0)),
                "previous": int(prev_by_code.get(c, 0)),
                "wow_pct": round(
                    (cur_by_code.get(c, 0) - prev_by_code.get(c, 0)) / max(prev_by_code.get(c, 1), 1) * 100, 1
                ),
                "share_pct": round(cur_by_code.get(c, 0) / grand_total * 100, 3),
            }
            for c in codes
        ],
        key=lambda r: -r["current"],
    )

    # -- Spike detection (z-score anomaly detection) --
    analysis["spikes"] = {
        "5xx_by_endpoint": _spike_windows(current[current["status_code"].between(500, 599)], "endpoint"),
        "4xx_by_endpoint": _spike_windows(current[current["status_code"].between(400, 499)], "endpoint"),
    }

    return analysis


# -- Report formatting (deterministic) ----------------------------------------


def format_analysis(data: dict) -> str:
    """Format computed analysis into concise text for the LLM prompt."""
    lines: list[str] = []

    lines.append("Request Volume by Endpoint (WoW):")
    for ep, v in data["volume_by_endpoint"].items():
        lines.append(f"  {ep}: {v['current']:,} ({v['wow_pct']:+.1f}% WoW)")
    lines.append("")

    lines.append("Error Rates by Endpoint:")
    for ep, v in data["error_rates"].items():
        lines.append(f"  {ep}:")
        lines.append(f"    4xx: {v['4xx_current_pct']:.3f}% (prev: {v['4xx_previous_pct']:.3f}%)")
        lines.append(f"    5xx: {v['5xx_current_pct']:.3f}% (prev: {v['5xx_previous_pct']:.3f}%)")
    lines.append("")

    lines.append("Latency by Endpoint (ms, weighted mean / peak):")
    for ep, v in data["latency"].items():
        lines.append(f"  {ep}:")
        for pct in ("p50", "p95", "p99"):
            cur = f"{v[f'{pct}_mean_current_ms']:.0f}/{v[f'{pct}_peak_current_ms']:.0f}"
            prev = f"{v[f'{pct}_mean_previous_ms']:.0f}/{v[f'{pct}_peak_previous_ms']:.0f}"
            lines.append(f"    {pct}: {cur} (prev: {prev})")
    lines.append("")

    lines.append("Status Code Breakdown (global):")
    for row in data["status_codes"]:
        if row["current"] == 0 and row["previous"] == 0:
            continue
        lines.append(
            f"  HTTP {row['code']}: {row['current']:,} ({row['share_pct']:.3f}% share, {row['wow_pct']:+.1f}% WoW)"
        )
    lines.append("")

    spikes = data.get("spikes", {})
    if any(spikes.values()):
        lines.append("Anomalous hourly windows (z-score >= 2.5 above weekly baseline):")
        for series, windows in spikes.items():
            if not windows:
                continue
            lines.append(f"  {series}:")
            for w in windows[:5]:
                lines.append(f"    {w['timestamp']}  {w['group']}  count={w['value']}  z={w['z_score']}")
        lines.append("")

    return "\n".join(lines)


_SYSTEM_PROMPT = """\
You are a senior SRE reviewing weekly production metrics for a web service. \
Your job is to identify anomalies, trends, and anything that warrants \
operational attention.

Be concise, precise, and technical. Lead with the most important findings. \
Respond only with valid JSON, no markdown fences.\
"""

_ANALYSIS_PROMPT = """\
Week-over-week production metrics for a web API.

{formatted_data}

Respond with a JSON object with these keys:
  - "executive_summary": string — 2-3 sentences covering the most important findings
  - "anomalies": array of objects sorted by severity, each with:
      - "severity": "high", "medium", or "low"
      - "description": string — what was observed
      - "metric": string — which metric or endpoint is affected
  - "improvements": array of strings — notable positive trends
  - "follow_up_items": array of strings — items warranting further investigation

Focus on operationally meaningful signals. Distinguish between statistical \
noise and genuine trends.\
"""


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


# -- Graph state and nodes ----------------------------------------------------


class ReviewState(TypedDict):
    current: Any  # pd.DataFrame (not serializable, ephemeral task)
    previous: Any  # pd.DataFrame
    analysis: dict | None
    formatted: str
    summary: dict | None
    report: str


def _build_graph(llm) -> StateGraph:
    """Wire the review graph. 3 deterministic nodes, 1 LLM node."""

    def fetch_metrics(state: ReviewState) -> dict:
        current, previous = generate_metrics()
        return {"current": current, "previous": previous}

    def analyze_metrics(state: ReviewState) -> dict:
        result = analyze(state["current"], state["previous"])
        return {"analysis": result, "formatted": format_analysis(result)}

    async def summarize(state: ReviewState) -> dict:
        prompt = _ANALYSIS_PROMPT.format(formatted_data=state["formatted"])
        response = await llm.ainvoke(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
        try:
            summary = json.loads(_strip_json_fences(response.content))
        except (json.JSONDecodeError, TypeError):
            summary = {"executive_summary": response.content}
        return {"summary": summary}

    def compile_report(state: ReviewState) -> dict:
        s = state["summary"] or {}
        lines: list[str] = ["=== Weekly Ops Review ===", ""]

        if s.get("executive_summary"):
            lines += ["Executive Summary:", s["executive_summary"], ""]

        if s.get("anomalies"):
            lines.append("Anomalies:")
            for a in s["anomalies"]:
                sev = (a.get("severity") or "?").upper()
                lines.append(f"  [{sev}] {a.get('metric', '?')}: {a.get('description', '')}")
            lines.append("")

        if s.get("improvements"):
            lines.append("Improvements:")
            for item in s["improvements"]:
                lines.append(f"  - {item}")
            lines.append("")

        if s.get("follow_up_items"):
            lines.append("Follow-up:")
            for item in s["follow_up_items"]:
                lines.append(f"  - {item}")
            lines.append("")

        return {"report": "\n".join(lines)}

    graph = StateGraph(ReviewState)
    graph.add_node("fetch_metrics", fetch_metrics)  # deterministic
    graph.add_node("analyze_metrics", analyze_metrics)  # deterministic
    graph.add_node("summarize", summarize)  # ← the ONE LLM call
    graph.add_node("compile_report", compile_report)  # deterministic

    graph.add_edge(START, "fetch_metrics")
    graph.add_edge("fetch_metrics", "analyze_metrics")
    graph.add_edge("analyze_metrics", "summarize")
    graph.add_edge("summarize", "compile_report")
    graph.add_edge("compile_report", END)

    return graph


# -- Task entry point ---------------------------------------------------------


class ReviewTask(Task):
    name = "review"
    description = "Weekly ops review — service health analysis with mock NewRelic data"

    async def run(self, ctx: AgentContext) -> None:
        llm_config = ctx.config.get("llm", {})
        api_key = llm_config.get("api_key")
        if not api_key:
            ctx.fail("No LLM API key configured. Set llm.api_key in ~/.devops/config.toml")
            return

        model = llm_config.get("model", "claude-sonnet-4-20250514")
        llm = build_llm(model, api_key, llm_config.get("base_url"))

        ctx.progress(f"Starting ops review (model: {model})")
        ctx.progress("Generating mock NewRelic metrics (2 weeks, 5-min granularity)...")

        graph = _build_graph(llm).compile(checkpointer=ctx.checkpointer)
        config = {"configurable": {"thread_id": ctx.task_id}}

        initial: ReviewState = {
            "current": None,
            "previous": None,
            "analysis": None,
            "formatted": "",
            "summary": None,
            "report": "",
        }

        result = await graph.ainvoke(initial, config)
        ctx.complete(result["report"])
