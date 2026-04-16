from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .client import EODHDClient
from .data_provider import DataProvider
from .utils import to_float, utc_today_ts

ClientLike = EODHDClient | DataProvider

CYCLICAL_SECTORS = {
    "Energy",
    "Industrials",
    "Basic Materials",
    "Consumer Cyclical",
    "Technology",
    "Communication Services",
}

DEFENSIVE_SECTORS = {
    "Healthcare",
    "Consumer Defensive",
    "Utilities",
}


@dataclass
class MacroDecision:
    as_of_date: str
    country: str
    regime: str
    macro_score: float
    indicator_values: Dict[str, Optional[float]]
    notes: List[str]
    sector_bonuses: Dict[str, float]


def _coerce_macro_rows(payload: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]

    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            return [x for x in payload["data"] if isinstance(x, dict)]

        for k, v in payload.items():
            if isinstance(v, dict):
                rows.append(v)
            elif isinstance(v, (int, float, str)):
                rows.append({"date": k, "value": v})

    return rows


def _extract_date_value(row: Dict[str, Any]) -> Tuple[pd.Timestamp, Optional[float]]:
    raw_date = (
        row.get("date")
        or row.get("Date")
        or row.get("period")
        or row.get("Period")
        or row.get("year")
        or row.get("Year")
    )
    raw_value = (
        row.get("value")
        or row.get("Value")
        or row.get("close")
        or row.get("Close")
        or row.get("indicator")
        or row.get("Indicator")
    )

    dt = pd.to_datetime(raw_date, errors="coerce")
    val = to_float(raw_value)
    return dt, val


def _latest_and_previous(payload: Any) -> Tuple[Optional[float], Optional[float]]:
    rows = _coerce_macro_rows(payload)
    if not rows:
        return None, None

    parsed = []
    for row in rows:
        dt, val = _extract_date_value(row)
        if pd.isna(dt) or val is None:
            continue
        parsed.append((dt, val))

    if not parsed:
        return None, None

    parsed.sort(key=lambda x: x[0])
    latest = parsed[-1][1]
    prev = parsed[-2][1] if len(parsed) >= 2 else None
    return latest, prev


def _coerce_event_rows(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        if isinstance(payload.get("events"), list):
            return [x for x in payload["events"] if isinstance(x, dict)]
        return [x for x in payload.values() if isinstance(x, dict)]
    return []


def _event_label(event: Dict[str, Any]) -> str:
    parts = [
        str(event.get("event") or ""),
        str(event.get("type") or ""),
        str(event.get("title") or ""),
        str(event.get("description") or ""),
    ]
    return " ".join(x for x in parts if x).strip().lower()


def _recent_event_surprise_score(client: ClientLike, country: str, as_of_date: pd.Timestamp) -> float:
    start_date = (as_of_date - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    end_date = as_of_date.strftime("%Y-%m-%d")

    try:
        rows = _coerce_event_rows(client.get_economic_events(start_date, end_date, country=country))
    except Exception:
        return 0.0

    signals: List[float] = []

    for event in rows:
        label = _event_label(event)
        actual = to_float(event.get("actual") or event.get("actual_value"))
        estimate = to_float(event.get("estimate") or event.get("consensus") or event.get("forecast"))
        previous = to_float(event.get("previous"))

        benchmark = estimate if estimate is not None else previous
        if actual is None or benchmark is None:
            continue

        if any(k in label for k in ["pmi", "payroll", "retail sales", "durable goods", "industrial production"]):
            signals.append(1.0 if actual > benchmark else -1.0)
        elif any(k in label for k in ["cpi", "inflation", "jobless claims", "unemployment", "rate decision", "interest rate"]):
            signals.append(1.0 if actual < benchmark else -1.0)

    if not signals:
        return 0.0

    score = sum(signals) / len(signals)
    return max(-1.0, min(1.0, score))


def infer_macro_decision(
    client: ClientLike,
    country: str = "USA",
    as_of_date: str | None = None,
    tilt_strength: float = 0.10,
) -> MacroDecision:
    as_of = pd.Timestamp(as_of_date).normalize() if as_of_date else utc_today_ts().tz_localize(None)

    indicator_values: Dict[str, Optional[float]] = {}
    notes: List[str] = []
    score = 0.0

    def fetch(indicator: str) -> Tuple[Optional[float], Optional[float]]:
        try:
            return _latest_and_previous(client.get_macro_indicator(country, indicator))
        except Exception:
            return None, None

    gdp_latest, gdp_prev = fetch("gdp_growth_annual")
    inflation_latest, inflation_prev = fetch("inflation_consumer_prices_annual")
    unemployment_latest, unemployment_prev = fetch("unemployment_total_percent")
    real_rate_latest, _ = fetch("real_interest_rate")

    indicator_values["gdp_growth_annual"] = gdp_latest
    indicator_values["inflation_consumer_prices_annual"] = inflation_latest
    indicator_values["unemployment_total_percent"] = unemployment_latest
    indicator_values["real_interest_rate"] = real_rate_latest

    if gdp_latest is not None:
        if gdp_latest >= 2.0:
            score += 1.0
        elif gdp_latest <= 1.0:
            score -= 1.0
        if gdp_prev is not None:
            if gdp_latest > gdp_prev:
                score += 0.5
            elif gdp_latest < gdp_prev:
                score -= 0.5
        notes.append(f"GDP growth: {gdp_latest:.2f}%")

    if inflation_latest is not None:
        if inflation_latest <= 3.0:
            score += 1.0
        elif inflation_latest >= 4.0:
            score -= 1.0
        if inflation_prev is not None:
            if inflation_latest < inflation_prev:
                score += 0.5
            elif inflation_latest > inflation_prev:
                score -= 0.5
        notes.append(f"Inflation: {inflation_latest:.2f}%")

    if unemployment_latest is not None:
        if unemployment_latest <= 4.5:
            score += 1.0
        elif unemployment_latest >= 5.5:
            score -= 1.0
        if unemployment_prev is not None:
            if unemployment_latest < unemployment_prev:
                score += 0.5
            elif unemployment_latest > unemployment_prev:
                score -= 0.5
        notes.append(f"Unemployment: {unemployment_latest:.2f}%")

    if real_rate_latest is not None:
        if real_rate_latest > 0:
            score += 0.5
        elif real_rate_latest < -0.5:
            score -= 0.5
        notes.append(f"Real interest rate: {real_rate_latest:.2f}%")

    event_score = _recent_event_surprise_score(client, country, as_of)
    score += 0.75 * event_score
    notes.append(f"Recent event surprise score: {event_score:.2f}")

    if score >= 2.0:
        regime = "risk_on"
    elif score <= -2.0:
        regime = "risk_off"
    else:
        regime = "neutral"

    sector_bonuses: Dict[str, float] = {}
    if regime == "risk_on":
        sector_bonuses.update({s: tilt_strength for s in CYCLICAL_SECTORS})
        sector_bonuses.update({s: -tilt_strength / 2.0 for s in DEFENSIVE_SECTORS})
    elif regime == "risk_off":
        sector_bonuses.update({s: tilt_strength for s in DEFENSIVE_SECTORS})
        sector_bonuses.update({s: -tilt_strength / 2.0 for s in CYCLICAL_SECTORS})

    return MacroDecision(
        as_of_date=as_of.strftime("%Y-%m-%d"),
        country=country,
        regime=regime,
        macro_score=round(score, 4),
        indicator_values=indicator_values,
        notes=notes,
        sector_bonuses=sector_bonuses,
    )


def apply_macro_sector_tilts(ranked: pd.DataFrame, decision: MacroDecision) -> pd.DataFrame:
    if ranked.empty:
        return ranked.copy()

    out = ranked.copy()
    out["macro_regime"] = decision.regime
    out["macro_score"] = decision.macro_score
    out["composite_score_pre_macro"] = out["composite_score"]
    out["macro_sector_bonus"] = out["sector"].map(lambda x: decision.sector_bonuses.get(str(x), 0.0))
    out["composite_score"] = out["composite_score"] + out["macro_sector_bonus"]

    out = out.sort_values(
        by=["composite_score", "shareholder_yield", "gross_profitability"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)
    return out