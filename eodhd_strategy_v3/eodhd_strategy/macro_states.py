from __future__ import annotations

from typing import Any, Dict, Optional


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def classify_macro_state(
    gdp_growth: Any,
    inflation: Any,
    unemployment: Any,
    real_interest_rate: Any,
    recent_event_surprise_score: Any = None,
    fallback_regime: str = "neutral",
) -> str:
    g = _to_float(gdp_growth)
    inf = _to_float(inflation)
    un = _to_float(unemployment)
    rr = _to_float(real_interest_rate)
    surprise = _to_float(recent_event_surprise_score)

    # fallback when data is incomplete
    if g is None or inf is None or un is None or rr is None:
        fallback = (fallback_regime or "neutral").lower()
        if fallback == "risk_on":
            return "expansion"
        if fallback == "risk_off":
            return "defensive"
        return "neutral"

    # high inflation / poor real-rate backdrop
    if inf >= 3.5 and rr <= 0.5:
        return "inflation_stress"
    if inf >= 4.0:
        return "inflation_stress"

    # slowdown / protection mode
    if g <= 1.0:
        return "defensive"
    if un >= 5.0:
        return "defensive"
    if rr >= 2.0 and g < 2.0:
        return "defensive"

    # healthy expansion
    if g >= 2.0 and un <= 4.5 and inf < 3.5 and rr < 2.0:
        if surprise is None or surprise >= -0.5:
            return "expansion"

    return "neutral"


def get_macro_factor_weights(macro_state: str, fallback_regime: str = "neutral") -> Dict[str, float]:
    state = (macro_state or "").lower()

    if state == "expansion":
        return {
            "shareholder_yield": 0.45,
            "gross_profitability": 0.20,
            "adjusted_book_to_market": 0.35,
        }

    if state == "defensive":
        return {
            "shareholder_yield": 0.20,
            "gross_profitability": 0.60,
            "adjusted_book_to_market": 0.20,
        }

    if state == "inflation_stress":
        return {
            "shareholder_yield": 0.25,
            "gross_profitability": 0.30,
            "adjusted_book_to_market": 0.45,
        }

    if state == "neutral":
        return {
            "shareholder_yield": 0.40,
            "gross_profitability": 0.35,
            "adjusted_book_to_market": 0.25,
        }

    # fallback to old regime logic
    regime = (fallback_regime or "neutral").lower()
    if regime == "risk_on":
        return {
            "shareholder_yield": 0.50,
            "gross_profitability": 0.20,
            "adjusted_book_to_market": 0.30,
        }
    if regime == "risk_off":
        return {
            "shareholder_yield": 0.20,
            "gross_profitability": 0.60,
            "adjusted_book_to_market": 0.20,
        }
    return {
        "shareholder_yield": 0.45,
        "gross_profitability": 0.35,
        "adjusted_book_to_market": 0.20,
    }