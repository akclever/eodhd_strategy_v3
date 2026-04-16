from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from .advanced_factors import balance_sheet_records, cash_flow_records, income_statement_records
from .client import EODHDClient
from .data_provider import DataProvider
from .utils import pick_first, to_float

ClientLike = EODHDClient | DataProvider

DEFAULT_TOP_N = 20
RECENT_PRICE_LOOKBACK_DAYS = 45
MAX_REASONABLE_CURRENT_ANCHOR_MULTIPLE = 80.0

EPS_BASE_MULTIPLE_BY_SECTOR: dict[str, float] = {
    "technology": 18.0,
    "healthcare": 17.0,
    "consumer cyclical": 14.0,
    "consumer defensive": 16.0,
    "industrials": 15.0,
    "communication services": 16.0,
    "basic materials": 12.0,
    "utilities": 13.0,
    "energy": 10.0,
}

FCF_BASE_MULTIPLE_BY_SECTOR: dict[str, float] = {
    "technology": 20.0,
    "healthcare": 18.0,
    "consumer cyclical": 15.0,
    "consumer defensive": 17.0,
    "industrials": 16.0,
    "communication services": 17.0,
    "basic materials": 13.0,
    "utilities": 14.0,
    "energy": 11.0,
}

CYCLICAL_SECTORS = {
    "basic materials",
    "consumer cyclical",
    "energy",
    "industrials",
}


def _clip(value: float, lower: float, upper: float) -> float:
    return float(min(upper, max(lower, value)))


def _safe_positive_float(value: Any) -> Optional[float]:
    numeric = to_float(value)
    if numeric is None or numeric <= 0:
        return None
    return float(numeric)


def _iter_price_history_rows(price_history: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(price_history, list):
        for item in price_history:
            if isinstance(item, dict):
                yield item
        return

    if isinstance(price_history, dict):
        for key, value in price_history.items():
            if not isinstance(value, dict):
                continue
            item = dict(value)
            item.setdefault("date", key)
            yield item


def _latest_history_price(price_history: Any) -> tuple[Optional[float], Optional[str], Optional[str]]:
    latest_date: Optional[pd.Timestamp] = None
    latest_price: Optional[float] = None
    latest_source: Optional[str] = None

    for item in _iter_price_history_rows(price_history):
        parsed_date = pd.to_datetime(
            item.get("date")
            or item.get("datetime")
            or item.get("timestamp")
            or item.get("dateFormatted"),
            errors="coerce",
        )
        if pd.isna(parsed_date):
            continue

        adjusted_close = _safe_positive_float(item.get("adjusted_close") or item.get("adjustedClose"))
        close = _safe_positive_float(item.get("close"))
        if adjusted_close is not None:
            price = adjusted_close
            source = "history_adjusted_close"
        elif close is not None:
            price = close
            source = "history_close"
        else:
            continue

        if latest_date is None or parsed_date > latest_date:
            latest_date = parsed_date
            latest_price = price
            latest_source = source

    if latest_price is None or latest_date is None:
        return None, None, None
    return float(latest_price), latest_source, latest_date.strftime("%Y-%m-%d")


def _resolve_current_price(
    row: pd.Series,
    *,
    price_history: Any = None,
) -> tuple[Optional[float], Optional[str], Optional[str], Optional[float], Optional[float]]:
    price_proxy = _safe_positive_float(row.get("price_proxy"))

    history_price, history_source, history_date = _latest_history_price(price_history)
    if history_price is not None:
        proxy_gap = None
        if price_proxy is not None:
            proxy_gap = float(price_proxy / history_price - 1.0)
        return float(history_price), history_source, history_date, price_proxy, proxy_gap

    for field_name in ("current_price", "price", "close", "adjusted_close"):
        candidate = _safe_positive_float(row.get(field_name))
        if candidate is not None:
            proxy_gap = None
            if price_proxy is not None:
                proxy_gap = float(price_proxy / candidate - 1.0)
            return candidate, f"row_{field_name}", None, price_proxy, proxy_gap

    if price_proxy is not None:
        return price_proxy, "price_proxy", None, price_proxy, 0.0

    return None, None, None, None, None


def _base_valuation_row(
    row: pd.Series,
    *,
    valuation_symbol: str,
    valuation_currency_code: str,
    current_price: Optional[float],
    current_price_source: Optional[str],
    current_price_as_of_date: Optional[str],
    current_price_proxy: Optional[float],
    price_proxy_gap_pct: Optional[float],
    price_history_fetch_error: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "rank": row.get("rank"),
        "symbol": _text_value(row.get("symbol")),
        "analysis_symbol": _text_value(row.get("analysis_symbol")) or _text_value(row.get("symbol")),
        "company_name": row.get("company_name"),
        "sector": row.get("sector"),
        "industry": row.get("industry"),
        "valuation_symbol": valuation_symbol,
        "valuation_currency_code": valuation_currency_code,
        "current_price": current_price,
        "current_price_source": current_price_source,
        "current_price_as_of_date": current_price_as_of_date,
        "current_price_proxy": current_price_proxy,
        "price_proxy_gap_pct": price_proxy_gap_pct,
        "price_history_fetch_error": price_history_fetch_error,
    }


def _text_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _series_from_records(
    records: Iterable[Dict[str, Any]],
    *,
    field: str,
    fallback_fields: tuple[str, ...] = (),
) -> list[Optional[float]]:
    values: list[Optional[float]] = []
    for record in records:
        values.append(pick_first(record, field, *fallback_fields))
    return values


def _weighted_anchor(values: list[Optional[float]]) -> Optional[float]:
    numeric = pd.Series([to_float(value) for value in values], dtype="float64").dropna()
    if numeric.empty:
        return None
    if len(numeric) >= 4:
        lo = float(numeric.quantile(0.15))
        hi = float(numeric.quantile(0.85))
        numeric = numeric.clip(lower=lo, upper=hi)
    weights = np.arange(len(numeric), 0, -1, dtype=float)
    weights = weights / weights.sum()
    return float(np.dot(numeric.to_numpy(dtype=float), weights))


def _positive_ratio(values: list[Optional[float]]) -> float:
    numeric = pd.Series([to_float(value) for value in values], dtype="float64").dropna()
    if numeric.empty:
        return 0.0
    return float((numeric > 0.0).mean())


def _stability_score(values: list[Optional[float]], scale: float) -> float:
    numeric = pd.Series([to_float(value) for value in values], dtype="float64").dropna()
    if len(numeric) <= 1:
        return 0.0
    return _clip(1.0 - float(numeric.std(ddof=0)) / max(scale, 1e-9), 0.0, 1.0)


def _growth_cagr(values: list[Optional[float]], lookback_years: int = 3) -> Optional[float]:
    numeric = [to_float(value) for value in values]
    usable = [(idx, value) for idx, value in enumerate(numeric) if value is not None and value > 0]
    if len(usable) < 2:
        return None

    latest_idx, latest_value = usable[0]
    for older_idx, older_value in usable[1:]:
        periods = older_idx - latest_idx
        if periods >= lookback_years:
            return float((latest_value / older_value) ** (1.0 / periods) - 1.0)

    oldest_idx, oldest_value = usable[-1]
    periods = oldest_idx - latest_idx
    if periods <= 0:
        return None
    return float((latest_value / oldest_value) ** (1.0 / periods) - 1.0)


def _latest_debt_to_assets(balance_records: list[Dict[str, Any]]) -> Optional[float]:
    if not balance_records:
        return None
    latest = balance_records[0]
    total_assets = pick_first(latest, "totalAssets")
    total_debt = pick_first(latest, "longTermDebt", "shortLongTermDebtTotal", "totalDebt")
    if total_assets is None or total_assets <= 0 or total_debt is None:
        return None
    return float(total_debt / total_assets)


def _sector_key(row: pd.Series) -> str:
    return str(row.get("sector") or "").strip().lower()


def _confidence_label(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.50:
        return "medium"
    return "low"


def _truthy_price_source(source: Optional[str]) -> bool:
    return bool(source) and source not in {"price_proxy", "row_price", "row_close", "row_adjusted_close", "row_current_price"}


def _valuation_reason_code(
    *,
    method: str,
    sector: str,
    life_cycle_stage: str,
    confidence_label: str,
) -> str:
    if method == "unavailable":
        return "insufficient_history"
    if life_cycle_stage == "recovery":
        return "recovery_wide_band"
    if sector in CYCLICAL_SECTORS:
        return "cyclical_normalized"
    if method == "blended_fcf_eps":
        return f"blended_{confidence_label}"
    if method == "normalized_fcf_multiple":
        return "fcf_anchor"
    return "eps_anchor"


def _guardrail_reason(
    *,
    current_price: Optional[float],
    current_anchor_multiple: Optional[float],
) -> Optional[str]:
    if current_price is None or current_price <= 0:
        return "missing_current_price"
    if current_anchor_multiple is not None and current_anchor_multiple > MAX_REASONABLE_CURRENT_ANCHOR_MULTIPLE:
        return "anchor_multiple_outlier"
    return None


def _reversion_dependency(
    *,
    multiple_reversion_component: Optional[float],
    expected_3y_cagr: Optional[float],
) -> Optional[float]:
    if multiple_reversion_component is None or expected_3y_cagr is None:
        return None
    denominator = max(abs(float(expected_3y_cagr)), 1e-6)
    return float(abs(float(multiple_reversion_component)) / denominator)


def _extract_yearly_financial_frame(fundamentals: Dict[str, Any]) -> pd.DataFrame:
    income = {str(record.get("dateFormatted") or record.get("date") or record.get("filing_date") or ""): record for record in income_statement_records(fundamentals, "yearly")}
    balance = {str(record.get("dateFormatted") or record.get("date") or record.get("filing_date") or ""): record for record in balance_sheet_records(fundamentals, "yearly")}
    cashflow = {str(record.get("dateFormatted") or record.get("date") or record.get("filing_date") or ""): record for record in cash_flow_records(fundamentals, "yearly")}

    keys = [key for key in income.keys() if key and key in balance and key in cashflow]
    keys.sort(reverse=True)

    rows: list[dict[str, Any]] = []
    for key in keys[:5]:
        inc = income[key]
        bal = balance[key]
        cf = cashflow[key]
        revenue = pick_first(inc, "totalRevenue")
        gross_profit = pick_first(inc, "grossProfit")
        operating_income = pick_first(inc, "operatingIncome", "ebit")
        net_income = pick_first(inc, "netIncome")
        cfo = pick_first(cf, "totalCashFromOperatingActivities", "operatingCashFlow")
        capex_raw = pick_first(cf, "capitalExpenditures", "capitalExpenditure")
        capex_abs = abs(float(capex_raw)) if capex_raw is not None else None
        shares = pick_first(bal, "commonStockSharesOutstanding")
        if shares is not None and shares <= 0:
            shares = None

        gross_margin = gross_profit / revenue if gross_profit is not None and revenue and revenue > 0 else None
        operating_margin = operating_income / revenue if operating_income is not None and revenue and revenue > 0 else None
        eps = net_income / shares if net_income is not None and shares and shares > 0 else None
        fcf = float(cfo - capex_abs) if cfo is not None and capex_abs is not None else None
        fcf_per_share = fcf / shares if fcf is not None and shares and shares > 0 else None

        rows.append(
            {
                "period": key,
                "revenue": revenue,
                "gross_margin": gross_margin,
                "operating_margin": operating_margin,
                "eps": eps,
                "fcf_per_share": fcf_per_share,
            }
        )

    return pd.DataFrame(rows)


def _target_multiple(
    *,
    method: str,
    row: pd.Series,
    margin_stability: float,
    leverage_penalty: float,
    growth_rate: float,
) -> float:
    sector = _sector_key(row)
    life_cycle_stage = str(row.get("life_cycle_stage") or "").strip().lower()
    forensic_penalty = _clip(float(to_float(row.get("forensic_penalty")) or 0.0) / 1.25, 0.0, 1.0)
    compounder = float(to_float(row.get("compounder_persistence_signal")) or 0.0)
    accrual = float(to_float(row.get("accrual_quality_signal")) or 0.0)
    capital_allocation = float(to_float(row.get("capital_allocation_quality_signal")) or 0.0)
    investment_restraint = float(to_float(row.get("investment_restraint_signal")) or 0.0)
    roic = to_float(row.get("return_on_invested_capital"))
    roic_score = _clip(((roic or 0.08) - 0.08) / 0.12, -1.0, 1.0) if roic is not None else 0.0
    quality_score = (
        0.30 * compounder
        + 0.20 * accrual
        + 0.20 * capital_allocation
        + 0.10 * investment_restraint
        + 0.20 * roic_score
    )
    growth_score = (
        0.55 * _clip(growth_rate / 0.15, -1.0, 1.0)
        + 0.25 * float(to_float(row.get("revision_impulse_signal")) or 0.0)
        + 0.20 * float(to_float(row.get("estimate_term_structure_signal")) or 0.0)
    )

    if method == "normalized_fcf_multiple":
        base_multiple = FCF_BASE_MULTIPLE_BY_SECTOR.get(sector, 15.0)
        lower_bound, upper_bound = 8.0, 32.0
    else:
        base_multiple = EPS_BASE_MULTIPLE_BY_SECTOR.get(sector, 14.0)
        lower_bound, upper_bound = 7.0, 30.0

    target = (
        base_multiple
        + 3.4 * quality_score
        + 2.2 * growth_score
        + 1.1 * margin_stability
        - 2.6 * leverage_penalty
        - 1.8 * forensic_penalty
    )
    if life_cycle_stage == "recovery":
        target -= 1.5
    if sector in CYCLICAL_SECTORS and growth_score > 0.0:
        target -= 0.9 * growth_score

    return _clip(target, lower_bound, upper_bound)


def _expected_growth_rate(row: pd.Series, revenue_cagr: Optional[float]) -> float:
    revenue_growth_yoy = to_float(row.get("revenue_growth_yoy"))
    revision_signal = float(to_float(row.get("revision_impulse_signal")) or 0.0)
    estimate_signal = float(to_float(row.get("estimate_term_structure_signal")) or 0.0)
    compounder = float(to_float(row.get("compounder_persistence_signal")) or 0.0)
    base_growth = revenue_cagr if revenue_cagr is not None else revenue_growth_yoy
    if base_growth is None:
        base_growth = 0.04 + 0.03 * compounder
    growth = (
        0.65 * float(base_growth)
        + 0.15 * 0.10 * revision_signal
        + 0.10 * 0.08 * estimate_signal
        + 0.10 * 0.04 * compounder
    )
    return _clip(growth, -0.05, 0.20)


def _valuation_row_from_fundamentals(
    row: pd.Series,
    fundamentals: Dict[str, Any],
    *,
    current_price: Optional[float],
    current_price_source: Optional[str],
    current_price_as_of_date: Optional[str],
    current_price_proxy: Optional[float],
    price_proxy_gap_pct: Optional[float],
    price_history_fetch_error: Optional[str] = None,
) -> Dict[str, Any]:
    frame = _extract_yearly_financial_frame(fundamentals)
    symbol = _text_value(row.get("symbol"))
    analysis_symbol = _text_value(row.get("analysis_symbol")) or symbol
    valuation_symbol = analysis_symbol or symbol
    valuation_currency_code = str(row.get("analysis_currency_code") or row.get("currency_code") or "")
    sector = _sector_key(row)
    life_cycle_stage = str(row.get("life_cycle_stage") or "").strip().lower()
    base_row = _base_valuation_row(
        row,
        valuation_symbol=valuation_symbol,
        valuation_currency_code=valuation_currency_code,
        current_price=current_price,
        current_price_source=current_price_source,
        current_price_as_of_date=current_price_as_of_date,
        current_price_proxy=current_price_proxy,
        price_proxy_gap_pct=price_proxy_gap_pct,
        price_history_fetch_error=price_history_fetch_error,
    )

    if frame.empty:
        return {
            **base_row,
            "fair_value_low": None,
            "fair_value_base": None,
            "fair_value_high": None,
            "upside_to_base": None,
            "expected_3y_cagr": None,
            "valuation_confidence": 0.0,
            "valuation_confidence_label": "low",
            "valuation_method": "unavailable",
            "valuation_reason_code": "insufficient_history",
            "normalized_eps_per_share": None,
            "normalized_fcf_per_share": None,
            "current_anchor_multiple": None,
            "target_multiple_low": None,
            "target_multiple_base": None,
            "target_multiple_high": None,
            "expected_growth_rate": None,
            "shareholder_yield_component": to_float(row.get("shareholder_yield")),
            "multiple_reversion_component": None,
            "reversion_dependency": None,
            "band_width_ratio": None,
        }

    eps_values = frame["eps"].tolist()
    fcf_values = frame["fcf_per_share"].tolist()
    revenue_values = frame["revenue"].tolist()
    gross_margin_values = frame["gross_margin"].tolist()
    operating_margin_values = frame["operating_margin"].tolist()

    normalized_eps = _weighted_anchor(eps_values)
    normalized_fcf = _weighted_anchor(fcf_values)
    eps_positive_ratio = _positive_ratio(eps_values)
    fcf_positive_ratio = _positive_ratio(fcf_values)
    eps_count = int(pd.Series(eps_values, dtype="float64").notna().sum())
    fcf_count = int(pd.Series(fcf_values, dtype="float64").notna().sum())
    margin_stability = 0.55 * _stability_score(gross_margin_values, 0.12) + 0.45 * _stability_score(
        operating_margin_values, 0.10
    )
    revenue_cagr = _growth_cagr(revenue_values)
    growth_rate = _expected_growth_rate(row, revenue_cagr)

    balance_records = balance_sheet_records(fundamentals, "yearly")
    debt_to_assets = _latest_debt_to_assets(balance_records)
    leverage_penalty = 0.0 if debt_to_assets is None else _clip((float(debt_to_assets) - 0.35) / 0.35, 0.0, 1.0)

    method_weights: dict[str, float] = {}
    if normalized_eps is not None and eps_count >= 3 and eps_positive_ratio >= 0.50:
        method_weights["normalized_eps_multiple"] = 0.85 + 0.35 * eps_positive_ratio + 0.10 * margin_stability
    if normalized_fcf is not None and fcf_count >= 3 and fcf_positive_ratio >= 0.50:
        method_weights["normalized_fcf_multiple"] = 0.95 + 0.40 * fcf_positive_ratio + 0.15 * margin_stability

    if not method_weights:
        method = "unavailable"
        confidence = 0.0
        confidence_label = "low"
        reason_code = "insufficient_history"
        return {
            **base_row,
            "fair_value_low": None,
            "fair_value_base": None,
            "fair_value_high": None,
            "upside_to_base": None,
            "expected_3y_cagr": None,
            "valuation_confidence": confidence,
            "valuation_confidence_label": confidence_label,
            "valuation_method": method,
            "valuation_reason_code": reason_code,
            "normalized_eps_per_share": normalized_eps,
            "normalized_fcf_per_share": normalized_fcf,
            "current_anchor_multiple": None,
            "target_multiple_low": None,
            "target_multiple_base": None,
            "target_multiple_high": None,
            "expected_growth_rate": growth_rate,
            "shareholder_yield_component": to_float(row.get("shareholder_yield")),
            "multiple_reversion_component": None,
            "reversion_dependency": None,
            "band_width_ratio": None,
        }

    total_weight = sum(method_weights.values())
    normalized_weights = {name: value / total_weight for name, value in method_weights.items()}
    method = "blended_fcf_eps" if len(normalized_weights) > 1 else next(iter(normalized_weights))

    target_multiple_base = 0.0
    target_multiple_low = 0.0
    target_multiple_high = 0.0
    fair_value_base = 0.0
    fair_value_low = 0.0
    fair_value_high = 0.0
    blended_anchor = 0.0
    anchor_weight = 0.0

    history_depth = _clip(max(frame["eps"].notna().mean(), frame["fcf_per_share"].notna().mean()), 0.0, 1.0)
    positive_ratio = max(eps_positive_ratio, fcf_positive_ratio)
    confidence = (
        0.30 * history_depth
        + 0.25 * margin_stability
        + 0.20 * positive_ratio
        + 0.15 * (1.0 - leverage_penalty)
        + 0.10 * (1.0 if method != "unavailable" else 0.0)
    )
    if current_price_source == "price_proxy":
        confidence *= 0.45
    elif current_price_source and current_price_source.startswith("row_"):
        confidence *= 0.65
    if life_cycle_stage == "recovery":
        confidence *= 0.78
    confidence = _clip(confidence, 0.0, 1.0)

    spread = 0.20 + 0.50 * (1.0 - confidence)
    metric_adjustment = 0.03 + 0.10 * (1.0 - confidence)

    for method_name, weight in normalized_weights.items():
        anchor = normalized_fcf if method_name == "normalized_fcf_multiple" else normalized_eps
        if anchor is None or anchor <= 0:
            continue
        target = _target_multiple(
            method=method_name,
            row=row,
            margin_stability=margin_stability,
            leverage_penalty=leverage_penalty,
            growth_rate=growth_rate,
        )
        low_target = target * (1.0 - 0.5 * spread)
        high_target = target * (1.0 + 0.5 * spread)

        target_multiple_base += weight * target
        target_multiple_low += weight * low_target
        target_multiple_high += weight * high_target
        fair_value_base += weight * anchor * target
        fair_value_low += weight * anchor * (1.0 - metric_adjustment) * low_target
        fair_value_high += weight * anchor * (1.0 + metric_adjustment) * high_target
        blended_anchor += weight * anchor
        anchor_weight += weight

    current_anchor_multiple = None
    multiple_reversion_component = None
    upside_to_base = None
    expected_3y_cagr = None
    shareholder_yield = float(to_float(row.get("shareholder_yield")) or 0.0)

    if current_price is not None and fair_value_base > 0:
        upside_to_base = float(fair_value_base / current_price - 1.0)
    if current_price is not None and blended_anchor > 0 and anchor_weight > 0:
        current_anchor_multiple = float(current_price / blended_anchor)
        if current_anchor_multiple > 0 and target_multiple_base > 0:
            multiple_reversion_component = float((target_multiple_base / current_anchor_multiple) ** (1.0 / 3.0) - 1.0)
            expected_3y_cagr = float(
                (1.0 + growth_rate) * (1.0 + shareholder_yield) * (1.0 + multiple_reversion_component) - 1.0
            )

    reversion_dependency = _reversion_dependency(
        multiple_reversion_component=multiple_reversion_component,
        expected_3y_cagr=expected_3y_cagr,
    )
    band_width_ratio = None
    if fair_value_base > 0 and fair_value_high > 0 and fair_value_low > 0:
        band_width_ratio = float((fair_value_high - fair_value_low) / fair_value_base)

    if reversion_dependency is not None:
        reversion_penalty = _clip((reversion_dependency - 0.35) / 0.90, 0.0, 1.0)
        confidence *= 1.0 - 0.45 * reversion_penalty
    if band_width_ratio is not None:
        band_penalty = _clip((band_width_ratio - 0.45) / 0.75, 0.0, 1.0)
        confidence *= 1.0 - 0.30 * band_penalty
    if current_anchor_multiple is not None and target_multiple_high > 0:
        anchor_stretch = float(current_anchor_multiple / target_multiple_high)
        anchor_penalty = _clip((anchor_stretch - 1.80) / 1.20, 0.0, 1.0)
        confidence *= 1.0 - 0.40 * anchor_penalty
    confidence = _clip(confidence, 0.0, 1.0)

    guardrail_reason = _guardrail_reason(
        current_price=current_price,
        current_anchor_multiple=current_anchor_multiple,
    )
    if guardrail_reason:
        return {
            **base_row,
            "fair_value_low": None,
            "fair_value_base": None,
            "fair_value_high": None,
            "upside_to_base": None,
            "expected_3y_cagr": None,
            "valuation_confidence": 0.0,
            "valuation_confidence_label": "low",
            "valuation_method": "unavailable",
            "valuation_reason_code": guardrail_reason,
            "normalized_eps_per_share": normalized_eps,
            "normalized_fcf_per_share": normalized_fcf,
            "current_anchor_multiple": current_anchor_multiple,
            "target_multiple_low": float(target_multiple_low) if target_multiple_low > 0 else None,
            "target_multiple_base": float(target_multiple_base) if target_multiple_base > 0 else None,
            "target_multiple_high": float(target_multiple_high) if target_multiple_high > 0 else None,
            "expected_growth_rate": growth_rate,
            "shareholder_yield_component": shareholder_yield,
            "multiple_reversion_component": None,
            "reversion_dependency": reversion_dependency,
            "band_width_ratio": band_width_ratio,
        }

    confidence_label = _confidence_label(confidence)
    reason_code = _valuation_reason_code(
        method=method,
        sector=sector,
        life_cycle_stage=life_cycle_stage,
        confidence_label=confidence_label,
    )

    return {
        **base_row,
        "fair_value_low": float(fair_value_low) if fair_value_low > 0 else None,
        "fair_value_base": float(fair_value_base) if fair_value_base > 0 else None,
        "fair_value_high": float(fair_value_high) if fair_value_high > 0 else None,
        "upside_to_base": upside_to_base,
        "expected_3y_cagr": expected_3y_cagr,
        "valuation_confidence": confidence,
        "valuation_confidence_label": confidence_label,
        "valuation_method": method,
        "valuation_reason_code": reason_code,
        "normalized_eps_per_share": normalized_eps,
        "normalized_fcf_per_share": normalized_fcf,
        "current_anchor_multiple": current_anchor_multiple,
        "target_multiple_low": float(target_multiple_low) if target_multiple_low > 0 else None,
        "target_multiple_base": float(target_multiple_base) if target_multiple_base > 0 else None,
        "target_multiple_high": float(target_multiple_high) if target_multiple_high > 0 else None,
        "expected_growth_rate": growth_rate,
        "shareholder_yield_component": shareholder_yield,
        "multiple_reversion_component": multiple_reversion_component,
        "reversion_dependency": reversion_dependency,
        "band_width_ratio": band_width_ratio,
    }


def _add_valuation_summary_fields(report: pd.DataFrame) -> pd.DataFrame:
    if report.empty:
        return report

    out = report.copy()

    numeric_cols = [
        "rank",
        "current_price",
        "fair_value_low",
        "fair_value_base",
        "fair_value_high",
        "current_anchor_multiple",
        "target_multiple_high",
        "valuation_confidence",
        "multiple_reversion_component",
        "expected_3y_cagr",
        "reversion_dependency",
        "band_width_ratio",
    ]
    for column in numeric_cols:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    out["cheap_to_low_band"] = False
    mask_low = out["current_price"].notna() & out["fair_value_low"].notna()
    out.loc[mask_low, "cheap_to_low_band"] = out.loc[mask_low, "current_price"] < out.loc[mask_low, "fair_value_low"]

    out["cheap_to_base"] = False
    mask_base = out["current_price"].notna() & out["fair_value_base"].notna()
    out.loc[mask_base, "cheap_to_base"] = out.loc[mask_base, "current_price"] < out.loc[mask_base, "fair_value_base"]

    out["valuation_gap_pct"] = np.nan
    out.loc[mask_base, "valuation_gap_pct"] = (
        out.loc[mask_base, "current_price"] / out.loc[mask_base, "fair_value_base"] - 1.0
    )

    out["reversion_dependency"] = pd.to_numeric(out.get("reversion_dependency"), errors="coerce")
    missing_reversion = out["reversion_dependency"].isna()
    valid_components = (
        out["multiple_reversion_component"].notna()
        & out["expected_3y_cagr"].notna()
    )
    recompute_mask = missing_reversion & valid_components
    out.loc[recompute_mask, "reversion_dependency"] = (
        out.loc[recompute_mask, "multiple_reversion_component"].abs()
        / out.loc[recompute_mask, "expected_3y_cagr"].abs().clip(lower=1e-6)
    )

    sector_anchor_p90 = pd.Series(np.nan, index=out.index, dtype="float64")
    if "sector" in out.columns:
        grouped = out.groupby(out["sector"].fillna("").astype(str))["current_anchor_multiple"]
        sector_anchor_p90 = grouped.transform(lambda series: float(series.quantile(0.90)) if series.notna().any() else np.nan)

    anchor_stretch = out["current_anchor_multiple"] / out["target_multiple_high"].replace(0.0, np.nan)
    out["extreme_anchor_flag"] = False
    extreme_mask = out["current_anchor_multiple"].notna() & (
        (sector_anchor_p90.notna() & (out["current_anchor_multiple"] > sector_anchor_p90))
        | (anchor_stretch.notna() & (anchor_stretch > 2.5))
    )
    out.loc[extreme_mask, "extreme_anchor_flag"] = True

    rank_numeric = out["rank"].astype(float)
    if len(out) == 1:
        out["rank_percentile"] = 1.0
    else:
        out["rank_percentile"] = 1.0 - (rank_numeric - 1.0) / max(len(out) - 1.0, 1.0)

    out["valuation_percentile"] = np.nan
    valid_valuation = out["valuation_gap_pct"].notna()
    if valid_valuation.any():
        out.loc[valid_valuation, "valuation_percentile"] = (
            (-out.loc[valid_valuation, "valuation_gap_pct"]).rank(method="average", pct=True)
        )

    out["valuation_tension"] = out["rank_percentile"] - out["valuation_percentile"]

    out["valuation_actionable"] = False
    actionable_mask = (
        out["fair_value_base"].notna()
        & out["valuation_confidence"].notna()
        & (out["valuation_confidence"] >= 0.50)
        & out["current_price_source"].fillna("").astype(str).apply(_truthy_price_source)
    )
    out.loc[actionable_mask, "valuation_actionable"] = True

    return out


def build_valuation_report(
    client: ClientLike,
    ranked: pd.DataFrame,
    *,
    top_n: int = DEFAULT_TOP_N,
) -> pd.DataFrame:
    if ranked.empty or top_n <= 0:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for _, row in ranked.head(int(top_n)).iterrows():
        valuation_symbol = (_text_value(row.get("analysis_symbol")) or _text_value(row.get("symbol"))).upper()
        if not valuation_symbol:
            continue

        price_history = None
        price_history_fetch_error = None
        from_date = (pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=RECENT_PRICE_LOOKBACK_DAYS)).strftime(
            "%Y-%m-%d"
        )
        try:
            price_history = client.get_price_history(valuation_symbol, from_date=from_date)
        except Exception as exc:
            price_history_fetch_error = f"{type(exc).__name__}: {exc}"[:240]

        current_price, current_price_source, current_price_as_of_date, current_price_proxy, price_proxy_gap_pct = (
            _resolve_current_price(row, price_history=price_history)
        )
        valuation_currency_code = str(row.get("analysis_currency_code") or row.get("currency_code") or "")
        base_row = _base_valuation_row(
            row,
            valuation_symbol=valuation_symbol,
            valuation_currency_code=valuation_currency_code,
            current_price=current_price,
            current_price_source=current_price_source,
            current_price_as_of_date=current_price_as_of_date,
            current_price_proxy=current_price_proxy,
            price_proxy_gap_pct=price_proxy_gap_pct,
            price_history_fetch_error=price_history_fetch_error,
        )

        try:
            fundamentals = client.get_fundamentals(valuation_symbol)
            valuation_row = _valuation_row_from_fundamentals(
                row,
                fundamentals,
                current_price=current_price,
                current_price_source=current_price_source,
                current_price_as_of_date=current_price_as_of_date,
                current_price_proxy=current_price_proxy,
                price_proxy_gap_pct=price_proxy_gap_pct,
                price_history_fetch_error=price_history_fetch_error,
            )
        except Exception as exc:
            valuation_row = {
                **base_row,
                "fair_value_low": None,
                "fair_value_base": None,
                "fair_value_high": None,
                "upside_to_base": None,
                "expected_3y_cagr": None,
                "valuation_confidence": 0.0,
                "valuation_confidence_label": "low",
                "valuation_method": "unavailable",
                "valuation_reason_code": "fetch_error",
                "normalized_eps_per_share": None,
                "normalized_fcf_per_share": None,
                "current_anchor_multiple": None,
                "target_multiple_low": None,
                "target_multiple_base": None,
                "target_multiple_high": None,
                "expected_growth_rate": None,
                "shareholder_yield_component": to_float(row.get("shareholder_yield")),
                "multiple_reversion_component": None,
                "reversion_dependency": None,
                "band_width_ratio": None,
                "valuation_error": f"{type(exc).__name__}: {exc}"[:240],
            }
        records.append(valuation_row)

    if not records:
        return pd.DataFrame()

    out = pd.DataFrame(records)
    sort_cols = [col for col in ["rank", "symbol"] if col in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    return _add_valuation_summary_fields(out)
