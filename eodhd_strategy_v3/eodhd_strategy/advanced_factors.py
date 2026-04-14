from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .utils import normalize_records, pick_first, to_float, utc_today_ts

INTANGIBLE_ADJUSTMENT_SECTORS = {
    "technology",
    "healthcare",
    "communication services",
}


def _record_key(record: Dict[str, Any]) -> str:
    return str(record.get("dateFormatted") or record.get("date") or record.get("filing_date") or "")


def _statement_records(fundamentals: Dict[str, Any], section_name: str, frequency: str) -> List[Dict[str, Any]]:
    section = (((fundamentals.get("Financials") or {}).get(section_name) or {}).get(frequency) or {})
    return normalize_records(section)


def income_statement_records(fundamentals: Dict[str, Any], frequency: str = "yearly") -> List[Dict[str, Any]]:
    return _statement_records(fundamentals, "Income_Statement", frequency)


def balance_sheet_records(fundamentals: Dict[str, Any], frequency: str = "yearly") -> List[Dict[str, Any]]:
    return _statement_records(fundamentals, "Balance_Sheet", frequency)


def cash_flow_records(fundamentals: Dict[str, Any], frequency: str = "yearly") -> List[Dict[str, Any]]:
    return _statement_records(fundamentals, "Cash_Flow", frequency)


def outstanding_share_records(fundamentals: Dict[str, Any], frequency: str = "yearly") -> List[Dict[str, Any]]:
    outstanding = fundamentals.get("outstandingShares") or {}
    section = outstanding.get(frequency) if isinstance(outstanding, dict) else None
    return normalize_records(section)


def _matched_statement_sets(fundamentals: Dict[str, Any], frequency: str) -> List[Dict[str, Any]]:
    income = {_record_key(rec): rec for rec in income_statement_records(fundamentals, frequency)}
    balance = {_record_key(rec): rec for rec in balance_sheet_records(fundamentals, frequency)}
    cashflow = {_record_key(rec): rec for rec in cash_flow_records(fundamentals, frequency)}

    common_keys = [key for key in income.keys() if key and key in balance and key in cashflow]
    common_keys.sort(key=lambda key: pd.to_datetime(key, errors="coerce"), reverse=True)

    return [
        {"period": key, "income": income[key], "balance": balance[key], "cashflow": cashflow[key]}
        for key in common_keys
    ]


def compute_piotroski_f_score(fundamentals: Dict[str, Any]) -> Optional[int]:
    matched = _matched_statement_sets(fundamentals, "yearly")
    if len(matched) < 2:
        return None

    current = matched[0]
    previous = matched[1]

    inc0, inc1 = current["income"], previous["income"]
    bal0, bal1 = current["balance"], previous["balance"]
    cf0, cf1 = current["cashflow"], previous["cashflow"]

    ni0 = pick_first(inc0, "netIncome")
    ni1 = pick_first(inc1, "netIncome")

    ta0 = pick_first(bal0, "totalAssets")
    ta1 = pick_first(bal1, "totalAssets")

    cfo0 = pick_first(cf0, "totalCashFromOperatingActivities", "operatingCashFlow")
    cfo1 = pick_first(cf1, "totalCashFromOperatingActivities", "operatingCashFlow")

    ltd0 = pick_first(bal0, "longTermDebt", "shortLongTermDebtTotal", "totalDebt")
    ltd1 = pick_first(bal1, "longTermDebt", "shortLongTermDebtTotal", "totalDebt")

    ca0 = pick_first(bal0, "totalCurrentAssets")
    ca1 = pick_first(bal1, "totalCurrentAssets")
    cl0 = pick_first(bal0, "totalCurrentLiabilities")
    cl1 = pick_first(bal1, "totalCurrentLiabilities")

    sh0 = pick_first(bal0, "commonStockSharesOutstanding")
    sh1 = pick_first(bal1, "commonStockSharesOutstanding")

    gp0 = pick_first(inc0, "grossProfit")
    gp1 = pick_first(inc1, "grossProfit")
    rev0 = pick_first(inc0, "totalRevenue")
    rev1 = pick_first(inc1, "totalRevenue")

    score = 0

    roa0 = (ni0 / ta0) if ni0 is not None and ta0 and ta0 > 0 else None
    roa1 = (ni1 / ta1) if ni1 is not None and ta1 and ta1 > 0 else None

    if roa0 is not None and roa0 > 0:
        score += 1
    if cfo0 is not None and cfo0 > 0:
        score += 1
    if roa0 is not None and roa1 is not None and roa0 > roa1:
        score += 1
    if cfo0 is not None and ni0 is not None and cfo0 > ni0:
        score += 1

    lev0 = (ltd0 / ta0) if ltd0 is not None and ta0 and ta0 > 0 else None
    lev1 = (ltd1 / ta1) if ltd1 is not None and ta1 and ta1 > 0 else None
    if lev0 is not None and lev1 is not None and lev0 < lev1:
        score += 1

    cr0 = (ca0 / cl0) if ca0 is not None and cl0 and cl0 > 0 else None
    cr1 = (ca1 / cl1) if ca1 is not None and cl1 and cl1 > 0 else None
    if cr0 is not None and cr1 is not None and cr0 > cr1:
        score += 1

    if sh0 is not None and sh1 is not None and sh0 <= sh1:
        score += 1

    gm0 = (gp0 / rev0) if gp0 is not None and rev0 and rev0 > 0 else None
    gm1 = (gp1 / rev1) if gp1 is not None and rev1 and rev1 > 0 else None
    if gm0 is not None and gm1 is not None and gm0 > gm1:
        score += 1

    at0 = (rev0 / ta0) if rev0 is not None and ta0 and ta0 > 0 else None
    at1 = (rev1 / ta1) if rev1 is not None and ta1 and ta1 > 0 else None
    if at0 is not None and at1 is not None and at0 > at1:
        score += 1

    return int(score)


def earnings_history_records(fundamentals: Dict[str, Any]) -> List[Dict[str, Any]]:
    section = ((fundamentals.get("Earnings") or {}).get("History") or {})
    records = normalize_records(section)
    records.sort(
        key=lambda rec: pd.to_datetime(
            rec.get("reportDate") or rec.get("report_date") or rec.get("date"),
            errors="coerce",
            utc=True,
        ),
        reverse=True,
    )
    return records


def earnings_trend_records(fundamentals: Dict[str, Any]) -> List[Dict[str, Any]]:
    section = ((fundamentals.get("Earnings") or {}).get("Trend") or {})
    records = normalize_records(section)
    records.sort(
        key=lambda rec: pd.to_datetime(rec.get("date"), errors="coerce"),
        reverse=True,
    )
    return records


def latest_earnings_trend_record(fundamentals: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for record in earnings_trend_records(fundamentals):
        if any(
            to_float(record.get(key)) is not None
            for key in [
                "earningsEstimateAvg",
                "epsTrendCurrent",
                "epsTrend7daysAgo",
                "epsTrend30daysAgo",
                "epsRevisionsUpLast7days",
                "epsRevisionsDownLast7days",
                "epsRevisionsUpLast30days",
                "epsRevisionsDownLast30days",
            ]
        ):
            return record
    return None


def _percentage_points_to_fraction(value: Any) -> Optional[float]:
    numeric = to_float(value)
    if numeric is None:
        return None
    return float(numeric) / 100.0


def _first_present_numeric(mapping: Dict[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        value = to_float(mapping.get(key))
        if value is not None:
            return float(value)
    return None


def _latest_share_count(fundamentals: Dict[str, Any]) -> Optional[float]:
    quarterly_history = _share_count_history(fundamentals, "quarterly")
    if quarterly_history:
        return float(quarterly_history[0]["shares"])
    yearly_history = _share_count_history(fundamentals, "yearly")
    if yearly_history:
        return float(yearly_history[0]["shares"])
    return None


def compute_short_interest_metrics_from_fundamentals(fundamentals: Dict[str, Any]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "short_interest_shares": None,
        "short_interest_shares_prev": None,
        "short_interest_ratio": None,
        "short_interest_pct_float": None,
        "short_interest_pct_outstanding": None,
        "short_interest_change": None,
    }

    shares_stats = fundamentals.get("SharesStats") or {}
    technicals = fundamentals.get("Technicals") or {}

    shares = _first_present_numeric(shares_stats, "SharesShort")
    if shares is None:
        shares = _first_present_numeric(technicals, "SharesShort")
    previous_shares = _first_present_numeric(shares_stats, "SharesShortPriorMonth")
    if previous_shares is None:
        previous_shares = _first_present_numeric(technicals, "SharesShortPriorMonth")
    short_ratio = _first_present_numeric(shares_stats, "ShortRatio")
    if short_ratio is None:
        short_ratio = _first_present_numeric(technicals, "ShortRatio")
    short_pct_float = _first_present_numeric(shares_stats, "ShortPercentFloat")
    short_pct_outstanding = _first_present_numeric(shares_stats, "ShortPercentOutstanding")
    if short_pct_outstanding is None:
        short_pct_outstanding = _first_present_numeric(technicals, "ShortPercent")

    shares_float = _first_present_numeric(shares_stats, "SharesFloat")
    shares_outstanding = _first_present_numeric(shares_stats, "SharesOutstanding")
    if shares_outstanding is None:
        shares_outstanding = _latest_share_count(fundamentals)

    if short_pct_float is None and shares is not None and shares_float not in (None, 0):
        short_pct_float = float(shares) / max(abs(float(shares_float)), 1.0)
    if short_pct_outstanding is None and shares is not None and shares_outstanding not in (None, 0):
        short_pct_outstanding = float(shares) / max(abs(float(shares_outstanding)), 1.0)

    short_interest_change = _safe_divide(
        (shares - previous_shares) if shares is not None and previous_shares is not None else None,
        abs(float(previous_shares)) if previous_shares not in (None, 0) else None,
    )

    out["short_interest_shares"] = shares
    out["short_interest_shares_prev"] = previous_shares
    out["short_interest_ratio"] = short_ratio
    out["short_interest_pct_float"] = short_pct_float
    out["short_interest_pct_outstanding"] = short_pct_outstanding
    out["short_interest_change"] = short_interest_change
    return out


def _holder_records(fundamentals: Dict[str, Any], holder_type: str) -> List[Dict[str, Any]]:
    holders = fundamentals.get("Holders") or {}
    section = holders.get(holder_type) if isinstance(holders, dict) else None
    return normalize_records(section)


def _holder_groups_by_date(records: List[Dict[str, Any]]) -> List[tuple[pd.Timestamp, List[Dict[str, Any]]]]:
    grouped: dict[pd.Timestamp, List[Dict[str, Any]]] = {}
    for record in records:
        date = pd.to_datetime(record.get("date"), errors="coerce")
        if pd.isna(date):
            continue
        key = pd.Timestamp(date).normalize()
        grouped.setdefault(key, []).append(record)
    return sorted(grouped.items(), key=lambda item: item[0], reverse=True)


def _holder_total_share_fraction(records: List[Dict[str, Any]]) -> float:
    total = 0.0
    for record in records:
        total_shares = _percentage_points_to_fraction(record.get("totalShares"))
        if total_shares is not None:
            total += float(total_shares)
    return total


def _holder_top_n_concentration(records: List[Dict[str, Any]], n: int = 5) -> float:
    weights = [
        float(weight)
        for weight in (_percentage_points_to_fraction(record.get("totalShares")) for record in records)
        if weight is not None
    ]
    weights.sort(reverse=True)
    return float(sum(weights[:n]))


def compute_institutional_breadth_metrics_from_fundamentals(
    fundamentals: Dict[str, Any],
) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "percent_institutions": _percentage_points_to_fraction((fundamentals.get("SharesStats") or {}).get("PercentInstitutions")),
        "institution_holder_count_latest": None,
        "institution_holder_count_prev": None,
        "institutional_breadth_delta": None,
        "institutional_ownership_from_holders": None,
        "institutional_ownership_delta": None,
        "institutional_top5_concentration_delta": None,
    }

    records = _holder_records(fundamentals, "Institutions")
    if not records:
        records = _holder_records(fundamentals, "Funds")
    if not records:
        return out

    grouped_records = _holder_groups_by_date(records)
    latest_rows = grouped_records[0][1]
    out["institution_holder_count_latest"] = float(len(latest_rows))
    out["institutional_ownership_from_holders"] = _holder_total_share_fraction(latest_rows)

    if len(grouped_records) < 2:
        return out

    previous_rows = grouped_records[1][1]
    prev_count = float(len(previous_rows))
    latest_count = float(len(latest_rows))
    ownership_latest = _holder_total_share_fraction(latest_rows)
    ownership_prev = _holder_total_share_fraction(previous_rows)
    top5_latest = _holder_top_n_concentration(latest_rows)
    top5_prev = _holder_top_n_concentration(previous_rows)

    out["institution_holder_count_prev"] = prev_count
    out["institutional_breadth_delta"] = _safe_divide(latest_count - prev_count, max(prev_count, 1.0))
    out["institutional_ownership_delta"] = float(ownership_latest - ownership_prev)
    out["institutional_top5_concentration_delta"] = float(top5_latest - top5_prev)
    return out


def _share_count_history(fundamentals: Dict[str, Any], frequency: str) -> List[Dict[str, float]]:
    preferred_records = outstanding_share_records(fundamentals, frequency)
    fallback_records = balance_sheet_records(fundamentals, frequency)
    records = preferred_records if preferred_records else fallback_records

    history: List[Dict[str, float]] = []
    seen_dates: set[pd.Timestamp] = set()
    for record in records:
        date = pd.to_datetime(_record_key(record), errors="coerce")
        shares = _share_count_from_record(record)
        if pd.isna(date) or shares is None or shares <= 0:
            continue
        normalized_date = pd.Timestamp(date).normalize()
        if normalized_date in seen_dates:
            continue
        seen_dates.add(normalized_date)
        history.append({"date": normalized_date, "shares": float(shares)})
    return history


def _share_drift(current_shares: Optional[float], previous_shares: Optional[float]) -> Optional[float]:
    if current_shares is None or previous_shares in (None, 0):
        return None
    return (float(previous_shares) - float(current_shares)) / max(abs(float(previous_shares)), 1.0)


def compute_share_drift_metrics_from_fundamentals(
    fundamentals: Dict[str, Any],
    alpha_factor_spec: str = "legacy",
) -> Dict[str, Optional[float]]:
    quarterly_history = _share_count_history(fundamentals, "quarterly")
    yearly_history = _share_count_history(fundamentals, "yearly")

    share_drift_1q = None
    if len(quarterly_history) >= 2:
        share_drift_1q = _share_drift(quarterly_history[0]["shares"], quarterly_history[1]["shares"])

    share_drift_4q = None
    if len(quarterly_history) >= 5:
        share_drift_4q = _share_drift(quarterly_history[0]["shares"], quarterly_history[4]["shares"])
    elif len(yearly_history) >= 2:
        share_drift_4q = _share_drift(yearly_history[0]["shares"], yearly_history[1]["shares"])

    persistence = None
    available_quarterly_transitions = min(4, max(0, len(quarterly_history) - 1))
    if available_quarterly_transitions > 0:
        positive_quarters = 0
        for idx in range(available_quarterly_transitions):
            drift = _share_drift(quarterly_history[idx]["shares"], quarterly_history[idx + 1]["shares"])
            if drift is not None and drift > 0:
                positive_quarters += 1
        persistence = float(positive_quarters / available_quarterly_transitions)

    legacy_buyback_yield = None
    if len(yearly_history) >= 2:
        legacy_buyback_yield = _share_drift(yearly_history[0]["shares"], yearly_history[1]["shares"])
    elif share_drift_1q is not None:
        legacy_buyback_yield = share_drift_1q

    buyback_yield = legacy_buyback_yield
    if str(alpha_factor_spec).lower() == "v2":
        weighted_components: List[tuple[float, float]] = []
        if share_drift_4q is not None:
            weighted_components.append((0.70, float(share_drift_4q)))
        if share_drift_1q is not None:
            weighted_components.append((0.30, float(share_drift_1q)))
        if weighted_components:
            total_weight = float(sum(weight for weight, _ in weighted_components))
            buyback_yield = float(sum(weight * value for weight, value in weighted_components) / total_weight)

    return {
        "buyback_yield": buyback_yield,
        "share_drift_1q": share_drift_1q,
        "share_drift_4q": share_drift_4q,
        "share_drift_persistence": persistence,
    }


def _cost_of_revenue_value(income: Dict[str, Any]) -> Optional[float]:
    cost_of_revenue = pick_first(income, "costOfRevenue")
    if cost_of_revenue is None:
        revenue = pick_first(income, "totalRevenue")
        gross_profit = pick_first(income, "grossProfit")
        if revenue is not None and gross_profit is not None:
            cost_of_revenue = float(revenue - gross_profit)
    if cost_of_revenue is None:
        return None
    return abs(float(cost_of_revenue)) if abs(float(cost_of_revenue)) > 1e-12 else None


def _average_balance(current_value: Optional[float], previous_value: Optional[float]) -> Optional[float]:
    if current_value is None:
        return None
    if previous_value is None:
        return float(current_value)
    return float(current_value + previous_value) / 2.0


def _turnover_days_from_entry(
    income: Dict[str, Any],
    balance: Dict[str, Any],
    previous_balance: Optional[Dict[str, Any]],
    period_days: float,
) -> Dict[str, Optional[float]]:
    revenue = pick_first(income, "totalRevenue")
    cost_of_revenue = _cost_of_revenue_value(income)

    receivables = pick_first(balance, "netReceivables", "accountsReceivable")
    previous_receivables = pick_first(previous_balance or {}, "netReceivables", "accountsReceivable")
    inventory = _inventory_value(balance)
    previous_inventory = _inventory_value(previous_balance or {})
    payables = _accounts_payable_value(balance)
    previous_payables = _accounts_payable_value(previous_balance or {})

    receivables_days = None
    average_receivables = _average_balance(receivables, previous_receivables)
    if average_receivables is not None and revenue not in (None, 0):
        receivables_days = float(average_receivables / max(abs(float(revenue)), 1.0) * float(period_days))

    inventory_days = None
    average_inventory = _average_balance(inventory, previous_inventory)
    if average_inventory is not None and cost_of_revenue not in (None, 0):
        inventory_days = float(average_inventory / max(abs(float(cost_of_revenue)), 1.0) * float(period_days))

    payables_days = None
    average_payables = _average_balance(payables, previous_payables)
    if average_payables is not None and cost_of_revenue not in (None, 0):
        payables_days = float(average_payables / max(abs(float(cost_of_revenue)), 1.0) * float(period_days))

    return {
        "receivables_days": receivables_days,
        "inventory_days": inventory_days,
        "payables_days": payables_days,
    }


def _cash_conversion_cycle_days(
    receivables_days: Optional[float],
    inventory_days: Optional[float],
    payables_days: Optional[float],
) -> Optional[float]:
    if receivables_days is None or inventory_days is None or payables_days is None:
        return None
    return float(receivables_days + inventory_days - payables_days)


def compute_turnover_day_metrics_from_fundamentals(fundamentals: Dict[str, Any]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "receivables_days": None,
        "receivables_days_prev": None,
        "receivables_days_delta": None,
        "inventory_days": None,
        "inventory_days_prev": None,
        "inventory_days_delta": None,
        "payables_days": None,
        "payables_days_prev": None,
        "payables_days_delta": None,
        "cash_conversion_cycle_days": None,
        "cash_conversion_cycle_days_prev": None,
        "cash_conversion_cycle_days_delta": None,
        "cash_conversion_cycle_convexity": None,
    }

    matched = _matched_statement_sets(fundamentals, "quarterly")
    period_days = 90.0
    if len(matched) < 2:
        matched = _matched_statement_sets(fundamentals, "yearly")
        period_days = 365.0
    if len(matched) < 2:
        return out

    current = matched[0]
    previous = matched[1]
    earlier = matched[2] if len(matched) >= 3 else None
    earlier_balance = earlier["balance"] if earlier is not None else None
    earliest_balance = matched[3]["balance"] if len(matched) >= 4 else None

    current_days = _turnover_days_from_entry(
        current["income"],
        current["balance"],
        previous["balance"],
        period_days,
    )
    previous_days = _turnover_days_from_entry(
        previous["income"],
        previous["balance"],
        earlier_balance,
        period_days,
    )
    earlier_days = (
        _turnover_days_from_entry(
            earlier["income"],
            earlier["balance"],
            earliest_balance,
            period_days,
        )
        if earlier is not None and earliest_balance is not None
        else {}
    )

    for key in ["receivables_days", "inventory_days", "payables_days"]:
        current_value = current_days.get(key)
        previous_value = previous_days.get(key)
        out[key] = current_value
        out[f"{key}_prev"] = previous_value
        if current_value is not None and previous_value is not None:
            out[f"{key}_delta"] = float(current_value - previous_value)

    current_cycle = _cash_conversion_cycle_days(
        out.get("receivables_days"),
        out.get("inventory_days"),
        out.get("payables_days"),
    )
    previous_cycle = _cash_conversion_cycle_days(
        out.get("receivables_days_prev"),
        out.get("inventory_days_prev"),
        out.get("payables_days_prev"),
    )
    earlier_cycle = _cash_conversion_cycle_days(
        to_float(earlier_days.get("receivables_days")),
        to_float(earlier_days.get("inventory_days")),
        to_float(earlier_days.get("payables_days")),
    )
    out["cash_conversion_cycle_days"] = current_cycle
    out["cash_conversion_cycle_days_prev"] = previous_cycle
    if current_cycle is not None and previous_cycle is not None:
        current_delta = float(current_cycle - previous_cycle)
        out["cash_conversion_cycle_days_delta"] = current_delta
        if earlier_cycle is not None:
            previous_delta = float(previous_cycle - earlier_cycle)
            out["cash_conversion_cycle_convexity"] = float(current_delta - previous_delta)
    return out


def _safe_divide(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or abs(float(denominator)) < 1e-12:
        return None
    return float(numerator) / float(denominator)


def _clip_unit(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return max(-1.0, min(1.0, float(value)))


def _parse_surprise_pct(item: Dict[str, Any]) -> Optional[float]:
    pct = to_float(item.get("surprisePercent") or item.get("percent"))
    actual = to_float(item.get("epsActual") or item.get("actual"))
    estimate = to_float(item.get("epsEstimate") or item.get("estimate"))
    difference = to_float(item.get("epsDifference") or item.get("difference"))

    if pct is None:
        if difference is None and actual is not None and estimate is not None:
            difference = actual - estimate
        if difference is not None and estimate not in (None, 0):
            pct = (difference / abs(float(estimate))) * 100.0

    if pct is None or estimate in (None, 0) or abs(float(estimate)) < 0.02:
        return None

    return max(-100.0, min(100.0, float(pct)))


def compute_pead_signal_from_surprise(
    earnings_surprise_pct: Optional[float],
    earnings_report_date: Any,
    half_life_days: int = 30,
    max_abs_surprise_pct: float = 100.0,
    max_age_days: int = 45,
) -> Optional[float]:
    surprise = to_float(earnings_surprise_pct)
    if surprise is None:
        return None

    clipped = max(-max_abs_surprise_pct, min(max_abs_surprise_pct, float(surprise)))
    scaled = math.tanh(clipped / 25.0)

    report_ts = pd.to_datetime(earnings_report_date, errors="coerce", utc=True)
    if pd.isna(report_ts):
        return float(scaled)

    age_days = max(0.0, float((utc_today_ts() - report_ts.normalize()).days))
    if age_days > float(max_age_days):
        return 0.0

    decay = 0.5 ** (age_days / max(1.0, float(half_life_days)))
    return float(scaled * decay)


def compute_pead_metrics_from_fundamentals(
    fundamentals: Dict[str, Any],
    min_pead_analysts: int,
    half_life_days: int,
    max_abs_surprise_pct: float,
    max_age_days: int,
    alpha_factor_spec: str = "legacy",
) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "earnings_surprise_pct": None,
        "earnings_report_date": None,
        "pead_signal": None,
        "pead_signal_v2": None,
        "pead_surprise_component": None,
        "pead_decay_component": None,
        "pead_breadth_component": None,
        "pead_revision_component": None,
        "pead_analyst_count": None,
        "pead_filter_pass": 1.0,
        "pead_has_setup_coverage": 0.0,
    }

    today = utc_today_ts()
    use_v2 = str(alpha_factor_spec).lower() == "v2"
    effective_half_life_days = max(1.0, float(half_life_days) * (0.75 if use_v2 else 1.0))
    effective_max_age_days = max(5.0, float(max_age_days) * (0.85 if use_v2 else 1.0))
    chosen_event: Optional[Dict[str, Any]] = None

    for item in earnings_history_records(fundamentals):
        report_ts = pd.to_datetime(
            item.get("reportDate") or item.get("report_date") or item.get("date"),
            errors="coerce",
            utc=True,
        )
        actual = to_float(item.get("epsActual") or item.get("actual"))
        estimate = to_float(item.get("epsEstimate") or item.get("estimate"))
        surprise_pct = _parse_surprise_pct(item)

        if pd.isna(report_ts) or report_ts.normalize() > today:
            continue
        if actual is None or estimate is None or surprise_pct is None:
            continue

        chosen_event = item
        break

    if chosen_event is None:
        return out

    report_ts = pd.to_datetime(
        chosen_event.get("reportDate") or chosen_event.get("report_date") or chosen_event.get("date"),
        errors="coerce",
        utc=True,
    )
    surprise_pct = _parse_surprise_pct(chosen_event)
    if pd.isna(report_ts) or surprise_pct is None:
        return out

    out["earnings_surprise_pct"] = float(surprise_pct)
    out["earnings_report_date"] = report_ts.strftime("%Y-%m-%d")

    clipped_surprise = max(-max_abs_surprise_pct, min(max_abs_surprise_pct, float(surprise_pct)))
    surprise_component = math.tanh(clipped_surprise / 25.0)
    age_days = max(0.0, float((today - report_ts.normalize()).days))
    decay_component = 0.0 if age_days > effective_max_age_days else 0.5 ** (age_days / effective_half_life_days)

    out["pead_surprise_component"] = float(surprise_component)
    out["pead_decay_component"] = float(decay_component)

    chosen_period = str(chosen_event.get("date") or "")
    trend_map = {str(record.get("date") or ""): record for record in earnings_trend_records(fundamentals)}
    trend = trend_map.get(chosen_period)
    if not trend:
        return out

    analyst_count = to_float(trend.get("earningsEstimateNumberOfAnalysts"))
    estimate_avg = to_float(trend.get("earningsEstimateAvg"))
    eps_current = to_float(trend.get("epsTrendCurrent"))
    eps_30d = to_float(trend.get("epsTrend30daysAgo"))
    rev_up_7d = to_float(trend.get("epsRevisionsUpLast7days"))
    rev_down_30d = to_float(trend.get("epsRevisionsDownLast30days"))

    out["pead_analyst_count"] = analyst_count

    if analyst_count is None or eps_current is None or eps_30d is None or rev_up_7d is None or rev_down_30d is None:
        return out

    drift_component = _clip_unit((eps_current - eps_30d) / max(abs(float(estimate_avg or 0.0)), 0.05))
    breadth_revision = _clip_unit((rev_up_7d - rev_down_30d) / max(float(analyst_count), 1.0))
    if drift_component is None or breadth_revision is None:
        return out

    revision_component = _clip_unit((drift_component + breadth_revision) / 2.0)
    breadth_component = min(1.0, float(analyst_count) / max(1.0, float(min_pead_analysts)))

    out["pead_has_setup_coverage"] = 1.0
    out["pead_breadth_component"] = float(breadth_component)
    out["pead_revision_component"] = revision_component
    out["pead_filter_pass"] = 1.0 if revision_component is not None and revision_component > 0 else 0.0

    if revision_component is None:
        return out

    signal = surprise_component * decay_component * breadth_component * revision_component
    signal = max(-1.0, min(1.0, float(signal)))
    out["pead_signal"] = signal
    out["pead_signal_v2"] = signal
    return out


def _normalize_estimate_growth(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None

    normalized = float(value)
    if abs(normalized) > 3.0:
        normalized = normalized / 100.0
    return normalized


def _clip_score(value: Optional[float], scale: float, limit: float = 1.0) -> Optional[float]:
    if value is None or scale <= 0:
        return None
    return max(-float(limit), min(float(limit), float(value) / float(scale)))


def _mean_available(values: List[Optional[float]]) -> Optional[float]:
    usable = [float(value) for value in values if value is not None]
    if not usable:
        return None
    return float(sum(usable) / len(usable))


def _winsorize_series(series: pd.Series, lower: float = 0.05, upper: float = 0.95) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return numeric
    lo = float(valid.quantile(lower))
    hi = float(valid.quantile(upper))
    return numeric.clip(lower=lo, upper=hi)


def _capitalize_expense(records: List[Dict[str, Any]], field: str, years: int, scale: float = 1.0) -> Optional[float]:
    capital = 0.0
    found_any = False
    for age, record in enumerate(records[:years]):
        expense = to_float(record.get(field))
        if expense is None or expense <= 0:
            continue
        found_any = True
        remaining_life = max(0.0, 1.0 - (age / max(1, years)))
        capital += float(expense) * float(scale) * remaining_life
    return capital if found_any else None


def _intangible_adjustment_eligible(fundamentals: Dict[str, Any]) -> bool:
    general = fundamentals.get("General") or {}
    sector_text = str(general.get("Sector") or "").strip().lower()
    latest_income = next(iter(income_statement_records(fundamentals, "yearly")), {})
    revenue = pick_first(latest_income, "totalRevenue")
    rd_expense = pick_first(latest_income, "researchDevelopment")
    rd_ratio = _safe_divide(rd_expense, revenue)
    return bool((rd_ratio is not None and rd_ratio >= 0.02) or sector_text in INTANGIBLE_ADJUSTMENT_SECTORS)


def _capitalized_intangible_assets(fundamentals: Dict[str, Any], use_intangible_adjustments: bool) -> float:
    if not use_intangible_adjustments or not _intangible_adjustment_eligible(fundamentals):
        return 0.0
    records = income_statement_records(fundamentals, "yearly")
    return float(
        (_capitalize_expense(records, "researchDevelopment", years=3, scale=1.0) or 0.0)
        + (_capitalize_expense(records, "sellingGeneralAdministrative", years=2, scale=0.30) or 0.0)
    )


def _share_count_from_record(record: Dict[str, Any]) -> Optional[float]:
    return pick_first(
        record,
        "commonStockSharesOutstanding",
        "shares",
        "sharesMln",
        "weightedAverageShsOut",
        "weightedAverageShsOutDil",
    )


def _gross_margin(income: Dict[str, Any]) -> Optional[float]:
    revenue = pick_first(income, "totalRevenue")
    gross_profit = pick_first(income, "grossProfit")
    cost_of_revenue = pick_first(income, "costOfRevenue")
    if revenue is None or revenue <= 0:
        return None
    if gross_profit is None and cost_of_revenue is not None:
        gross_profit = revenue - cost_of_revenue
    if gross_profit is None:
        return None
    return float(gross_profit / revenue)


def _asset_turnover(income: Dict[str, Any], balance: Dict[str, Any], previous_balance: Optional[Dict[str, Any]] = None) -> Optional[float]:
    revenue = pick_first(income, "totalRevenue")
    assets = pick_first(balance, "totalAssets")
    previous_assets = pick_first(previous_balance or {}, "totalAssets")
    if revenue is None:
        return None
    denominator = assets
    if assets is not None and previous_assets is not None and (assets + previous_assets) > 0:
        denominator = (assets + previous_assets) / 2.0
    if denominator is None or denominator <= 0:
        return None
    return float(revenue / denominator)


def _cfo_ni_consistency(net_income: Optional[float], cfo: Optional[float]) -> Optional[float]:
    if net_income is None or cfo is None:
        return None
    denominator = max(abs(float(net_income)), abs(float(cfo)), 1.0)
    gap = abs(float(cfo) - float(net_income)) / denominator
    return float(max(-1.0, min(1.0, 1.0 - gap)))


def _capital_expenditure_abs(cashflow: Dict[str, Any]) -> Optional[float]:
    capex = pick_first(cashflow, "capitalExpenditures", "capitalExpenditure")
    if capex is None:
        return None
    return abs(float(capex))


def _inventory_value(balance: Dict[str, Any]) -> Optional[float]:
    return pick_first(balance, "inventory", "totalInventory")


def _accounts_payable_value(balance: Dict[str, Any]) -> Optional[float]:
    return pick_first(balance, "accountsPayable", "accountPayables", "payables")


def _ebit_value(income: Dict[str, Any]) -> Optional[float]:
    return pick_first(
        income,
        "ebit",
        "EBIT",
        "operatingIncome",
        "operatingIncomeLoss",
        "incomeBeforeTax",
        "netIncome",
    )


def _goodwill_and_intangible_assets(balance: Dict[str, Any]) -> Optional[float]:
    combined = pick_first(
        balance,
        "goodWillAndOtherIntangibleAssets",
        "goodWillAndIntangibleAssets",
        "goodwillAndOtherIntangibleAssets",
        "goodwillAndIntangibleAssets",
    )
    if combined is not None:
        return float(combined)

    goodwill = pick_first(balance, "goodWill", "goodwill")
    intangible_assets = pick_first(
        balance,
        "otherIntangibleAssets",
        "intangibleAssets",
        "otherAssetsIntangible",
    )
    if goodwill is None and intangible_assets is None:
        return None
    return float((goodwill or 0.0) + (intangible_assets or 0.0))


def _accrual_ratio_from_set(current: Dict[str, Any], previous: Optional[Dict[str, Any]]) -> Optional[float]:
    income = current.get("income", {})
    cashflow = current.get("cashflow", {})
    balance = current.get("balance", {})
    previous_balance = (previous or {}).get("balance", {})

    net_income = pick_first(income, "netIncome")
    cfo = pick_first(cashflow, "totalCashFromOperatingActivities", "operatingCashFlow")
    total_assets = pick_first(balance, "totalAssets")
    previous_assets = pick_first(previous_balance, "totalAssets")
    if net_income is None or cfo is None or total_assets is None:
        return None

    denominator = float(total_assets)
    if previous_assets is not None and (total_assets + previous_assets) > 0:
        denominator = float((total_assets + previous_assets) / 2.0)
    if denominator <= 0:
        return None
    return float((net_income - cfo) / denominator)


def compute_estimate_term_structure_metrics_from_fundamentals(
    fundamentals: Dict[str, Any],
    min_revision_analysts: int,
    alpha_factor_spec: str = "legacy",
) -> Dict[str, Optional[float]]:
    if str(alpha_factor_spec).lower() == "v2":
        return _compute_estimate_term_structure_metrics_v2(
            fundamentals,
            min_revision_analysts=min_revision_analysts,
        )

    out: Dict[str, Optional[float]] = {
        "estimate_term_structure_signal": None,
        "estimate_term_structure_has_coverage": 0.0,
        "estimate_term_structure_record_count": 0.0,
        "estimate_term_structure_persistence": None,
        "estimate_term_structure_improvement": None,
        "estimate_term_structure_disagreement_trend": None,
        "estimate_term_structure_coverage_component": None,
    }

    rows: List[Dict[str, float]] = []
    for trend in earnings_trend_records(fundamentals)[:6]:
        analyst_count = to_float(trend.get("earningsEstimateNumberOfAnalysts"))
        estimate_avg = to_float(trend.get("earningsEstimateAvg"))
        eps_current = to_float(trend.get("epsTrendCurrent"))
        eps_7d = to_float(trend.get("epsTrend7daysAgo"))
        eps_30d = to_float(trend.get("epsTrend30daysAgo"))
        rev_up_7d = to_float(trend.get("epsRevisionsUpLast7days"))
        rev_down_30d = to_float(trend.get("epsRevisionsDownLast30days"))
        estimate_high = to_float(trend.get("earningsEstimateHigh"))
        estimate_low = to_float(trend.get("earningsEstimateLow"))

        if analyst_count is None or analyst_count <= 0 or estimate_avg is None or eps_current is None:
            continue

        denominator = max(abs(float(estimate_avg)), 0.05)
        drift_7d = (float(eps_current) - float(eps_7d)) / denominator if eps_7d is not None else None
        drift_30d = (float(eps_current) - float(eps_30d)) / denominator if eps_30d is not None else None
        breadth = (
            (float(rev_up_7d) - float(rev_down_30d)) / max(float(analyst_count), 1.0)
            if rev_up_7d is not None and rev_down_30d is not None
            else None
        )
        disagreement = (
            _safe_divide((estimate_high - estimate_low), denominator)
            if estimate_high is not None and estimate_low is not None
            else None
        )

        row_signal = _mean_available(
            [
                _clip_score(drift_7d, 0.20),
                _clip_score(drift_30d, 0.25),
                _clip_score(breadth, 0.40),
            ]
        )
        if row_signal is None:
            continue

        rows.append(
            {
                "signal": float(row_signal),
                "disagreement": float(disagreement) if disagreement is not None else float("nan"),
                "analyst_count": float(analyst_count),
            }
        )
        if len(rows) >= 3:
            break

    if len(rows) < 2:
        return out

    frame = pd.DataFrame(rows)
    persistence = float(frame["signal"].mean())
    improvement = float(frame["signal"].iloc[0] - frame["signal"].iloc[-1])
    disagreement_values = frame["disagreement"].replace([np.inf, -np.inf], np.nan)
    disagreement_trend = None
    if disagreement_values.notna().sum() >= 2:
        disagreement_trend = float(disagreement_values.iloc[-1] - disagreement_values.iloc[0])

    coverage_component = float(
        min(1.0, frame["analyst_count"].mean() / max(1.0, float(min_revision_analysts)))
    )
    signal = _mean_available(
        [
            _clip_score(persistence, 0.60),
            _clip_score(improvement, 0.35),
            _clip_score(disagreement_trend, 0.75),
        ]
    )
    if signal is None:
        return out

    signal = float(max(-1.0, min(1.0, signal * coverage_component)))
    out["estimate_term_structure_signal"] = signal
    out["estimate_term_structure_has_coverage"] = 1.0
    out["estimate_term_structure_record_count"] = float(len(frame))
    out["estimate_term_structure_persistence"] = persistence
    out["estimate_term_structure_improvement"] = improvement
    out["estimate_term_structure_disagreement_trend"] = disagreement_trend
    out["estimate_term_structure_coverage_component"] = coverage_component
    return out


def compute_compounder_persistence_metrics_from_fundamentals(
    fundamentals: Dict[str, Any],
    alpha_factor_spec: str = "legacy",
    use_intangible_adjustments: bool = False,
) -> Dict[str, Optional[float]]:
    if str(alpha_factor_spec).lower() == "v2":
        return _compute_compounder_persistence_metrics_v2(
            fundamentals,
            use_intangible_adjustments=use_intangible_adjustments,
        )

    out: Dict[str, Optional[float]] = {
        "compounder_persistence_signal": None,
        "compounder_persistence_has_coverage": 0.0,
        "compounder_persistence_measure_count": 0.0,
        "compounder_persistence_level_component": None,
        "compounder_persistence_stability_component": None,
        "compounder_persistence_trend_component": None,
        "compounder_persistence_periodicity": None,
    }

    matched = _matched_statement_sets(fundamentals, "quarterly")
    periodicity = 1.0
    if len(matched) < 4:
        matched = _matched_statement_sets(fundamentals, "yearly")
        periodicity = 0.0
    if len(matched) < 4:
        return out

    matched = list(reversed(matched[:8]))
    rows: List[Dict[str, float]] = []
    for idx, entry in enumerate(matched):
        previous_balance = matched[idx - 1]["balance"] if idx > 0 else None
        gross_margin = _gross_margin(entry["income"])
        asset_turnover = _asset_turnover(entry["income"], entry["balance"], previous_balance)
        net_income = pick_first(entry["income"], "netIncome")
        cfo = pick_first(entry["cashflow"], "totalCashFromOperatingActivities", "operatingCashFlow")
        cfo_consistency = _cfo_ni_consistency(net_income, cfo)
        share_count = _share_count_from_record(entry["balance"])
        accrual_ratio = _accrual_ratio_from_set(entry, matched[idx - 1] if idx > 0 else None)

        row = {
            "gross_margin": float(gross_margin) if gross_margin is not None else np.nan,
            "asset_turnover": float(asset_turnover) if asset_turnover is not None else np.nan,
            "cfo_consistency": float(cfo_consistency) if cfo_consistency is not None else np.nan,
            "share_count": float(share_count) if share_count is not None else np.nan,
            "accrual_ratio": float(accrual_ratio) if accrual_ratio is not None else np.nan,
        }
        if pd.isna(pd.Series(row)).all():
            continue
        rows.append(row)

    if len(rows) < 4:
        return out

    frame = pd.DataFrame(rows)
    level_components = [
        _clip_score(frame["gross_margin"].iloc[-1] if frame["gross_margin"].notna().any() else None, 0.60),
        _clip_score(frame["asset_turnover"].iloc[-1] if frame["asset_turnover"].notna().any() else None, 1.20),
        frame["cfo_consistency"].iloc[-1] if frame["cfo_consistency"].notna().any() else None,
        _clip_score(
            None
            if frame["share_count"].notna().sum() < 2
            else (frame["share_count"].iloc[0] - frame["share_count"].iloc[-1]) / max(abs(frame["share_count"].iloc[0]), 1.0),
            0.08,
        ),
        _clip_score(
            None if frame["accrual_ratio"].notna().sum() < 1 else -float(frame["accrual_ratio"].iloc[-1]),
            0.10,
        ),
    ]
    level_score = _mean_available(level_components)

    stability_components = []
    for column, scale in [
        ("gross_margin", 0.10),
        ("asset_turnover", 0.25),
        ("cfo_consistency", 0.30),
        ("accrual_ratio", 0.06),
    ]:
        series = frame[column].dropna()
        if len(series) >= 3:
            stability_components.append(_clip_score(1.0 - float(series.std(ddof=0) / scale), 1.0))
    if frame["share_count"].notna().sum() >= 3:
        share_std = frame["share_count"].pct_change(fill_method=None).dropna().std(ddof=0)
        if pd.notna(share_std):
            stability_components.append(_clip_score(1.0 - float(share_std / 0.04), 1.0))
    stability_score = _mean_available(stability_components)

    trend_components = []
    for column, scale in [
        ("gross_margin", 0.08),
        ("asset_turnover", 0.20),
        ("cfo_consistency", 0.35),
    ]:
        series = frame[column].dropna()
        if len(series) >= 2:
            trend_components.append(_clip_score(float(series.iloc[-1] - series.iloc[0]), scale))
    share_series = frame["share_count"].dropna()
    if len(share_series) >= 2:
        trend_components.append(
            _clip_score((float(share_series.iloc[0]) - float(share_series.iloc[-1])) / max(abs(float(share_series.iloc[0])), 1.0), 0.08)
        )
    accrual_series = frame["accrual_ratio"].dropna()
    if len(accrual_series) >= 2:
        trend_components.append(_clip_score(float(accrual_series.iloc[0] - accrual_series.iloc[-1]), 0.06))
    trend_score = _mean_available(trend_components)

    available_component_count = sum(component is not None for component in [level_score, stability_score, trend_score])
    if available_component_count < 3:
        return out

    coverage = float(min(1.0, len(frame.columns[frame.notna().any()]) / 5.0))
    signal = _mean_available([level_score, stability_score, trend_score])
    if signal is None:
        return out

    out["compounder_persistence_signal"] = float(max(-1.0, min(1.0, signal * coverage)))
    out["compounder_persistence_has_coverage"] = coverage
    out["compounder_persistence_measure_count"] = float(len(frame))
    out["compounder_persistence_level_component"] = level_score
    out["compounder_persistence_stability_component"] = stability_score
    out["compounder_persistence_trend_component"] = trend_score
    out["compounder_persistence_periodicity"] = periodicity
    return out


def compute_working_capital_stress_metrics(
    fundamentals: Dict[str, Any],
    alpha_factor_spec: str = "legacy",
) -> Dict[str, Optional[float]]:
    if str(alpha_factor_spec).lower() == "v2":
        return _compute_working_capital_stress_metrics_v2(fundamentals)

    out: Dict[str, Optional[float]] = {
        "working_capital_stress_penalty": 0.0,
        "working_capital_stress_has_coverage": 0.0,
        "working_capital_receivables_stress": None,
        "working_capital_inventory_stress": None,
        "working_capital_payables_stress": None,
        "working_capital_cfo_stress": None,
        "working_capital_cycle_stress": None,
    }

    matched = _matched_statement_sets(fundamentals, "yearly")
    if len(matched) < 2:
        matched = _matched_statement_sets(fundamentals, "quarterly")
    if len(matched) < 2:
        return out

    current = matched[0]
    previous = matched[1]
    inc0, inc1 = current["income"], previous["income"]
    bal0, bal1 = current["balance"], previous["balance"]
    cf0 = current["cashflow"]

    sales_0 = pick_first(inc0, "totalRevenue")
    sales_1 = pick_first(inc1, "totalRevenue")
    receivables_0 = pick_first(bal0, "netReceivables", "accountsReceivable")
    receivables_1 = pick_first(bal1, "netReceivables", "accountsReceivable")
    inventory_0 = _inventory_value(bal0)
    inventory_1 = _inventory_value(bal1)
    payables_0 = _accounts_payable_value(bal0)
    payables_1 = _accounts_payable_value(bal1)
    net_income_0 = pick_first(inc0, "netIncome")
    cfo_0 = pick_first(cf0, "totalCashFromOperatingActivities", "operatingCashFlow")

    sales_growth = _safe_divide((sales_0 - sales_1) if sales_0 is not None and sales_1 is not None else None, abs(float(sales_1)) if sales_1 not in (None, 0) else None)
    receivables_growth = _safe_divide((receivables_0 - receivables_1) if receivables_0 is not None and receivables_1 is not None else None, abs(float(receivables_1)) if receivables_1 not in (None, 0) else None)
    inventory_growth = _safe_divide((inventory_0 - inventory_1) if inventory_0 is not None and inventory_1 is not None else None, abs(float(inventory_1)) if inventory_1 not in (None, 0) else None)
    payables_growth = _safe_divide((payables_0 - payables_1) if payables_0 is not None and payables_1 is not None else None, abs(float(payables_1)) if payables_1 not in (None, 0) else None)
    cfo_gap = _safe_divide((net_income_0 - cfo_0) if net_income_0 is not None and cfo_0 is not None else None, max(abs(float(net_income_0)), 1.0) if net_income_0 is not None else None)

    receivables_stress = max(0.0, float(receivables_growth - sales_growth)) if receivables_growth is not None and sales_growth is not None else None
    inventory_stress = max(0.0, float(inventory_growth - sales_growth)) if inventory_growth is not None and sales_growth is not None else None
    payables_stress = max(0.0, float(payables_growth - sales_growth - 0.10)) if payables_growth is not None and sales_growth is not None else None
    cfo_stress = max(0.0, float(cfo_gap)) if cfo_gap is not None else None

    penalties = [
        _clip_score(receivables_stress, 0.30),
        _clip_score(inventory_stress, 0.30),
        _clip_score(payables_stress, 0.30),
        _clip_score(cfo_stress, 0.40),
    ]
    usable_penalties = [float(penalty) for penalty in penalties if penalty is not None]
    if len(usable_penalties) < 2:
        return out

    out["working_capital_stress_penalty"] = float(min(0.06, max(0.0, 0.06 * sum(usable_penalties) / len(usable_penalties))))
    out["working_capital_stress_has_coverage"] = float(min(1.0, len(usable_penalties) / 4.0))
    out["working_capital_receivables_stress"] = receivables_stress
    out["working_capital_inventory_stress"] = inventory_stress
    out["working_capital_payables_stress"] = payables_stress
    out["working_capital_cfo_stress"] = cfo_stress
    return out


def compute_capital_allocation_quality_metrics(
    fundamentals: Dict[str, Any],
    alpha_factor_spec: str = "legacy",
) -> Dict[str, Optional[float]]:
    if str(alpha_factor_spec).lower() == "v2":
        return _compute_capital_allocation_quality_metrics_v2(fundamentals)

    out: Dict[str, Optional[float]] = {
        "capital_allocation_quality_signal": None,
        "capital_allocation_quality_has_coverage": 0.0,
        "capital_allocation_buyback_component": None,
        "capital_allocation_funding_component": None,
        "capital_allocation_debt_component": None,
        "capital_allocation_payout_component": None,
        "capital_allocation_reinvestment_component": None,
        "capital_allocation_financing_dependency_component": None,
        "financing_dependency_stress": None,
        "financing_dependency_burn_component": None,
        "financing_dependency_dilution_component": None,
        "financing_dependency_debt_component": None,
        "financing_dependency_revision_component": None,
    }

    matched = _matched_statement_sets(fundamentals, "yearly")
    if len(matched) < 2:
        return out

    current = matched[0]
    previous = matched[1]
    inc0, bal0, cf0 = current["income"], current["balance"], current["cashflow"]
    bal1 = previous["balance"]

    current_shares = _share_count_from_record(bal0)
    previous_shares = _share_count_from_record(bal1)
    share_change = (
        (float(previous_shares) - float(current_shares)) / max(abs(float(previous_shares)), 1.0)
        if current_shares is not None and previous_shares not in (None, 0)
        else None
    )

    total_assets = pick_first(bal0, "totalAssets")
    cfo = pick_first(cf0, "totalCashFromOperatingActivities", "operatingCashFlow")
    capex = _capital_expenditure_abs(cf0)
    fcf = float(cfo - capex) if cfo is not None and capex is not None else None
    fcf_scale = max(abs(float(total_assets)) * 0.05, 1.0) if total_assets is not None else None

    debt_current = _total_debt(bal0)
    debt_previous = _total_debt(bal1)
    debt_improvement = (
        (float(debt_previous) - float(debt_current)) / max(abs(float(debt_previous)), 1.0)
        if debt_current is not None and debt_previous not in (None, 0)
        else None
    )

    payout_ratio = pick_first(fundamentals.get("Highlights") or {}, "PayoutRatio")
    buyback_component = _clip_score(share_change, 0.08)
    funding_component = None
    if share_change is not None and fcf is not None and fcf_scale is not None:
        funding_component = _clip_score((float(fcf) / fcf_scale) + 0.75 * float(share_change), 1.0)
        if debt_improvement is not None and debt_improvement < 0 and share_change > 0:
            funding_component = max(-1.0, float(funding_component) + float(debt_improvement))
    debt_component = _clip_score(debt_improvement, 0.12)
    payout_component = _clip_score((1.0 - float(payout_ratio)) if payout_ratio is not None else None, 0.60)

    components = [buyback_component, funding_component, debt_component, payout_component]
    usable_components = [float(component) for component in components if component is not None]
    if len(usable_components) < 2:
        return out

    out["capital_allocation_quality_signal"] = float(sum(usable_components) / len(usable_components))
    out["capital_allocation_quality_has_coverage"] = float(min(1.0, len(usable_components) / 4.0))
    out["capital_allocation_buyback_component"] = buyback_component
    out["capital_allocation_funding_component"] = funding_component
    out["capital_allocation_debt_component"] = debt_component
    out["capital_allocation_payout_component"] = payout_component
    return out


def compute_recovery_fundamental_metrics(
    fundamentals: Dict[str, Any],
    alpha_factor_spec: str = "legacy",
) -> Dict[str, Optional[float]]:
    if str(alpha_factor_spec).lower() == "v2":
        return _compute_recovery_fundamental_metrics_v2(fundamentals)

    out: Dict[str, Optional[float]] = {
        "recovery_margin_inflection": None,
        "recovery_leverage_improvement": None,
        "recovery_accrual_improvement": None,
    }

    quarterly = _matched_statement_sets(fundamentals, "quarterly")
    if len(quarterly) >= 4:
        margin_series: List[float] = []
        leverage_series: List[float] = []
        for entry in quarterly[:4]:
            income = entry["income"]
            balance = entry["balance"]
            margin = _gross_margin(income)
            debt = _total_debt(balance)
            assets = pick_first(balance, "totalAssets")
            leverage = _safe_divide(debt, assets)
            if margin is not None:
                margin_series.append(float(margin))
            if leverage is not None:
                leverage_series.append(float(leverage))

        if len(margin_series) >= 2:
            out["recovery_margin_inflection"] = float(margin_series[0] - np.mean(margin_series[1:]))
        if len(leverage_series) >= 2:
            out["recovery_leverage_improvement"] = float(np.mean(leverage_series[1:]) - leverage_series[0])
        return out

    yearly = _matched_statement_sets(fundamentals, "yearly")
    if len(yearly) < 2:
        return out

    current = yearly[0]
    previous = yearly[1]
    margin_current = _gross_margin(current["income"])
    margin_previous = _gross_margin(previous["income"])
    leverage_current = _safe_divide(_total_debt(current["balance"]), pick_first(current["balance"], "totalAssets"))
    leverage_previous = _safe_divide(_total_debt(previous["balance"]), pick_first(previous["balance"], "totalAssets"))

    if margin_current is not None and margin_previous is not None:
        out["recovery_margin_inflection"] = float(margin_current - margin_previous)
    if leverage_current is not None and leverage_previous is not None:
        out["recovery_leverage_improvement"] = float(leverage_previous - leverage_current)

    return out


def _cash_like_assets(balance: Dict[str, Any]) -> Optional[float]:
    combined = pick_first(balance, "cashAndShortTermInvestments")
    if combined is not None:
        return float(combined)

    cash = pick_first(
        balance,
        "cashAndCashEquivalents",
        "cash",
        "cashAndEquivalents",
    )
    short_term_investments = pick_first(balance, "shortTermInvestments")
    if cash is not None and short_term_investments is not None:
        return float(cash + short_term_investments)
    if cash is not None:
        return float(cash)
    if short_term_investments is not None:
        return float(short_term_investments)
    return None


def _ttm_cash_flow_snapshot(fundamentals: Dict[str, Any]) -> Dict[str, Any]:
    quarterly = _matched_statement_sets(fundamentals, "quarterly")
    if len(quarterly) >= 4:
        latest = quarterly[0]
        previous = quarterly[1] if len(quarterly) >= 2 else None
        cfo_values = [
            pick_first(entry["cashflow"], "totalCashFromOperatingActivities", "operatingCashFlow")
            for entry in quarterly[:4]
        ]
        capex_values = [_capital_expenditure_abs(entry["cashflow"]) for entry in quarterly[:4]]
        ttm_cfo = float(sum(float(value) for value in cfo_values)) if all(value is not None for value in cfo_values) else None
        ttm_capex = float(sum(float(value or 0.0) for value in capex_values)) if capex_values else None
        ttm_fcf = float(ttm_cfo - (ttm_capex or 0.0)) if ttm_cfo is not None else None
        latest_cfo = cfo_values[0]
        debt_latest = _total_debt(latest["balance"])
        debt_previous = _total_debt(previous["balance"]) if previous is not None else None
        debt_change = (
            (float(debt_previous) - float(debt_latest)) / max(abs(float(debt_previous)), 1.0)
            if debt_latest is not None and debt_previous not in (None, 0)
            else None
        )
        return {
            "ttm_fcf": ttm_fcf,
            "latest_cfo": latest_cfo,
            "latest_balance": latest["balance"],
            "debt_change": debt_change,
        }

    yearly = _matched_statement_sets(fundamentals, "yearly")
    if not yearly:
        return {
            "ttm_fcf": None,
            "latest_cfo": None,
            "latest_balance": None,
            "debt_change": None,
        }

    latest = yearly[0]
    previous = yearly[1] if len(yearly) >= 2 else None
    latest_cfo = pick_first(latest["cashflow"], "totalCashFromOperatingActivities", "operatingCashFlow")
    latest_capex = _capital_expenditure_abs(latest["cashflow"])
    ttm_fcf = float(latest_cfo - (latest_capex or 0.0)) if latest_cfo is not None else None
    debt_latest = _total_debt(latest["balance"])
    debt_previous = _total_debt(previous["balance"]) if previous is not None else None
    debt_change = (
        (float(debt_previous) - float(debt_latest)) / max(abs(float(debt_previous)), 1.0)
        if debt_latest is not None and debt_previous not in (None, 0)
        else None
    )
    return {
        "ttm_fcf": ttm_fcf,
        "latest_cfo": latest_cfo,
        "latest_balance": latest["balance"],
        "debt_change": debt_change,
    }


def _effective_total_assets(
    balance: Dict[str, Any],
    *,
    intangible_asset: float = 0.0,
) -> Optional[float]:
    total_assets = pick_first(balance, "totalAssets")
    if total_assets is None:
        return None
    adjusted_assets = float(total_assets) + max(0.0, float(intangible_asset))
    return adjusted_assets if adjusted_assets > 0 else None


def _effective_invested_capital(
    balance: Dict[str, Any],
    *,
    intangible_asset: float = 0.0,
) -> Optional[float]:
    equity = pick_first(
        balance,
        "totalStockholderEquity",
        "commonStockTotalEquity",
        "totalEquity",
        "bookValue",
    )
    debt = _total_debt(balance)
    cash_like = _cash_like_assets(balance)
    invested_capital = None
    if equity is not None or debt is not None:
        invested_capital = float((equity or 0.0) + (debt or 0.0) - (cash_like or 0.0))
    if invested_capital is None or invested_capital <= 0:
        total_assets = pick_first(balance, "totalAssets")
        current_liabilities = pick_first(balance, "totalCurrentLiabilities")
        if total_assets is not None:
            invested_capital = float(total_assets - (current_liabilities or 0.0) - (cash_like or 0.0))
    if invested_capital is None:
        return None
    invested_capital += max(0.0, float(intangible_asset))
    return invested_capital if invested_capital > 0 else None


def _free_cash_flow_conversion(net_income: Optional[float], cfo: Optional[float], capex_abs: Optional[float]) -> Optional[float]:
    if net_income is None or cfo is None:
        return None
    fcf = float(cfo) - float(capex_abs or 0.0)
    denominator = max(abs(float(net_income)), 1.0)
    return float(fcf / denominator)


def _series_inverse_std(series: pd.Series, scale: float) -> Optional[float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < 3 or scale <= 0:
        return None
    return _clip_score(1.0 - float(clean.std(ddof=0) / scale), 1.0)


def _series_delta(series: pd.Series, scale: float) -> Optional[float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < 2 or scale <= 0:
        return None
    return _clip_score(float(clean.iloc[-1] - clean.iloc[0]), scale)


def _series_latest(series: pd.Series, scale: float) -> Optional[float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty or scale <= 0:
        return None
    return _clip_score(float(clean.iloc[-1]), scale)


def _compute_estimate_term_structure_metrics_v2(
    fundamentals: Dict[str, Any],
    *,
    min_revision_analysts: int,
) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "estimate_term_structure_signal": None,
        "estimate_term_structure_has_coverage": 0.0,
        "estimate_term_structure_record_count": 0.0,
        "estimate_term_structure_persistence": None,
        "estimate_term_structure_improvement": None,
        "estimate_term_structure_disagreement_trend": None,
        "estimate_term_structure_coverage_component": None,
    }

    rows: List[Dict[str, float]] = []
    for trend in earnings_trend_records(fundamentals)[:6]:
        analyst_count = to_float(trend.get("earningsEstimateNumberOfAnalysts"))
        estimate_avg = to_float(trend.get("earningsEstimateAvg"))
        eps_current = to_float(trend.get("epsTrendCurrent"))
        eps_7d = to_float(trend.get("epsTrend7daysAgo"))
        eps_30d = to_float(trend.get("epsTrend30daysAgo"))
        rev_up_7d = to_float(trend.get("epsRevisionsUpLast7days"))
        rev_down_30d = to_float(trend.get("epsRevisionsDownLast30days"))
        estimate_high = to_float(trend.get("earningsEstimateHigh"))
        estimate_low = to_float(trend.get("earningsEstimateLow"))

        if analyst_count is None or analyst_count <= 0 or estimate_avg is None or eps_current is None:
            continue

        denominator = max(abs(float(estimate_avg)), 0.05)
        drift_7d = (float(eps_current) - float(eps_7d)) / denominator if eps_7d is not None else None
        drift_30d = (float(eps_current) - float(eps_30d)) / denominator if eps_30d is not None else None
        breadth = (
            (float(rev_up_7d) - float(rev_down_30d)) / max(float(analyst_count), 1.0)
            if rev_up_7d is not None and rev_down_30d is not None
            else None
        )
        disagreement = (
            _safe_divide((estimate_high - estimate_low), denominator)
            if estimate_high is not None and estimate_low is not None
            else None
        )
        row_signal = _mean_available(
            [
                _clip_score(drift_7d, 0.18),
                _clip_score(drift_30d, 0.22),
                _clip_score(breadth, 0.35),
            ]
        )
        if row_signal is None:
            continue

        rows.append(
            {
                "signal": float(row_signal),
                "breadth": float(breadth) if breadth is not None else float("nan"),
                "disagreement": float(disagreement) if disagreement is not None else float("nan"),
                "analyst_count": float(analyst_count),
            }
        )
        if len(rows) >= 4:
            break

    if len(rows) < 2:
        return out

    frame = pd.DataFrame(rows)
    signal_series = _winsorize_series(frame["signal"])
    breadth_series = _winsorize_series(frame["breadth"])
    disagreement_series = _winsorize_series(frame["disagreement"])

    persistence = float(signal_series.mean())
    improvement = float(signal_series.iloc[0] - signal_series.iloc[-1])
    disagreement_trend = None
    valid_disagreement = disagreement_series.dropna()
    if len(valid_disagreement) >= 2:
        disagreement_trend = float(valid_disagreement.iloc[-1] - valid_disagreement.iloc[0])
    breadth_level = float(breadth_series.dropna().mean()) if breadth_series.notna().any() else None
    coverage_component = float(
        min(1.0, frame["analyst_count"].mean() / max(1.0, float(min_revision_analysts)))
    )

    signal = _mean_available(
        [
            _clip_score(persistence, 0.55),
            _clip_score(improvement, 0.25),
            _clip_score(breadth_level, 0.30),
            _clip_score(disagreement_trend, 0.60),
        ]
    )
    if signal is None:
        return out

    out["estimate_term_structure_signal"] = float(np.clip(signal * coverage_component, -1.0, 1.0))
    out["estimate_term_structure_has_coverage"] = 1.0
    out["estimate_term_structure_record_count"] = float(len(frame))
    out["estimate_term_structure_persistence"] = persistence
    out["estimate_term_structure_improvement"] = improvement
    out["estimate_term_structure_disagreement_trend"] = disagreement_trend
    out["estimate_term_structure_coverage_component"] = coverage_component
    return out


def _compute_compounder_persistence_metrics_v2(
    fundamentals: Dict[str, Any],
    *,
    use_intangible_adjustments: bool,
) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "compounder_persistence_signal": None,
        "compounder_persistence_has_coverage": 0.0,
        "compounder_persistence_measure_count": 0.0,
        "compounder_persistence_level_component": None,
        "compounder_persistence_stability_component": None,
        "compounder_persistence_trend_component": None,
        "compounder_persistence_periodicity": None,
    }

    matched = _matched_statement_sets(fundamentals, "quarterly")
    periodicity = 1.0
    if len(matched) < 4:
        matched = _matched_statement_sets(fundamentals, "yearly")
        periodicity = 0.0
    if len(matched) < 4:
        return out

    intangible_asset = _capitalized_intangible_assets(fundamentals, use_intangible_adjustments)
    matched = list(reversed(matched[:8]))
    rows: List[Dict[str, float]] = []
    for idx, entry in enumerate(matched):
        previous_balance = matched[idx - 1]["balance"] if idx > 0 else None
        gross_margin = _gross_margin(entry["income"])
        asset_turnover = _asset_turnover(entry["income"], entry["balance"], previous_balance)
        net_income = pick_first(entry["income"], "netIncome")
        cfo = pick_first(entry["cashflow"], "totalCashFromOperatingActivities", "operatingCashFlow")
        cfo_consistency = _cfo_ni_consistency(net_income, cfo)
        capex_abs = _capital_expenditure_abs(entry["cashflow"])
        fcf_conversion = _free_cash_flow_conversion(net_income, cfo, capex_abs)
        effective_assets = _effective_total_assets(entry["balance"], intangible_asset=intangible_asset)
        effective_invested_capital = _effective_invested_capital(entry["balance"], intangible_asset=intangible_asset)
        roa = _safe_divide(net_income, effective_assets)
        roic = _safe_divide(net_income, effective_invested_capital)
        share_count = _share_count_from_record(entry["balance"])
        accrual_ratio = _accrual_ratio_from_set(entry, matched[idx - 1] if idx > 0 else None)

        row = {
            "gross_margin": float(gross_margin) if gross_margin is not None else np.nan,
            "asset_turnover": float(asset_turnover) if asset_turnover is not None else np.nan,
            "cfo_consistency": float(cfo_consistency) if cfo_consistency is not None else np.nan,
            "fcf_conversion": float(fcf_conversion) if fcf_conversion is not None else np.nan,
            "roa": float(roa) if roa is not None else np.nan,
            "roic": float(roic) if roic is not None else np.nan,
            "share_count": float(share_count) if share_count is not None else np.nan,
            "accrual_ratio": float(accrual_ratio) if accrual_ratio is not None else np.nan,
        }
        if pd.isna(pd.Series(row)).all():
            continue
        rows.append(row)

    if len(rows) < 4:
        return out

    frame = pd.DataFrame(rows)
    share_series = frame["share_count"].dropna()
    share_discipline_level = None
    if len(share_series) >= 2:
        share_discipline_level = _clip_score(
            (float(share_series.iloc[0]) - float(share_series.iloc[-1])) / max(abs(float(share_series.iloc[0])), 1.0),
            0.08,
        )

    level_components = [
        _series_latest(frame["gross_margin"], 0.60),
        _series_latest(frame["asset_turnover"], 1.20),
        _series_latest(frame["cfo_consistency"], 1.00),
        _series_latest(frame["fcf_conversion"], 1.20),
        _series_latest(frame["roa"], 0.12),
        _series_latest(frame["roic"], 0.18),
        share_discipline_level,
        _clip_score(
            None if frame["accrual_ratio"].dropna().empty else -float(frame["accrual_ratio"].dropna().iloc[-1]),
            0.10,
        ),
    ]
    level_score = _mean_available(level_components)

    stability_components = [
        _series_inverse_std(frame["gross_margin"], 0.08),
        _series_inverse_std(frame["asset_turnover"], 0.18),
        _series_inverse_std(frame["cfo_consistency"], 0.25),
        _series_inverse_std(frame["fcf_conversion"], 0.45),
        _series_inverse_std(frame["roa"], 0.04),
        _series_inverse_std(frame["roic"], 0.06),
        _series_inverse_std(frame["accrual_ratio"], 0.05),
    ]
    if len(share_series) >= 3:
        share_std = share_series.pct_change(fill_method=None).dropna().std(ddof=0)
        if pd.notna(share_std):
            stability_components.append(_clip_score(1.0 - float(share_std / 0.04), 1.0))
    stability_score = _mean_available(stability_components)

    trend_components = [
        _series_delta(frame["gross_margin"], 0.06),
        _series_delta(frame["asset_turnover"], 0.15),
        _series_delta(frame["cfo_consistency"], 0.25),
        _series_delta(frame["fcf_conversion"], 0.50),
        _series_delta(frame["roa"], 0.04),
        _series_delta(frame["roic"], 0.05),
        _clip_score(
            None
            if len(share_series) < 2
            else (float(share_series.iloc[0]) - float(share_series.iloc[-1])) / max(abs(float(share_series.iloc[0])), 1.0),
            0.08,
        ),
        _clip_score(
            None
            if frame["accrual_ratio"].dropna().shape[0] < 2
            else float(frame["accrual_ratio"].dropna().iloc[0] - frame["accrual_ratio"].dropna().iloc[-1]),
            0.05,
        ),
    ]
    trend_score = _mean_available(trend_components)

    family_coverage = {
        "margins": frame["gross_margin"].notna().sum() >= 2,
        "efficiency": frame["asset_turnover"].notna().sum() >= 2,
        "cash_conversion": frame["cfo_consistency"].notna().sum() >= 2 or frame["fcf_conversion"].notna().sum() >= 2,
        "returns": frame["roa"].notna().sum() >= 2 or frame["roic"].notna().sum() >= 2,
        "dilution": frame["share_count"].notna().sum() >= 2,
        "accruals": frame["accrual_ratio"].notna().sum() >= 2,
    }
    family_count = sum(bool(flag) for flag in family_coverage.values())
    if family_count < 3:
        return out

    signal = _mean_available([level_score, stability_score, trend_score])
    if signal is None:
        return out

    coverage = float(min(1.0, family_count / 6.0))
    out["compounder_persistence_signal"] = float(np.clip(signal * coverage, -1.0, 1.0))
    out["compounder_persistence_has_coverage"] = coverage
    out["compounder_persistence_measure_count"] = float(len(frame))
    out["compounder_persistence_level_component"] = level_score
    out["compounder_persistence_stability_component"] = stability_score
    out["compounder_persistence_trend_component"] = trend_score
    out["compounder_persistence_periodicity"] = periodicity
    return out


def _compute_working_capital_stress_metrics_v2(
    fundamentals: Dict[str, Any],
) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "working_capital_stress_penalty": 0.0,
        "working_capital_stress_has_coverage": 0.0,
        "working_capital_receivables_stress": None,
        "working_capital_inventory_stress": None,
        "working_capital_payables_stress": None,
        "working_capital_cfo_stress": None,
        "working_capital_cycle_stress": None,
    }

    matched = _matched_statement_sets(fundamentals, "quarterly")
    period_days = 90.0
    if len(matched) < 2:
        matched = _matched_statement_sets(fundamentals, "yearly")
        period_days = 365.0
    if len(matched) < 2:
        return out

    window = list(reversed(matched[:5]))
    rows: List[Dict[str, float]] = []
    for idx, entry in enumerate(window):
        previous = window[idx - 1] if idx > 0 else None
        turnover_days = _turnover_days_from_entry(
            entry["income"],
            entry["balance"],
            previous["balance"] if previous is not None else None,
            period_days,
        )
        cycle_days = _cash_conversion_cycle_days(
            turnover_days["receivables_days"],
            turnover_days["inventory_days"],
            turnover_days["payables_days"],
        )
        net_income = pick_first(entry["income"], "netIncome")
        cfo = pick_first(entry["cashflow"], "totalCashFromOperatingActivities", "operatingCashFlow")
        cfo_gap = _safe_divide(
            (net_income - cfo) if net_income is not None and cfo is not None else None,
            max(abs(float(net_income)), 1.0) if net_income is not None else None,
        )

        row = {
            "receivables_days": float(turnover_days["receivables_days"]) if turnover_days["receivables_days"] is not None else np.nan,
            "inventory_days": float(turnover_days["inventory_days"]) if turnover_days["inventory_days"] is not None else np.nan,
            "payables_days": float(turnover_days["payables_days"]) if turnover_days["payables_days"] is not None else np.nan,
            "cash_conversion_cycle_days": float(cycle_days) if cycle_days is not None else np.nan,
            "cfo_stress": max(0.0, float(cfo_gap)) if cfo_gap is not None else np.nan,
        }
        if pd.isna(pd.Series(row)).all():
            continue
        rows.append(row)

    if len(rows) < 2:
        return out

    frame = pd.DataFrame(rows)
    delta_frame = pd.DataFrame(
        {
            "receivables_stress": frame["receivables_days"].diff(),
            "inventory_stress": frame["inventory_days"].diff(),
            "payables_stress": frame["payables_days"].diff(),
            "cash_conversion_cycle_delta": frame["cash_conversion_cycle_days"].diff(),
        }
    )

    penalties: List[float] = []
    averages: Dict[str, Optional[float]] = {}
    for column, scale in [
        ("receivables_stress", 18.0),
        ("inventory_stress", 22.0),
        ("payables_stress", 15.0),
    ]:
        series = pd.to_numeric(delta_frame[column], errors="coerce").dropna()
        if series.empty:
            averages[column] = None
            continue
        positive_series = series.clip(lower=0.0)
        averages[column] = float(positive_series.iloc[-1])
        persistence = float((series > 0.0).mean())
        composite = float(positive_series.mean()) * (0.70 + 0.30 * persistence)
        penalty = _clip_score(composite, scale)
        if penalty is not None:
            penalties.append(float(max(0.0, penalty)))

    cfo_series = pd.to_numeric(frame["cfo_stress"], errors="coerce").dropna()
    averages["cfo_stress"] = float(cfo_series.iloc[-1]) if not cfo_series.empty else None
    if not cfo_series.empty:
        cfo_persistence = float((cfo_series > 0.05).mean())
        cfo_composite = float(cfo_series.mean()) * (0.75 + 0.25 * cfo_persistence)
        cfo_penalty = _clip_score(cfo_composite, 0.25)
        if cfo_penalty is not None:
            penalties.append(float(max(0.0, cfo_penalty)))

    ccc_delta_series = pd.to_numeric(delta_frame["cash_conversion_cycle_delta"], errors="coerce").dropna()
    ccc_convexity_series = ccc_delta_series.diff().dropna()
    latest_ccc_delta = float(ccc_delta_series.iloc[-1]) if not ccc_delta_series.empty else None
    latest_ccc_convexity = float(ccc_convexity_series.iloc[-1]) if not ccc_convexity_series.empty else None
    working_capital_cycle_stress = _mean_available(
        [
            _clip_score(max(0.0, latest_ccc_delta), 20.0) if latest_ccc_delta is not None else None,
            _clip_score(max(0.0, latest_ccc_convexity), 12.0) if latest_ccc_convexity is not None else None,
        ]
    )
    if working_capital_cycle_stress is not None:
        penalties.append(float(max(0.0, working_capital_cycle_stress)))

    if len(penalties) < 2:
        return out

    out["working_capital_stress_penalty"] = float(min(0.06, max(0.0, 0.06 * sum(penalties) / len(penalties))))
    out["working_capital_stress_has_coverage"] = float(min(1.0, len(penalties) / 5.0))
    out["working_capital_receivables_stress"] = averages["receivables_stress"]
    out["working_capital_inventory_stress"] = averages["inventory_stress"]
    out["working_capital_payables_stress"] = averages["payables_stress"]
    out["working_capital_cfo_stress"] = averages["cfo_stress"]
    out["working_capital_cycle_stress"] = working_capital_cycle_stress
    return out


def _compute_capital_allocation_quality_metrics_v2(
    fundamentals: Dict[str, Any],
) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "capital_allocation_quality_signal": None,
        "capital_allocation_quality_has_coverage": 0.0,
        "capital_allocation_buyback_component": None,
        "capital_allocation_funding_component": None,
        "capital_allocation_debt_component": None,
        "capital_allocation_payout_component": None,
        "capital_allocation_reinvestment_component": None,
        "capital_allocation_financing_dependency_component": None,
        "financing_dependency_stress": None,
        "financing_dependency_burn_component": None,
        "financing_dependency_dilution_component": None,
        "financing_dependency_debt_component": None,
        "financing_dependency_revision_component": None,
    }

    matched = _matched_statement_sets(fundamentals, "yearly")
    if len(matched) < 2:
        return out

    latest = matched[0]
    previous = matched[1]
    share_drift_metrics = compute_share_drift_metrics_from_fundamentals(
        fundamentals,
        alpha_factor_spec="v2",
    )
    share_drift_1q = to_float(share_drift_metrics.get("share_drift_1q"))
    share_drift_4q = to_float(share_drift_metrics.get("share_drift_4q"))
    share_drift_persistence = to_float(share_drift_metrics.get("share_drift_persistence"))

    drift_components: List[tuple[float, float]] = []
    if share_drift_4q is not None:
        drift_components.append((0.70, float(share_drift_4q)))
    if share_drift_1q is not None:
        drift_components.append((0.30, float(share_drift_1q)))
    share_change = None
    if drift_components:
        total_weight = float(sum(weight for weight, _ in drift_components))
        share_change = float(sum(weight * value for weight, value in drift_components) / total_weight)

    debt_latest = _total_debt(latest["balance"])
    debt_previous = _total_debt(previous["balance"])
    debt_change = (
        (float(debt_previous) - float(debt_latest)) / max(abs(float(debt_previous)), 1.0)
        if debt_latest is not None and debt_previous not in (None, 0)
        else None
    )
    cfo = pick_first(latest["cashflow"], "totalCashFromOperatingActivities", "operatingCashFlow")
    capex_abs = _capital_expenditure_abs(latest["cashflow"])
    net_income = pick_first(latest["income"], "netIncome")
    total_assets = pick_first(latest["balance"], "totalAssets")
    revenue_latest = pick_first(latest["income"], "totalRevenue")
    revenue_previous = pick_first(previous["income"], "totalRevenue")
    revenue_growth = _safe_divide(
        (revenue_latest - revenue_previous) if revenue_latest is not None and revenue_previous is not None else None,
        abs(float(revenue_previous)) if revenue_previous not in (None, 0) else None,
    )
    fcf = float(cfo - (capex_abs or 0.0)) if cfo is not None else None
    fcf_conversion = _free_cash_flow_conversion(net_income, cfo, capex_abs)
    capex_intensity = _safe_divide(capex_abs, total_assets)
    payout_ratio = pick_first(fundamentals.get("Highlights") or {}, "PayoutRatio")
    financing_snapshot = _ttm_cash_flow_snapshot(fundamentals)
    ttm_fcf = to_float(financing_snapshot.get("ttm_fcf"))
    latest_cfo = to_float(financing_snapshot.get("latest_cfo"))
    financing_balance = financing_snapshot.get("latest_balance")
    financing_debt_change = to_float(financing_snapshot.get("debt_change"))
    cash_like = _cash_like_assets(financing_balance) if isinstance(financing_balance, dict) else None
    revision_metrics = compute_revision_impulse_metrics_from_fundamentals(
        fundamentals,
        min_revision_analysts=1,
        alpha_factor_spec="v2",
    )
    revision_breadth_7d = to_float(revision_metrics.get("revision_breadth_7d"))
    drift_30d = to_float(revision_metrics.get("revision_impulse_drift_30d"))

    buyback_signal = share_change
    if buyback_signal is not None and share_drift_persistence is not None:
        buyback_signal = float(buyback_signal) * (0.70 + 0.30 * float(share_drift_persistence))
    buyback_component = _clip_score(buyback_signal, 0.08)
    funding_component = None
    if share_change is not None and fcf is not None:
        funding_signal = float(fcf / max(abs(float(total_assets or 0.0)) * 0.04, 1.0)) + 0.60 * float(share_change)
        if debt_change is not None and share_change > 0:
            funding_signal += 0.50 * float(debt_change)
        if fcf_conversion is not None:
            funding_signal += 0.30 * float(np.clip(fcf_conversion, -1.0, 1.0))
        if share_drift_persistence is not None:
            funding_signal += 0.20 * float(np.clip(share_drift_persistence - 0.50, -1.0, 1.0))
        funding_component = _clip_score(funding_signal, 1.0)
    debt_component = _clip_score(debt_change, 0.10)
    payout_component = None
    if payout_ratio is not None:
        payout_quality = 1.0 - float(payout_ratio)
        if fcf_conversion is not None and fcf_conversion < 0:
            payout_quality += float(fcf_conversion)
        payout_component = _clip_score(payout_quality, 0.60)
    reinvestment_component = None
    if capex_intensity is not None:
        reinvestment_signal = float(np.clip(capex_intensity / 0.10, -1.0, 1.0))
        if revenue_growth is not None:
            reinvestment_signal = 0.55 * float(np.clip(revenue_growth / 0.20, -1.0, 1.0)) + 0.45 * reinvestment_signal
        reinvestment_component = float(np.clip(reinvestment_signal, -1.0, 1.0))

    financing_burn_component = None
    if ttm_fcf is not None:
        burn_ratio = max(0.0, -float(ttm_fcf) / max(abs(float(cash_like or 0.0)), 1.0))
        financing_burn_component = _clip_score(burn_ratio, 0.75)
    financing_dilution_component = _clip_score(max(0.0, -float(share_change)), 0.08) if share_change is not None else None
    financing_debt_component = (
        _clip_score(max(0.0, -float(financing_debt_change)), 0.25)
        if financing_debt_change is not None
        else None
    )
    financing_revision_component = _mean_available(
        [
            _clip_score(max(0.0, -float(revision_breadth_7d)), 0.60) if revision_breadth_7d is not None else None,
            _clip_score(max(0.0, -float(drift_30d)), 0.20) if drift_30d is not None else None,
        ]
    )

    financing_dependency_stress = None
    if ttm_fcf is not None and latest_cfo is not None and ttm_fcf >= 0 and latest_cfo >= 0:
        financing_dependency_stress = 0.0
    elif financing_burn_component is not None:
        secondary_components = [
            component
            for component in [
                financing_dilution_component,
                financing_debt_component,
                financing_revision_component,
            ]
            if component is not None
        ]
        if secondary_components:
            financing_dependency_stress = _mean_available(
                [financing_burn_component] + [float(component) for component in secondary_components]
            )
    financing_dependency_component = (
        -float(financing_dependency_stress)
        if financing_dependency_stress is not None
        else None
    )

    components = [
        buyback_component,
        funding_component,
        debt_component,
        payout_component,
        reinvestment_component,
        financing_dependency_component,
    ]
    usable_components = [float(component) for component in components if component is not None]
    if len(usable_components) < 2:
        return out

    out["capital_allocation_quality_signal"] = float(sum(usable_components) / len(usable_components))
    out["capital_allocation_quality_has_coverage"] = float(min(1.0, len(usable_components) / 6.0))
    out["capital_allocation_buyback_component"] = buyback_component
    out["capital_allocation_funding_component"] = funding_component
    out["capital_allocation_debt_component"] = debt_component
    out["capital_allocation_payout_component"] = payout_component
    out["capital_allocation_reinvestment_component"] = reinvestment_component
    out["capital_allocation_financing_dependency_component"] = financing_dependency_component
    out["financing_dependency_stress"] = financing_dependency_stress
    out["financing_dependency_burn_component"] = financing_burn_component
    out["financing_dependency_dilution_component"] = financing_dilution_component
    out["financing_dependency_debt_component"] = financing_debt_component
    out["financing_dependency_revision_component"] = financing_revision_component
    return out


def _compute_recovery_fundamental_metrics_v2(
    fundamentals: Dict[str, Any],
) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "recovery_margin_inflection": None,
        "recovery_leverage_improvement": None,
        "recovery_accrual_improvement": None,
    }

    matched = _matched_statement_sets(fundamentals, "quarterly")
    if len(matched) < 4:
        matched = _matched_statement_sets(fundamentals, "yearly")
    if len(matched) < 2:
        return out

    window = matched[:4]
    margin_series: List[float] = []
    leverage_series: List[float] = []
    accrual_series: List[float] = []
    for idx, entry in enumerate(window):
        margin = _gross_margin(entry["income"])
        leverage = _safe_divide(_total_debt(entry["balance"]), pick_first(entry["balance"], "totalAssets"))
        accrual = _accrual_ratio_from_set(entry, window[idx + 1] if idx + 1 < len(window) else None)
        if margin is not None:
            margin_series.append(float(margin))
        if leverage is not None:
            leverage_series.append(float(leverage))
        if accrual is not None:
            accrual_series.append(float(accrual))

    if len(margin_series) >= 2:
        out["recovery_margin_inflection"] = float(margin_series[0] - np.mean(margin_series[1:]))
    if len(leverage_series) >= 2:
        out["recovery_leverage_improvement"] = float(np.mean(leverage_series[1:]) - leverage_series[0])
    if len(accrual_series) >= 2:
        out["recovery_accrual_improvement"] = float(np.mean(accrual_series[1:]) - accrual_series[0])

    return out


def compute_investment_restraint_metrics(
    fundamentals: Dict[str, Any],
) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "investment_restraint_signal": None,
        "investment_restraint_has_coverage": 0.0,
        "investment_restraint_measure_count": 0.0,
        "investment_restraint_asset_growth": None,
        "investment_restraint_noa_growth": None,
        "investment_restraint_acquisition_intensity": None,
        "investment_restraint_capex_intensity": None,
        "investment_restraint_share_issuance": None,
        "investment_restraint_debt_funded_expansion": None,
    }

    matched = _matched_statement_sets(fundamentals, "yearly")
    if len(matched) < 2:
        return out

    current = matched[0]
    previous = matched[1]
    income_0, income_1 = current["income"], previous["income"]
    balance_0, balance_1 = current["balance"], previous["balance"]
    cashflow_0 = current["cashflow"]

    assets_0 = pick_first(balance_0, "totalAssets")
    assets_1 = pick_first(balance_1, "totalAssets")
    asset_growth = _safe_divide(
        (assets_0 - assets_1) if assets_0 is not None and assets_1 is not None else None,
        abs(float(assets_1)) if assets_1 not in (None, 0) else None,
    )

    noa_0 = _effective_invested_capital(balance_0, intangible_asset=0.0)
    noa_1 = _effective_invested_capital(balance_1, intangible_asset=0.0)
    noa_growth = _safe_divide(
        (noa_0 - noa_1) if noa_0 is not None and noa_1 is not None else None,
        abs(float(noa_1)) if noa_1 not in (None, 0) else None,
    )

    goodwill_0 = _goodwill_and_intangible_assets(balance_0)
    goodwill_1 = _goodwill_and_intangible_assets(balance_1)
    acquisition_intensity = _safe_divide(
        (goodwill_0 - goodwill_1) if goodwill_0 is not None and goodwill_1 is not None else None,
        max(abs(float(assets_1)), 1.0) if assets_1 is not None else None,
    )

    debt_0 = _total_debt(balance_0)
    debt_1 = _total_debt(balance_1)
    debt_growth = _safe_divide(
        (debt_0 - debt_1) if debt_0 is not None and debt_1 is not None else None,
        abs(float(debt_1)) if debt_1 not in (None, 0) else None,
    )

    shares_0 = _share_count_from_record(balance_0)
    shares_1 = _share_count_from_record(balance_1)
    share_issuance = None
    if shares_0 is not None and shares_1 not in (None, 0):
        share_issuance = max(0.0, float((float(shares_0) - float(shares_1)) / max(abs(float(shares_1)), 1.0)))

    capex_abs = _capital_expenditure_abs(cashflow_0)
    capex_intensity = _safe_divide(capex_abs, assets_0)

    revenue_0 = pick_first(income_0, "totalRevenue")
    revenue_1 = pick_first(income_1, "totalRevenue")
    revenue_growth = _safe_divide(
        (revenue_0 - revenue_1) if revenue_0 is not None and revenue_1 is not None else None,
        abs(float(revenue_1)) if revenue_1 not in (None, 0) else None,
    )

    debt_funded_expansion = None
    if asset_growth is not None or debt_growth is not None:
        debt_funded_expansion = max(0.0, float(debt_growth or 0.0)) + 0.50 * max(0.0, float(asset_growth or 0.0))

    capex_discipline_component = None
    if capex_intensity is not None:
        discipline_signal = float(revenue_growth or 0.0) - 0.70 * float(capex_intensity)
        if debt_growth is not None and debt_growth > 0:
            discipline_signal -= 0.50 * float(debt_growth)
        if acquisition_intensity is not None and acquisition_intensity > 0:
            discipline_signal -= 0.50 * float(acquisition_intensity)
        capex_discipline_component = _clip_score(discipline_signal, 0.20)

    components = [
        _clip_score(-max(0.0, float(asset_growth)), 0.25) if asset_growth is not None else None,
        _clip_score(-max(0.0, float(noa_growth)), 0.22) if noa_growth is not None else None,
        _clip_score(-max(0.0, float(acquisition_intensity)), 0.12) if acquisition_intensity is not None else None,
        _clip_score(-float(debt_funded_expansion), 0.25) if debt_funded_expansion is not None else None,
        capex_discipline_component,
        _clip_score(-float(share_issuance), 0.05) if share_issuance is not None else None,
    ]
    usable_components = [float(component) for component in components if component is not None]
    if len(usable_components) < 3:
        return out

    out["investment_restraint_signal"] = float(np.clip(sum(usable_components) / len(usable_components), -1.0, 1.0))
    out["investment_restraint_has_coverage"] = float(min(1.0, len(usable_components) / 6.0))
    out["investment_restraint_measure_count"] = float(len(usable_components))
    out["investment_restraint_asset_growth"] = asset_growth
    out["investment_restraint_noa_growth"] = noa_growth
    out["investment_restraint_acquisition_intensity"] = acquisition_intensity
    out["investment_restraint_capex_intensity"] = capex_intensity
    out["investment_restraint_share_issuance"] = share_issuance
    out["investment_restraint_debt_funded_expansion"] = debt_funded_expansion
    return out


def compute_accrual_quality_metrics(
    fundamentals: Dict[str, Any],
    alpha_factor_spec: str = "legacy",
) -> Dict[str, Optional[float]]:
    if str(alpha_factor_spec).lower() == "v2":
        return _compute_accrual_quality_metrics_v2(fundamentals)

    out: Dict[str, Optional[float]] = {
        "accrual_quality_signal": None,
        "accrual_quality_has_coverage": 0.0,
        "accrual_quality_measure_count": 0.0,
        "accrual_quality_level_component": None,
        "accrual_quality_stability_component": None,
        "accrual_quality_trend_component": None,
        "accrual_quality_periodicity": None,
        "accrual_quality_cash_conversion": None,
        "accrual_quality_margin_gap": None,
        "accrual_quality_working_capital_stretch": None,
        "accrual_quality_cycle_convexity": None,
    }

    matched = _matched_statement_sets(fundamentals, "quarterly")
    periodicity = 1.0
    if len(matched) < 4:
        matched = _matched_statement_sets(fundamentals, "yearly")
        periodicity = 0.0
    if len(matched) < 4:
        return out

    matched = list(reversed(matched[:8]))
    rows: List[Dict[str, float]] = []
    for idx, entry in enumerate(matched):
        previous = matched[idx - 1] if idx > 0 else None
        income = entry["income"]
        balance = entry["balance"]
        cashflow = entry["cashflow"]

        accrual_ratio = _accrual_ratio_from_set(entry, previous)
        revenue = pick_first(income, "totalRevenue")
        cfo = pick_first(cashflow, "totalCashFromOperatingActivities", "operatingCashFlow")
        ebit = _ebit_value(income)
        cash_conversion = _safe_divide(cfo, ebit) if ebit not in (None, 0) else None
        cfo_margin = _safe_divide(cfo, revenue)
        ebit_margin = _safe_divide(ebit, revenue)
        margin_gap = (
            float(cfo_margin - ebit_margin)
            if cfo_margin is not None and ebit_margin is not None
            else None
        )

        working_capital_stretch = None
        if previous is not None:
            prev_income = previous["income"]
            prev_balance = previous["balance"]
            sales_growth = _safe_divide(
                (revenue - pick_first(prev_income, "totalRevenue"))
                if revenue is not None and pick_first(prev_income, "totalRevenue") is not None
                else None,
                abs(float(pick_first(prev_income, "totalRevenue")))
                if pick_first(prev_income, "totalRevenue") not in (None, 0)
                else None,
            )
            receivables_growth = _safe_divide(
                (pick_first(balance, "netReceivables", "accountsReceivable") - pick_first(prev_balance, "netReceivables", "accountsReceivable"))
                if pick_first(balance, "netReceivables", "accountsReceivable") is not None
                and pick_first(prev_balance, "netReceivables", "accountsReceivable") is not None
                else None,
                abs(float(pick_first(prev_balance, "netReceivables", "accountsReceivable")))
                if pick_first(prev_balance, "netReceivables", "accountsReceivable") not in (None, 0)
                else None,
            )
            inventory_growth = _safe_divide(
                (_inventory_value(balance) - _inventory_value(prev_balance))
                if _inventory_value(balance) is not None and _inventory_value(prev_balance) is not None
                else None,
                abs(float(_inventory_value(prev_balance))) if _inventory_value(prev_balance) not in (None, 0) else None,
            )
            stretch_components = [
                max(0.0, float(receivables_growth - sales_growth))
                if receivables_growth is not None and sales_growth is not None
                else None,
                max(0.0, float(inventory_growth - sales_growth))
                if inventory_growth is not None and sales_growth is not None
                else None,
            ]
            usable_stretch = [float(component) for component in stretch_components if component is not None]
            if usable_stretch:
                working_capital_stretch = float(sum(usable_stretch) / len(usable_stretch))

        row = {
            "accrual_ratio": float(accrual_ratio) if accrual_ratio is not None else np.nan,
            "cash_conversion": float(cash_conversion) if cash_conversion is not None else np.nan,
            "margin_gap": float(margin_gap) if margin_gap is not None else np.nan,
            "working_capital_stretch": float(working_capital_stretch) if working_capital_stretch is not None else np.nan,
        }
        if pd.isna(pd.Series(row)).all():
            continue
        rows.append(row)

    if len(rows) < 4:
        return out

    frame = pd.DataFrame(rows)
    latest_stretch = None if frame["working_capital_stretch"].dropna().empty else float(frame["working_capital_stretch"].dropna().iloc[-1])
    level_components = [
        _clip_score(
            None if frame["accrual_ratio"].dropna().empty else -float(frame["accrual_ratio"].dropna().iloc[-1]),
            0.08,
        ),
        _series_latest(frame["cash_conversion"], 1.20),
        _series_latest(frame["margin_gap"], 0.08),
        _clip_score(-latest_stretch, 0.12) if latest_stretch is not None else None,
    ]
    stability_components = [
        _series_inverse_std(frame["accrual_ratio"], 0.04),
        _series_inverse_std(frame["cash_conversion"], 0.45),
        _series_inverse_std(frame["margin_gap"], 0.05),
        _series_inverse_std(frame["working_capital_stretch"], 0.08),
    ]
    working_capital_series = frame["working_capital_stretch"].dropna()
    trend_components = [
        _clip_score(
            None
            if frame["accrual_ratio"].dropna().shape[0] < 2
            else float(frame["accrual_ratio"].dropna().iloc[0] - frame["accrual_ratio"].dropna().iloc[-1]),
            0.05,
        ),
        _series_delta(frame["cash_conversion"], 0.50),
        _series_delta(frame["margin_gap"], 0.05),
        _clip_score(
            None
            if len(working_capital_series) < 2
            else float(working_capital_series.iloc[0] - working_capital_series.iloc[-1]),
            0.10,
        ),
    ]

    level_score = _mean_available(level_components)
    stability_score = _mean_available(stability_components)
    trend_score = _mean_available(trend_components)
    family_coverage = {
        "accruals": frame["accrual_ratio"].notna().sum() >= 2,
        "cash_conversion": frame["cash_conversion"].notna().sum() >= 2,
        "margins": frame["margin_gap"].notna().sum() >= 2,
        "working_capital": frame["working_capital_stretch"].notna().sum() >= 2,
    }
    family_count = sum(bool(flag) for flag in family_coverage.values())
    if family_count < 3:
        return out

    signal = _mean_available([level_score, stability_score, trend_score])
    if signal is None:
        return out

    out["accrual_quality_signal"] = float(np.clip(signal * min(1.0, family_count / 4.0), -1.0, 1.0))
    out["accrual_quality_has_coverage"] = float(min(1.0, family_count / 4.0))
    out["accrual_quality_measure_count"] = float(len(frame))
    out["accrual_quality_level_component"] = level_score
    out["accrual_quality_stability_component"] = stability_score
    out["accrual_quality_trend_component"] = trend_score
    out["accrual_quality_periodicity"] = periodicity
    out["accrual_quality_cash_conversion"] = (
        None if frame["cash_conversion"].dropna().empty else float(frame["cash_conversion"].dropna().iloc[-1])
    )
    out["accrual_quality_margin_gap"] = (
        None if frame["margin_gap"].dropna().empty else float(frame["margin_gap"].dropna().iloc[-1])
    )
    out["accrual_quality_working_capital_stretch"] = latest_stretch
    return out


def _compute_accrual_quality_metrics_v2(
    fundamentals: Dict[str, Any],
) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "accrual_quality_signal": None,
        "accrual_quality_has_coverage": 0.0,
        "accrual_quality_measure_count": 0.0,
        "accrual_quality_level_component": None,
        "accrual_quality_stability_component": None,
        "accrual_quality_trend_component": None,
        "accrual_quality_periodicity": None,
        "accrual_quality_cash_conversion": None,
        "accrual_quality_margin_gap": None,
        "accrual_quality_working_capital_stretch": None,
        "accrual_quality_cycle_convexity": None,
    }

    matched = _matched_statement_sets(fundamentals, "quarterly")
    periodicity = 1.0
    period_days = 90.0
    if len(matched) < 4:
        matched = _matched_statement_sets(fundamentals, "yearly")
        periodicity = 0.0
        period_days = 365.0
    if len(matched) < 4:
        return out

    matched = list(reversed(matched[:8]))
    rows: List[Dict[str, float]] = []
    for idx, entry in enumerate(matched):
        previous = matched[idx - 1] if idx > 0 else None
        income = entry["income"]
        balance = entry["balance"]
        cashflow = entry["cashflow"]

        accrual_ratio = _accrual_ratio_from_set(entry, previous)
        revenue = pick_first(income, "totalRevenue")
        cfo = pick_first(cashflow, "totalCashFromOperatingActivities", "operatingCashFlow")
        ebit = _ebit_value(income)
        cash_conversion = _safe_divide(cfo, ebit) if ebit not in (None, 0) else None
        cfo_margin = _safe_divide(cfo, revenue)
        ebit_margin = _safe_divide(ebit, revenue)
        margin_gap = float(cfo_margin - ebit_margin) if cfo_margin is not None and ebit_margin is not None else None
        turnover_days = _turnover_days_from_entry(
            income,
            balance,
            previous["balance"] if previous is not None else None,
            period_days,
        )
        cycle_days = _cash_conversion_cycle_days(
            turnover_days["receivables_days"],
            turnover_days["inventory_days"],
            turnover_days["payables_days"],
        )

        row = {
            "accrual_ratio": float(accrual_ratio) if accrual_ratio is not None else np.nan,
            "cash_conversion": float(cash_conversion) if cash_conversion is not None else np.nan,
            "margin_gap": float(margin_gap) if margin_gap is not None else np.nan,
            "receivables_days": float(turnover_days["receivables_days"]) if turnover_days["receivables_days"] is not None else np.nan,
            "inventory_days": float(turnover_days["inventory_days"]) if turnover_days["inventory_days"] is not None else np.nan,
            "cash_conversion_cycle_days": float(cycle_days) if cycle_days is not None else np.nan,
        }
        if pd.isna(pd.Series(row)).all():
            continue
        rows.append(row)

    if len(rows) < 4:
        return out

    frame = pd.DataFrame(rows)
    receivables_delta = frame["receivables_days"].diff()
    inventory_delta = frame["inventory_days"].diff()
    cash_conversion_cycle_delta = frame["cash_conversion_cycle_days"].diff()
    cycle_convexity_series = cash_conversion_cycle_delta.diff()
    stretch_frame = pd.DataFrame(
        {
            "receivables": receivables_delta.clip(lower=0.0),
            "inventory": inventory_delta.clip(lower=0.0),
            "cash_conversion_cycle": cash_conversion_cycle_delta.clip(lower=0.0),
        }
    )
    frame["working_capital_stretch"] = stretch_frame.mean(axis=1, skipna=True)

    latest_stretch = None
    clean_stretch = frame["working_capital_stretch"].dropna()
    if not clean_stretch.empty:
        latest_stretch = float(clean_stretch.iloc[-1])
    latest_cycle_convexity = None
    clean_cycle_convexity = pd.to_numeric(cycle_convexity_series, errors="coerce").dropna()
    if not clean_cycle_convexity.empty:
        latest_cycle_convexity = float(clean_cycle_convexity.iloc[-1])

    level_components = [
        _clip_score(
            None if frame["accrual_ratio"].dropna().empty else -float(frame["accrual_ratio"].dropna().iloc[-1]),
            0.08,
        ),
        _series_latest(frame["cash_conversion"], 1.20),
        _series_latest(frame["margin_gap"], 0.08),
        _clip_score(-latest_stretch, 15.0) if latest_stretch is not None else None,
    ]
    stability_components = [
        _series_inverse_std(frame["accrual_ratio"], 0.04),
        _series_inverse_std(frame["cash_conversion"], 0.45),
        _series_inverse_std(frame["margin_gap"], 0.05),
        _series_inverse_std(frame["working_capital_stretch"], 8.0),
    ]
    working_capital_series = frame["working_capital_stretch"].dropna()
    trend_components = [
        _clip_score(
            None
            if frame["accrual_ratio"].dropna().shape[0] < 2
            else float(frame["accrual_ratio"].dropna().iloc[0] - frame["accrual_ratio"].dropna().iloc[-1]),
            0.05,
        ),
        _series_delta(frame["cash_conversion"], 0.50),
        _series_delta(frame["margin_gap"], 0.05),
        _clip_score(
            None
            if len(working_capital_series) < 2
            else float(working_capital_series.iloc[0] - working_capital_series.iloc[-1]),
            12.0,
        ),
        _clip_score(-latest_cycle_convexity, 12.0) if latest_cycle_convexity is not None else None,
    ]

    level_score = _mean_available(level_components)
    stability_score = _mean_available(stability_components)
    trend_score = _mean_available(trend_components)
    family_coverage = {
        "accruals": frame["accrual_ratio"].notna().sum() >= 2,
        "cash_conversion": frame["cash_conversion"].notna().sum() >= 2,
        "margins": frame["margin_gap"].notna().sum() >= 2,
        "working_capital": frame["working_capital_stretch"].notna().sum() >= 2,
    }
    family_count = sum(bool(flag) for flag in family_coverage.values())
    if family_count < 3:
        return out

    signal = _mean_available([level_score, stability_score, trend_score])
    if signal is None:
        return out

    out["accrual_quality_signal"] = float(np.clip(signal * min(1.0, family_count / 4.0), -1.0, 1.0))
    out["accrual_quality_has_coverage"] = float(min(1.0, family_count / 4.0))
    out["accrual_quality_measure_count"] = float(len(frame))
    out["accrual_quality_level_component"] = level_score
    out["accrual_quality_stability_component"] = stability_score
    out["accrual_quality_trend_component"] = trend_score
    out["accrual_quality_periodicity"] = periodicity
    out["accrual_quality_cash_conversion"] = (
        None if frame["cash_conversion"].dropna().empty else float(frame["cash_conversion"].dropna().iloc[-1])
    )
    out["accrual_quality_margin_gap"] = (
        None if frame["margin_gap"].dropna().empty else float(frame["margin_gap"].dropna().iloc[-1])
    )
    out["accrual_quality_working_capital_stretch"] = latest_stretch
    out["accrual_quality_cycle_convexity"] = latest_cycle_convexity
    return out


def _compute_quality_acceleration_metrics_v2(
    fundamentals: Dict[str, Any],
) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "quality_acceleration_signal": None,
        "quality_acceleration_has_coverage": 0.0,
        "quality_acceleration_measure_count": 0.0,
        "quality_acceleration_margin_delta": None,
        "quality_acceleration_return_delta": None,
        "quality_acceleration_turnover_delta": None,
        "quality_acceleration_cfo_margin_delta": None,
        "quality_acceleration_working_capital_delta": None,
        "quality_acceleration_periodicity": None,
    }

    matched = _matched_statement_sets(fundamentals, "quarterly")
    periodicity = 1.0
    period_days = 90.0
    if len(matched) < 3:
        matched = _matched_statement_sets(fundamentals, "yearly")
        periodicity = 0.0
        period_days = 365.0
    if len(matched) < 3:
        return out

    window = list(reversed(matched[:6]))
    rows: List[Dict[str, float]] = []
    for idx, entry in enumerate(window):
        previous = window[idx - 1] if idx > 0 else None
        income = entry["income"]
        balance = entry["balance"]
        cashflow = entry["cashflow"]
        previous_balance = previous["balance"] if previous is not None else None

        revenue = pick_first(income, "totalRevenue")
        gross_margin = _gross_margin(income)

        quality_return = None
        ebit = _ebit_value(income)
        invested_capital = _effective_invested_capital(balance, intangible_asset=0.0)
        previous_invested_capital = (
            _effective_invested_capital(previous_balance, intangible_asset=0.0)
            if previous_balance is not None
            else None
        )
        average_invested_capital = invested_capital
        if (
            invested_capital is not None
            and previous_invested_capital is not None
            and (invested_capital + previous_invested_capital) > 0
        ):
            average_invested_capital = (invested_capital + previous_invested_capital) / 2.0
        if ebit is not None and average_invested_capital not in (None, 0):
            quality_return = _safe_divide(ebit, average_invested_capital)

        if quality_return is None:
            net_income = pick_first(income, "netIncome")
            assets = pick_first(balance, "totalAssets")
            previous_assets = pick_first(previous_balance or {}, "totalAssets")
            average_assets = assets
            if assets is not None and previous_assets is not None and (assets + previous_assets) > 0:
                average_assets = (assets + previous_assets) / 2.0
            if net_income is not None and average_assets not in (None, 0):
                quality_return = _safe_divide(net_income, average_assets)

        asset_turnover = _asset_turnover(income, balance, previous_balance)
        cfo = pick_first(cashflow, "totalCashFromOperatingActivities", "operatingCashFlow")
        cfo_margin = _safe_divide(cfo, revenue)
        turnover_days = _turnover_days_from_entry(
            income,
            balance,
            previous_balance,
            period_days,
        )
        cycle_days = _cash_conversion_cycle_days(
            turnover_days["receivables_days"],
            turnover_days["inventory_days"],
            turnover_days["payables_days"],
        )

        row = {
            "gross_margin": float(gross_margin) if gross_margin is not None else np.nan,
            "quality_return": float(quality_return) if quality_return is not None else np.nan,
            "asset_turnover": float(asset_turnover) if asset_turnover is not None else np.nan,
            "cfo_margin": float(cfo_margin) if cfo_margin is not None else np.nan,
            "receivables_days": float(turnover_days["receivables_days"]) if turnover_days["receivables_days"] is not None else np.nan,
            "inventory_days": float(turnover_days["inventory_days"]) if turnover_days["inventory_days"] is not None else np.nan,
            "cash_conversion_cycle_days": float(cycle_days) if cycle_days is not None else np.nan,
        }
        if pd.isna(pd.Series(row)).all():
            continue
        rows.append(row)

    if len(rows) < 3:
        return out

    frame = pd.DataFrame(rows)

    def _latest_vs_prior_mean(series: pd.Series) -> Optional[float]:
        clean = pd.to_numeric(series, errors="coerce").dropna()
        if len(clean) < 2:
            return None
        baseline = float(clean.iloc[-2]) if len(clean) == 2 else float(clean.iloc[:-1].mean())
        return float(clean.iloc[-1] - baseline)

    margin_delta = _latest_vs_prior_mean(frame["gross_margin"])
    return_delta = _latest_vs_prior_mean(frame["quality_return"])
    turnover_delta = _latest_vs_prior_mean(frame["asset_turnover"])
    cfo_margin_delta = _latest_vs_prior_mean(frame["cfo_margin"])
    receivables_days_delta = _latest_vs_prior_mean(frame["receivables_days"])
    inventory_days_delta = _latest_vs_prior_mean(frame["inventory_days"])
    cash_conversion_cycle_days_delta = _latest_vs_prior_mean(frame["cash_conversion_cycle_days"])
    cash_conversion_cycle_convexity = _latest_vs_prior_mean(frame["cash_conversion_cycle_days"].diff())

    working_capital_delta = None
    wc_components = [
        float(value)
        for value in [
            receivables_days_delta,
            inventory_days_delta,
            cash_conversion_cycle_days_delta,
            cash_conversion_cycle_convexity,
        ]
        if value is not None
    ]
    if wc_components:
        working_capital_delta = float(sum(wc_components) / len(wc_components))

    weighted_components = [
        (0.25, _clip_score(margin_delta, 0.04)),
        (0.25, _clip_score(return_delta, 0.03)),
        (0.20, _clip_score(turnover_delta, 0.10)),
        (0.15, _clip_score(cfo_margin_delta, 0.04)),
        (0.15, _clip_score(-working_capital_delta, 18.0) if working_capital_delta is not None else None),
    ]
    usable_components = [(weight, value) for weight, value in weighted_components if value is not None]
    if len(usable_components) < 3:
        return out

    total_weight = float(sum(weight for weight, _ in usable_components))
    signal = float(sum(weight * float(value) for weight, value in usable_components) / total_weight)

    out["quality_acceleration_signal"] = float(np.clip(signal, -1.0, 1.0))
    out["quality_acceleration_has_coverage"] = float(min(1.0, total_weight))
    out["quality_acceleration_measure_count"] = float(len(frame))
    out["quality_acceleration_margin_delta"] = margin_delta
    out["quality_acceleration_return_delta"] = return_delta
    out["quality_acceleration_turnover_delta"] = turnover_delta
    out["quality_acceleration_cfo_margin_delta"] = cfo_margin_delta
    out["quality_acceleration_working_capital_delta"] = working_capital_delta
    out["quality_acceleration_periodicity"] = periodicity
    return out


def compute_quality_acceleration_metrics_from_fundamentals(
    fundamentals: Dict[str, Any],
    alpha_factor_spec: str = "legacy",
) -> Dict[str, Optional[float]]:
    if str(alpha_factor_spec).lower() == "v2":
        return _compute_quality_acceleration_metrics_v2(fundamentals)

    out: Dict[str, Optional[float]] = {
        "quality_acceleration_signal": None,
        "quality_acceleration_has_coverage": 0.0,
        "quality_acceleration_measure_count": 0.0,
        "quality_acceleration_margin_delta": None,
        "quality_acceleration_return_delta": None,
        "quality_acceleration_turnover_delta": None,
        "quality_acceleration_cfo_margin_delta": None,
        "quality_acceleration_working_capital_delta": None,
        "quality_acceleration_periodicity": None,
    }

    matched = _matched_statement_sets(fundamentals, "quarterly")
    periodicity = 1.0
    if len(matched) < 3:
        matched = _matched_statement_sets(fundamentals, "yearly")
        periodicity = 0.0
    if len(matched) < 3:
        return out

    window = list(reversed(matched[:6]))
    rows: List[Dict[str, float]] = []
    for idx, entry in enumerate(window):
        previous = window[idx - 1] if idx > 0 else None
        income = entry["income"]
        balance = entry["balance"]
        cashflow = entry["cashflow"]
        previous_balance = previous["balance"] if previous is not None else None

        revenue = pick_first(income, "totalRevenue")
        gross_margin = _gross_margin(income)

        quality_return = None
        ebit = _ebit_value(income)
        invested_capital = _effective_invested_capital(balance, intangible_asset=0.0)
        previous_invested_capital = (
            _effective_invested_capital(previous_balance, intangible_asset=0.0)
            if previous_balance is not None
            else None
        )
        average_invested_capital = invested_capital
        if (
            invested_capital is not None
            and previous_invested_capital is not None
            and (invested_capital + previous_invested_capital) > 0
        ):
            average_invested_capital = (invested_capital + previous_invested_capital) / 2.0
        if ebit is not None and average_invested_capital not in (None, 0):
            quality_return = _safe_divide(ebit, average_invested_capital)

        if quality_return is None:
            net_income = pick_first(income, "netIncome")
            assets = pick_first(balance, "totalAssets")
            previous_assets = pick_first(previous_balance or {}, "totalAssets")
            average_assets = assets
            if assets is not None and previous_assets is not None and (assets + previous_assets) > 0:
                average_assets = (assets + previous_assets) / 2.0
            if net_income is not None and average_assets not in (None, 0):
                quality_return = _safe_divide(net_income, average_assets)

        asset_turnover = _asset_turnover(income, balance, previous_balance)
        cfo = pick_first(cashflow, "totalCashFromOperatingActivities", "operatingCashFlow")
        cfo_margin = _safe_divide(cfo, revenue)
        receivables_to_sales = _safe_divide(
            pick_first(balance, "netReceivables", "accountsReceivable"),
            revenue,
        )
        inventory_to_sales = _safe_divide(_inventory_value(balance), revenue)

        row = {
            "gross_margin": float(gross_margin) if gross_margin is not None else np.nan,
            "quality_return": float(quality_return) if quality_return is not None else np.nan,
            "asset_turnover": float(asset_turnover) if asset_turnover is not None else np.nan,
            "cfo_margin": float(cfo_margin) if cfo_margin is not None else np.nan,
            "receivables_to_sales": float(receivables_to_sales) if receivables_to_sales is not None else np.nan,
            "inventory_to_sales": float(inventory_to_sales) if inventory_to_sales is not None else np.nan,
        }
        if pd.isna(pd.Series(row)).all():
            continue
        rows.append(row)

    if len(rows) < 3:
        return out

    frame = pd.DataFrame(rows)

    def _latest_vs_prior_mean(series: pd.Series) -> Optional[float]:
        clean = pd.to_numeric(series, errors="coerce").dropna()
        if len(clean) < 2:
            return None
        baseline = float(clean.iloc[-2]) if len(clean) == 2 else float(clean.iloc[:-1].mean())
        return float(clean.iloc[-1] - baseline)

    margin_delta = _latest_vs_prior_mean(frame["gross_margin"])
    return_delta = _latest_vs_prior_mean(frame["quality_return"])
    turnover_delta = _latest_vs_prior_mean(frame["asset_turnover"])
    cfo_margin_delta = _latest_vs_prior_mean(frame["cfo_margin"])
    receivables_delta = _latest_vs_prior_mean(frame["receivables_to_sales"])
    inventory_delta = _latest_vs_prior_mean(frame["inventory_to_sales"])

    working_capital_delta = None
    wc_components = [
        float(value)
        for value in [receivables_delta, inventory_delta]
        if value is not None
    ]
    if wc_components:
        working_capital_delta = float(sum(wc_components))

    weighted_components = [
        (0.25, _clip_score(margin_delta, 0.04)),
        (0.25, _clip_score(return_delta, 0.03)),
        (0.20, _clip_score(turnover_delta, 0.10)),
        (0.15, _clip_score(cfo_margin_delta, 0.04)),
        (0.15, _clip_score(-working_capital_delta, 0.08) if working_capital_delta is not None else None),
    ]
    usable_components = [(weight, value) for weight, value in weighted_components if value is not None]
    if len(usable_components) < 3:
        return out

    total_weight = float(sum(weight for weight, _ in usable_components))
    signal = float(sum(weight * float(value) for weight, value in usable_components) / total_weight)

    out["quality_acceleration_signal"] = float(np.clip(signal, -1.0, 1.0))
    out["quality_acceleration_has_coverage"] = float(min(1.0, total_weight))
    out["quality_acceleration_measure_count"] = float(len(frame))
    out["quality_acceleration_margin_delta"] = margin_delta
    out["quality_acceleration_return_delta"] = return_delta
    out["quality_acceleration_turnover_delta"] = turnover_delta
    out["quality_acceleration_cfo_margin_delta"] = cfo_margin_delta
    out["quality_acceleration_working_capital_delta"] = working_capital_delta
    out["quality_acceleration_periodicity"] = periodicity
    return out


def compute_revision_impulse_metrics_from_fundamentals(
    fundamentals: Dict[str, Any],
    min_revision_analysts: int,
    alpha_factor_spec: str = "legacy",
) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "revision_impulse_signal": None,
        "revision_impulse_has_coverage": 0.0,
        "revision_impulse_analyst_count": None,
        "revision_jerk_signal": None,
        "revision_jerk_has_coverage": 0.0,
        "revision_jerk_recent_velocity": None,
        "revision_jerk_prior_velocity": None,
        "revision_jerk_component": None,
        "revision_impulse_drift_7d": None,
        "revision_impulse_drift_30d": None,
        "revision_impulse_breadth": None,
        "revision_impulse_growth_component": None,
        "revision_impulse_coverage_component": None,
        "revision_impulse_disagreement": None,
        "revision_impulse_disagreement_penalty": None,
        "revision_short_divergence_component": None,
        "revision_breadth_7d": None,
        "revision_breadth_30d": None,
        "revision_breadth_acceleration": None,
        "float_absorption_signal": None,
        "squeeze_convexity_signal": None,
    }

    trend = latest_earnings_trend_record(fundamentals)
    if not trend:
        return out

    analyst_count = to_float(trend.get("earningsEstimateNumberOfAnalysts"))
    estimate_avg = to_float(trend.get("earningsEstimateAvg"))
    estimate_low = to_float(trend.get("earningsEstimateLow"))
    estimate_high = to_float(trend.get("earningsEstimateHigh"))
    estimate_growth = _normalize_estimate_growth(to_float(trend.get("earningsEstimateGrowth")))
    eps_current = to_float(trend.get("epsTrendCurrent"))
    eps_7d = to_float(trend.get("epsTrend7daysAgo"))
    eps_30d = to_float(trend.get("epsTrend30daysAgo"))
    rev_up_7d = to_float(trend.get("epsRevisionsUpLast7days"))
    rev_down_7d = to_float(trend.get("epsRevisionsDownLast7days"))
    rev_up_30d = to_float(trend.get("epsRevisionsUpLast30days"))
    rev_down_30d = to_float(trend.get("epsRevisionsDownLast30days"))

    out["revision_impulse_analyst_count"] = analyst_count
    if analyst_count is None or analyst_count <= 0 or estimate_avg is None or eps_current is None:
        return out

    denominator = max(abs(float(estimate_avg)), 0.05)
    drift_7d = _clip_unit((eps_current - eps_7d) / denominator) if eps_7d is not None else None
    drift_30d = _clip_unit((eps_current - eps_30d) / denominator) if eps_30d is not None else None
    breadth = (
        _clip_unit((rev_up_7d - rev_down_30d) / max(float(analyst_count), 1.0))
        if rev_up_7d is not None and rev_down_30d is not None
        else None
    )
    revision_breadth_7d = (
        _clip_unit((rev_up_7d - rev_down_7d) / max(float(rev_up_7d + rev_down_7d), 1.0))
        if rev_up_7d is not None and rev_down_7d is not None
        else None
    )
    revision_breadth_30d = (
        _clip_unit((rev_up_30d - rev_down_30d) / max(float(rev_up_30d + rev_down_30d), 1.0))
        if rev_up_30d is not None and rev_down_30d is not None
        else None
    )
    revision_breadth_acceleration = (
        float(revision_breadth_7d - revision_breadth_30d)
        if revision_breadth_7d is not None and revision_breadth_30d is not None
        else None
    )
    growth_component = _clip_unit(estimate_growth / 0.25) if estimate_growth is not None else None
    recent_velocity = _safe_divide((eps_current - eps_7d), denominator) if eps_7d is not None else None
    prior_velocity = _safe_divide((eps_7d - eps_30d), denominator) if eps_7d is not None and eps_30d is not None else None
    jerk_component = _clip_score(
        (recent_velocity - prior_velocity) if recent_velocity is not None and prior_velocity is not None else None,
        0.08,
    )

    disagreement = (
        _safe_divide((estimate_high - estimate_low), denominator)
        if estimate_high is not None and estimate_low is not None
        else None
    )
    disagreement_penalty = (
        max(0.0, min(1.0, float(disagreement) / 2.0))
        if disagreement is not None
        else 0.0
    )
    coverage_component = min(1.0, float(analyst_count) / max(1.0, float(min_revision_analysts)))
    short_divergence_component = None
    breadth_acceleration_component = None
    float_absorption_signal = None
    squeeze_convexity_signal = None
    if str(alpha_factor_spec).lower() == "v2":
        short_interest_metrics = compute_short_interest_metrics_from_fundamentals(fundamentals)
        share_drift_metrics = compute_share_drift_metrics_from_fundamentals(
            fundamentals,
            alpha_factor_spec="v2",
        )
        institutional_breadth_metrics = compute_institutional_breadth_metrics_from_fundamentals(fundamentals)
        short_components = [
            _clip_score(-to_float(short_interest_metrics.get("short_interest_pct_float")), 0.15),
            _clip_score(-to_float(short_interest_metrics.get("short_interest_ratio")), 8.0),
            _clip_score(-to_float(short_interest_metrics.get("short_interest_change")), 0.25),
        ]
        usable_short_components = [float(component) for component in short_components if component is not None]
        if len(usable_short_components) >= 2:
            short_divergence_component = float(sum(usable_short_components) / len(usable_short_components))
        breadth_acceleration_component = _clip_score(revision_breadth_acceleration, 0.75)

        absorption_components: List[float] = []
        has_short_input = False
        has_absorption_input = False
        short_pct_float_component = _clip_score(to_float(short_interest_metrics.get("short_interest_pct_float")), 0.18)
        short_ratio_component = _clip_score(to_float(short_interest_metrics.get("short_interest_ratio")), 8.0)
        share_drift_blended = to_float(share_drift_metrics.get("buyback_yield"))
        share_drift_component = _clip_score(share_drift_blended, 0.05)
        institutional_breadth_delta = to_float(institutional_breadth_metrics.get("institutional_breadth_delta"))
        institutional_breadth_component = _clip_score(institutional_breadth_delta, 0.25)

        for component in [short_pct_float_component, short_ratio_component]:
            if component is not None:
                absorption_components.append(float(component))
                has_short_input = True
        for component in [share_drift_component, institutional_breadth_component]:
            if component is not None:
                absorption_components.append(float(component))
                has_absorption_input = True
        if has_short_input and has_absorption_input and len(absorption_components) >= 3:
            float_absorption_signal = float(sum(absorption_components) / len(absorption_components))

        positive_revision_catalyst_components: List[float] = []
        if drift_7d is not None:
            positive_revision_catalyst_components.append(max(0.0, float(drift_7d)))
        if revision_breadth_7d is not None:
            positive_revision_catalyst_components.append(max(0.0, float(revision_breadth_7d)))
        if breadth_acceleration_component is not None:
            positive_revision_catalyst_components.append(max(0.0, float(breadth_acceleration_component)))
        positive_revision_catalyst = (
            float(sum(positive_revision_catalyst_components) / len(positive_revision_catalyst_components))
            if len(positive_revision_catalyst_components) >= 2
            else None
        )
        if float_absorption_signal is not None and positive_revision_catalyst is not None:
            squeeze_convexity_signal = float(
                np.clip(
                    max(0.0, float(float_absorption_signal)) * max(0.0, float(positive_revision_catalyst)),
                    0.0,
                    1.0,
                )
            )

    base_components = [
        component
        for component in [drift_7d, drift_30d, breadth, growth_component]
        if component is not None
    ]
    if str(alpha_factor_spec).lower() == "v2":
        for component in [revision_breadth_7d, breadth_acceleration_component]:
            if component is not None:
                base_components.append(float(component))
    if short_divergence_component is not None:
        base_components.append(short_divergence_component)
    if len(base_components) < 2:
        return out

    base_signal = sum(float(component) for component in base_components) / float(len(base_components))
    signal = base_signal * coverage_component * (1.0 - disagreement_penalty)
    if squeeze_convexity_signal is not None and str(alpha_factor_spec).lower() == "v2":
        signal = 0.85 * float(signal) + 0.15 * float(squeeze_convexity_signal)
    signal = max(-1.0, min(1.0, float(signal)))

    out["revision_impulse_has_coverage"] = 1.0
    out["revision_impulse_signal"] = signal
    out["revision_impulse_drift_7d"] = drift_7d
    out["revision_impulse_drift_30d"] = drift_30d
    out["revision_impulse_breadth"] = breadth
    out["revision_impulse_growth_component"] = growth_component
    out["revision_impulse_coverage_component"] = float(coverage_component)
    out["revision_impulse_disagreement"] = disagreement
    out["revision_impulse_disagreement_penalty"] = float(disagreement_penalty)
    out["revision_short_divergence_component"] = short_divergence_component
    out["revision_breadth_7d"] = revision_breadth_7d
    out["revision_breadth_30d"] = revision_breadth_30d
    out["revision_breadth_acceleration"] = revision_breadth_acceleration
    out["float_absorption_signal"] = float_absorption_signal
    out["squeeze_convexity_signal"] = squeeze_convexity_signal
    out["revision_jerk_recent_velocity"] = recent_velocity
    out["revision_jerk_prior_velocity"] = prior_velocity
    out["revision_jerk_component"] = jerk_component

    if jerk_component is not None:
        jerk_signal = (
            0.70 * float(jerk_component)
            + 0.20 * float(breadth or 0.0)
            + 0.10 * float(growth_component or 0.0)
        )
        jerk_signal = jerk_signal * coverage_component * max(0.0, 1.0 - 1.15 * disagreement_penalty)
        out["revision_jerk_signal"] = max(-1.0, min(1.0, float(jerk_signal)))
        out["revision_jerk_has_coverage"] = 1.0
    return out


def compute_sue_metrics_from_fundamentals(
    fundamentals: Dict[str, Any],
    min_history_quarters: int = 4,
) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "sue_signal": None,
        "sue_has_coverage": 0.0,
        "sue_surprise_raw": None,
        "sue_surprise_pct": None,
        "sue_std_error": None,
        "sue_report_date": None,
    }

    today = utc_today_ts()
    valid_events: List[Dict[str, Any]] = []
    for item in earnings_history_records(fundamentals):
        report_ts = pd.to_datetime(
            item.get("reportDate") or item.get("report_date") or item.get("date"),
            errors="coerce",
            utc=True,
        )
        actual = to_float(item.get("epsActual") or item.get("actual"))
        estimate = to_float(item.get("epsEstimate") or item.get("estimate"))
        if pd.isna(report_ts) or report_ts.normalize() > today or actual is None or estimate is None:
            continue

        valid_events.append(
            {
                "raw_surprise": float(actual - estimate),
                "surprise_pct": _parse_surprise_pct(item),
                "report_date": report_ts.strftime("%Y-%m-%d"),
            }
        )

    if not valid_events:
        return out

    current_event = valid_events[0]
    out["sue_surprise_raw"] = current_event["raw_surprise"]
    out["sue_surprise_pct"] = current_event["surprise_pct"]
    out["sue_report_date"] = current_event["report_date"]

    trailing_surprises = [event["raw_surprise"] for event in valid_events[1 : 1 + max(8, min_history_quarters)]]
    if len(trailing_surprises) < int(min_history_quarters):
        return out

    std_error = pd.Series(trailing_surprises, dtype=float).std(ddof=0)
    if std_error is None or pd.isna(std_error) or float(std_error) <= 1e-6:
        return out

    out["sue_has_coverage"] = 1.0
    out["sue_std_error"] = float(std_error)
    out["sue_signal"] = float(max(-8.0, min(8.0, current_event["raw_surprise"] / float(std_error))))
    return out


def compute_revenue_growth_metrics_from_fundamentals(
    fundamentals: Dict[str, Any],
) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "revenue_growth_yoy": None,
        "revenue_growth_yoy_prev": None,
        "revenue_acceleration": None,
        "revenue_growth_has_coverage": 0.0,
    }

    quarterly = income_statement_records(fundamentals, "quarterly")
    revenues = [pick_first(record, "totalRevenue") for record in quarterly]

    if len(revenues) >= 5:
        current = revenues[0]
        year_ago = revenues[4]
        if current is not None and year_ago is not None and year_ago > 0:
            out["revenue_growth_yoy"] = float(current / year_ago - 1.0)
            out["revenue_growth_has_coverage"] = 1.0

    if len(revenues) >= 6:
        prev_current = revenues[1]
        prev_year_ago = revenues[5]
        if prev_current is not None and prev_year_ago is not None and prev_year_ago > 0:
            out["revenue_growth_yoy_prev"] = float(prev_current / prev_year_ago - 1.0)

    if out["revenue_growth_yoy"] is not None and out["revenue_growth_yoy_prev"] is not None:
        out["revenue_acceleration"] = float(out["revenue_growth_yoy"] - out["revenue_growth_yoy_prev"])

    return out


def compute_price_momentum_metrics_from_history(
    price_history: Any,
) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "price_momentum_1m": None,
        "price_momentum_6m": None,
        "price_momentum_6m_ex_1m": None,
        "price_momentum_has_coverage": 0.0,
    }

    if not isinstance(price_history, list):
        return out

    rows = []
    for item in price_history:
        if not isinstance(item, dict):
            continue
        close = to_float(item.get("adjusted_close") or item.get("adjustedClose") or item.get("close"))
        date = pd.to_datetime(item.get("date"), errors="coerce")
        if close is None or close <= 0 or pd.isna(date):
            continue
        rows.append({"date": date, "close": float(close)})

    if not rows:
        return out

    frame = pd.DataFrame(rows).drop_duplicates(subset=["date"], keep="last").sort_values("date")
    closes = frame["close"].astype(float).reset_index(drop=True)

    if len(closes) >= 22:
        out["price_momentum_1m"] = float(closes.iloc[-1] / closes.iloc[-22] - 1.0)

    if len(closes) >= 127:
        out["price_momentum_6m"] = float(closes.iloc[-1] / closes.iloc[-127] - 1.0)
        out["price_momentum_6m_ex_1m"] = float(closes.iloc[-22] / closes.iloc[-127] - 1.0)
        out["price_momentum_has_coverage"] = 1.0

    return out


def compute_price_momentum_proxy_metrics(
    price_to_200dma: Any,
    recency_ratio: Any,
    distance_from_high: Any,
) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "price_momentum_effective_signal": None,
        "price_momentum_signal_coverage": 0.0,
        "price_momentum_proxy_used": 0.0,
    }

    trend_vs_ma = to_float(price_to_200dma)
    recency = to_float(recency_ratio)
    dist = to_float(distance_from_high)

    components: list[float] = []

    if trend_vs_ma is not None and trend_vs_ma > 0:
        components.append(max(-0.40, min(0.60, float(trend_vs_ma - 1.0))))

    if recency is not None and recency > 0:
        components.append(max(-0.45, min(0.10, float(recency - 1.0))))
    elif dist is not None:
        components.append(max(-0.45, min(0.10, float(dist))))

    if not components:
        return out

    if len(components) == 1:
        proxy_signal = components[0]
    else:
        proxy_signal = 0.65 * components[0] + 0.35 * components[1]

    out["price_momentum_effective_signal"] = float(proxy_signal)
    out["price_momentum_signal_coverage"] = 1.0
    out["price_momentum_proxy_used"] = 1.0
    return out


def _asset_quality_index(balance: Dict[str, Any]) -> Optional[float]:
    total_assets = pick_first(balance, "totalAssets")
    current_assets = pick_first(balance, "totalCurrentAssets")
    ppe = pick_first(balance, "propertyPlantAndEquipmentNet", "propertyPlantEquipment", "propertyPlantAndEquipmentGross")
    short_term_investments = pick_first(balance, "shortTermInvestments") or 0.0
    long_term_investments = pick_first(balance, "longTermInvestments") or 0.0

    if total_assets is None or total_assets <= 0 or current_assets is None or ppe is None:
        return None

    return 1.0 - ((current_assets + ppe + short_term_investments + long_term_investments) / total_assets)


def _depreciation_rate(income: Dict[str, Any], balance: Dict[str, Any]) -> Optional[float]:
    depreciation = pick_first(income, "depreciationAndAmortization", "depreciation")
    ppe = pick_first(balance, "propertyPlantAndEquipmentNet", "propertyPlantEquipment", "propertyPlantAndEquipmentGross")
    if depreciation is None or ppe is None or (depreciation + ppe) <= 0:
        return None
    return depreciation / (depreciation + ppe)


def _total_debt(balance: Dict[str, Any]) -> Optional[float]:
    total_debt = pick_first(balance, "totalDebt", "longTermDebtTotal")
    if total_debt is not None:
        return total_debt

    current_liab = pick_first(balance, "totalCurrentLiabilities")
    long_term_debt = pick_first(balance, "longTermDebt", "shortLongTermDebtTotal", "shortTermDebt")
    if long_term_debt is not None and current_liab is None:
        return long_term_debt
    if current_liab is None or long_term_debt is None:
        return None
    return current_liab + long_term_debt


def _empty_beneish_metrics(status: str) -> Dict[str, Any]:
    return {
        "beneish_m_score": None,
        "beneish_data_status": status,
        "beneish_is_missing": 1.0 if status == "missing" else 0.0,
        "beneish_is_pathological_clipped": 1.0 if status == "pathological_clipped" else 0.0,
    }


def compute_beneish_metrics(fundamentals: Dict[str, Any]) -> Dict[str, Any]:
    matched = _matched_statement_sets(fundamentals, "yearly")
    if len(matched) < 2:
        return _empty_beneish_metrics("missing")

    current = matched[0]
    previous = matched[1]

    inc0, inc1 = current["income"], previous["income"]
    bal0, bal1 = current["balance"], previous["balance"]
    cf0 = current["cashflow"]

    receivables_0 = pick_first(bal0, "netReceivables", "accountsReceivable")
    receivables_1 = pick_first(bal1, "netReceivables", "accountsReceivable")
    sales_0 = pick_first(inc0, "totalRevenue")
    sales_1 = pick_first(inc1, "totalRevenue")
    cogs_0 = pick_first(inc0, "costOfRevenue")
    cogs_1 = pick_first(inc1, "costOfRevenue")
    total_assets_0 = pick_first(bal0, "totalAssets")
    total_assets_1 = pick_first(bal1, "totalAssets")
    sga_0 = pick_first(inc0, "sellingGeneralAdministrative")
    sga_1 = pick_first(inc1, "sellingGeneralAdministrative")
    debt_0 = _total_debt(bal0)
    debt_1 = _total_debt(bal1)
    net_income_0 = pick_first(inc0, "netIncomeFromContinuingOps", "netIncome")
    cfo_0 = pick_first(cf0, "totalCashFromOperatingActivities", "operatingCashFlow")

    gross_margin_0 = _safe_divide((sales_0 - cogs_0) if sales_0 is not None and cogs_0 is not None else None, sales_0)
    gross_margin_1 = _safe_divide((sales_1 - cogs_1) if sales_1 is not None and cogs_1 is not None else None, sales_1)
    asset_quality_0 = _asset_quality_index(bal0)
    asset_quality_1 = _asset_quality_index(bal1)
    depreciation_rate_0 = _depreciation_rate(inc0, bal0)
    depreciation_rate_1 = _depreciation_rate(inc1, bal1)

    dsri = _safe_divide(_safe_divide(receivables_0, sales_0), _safe_divide(receivables_1, sales_1))
    gmi = _safe_divide(gross_margin_1, gross_margin_0)
    aqi = _safe_divide(asset_quality_0, asset_quality_1)
    sgi = _safe_divide(sales_0, sales_1)
    depi = _safe_divide(depreciation_rate_1, depreciation_rate_0)
    sgai = _safe_divide(_safe_divide(sga_0, sales_0), _safe_divide(sga_1, sales_1))
    lvgi = _safe_divide(_safe_divide(debt_0, total_assets_0), _safe_divide(debt_1, total_assets_1))
    tata = _safe_divide((net_income_0 - cfo_0) if net_income_0 is not None and cfo_0 is not None else None, total_assets_0)

    raw_inputs = [
        receivables_0,
        receivables_1,
        sales_0,
        sales_1,
        cogs_0,
        cogs_1,
        total_assets_0,
        total_assets_1,
        sga_0,
        sga_1,
        debt_0,
        debt_1,
        net_income_0,
        cfo_0,
    ]
    if any(value is None for value in raw_inputs):
        return _empty_beneish_metrics("missing")

    components = [dsri, gmi, aqi, sgi, depi, sgai, lvgi, tata]
    if any(component is None for component in components):
        return _empty_beneish_metrics("pathological_clipped")

    # Beneish component ratios should be positive and reasonably bounded.
    # If EODHD statement fields produce pathological ratios, treat the score as missing
    # rather than letting an absurd negative value look "super safe" in ranking.
    bounded_components = [dsri, gmi, aqi, sgi, depi, sgai, lvgi]
    if any(component <= 0 or component > 10 for component in bounded_components):
        return _empty_beneish_metrics("pathological_clipped")

    score = float(
        -4.84
        + 0.920 * dsri
        + 0.528 * gmi
        + 0.404 * aqi
        + 0.892 * sgi
        + 0.115 * depi
        - 0.172 * sgai
        + 4.679 * tata
        - 0.327 * lvgi
    )
    if score < -10.0 or score > 10.0:
        return _empty_beneish_metrics("pathological_clipped")

    return {
        "beneish_m_score": score,
        "beneish_data_status": "ok",
        "beneish_is_missing": 0.0,
        "beneish_is_pathological_clipped": 0.0,
    }


def compute_beneish_m_score(fundamentals: Dict[str, Any]) -> Optional[float]:
    return compute_beneish_metrics(fundamentals).get("beneish_m_score")


def _accrual_ratio_rows(fundamentals: Dict[str, Any], frequency: str, max_periods: int) -> List[Dict[str, float]]:
    matched = _matched_statement_sets(fundamentals, frequency)
    ratios: List[Dict[str, float]] = []

    for idx in range(min(len(matched) - 1, max_periods)):
        current = matched[idx]
        previous = matched[idx + 1]

        income = current["income"]
        cashflow = current["cashflow"]
        balance = current["balance"]
        previous_balance = previous["balance"]

        net_income = pick_first(income, "netIncome")
        cfo = pick_first(cashflow, "totalCashFromOperatingActivities", "operatingCashFlow")
        total_assets = pick_first(balance, "totalAssets")
        previous_assets = pick_first(previous_balance, "totalAssets")
        if net_income is None or cfo is None or total_assets is None or previous_assets is None:
            continue

        average_assets = (total_assets + previous_assets) / 2.0
        if average_assets <= 0:
            continue

        ratios.append(
            {
                "period": current["period"],
                "accrual_ratio": float((net_income - cfo) / average_assets),
            }
        )

    return ratios


def compute_accrual_metrics(fundamentals: Dict[str, Any]) -> Dict[str, Optional[float]]:
    quarterly = _accrual_ratio_rows(fundamentals, "quarterly", max_periods=8)
    if len(quarterly) >= 6:
        series = quarterly
        source = 1.0
    else:
        yearly = _accrual_ratio_rows(fundamentals, "yearly", max_periods=5)
        if len(yearly) < 4:
            return {
                "accrual_ratio": None,
                "accrual_volatility": None,
                "accrual_measure_count": 0.0,
                "accrual_is_quarterly": None,
            }
        series = yearly
        source = 0.0

    values = [row["accrual_ratio"] for row in series]
    volatility = float(pd.Series(values, dtype=float).std(ddof=0)) if len(values) >= 2 else None
    latest_ratio = float(series[0]["accrual_ratio"]) if series else None

    return {
        "accrual_ratio": latest_ratio,
        "accrual_volatility": volatility,
        "accrual_measure_count": float(len(series)),
        "accrual_is_quarterly": source,
    }


def passes_sentiment_coverage_gate(
    sentiment_count_days: Optional[float],
    sentiment_article_count_recent: Optional[float],
    sentiment_acceleration: Optional[float],
    min_count_days: int,
    min_article_count_recent: int,
    min_sentiment_accel: float,
) -> bool:
    count_days = to_float(sentiment_count_days)
    recent_articles = to_float(sentiment_article_count_recent)
    accel = to_float(sentiment_acceleration)

    if count_days is None or count_days < float(min_count_days):
        return True

    if recent_articles is None or recent_articles < float(min_article_count_recent):
        return True

    if accel is None:
        return True

    return accel >= float(min_sentiment_accel)
