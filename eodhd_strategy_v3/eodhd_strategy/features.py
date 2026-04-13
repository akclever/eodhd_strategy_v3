from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .advanced_factors import (
    balance_sheet_records,
    cash_flow_records,
    compute_accrual_metrics,
    compute_accrual_quality_metrics,
    compute_beneish_metrics,
    compute_capital_allocation_quality_metrics,
    compute_compounder_persistence_metrics_from_fundamentals,
    compute_estimate_term_structure_metrics_from_fundamentals,
    compute_investment_restraint_metrics,
    compute_pead_metrics_from_fundamentals,
    compute_price_momentum_metrics_from_history,
    compute_price_momentum_proxy_metrics,
    compute_pead_signal_from_surprise,
    compute_piotroski_f_score,
    compute_recovery_fundamental_metrics,
    compute_revenue_growth_metrics_from_fundamentals,
    compute_revision_impulse_metrics_from_fundamentals,
    compute_sue_metrics_from_fundamentals,
    compute_working_capital_stress_metrics,
    income_statement_records,
    passes_sentiment_coverage_gate,
)
from .client import EODHDClient
from .config import RankerConfig
from .exchanges import extract_listing_exchange_codes
from .utils import normalize_records, pick_first, to_float, utc_today_ts

NEWS_EVENT_CATEGORY_SPECS = {
    "earnings_positive": {
        "bias": 1.0,
        "terms": (
            "earnings surprise",
            "estimate revisions",
            "raises guidance",
            "raised guidance",
            "guidance raise",
            "guidance raised",
            "beats",
            "beat estimates",
            "strong buy",
            "upgrade",
            "upgraded",
            "price target raised",
            "revenue growth",
            "earnings growth",
        ),
    },
    "corporate_positive": {
        "bias": 1.0,
        "terms": (
            "mergers and acquisitions",
            "acquisition",
            "acquires",
            "partnership",
            "partnered",
            "contract win",
            "wins contract",
            "approved",
            "approval",
            "buyback",
            "share repurchase",
            "dividend raise",
            "dividend increase",
            "record backlog",
            "launches",
            "launch",
        ),
    },
    "earnings_negative": {
        "bias": -1.0,
        "terms": (
            "cuts guidance",
            "cut guidance",
            "guidance cut",
            "guidance lowered",
            "misses",
            "missed estimates",
            "downgrade",
            "downgraded",
            "price target cut",
            "weak outlook",
            "estimate cut",
            "warning",
        ),
    },
    "legal_regulatory": {
        "bias": -1.0,
        "terms": (
            "class action",
            "lawsuit",
            "investigation",
            "probe",
            "recall",
            "warning letter",
            "credit rating",
            "regulatory",
            "european regulatory news",
        ),
    },
    "financing_stress": {
        "bias": -1.0,
        "terms": (
            "offering",
            "dilution",
            "dilutive",
            "bankruptcy",
            "default",
            "restructuring",
            "going concern",
        ),
    },
}

NEWS_THEME_CATEGORY_SPECS_V2 = {
    "guidance_positive": {
        "bias": 1.0,
        "terms": (
            "raises guidance",
            "raised guidance",
            "guidance raise",
            "guidance raised",
            "outlook raised",
            "reaffirmed guidance",
        ),
    },
    "contracts_positive": {
        "bias": 1.0,
        "terms": (
            "contract win",
            "wins contract",
            "major order",
            "backlog",
            "partnership",
            "expands agreement",
        ),
    },
    "approvals_positive": {
        "bias": 1.0,
        "terms": (
            "approved",
            "approval",
            "clearance",
            "authorization",
        ),
    },
    "capital_return_positive": {
        "bias": 1.0,
        "terms": (
            "buyback",
            "share repurchase",
            "repurchase authorization",
            "dividend raise",
            "dividend increase",
        ),
    },
    "restructuring_positive": {
        "bias": 0.5,
        "terms": (
            "cost savings",
            "restructuring plan",
            "margin improvement",
            "turnaround plan",
        ),
    },
    "guidance_negative": {
        "bias": -1.0,
        "terms": (
            "cuts guidance",
            "cut guidance",
            "guidance cut",
            "guidance lowered",
            "weak outlook",
            "warning",
        ),
    },
    "dilution_negative": {
        "bias": -1.0,
        "terms": (
            "offering",
            "dilution",
            "dilutive",
            "convertible notes",
        ),
    },
    "litigation_regulatory_negative": {
        "bias": -1.0,
        "terms": (
            "lawsuit",
            "class action",
            "investigation",
            "probe",
            "regulatory",
            "warning letter",
        ),
    },
}

NEWS_TITLE_STOPWORDS = {
    "a",
    "an",
    "and",
    "after",
    "at",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "the",
    "to",
    "with",
}

INTANGIBLE_ADJUSTMENT_SECTORS = {
    "technology",
    "healthcare",
    "communication services",
}


def annual_outstanding_share_records(fundamentals: Dict[str, Any]) -> List[Dict[str, Any]]:
    outstanding = fundamentals.get("outstandingShares") or {}
    annual = outstanding.get("annual") if isinstance(outstanding, dict) else None
    return normalize_records(annual)


def capitalize_expense(records: List[Dict[str, Any]], field: str, years: int, scale: float = 1.0) -> Optional[float]:
    if not records:
        return None

    capital = 0.0
    found_any = False
    for age, record in enumerate(records[:years]):
        expense = to_float(record.get(field))
        if expense is None or expense <= 0:
            continue
        found_any = True
        remaining_life = max(0.0, 1.0 - (age / years))
        capital += expense * scale * remaining_life

    return capital if found_any else None


def compute_trailing_dividend_cash_per_share(client: EODHDClient, symbol: str) -> Optional[float]:
    start_date = (utc_today_ts() - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    try:
        dividends = client.get_dividends(symbol, start_date=start_date)
    except Exception:
        return None

    if not isinstance(dividends, list) or not dividends:
        return None

    cutoff = utc_today_ts() - pd.Timedelta(days=365)
    total = 0.0
    for item in dividends:
        if not isinstance(item, dict):
            continue
        ex_date = pd.to_datetime(
            item.get("date") or item.get("exDate") or item.get("ex_dividend_date"),
            errors="coerce",
            utc=True,
        )
        amount = to_float(item.get("value") or item.get("dividend") or item.get("amount"))
        if pd.isna(ex_date) or amount is None:
            continue
        if ex_date >= cutoff:
            total += amount

    return total if total > 0 else None


def compute_buyback_yield(fundamentals: Dict[str, Any]) -> Optional[float]:
    records = balance_sheet_records(fundamentals)
    if len(records) >= 2:
        current_shares = pick_first(records[0], "commonStockSharesOutstanding")
        previous_shares = pick_first(records[1], "commonStockSharesOutstanding")
        if current_shares and previous_shares and current_shares > 0 and previous_shares > 0:
            return 1.0 - (current_shares / previous_shares)

    annual_records = annual_outstanding_share_records(fundamentals)
    if len(annual_records) >= 2:
        current_shares = pick_first(annual_records[0], "shares", "sharesMln")
        previous_shares = pick_first(annual_records[1], "shares", "sharesMln")
        if current_shares and previous_shares and current_shares > 0 and previous_shares > 0:
            return 1.0 - (current_shares / previous_shares)

    return None


def compute_sentiment_metrics(client: EODHDClient, symbol: str, lookback_days: int) -> Dict[str, Optional[float]]:
    empty_result = {
        "sentiment_latest": None,
        "sentiment_speed": None,
        "sentiment_acceleration": None,
        "sentiment_count_days": 0,
        "sentiment_article_count_recent": 0.0,
        "sentiment_article_count_total": 0.0,
        "sentiment_latest_count": 0.0,
        "sentiment_fetch_status": "empty",
        "sentiment_fetch_error": 0.0,
        "sentiment_fetch_error_type": None,
    }

    end_date = utc_today_ts().strftime("%Y-%m-%d")
    start_date = (utc_today_ts() - pd.Timedelta(days=max(lookback_days * 2, 28))).strftime("%Y-%m-%d")

    try:
        payload = client.get_sentiments(symbol, start_date=start_date, end_date=end_date)
    except Exception as exc:
        return {
            **empty_result,
            "sentiment_fetch_status": "error",
            "sentiment_fetch_error": 1.0,
            "sentiment_fetch_error_type": type(exc).__name__,
        }

    series = None
    if isinstance(payload, dict):
        for key in [symbol, symbol.upper(), symbol.lower()]:
            if key in payload and isinstance(payload[key], list):
                series = payload[key]
                break
        if series is None and len(payload) == 1:
            only_value = next(iter(payload.values()))
            if isinstance(only_value, list):
                series = only_value

    if not isinstance(series, list) or not series:
        return empty_result

    rows = []
    for item in series:
        if not isinstance(item, dict):
            continue
        count_value = to_float(item.get("count"))
        rows.append(
            {
                "date": pd.to_datetime(item.get("date"), errors="coerce"),
                "normalized": to_float(item.get("normalized")),
                "count": float(count_value) if count_value is not None else 1.0,
            }
        )

    sdf = pd.DataFrame(rows).dropna(subset=["date", "normalized"]).sort_values("date")
    if sdf.empty:
        return empty_result

    normalized = sdf["normalized"].astype(float).reset_index(drop=True)
    counts = sdf["count"].clip(lower=0.0).astype(float).reset_index(drop=True)
    count_days = int(len(normalized))
    span = min(7, max(3, count_days))

    weighted_numerator = (normalized * counts).ewm(span=span, adjust=False).mean()
    weighted_denominator = counts.ewm(span=span, adjust=False).mean()
    weighted_ema = (weighted_numerator / weighted_denominator.replace(0.0, np.nan)).fillna(
        normalized.ewm(span=span, adjust=False).mean()
    )

    latest = float(weighted_ema.iloc[-1])
    speed = None
    acceleration = None
    if count_days >= 2:
        speed = float(weighted_ema.iloc[-1] - weighted_ema.iloc[-2])
    if count_days >= 3:
        previous_speed = float(weighted_ema.iloc[-2] - weighted_ema.iloc[-3])
        acceleration = float(speed - previous_speed) if speed is not None else None

    return {
        "sentiment_latest": latest,
        "sentiment_speed": speed,
        "sentiment_acceleration": acceleration,
        "sentiment_count_days": count_days,
        "sentiment_article_count_recent": float(counts.tail(5).sum()),
        "sentiment_article_count_total": float(counts.sum()),
        "sentiment_latest_count": float(counts.iloc[-1]),
        "sentiment_fetch_status": "ok",
        "sentiment_fetch_error": 0.0,
        "sentiment_fetch_error_type": None,
    }


def _normalize_news_text(article: Dict[str, Any]) -> str:
    parts = [str(article.get("title") or "")]
    tags = article.get("tags")
    if isinstance(tags, list):
        parts.extend(str(tag) for tag in tags if tag)
    return re.sub(r"\s+", " ", " ".join(parts)).strip().lower()


def _matched_news_categories(
    article: Dict[str, Any],
    category_specs: Dict[str, Dict[str, Any]] | None = None,
) -> List[str]:
    blob = _normalize_news_text(article)
    if not blob:
        return []

    active_specs = category_specs or NEWS_EVENT_CATEGORY_SPECS
    matched: List[str] = []
    for category, spec in active_specs.items():
        terms = spec.get("terms", ())
        if any(term in blob for term in terms):
            matched.append(category)
    return matched


def _canonical_news_title(article: Dict[str, Any]) -> str:
    title = str(article.get("title") or "").lower()
    tokens = re.findall(r"[a-z0-9]+", title)
    compact = [token for token in tokens if token not in NEWS_TITLE_STOPWORDS and len(token) > 2]
    return " ".join(compact)


def compute_news_event_metrics(client: EODHDClient, symbol: str, lookback_days: int) -> Dict[str, Optional[float]]:
    empty_result = {
        "news_event_signal": None,
        "news_event_breadth": 0.0,
        "news_article_count_recent": 0.0,
        "news_positive_article_share": 0.0,
        "news_negative_article_share": 0.0,
        "news_unique_title_ratio": 0.0,
        "news_novelty_score": 0.0,
        "news_saturation_score": 0.0,
        "news_fetch_status": "empty",
        "news_fetch_error": 0.0,
        "news_fetch_error_type": None,
    }

    as_of = utc_today_ts()
    end_date = as_of.strftime("%Y-%m-%d")
    start_date = (as_of - pd.Timedelta(days=max(int(lookback_days), 3))).strftime("%Y-%m-%d")
    limit = max(20, min(100, int(lookback_days) * 6))

    try:
        payload = client.get_news(symbol=symbol, start_date=start_date, end_date=end_date, limit=limit)
    except Exception as exc:
        return {
            **empty_result,
            "news_fetch_status": "error",
            "news_fetch_error": 1.0,
            "news_fetch_error_type": type(exc).__name__,
        }

    if not isinstance(payload, list) or not payload:
        return empty_result

    rows = []
    matched_categories_all: set[str] = set()
    canonical_titles: List[str] = []
    half_life_days = max(3.0, float(lookback_days) / 2.0)

    for article in payload:
        if not isinstance(article, dict):
            continue

        published_at = pd.to_datetime(article.get("date"), errors="coerce", utc=True)
        if pd.isna(published_at):
            continue
        if published_at.normalize() > as_of:
            continue

        sentiment = article.get("sentiment") if isinstance(article.get("sentiment"), dict) else {}
        polarity = to_float(sentiment.get("polarity")) if isinstance(sentiment, dict) else None
        polarity = float(polarity) if polarity is not None else 0.0

        categories = _matched_news_categories(article)
        matched_categories_all.update(categories)
        canonical_title = _canonical_news_title(article)
        if canonical_title:
            canonical_titles.append(canonical_title)
        raw_event_bias = sum(float(NEWS_EVENT_CATEGORY_SPECS[name]["bias"]) for name in categories)
        if categories:
            event_bias = float(np.clip(raw_event_bias / max(1.5, float(len(categories))), -1.0, 1.0))
        else:
            event_bias = 0.0

        article_signal = float(np.clip(0.65 * event_bias + 0.35 * polarity, -1.0, 1.0))

        mentioned_symbols = article.get("symbols")
        symbol_count = len(mentioned_symbols) if isinstance(mentioned_symbols, list) and mentioned_symbols else 1
        relevance_weight = 1.0 / np.sqrt(float(max(1, symbol_count)))

        age_days = max(0.0, float((as_of - published_at.normalize()).days))
        recency_weight = float(np.exp(-np.log(2.0) * age_days / half_life_days))
        article_weight = max(1e-6, relevance_weight * recency_weight)

        rows.append(
            {
                "signal": article_signal,
                "weight": article_weight,
                "direction": float(np.sign(article_signal)),
            }
        )

    if not rows:
        return empty_result

    news_df = pd.DataFrame(rows)
    total_weight = float(news_df["weight"].sum())
    if total_weight <= 0:
        return empty_result

    weighted_signal = float((news_df["signal"] * news_df["weight"]).sum() / total_weight)
    positive_share = float((news_df["signal"] > 0.15).mean())
    negative_share = float((news_df["signal"] < -0.15).mean())
    article_count = float(len(news_df))
    unique_titles = len(set(canonical_titles))
    unique_title_ratio = float(unique_titles / article_count) if article_count > 0 else 0.0
    category_diversity = float(min(1.0, len(matched_categories_all) / 3.0))
    novelty_score = float(np.clip(0.65 * unique_title_ratio + 0.35 * category_diversity, 0.0, 1.0))
    direction_concentration = float(abs(news_df["direction"].mean()))
    article_density = float(min(1.0, article_count / max(3.0, float(lookback_days) / 2.0)))
    saturation_score = float(np.clip(article_density * direction_concentration, 0.0, 1.0))

    return {
        "news_event_signal": weighted_signal,
        "news_event_breadth": float(len(matched_categories_all)),
        "news_article_count_recent": article_count,
        "news_positive_article_share": positive_share,
        "news_negative_article_share": negative_share,
        "news_unique_title_ratio": unique_title_ratio,
        "news_novelty_score": novelty_score,
        "news_saturation_score": saturation_score,
        "news_fetch_status": "ok",
        "news_fetch_error": 0.0,
        "news_fetch_error_type": None,
    }


def compute_news_theme_drift_metrics(
    client: EODHDClient,
    symbol: str,
    recent_window_days: int = 30,
    baseline_window_days: int = 90,
    alpha_factor_spec: str = "legacy",
    revision_support: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    empty_result = {
        "news_theme_drift_signal": None,
        "news_theme_drift_has_coverage": 0.0,
        "news_theme_drift_recent_intensity": 0.0,
        "news_theme_drift_baseline_intensity": 0.0,
        "news_theme_drift_fetch_status": "empty",
        "news_theme_drift_fetch_error": 0.0,
        "news_theme_drift_fetch_error_type": None,
    }

    as_of = utc_today_ts()
    end_date = as_of.strftime("%Y-%m-%d")
    start_date = (as_of - pd.Timedelta(days=max(int(baseline_window_days), int(recent_window_days), 30))).strftime(
        "%Y-%m-%d"
    )
    limit = max(40, min(250, int(max(baseline_window_days, recent_window_days)) * 4))

    try:
        payload = client.get_news(symbol=symbol, start_date=start_date, end_date=end_date, limit=limit)
    except Exception as exc:
        return {
            **empty_result,
            "news_theme_drift_fetch_status": "error",
            "news_theme_drift_fetch_error": 1.0,
            "news_theme_drift_fetch_error_type": type(exc).__name__,
        }

    if not isinstance(payload, list) or not payload:
        return empty_result

    use_v2 = str(alpha_factor_spec).lower() == "v2"
    category_specs = NEWS_THEME_CATEGORY_SPECS_V2 if use_v2 else NEWS_EVENT_CATEGORY_SPECS
    recent_scores: List[float] = []
    baseline_scores: List[float] = []
    for article in payload:
        if not isinstance(article, dict):
            continue

        published_at = pd.to_datetime(article.get("date"), errors="coerce", utc=True)
        if pd.isna(published_at) or published_at.normalize() > as_of:
            continue

        age_days = max(0.0, float((as_of - published_at.normalize()).days))
        if age_days > float(baseline_window_days):
            continue

        categories = _matched_news_categories(article, category_specs=category_specs)
        sentiment = article.get("sentiment") if isinstance(article.get("sentiment"), dict) else {}
        polarity = to_float(sentiment.get("polarity")) if isinstance(sentiment, dict) else None
        polarity = float(polarity) if polarity is not None else 0.0
        category_bias = sum(float(category_specs[name]["bias"]) for name in categories)
        if categories:
            theme_score = float(np.clip((category_bias / max(1.5, float(len(categories)))) + 0.35 * polarity, -1.0, 1.0))
        else:
            theme_score = float(np.clip(0.35 * polarity, -1.0, 1.0))

        symbol_count = len(article.get("symbols") or []) if isinstance(article.get("symbols"), list) else 1
        relevance_weight = 1.0 / np.sqrt(float(max(1, symbol_count)))
        if age_days <= float(recent_window_days):
            recent_scores.append(theme_score * relevance_weight)
        else:
            baseline_scores.append(theme_score * relevance_weight)

    if not recent_scores or not baseline_scores:
        return empty_result

    recent_intensity = float(np.mean(recent_scores))
    baseline_intensity = float(np.mean(baseline_scores))
    drift_signal = float(np.clip((recent_intensity - baseline_intensity) / 0.35, -1.0, 1.0))
    if use_v2:
        novelty_multiplier = 1.0 + 0.20 * min(1.0, len(recent_scores) / 6.0)
        revision_multiplier = 1.0 + 0.25 * float(np.clip(revision_support or 0.0, -1.0, 1.0))
        drift_signal = float(np.clip(drift_signal * novelty_multiplier * revision_multiplier, -1.0, 1.0))

    return {
        "news_theme_drift_signal": drift_signal,
        "news_theme_drift_has_coverage": 1.0,
        "news_theme_drift_recent_intensity": recent_intensity,
        "news_theme_drift_baseline_intensity": baseline_intensity,
        "news_theme_drift_fetch_status": "ok",
        "news_theme_drift_fetch_error": 0.0,
        "news_theme_drift_fetch_error_type": None,
    }


def _insider_role_weight(item: Dict[str, Any], *, strict: bool = False) -> float:
    blob = " ".join(
        [
            str(item.get("ownerRelationship") or ""),
            str(item.get("relationship") or ""),
            str(item.get("officerTitle") or ""),
            str(item.get("title") or ""),
        ]
    ).lower()

    if "chief executive" in blob or "ceo" in blob or "chief financial" in blob or "cfo" in blob:
        return 2.0
    if "director" in blob or "officer" in blob or "president" in blob or "vp" in blob:
        return 1.0
    return 0.5 if strict else 0.75


def compute_insider_conviction_metrics(
    client: EODHDClient,
    symbol: str,
    lookback_days: int = 90,
    alpha_factor_spec: str = "legacy",
    revision_support: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    empty_result = {
        "insider_conviction_signal": None,
        "insider_conviction_has_coverage": 0.0,
        "insider_conviction_buy_cluster": 0.0,
        "insider_conviction_sell_pressure": 0.0,
        "insider_fetch_status": "empty",
        "insider_fetch_error": 0.0,
        "insider_fetch_error_type": None,
    }

    as_of = utc_today_ts()
    end_date = as_of.strftime("%Y-%m-%d")
    start_date = (as_of - pd.Timedelta(days=max(int(lookback_days), 30))).strftime("%Y-%m-%d")

    try:
        payload = client.get_insider_transactions(symbol=symbol, start_date=start_date, end_date=end_date, limit=200)
    except Exception as exc:
        return {
            **empty_result,
            "insider_fetch_status": "error",
            "insider_fetch_error": 1.0,
            "insider_fetch_error_type": type(exc).__name__,
        }

    if not isinstance(payload, list) or not payload:
        return empty_result

    use_v2 = str(alpha_factor_spec).lower() == "v2"
    weighted_buy = 0.0
    weighted_sell = 0.0
    buy_people: set[str] = set()
    sell_people: set[str] = set()

    for item in payload:
        if not isinstance(item, dict):
            continue

        trade_date = pd.to_datetime(
            item.get("date") or item.get("transactionDate") or item.get("filingDate"),
            errors="coerce",
            utc=True,
        )
        if pd.isna(trade_date) or trade_date.normalize() > as_of:
            continue

        age_days = max(0.0, float((as_of - trade_date.normalize()).days))
        if age_days > float(lookback_days):
            continue

        code = str(
            item.get("transactionCode")
            or item.get("transactionType")
            or item.get("type")
            or item.get("acquisitionOrDisposition")
            or ""
        ).upper()
        is_buy = code in {"P", "A", "BUY", "ACQUIRE"}
        is_sell = code in {"S", "D", "SELL", "DISPOSE"}
        if not is_buy and not is_sell:
            continue

        shares = to_float(item.get("transactionAmount") or item.get("shares") or item.get("shareAmount"))
        price = to_float(item.get("transactionPrice") or item.get("price"))
        base_size = float(shares) if shares is not None else 1.0
        if price is not None and price > 0:
            base_size *= float(price)

        recency_weight = float(np.exp(-np.log(2.0) * age_days / max(20.0, float(lookback_days) / 2.0)))
        role_weight = _insider_role_weight(item, strict=use_v2)
        trade_weight = max(1.0, base_size ** 0.5) * recency_weight * role_weight
        owner = str(item.get("ownerName") or item.get("name") or "").strip().lower()

        if is_buy:
            weighted_buy += trade_weight
            if owner:
                buy_people.add(owner)
        elif is_sell:
            weighted_sell += trade_weight
            if owner:
                sell_people.add(owner)

    total_activity = weighted_buy + weighted_sell
    if total_activity <= 0:
        return empty_result

    buy_cluster = float(min(1.0, len(buy_people) / 3.0))
    buy_strength = float(np.tanh((weighted_buy / max(total_activity, 1.0)) * (1.0 + buy_cluster)))
    sell_pressure = 0.0
    if len(sell_people) >= 2 and weighted_sell > 1.25 * max(weighted_buy, 1.0):
        sell_pressure = float(np.tanh(weighted_sell / max(weighted_buy + 1.0, 1.0)))

    signal = float(np.clip(buy_strength - 0.85 * sell_pressure, -1.0, 1.0))
    if use_v2:
        revision_boost = 0.20 * float(np.clip(revision_support or 0.0, -1.0, 1.0))
        signal = float(np.clip(signal + revision_boost, -1.0, 1.0))
    return {
        "insider_conviction_signal": signal,
        "insider_conviction_has_coverage": 1.0,
        "insider_conviction_buy_cluster": buy_cluster,
        "insider_conviction_sell_pressure": sell_pressure,
        "insider_fetch_status": "ok",
        "insider_fetch_error": 0.0,
        "insider_fetch_error_type": None,
    }


def compute_pead_metrics_from_calendar(
    client: EODHDClient,
    symbol: Optional[str],
    lookback_days: int,
    half_life_days: int,
    max_abs_surprise_pct: float,
    max_age_days: int,
) -> Dict[str, Optional[float]]:
    from_date = (utc_today_ts() - pd.Timedelta(days=max(lookback_days * 2, 365))).strftime("%Y-%m-%d")
    to_date = utc_today_ts().strftime("%Y-%m-%d")

    try:
        payload = client.get_earnings_calendar(symbol, from_date=from_date, to_date=to_date)
    except Exception:
        return {
            "earnings_surprise_pct": None,
            "earnings_report_date": None,
            "pead_signal_calendar": None,
        }

    records = []
    if isinstance(payload, list):
        records = [item for item in payload if isinstance(item, dict)]
    elif isinstance(payload, dict):
        if isinstance(payload.get("earnings"), list):
            records = [item for item in payload["earnings"] if isinstance(item, dict)]
        elif symbol and isinstance(payload.get(symbol), list):
            records = [item for item in payload[symbol] if isinstance(item, dict)]
        else:
            for value in payload.values():
                if isinstance(value, list):
                    records = [item for item in value if isinstance(item, dict)]
                    if records:
                        break

    if not records:
        return {
            "earnings_surprise_pct": None,
            "earnings_report_date": None,
            "pead_signal_calendar": None,
        }

    def parse_report_date(item: Dict[str, Any]) -> pd.Timestamp:
        return pd.to_datetime(item.get("report_date") or item.get("date"), errors="coerce", utc=True)

    def parse_surprise_pct(item: Dict[str, Any]) -> Optional[float]:
        pct = to_float(item.get("percent"))
        actual = to_float(item.get("actual"))
        estimate = to_float(item.get("estimate"))
        difference = to_float(item.get("difference"))

        if pct is None and difference is None and actual is not None and estimate is not None:
            difference = actual - estimate
        if pct is None and difference is not None and estimate not in (None, 0):
            pct = (difference / abs(float(estimate))) * 100.0
        if pct is None or estimate in (None, 0) or abs(float(estimate)) < 0.02:
            return None
        return max(-100.0, min(100.0, float(pct)))

    records = sorted(records, key=parse_report_date, reverse=True)
    chosen = None
    for item in records:
        report_date = parse_report_date(item)
        if pd.isna(report_date) or report_date.normalize() > utc_today_ts():
            continue
        if parse_surprise_pct(item) is None:
            continue
        chosen = item
        break

    if chosen is None:
        return {
            "earnings_surprise_pct": None,
            "earnings_report_date": None,
            "pead_signal_calendar": None,
        }

    report_date = parse_report_date(chosen)
    surprise_pct = parse_surprise_pct(chosen)
    if pd.isna(report_date) or surprise_pct is None:
        return {
            "earnings_surprise_pct": None,
            "earnings_report_date": None,
            "pead_signal_calendar": None,
        }

    return {
        "earnings_surprise_pct": float(surprise_pct),
        "earnings_report_date": report_date.strftime("%Y-%m-%d"),
        "pead_signal_calendar": compute_pead_signal_from_surprise(
            earnings_surprise_pct=surprise_pct,
            earnings_report_date=report_date,
            half_life_days=half_life_days,
            max_abs_surprise_pct=max_abs_surprise_pct,
            max_age_days=max_age_days,
        ),
    }


def compute_fundamental_metrics(
    client: EODHDClient,
    symbol: str,
    fundamentals: Dict[str, Any],
    dividend_source: str,
    config: RankerConfig,
) -> Dict[str, Any]:
    alpha_factor_spec = str(getattr(config, "alpha_factor_spec", "legacy") or "legacy").lower()
    general = fundamentals.get("General") or {}
    highlights = fundamentals.get("Highlights") or {}
    splits_dividends = fundamentals.get("SplitsDividends") or {}
    technicals = fundamentals.get("Technicals") or {}
    shares_stats = fundamentals.get("SharesStats") or {}

    income_records = income_statement_records(fundamentals)
    balance_records = balance_sheet_records(fundamentals)
    latest_income = income_records[0] if income_records else {}
    latest_balance = balance_records[0] if balance_records else {}

    market_cap = pick_first(highlights, "MarketCapitalization")
    gross_profit = pick_first(latest_income, "grossProfit")
    total_assets = pick_first(latest_balance, "totalAssets")
    reported_gross_profitability = (
        gross_profit / total_assets if gross_profit is not None and total_assets and total_assets > 0 else None
    )

    total_revenue = pick_first(latest_income, "totalRevenue")
    full_time_employees = to_float(general.get("FullTimeEmployees"))
    revenue_per_employee = (
        total_revenue / full_time_employees
        if total_revenue is not None and full_time_employees and full_time_employees > 0
        else None
    )
    gross_profit_per_employee = (
        gross_profit / full_time_employees
        if gross_profit is not None and full_time_employees and full_time_employees > 0
        else None
    )

    base_equity = pick_first(
        latest_balance,
        "totalStockholderEquity",
        "commonStockTotalEquity",
        "totalEquity",
        "bookValue",
    )
    if base_equity is None:
        total_liabilities = pick_first(latest_balance, "totalLiab")
        if total_assets and total_liabilities is not None:
            base_equity = total_assets - total_liabilities

    reported_book_to_market = (
        base_equity / market_cap
        if base_equity is not None and market_cap and market_cap > 0
        else None
    )
    rd_asset = capitalize_expense(income_records, field="researchDevelopment", years=3, scale=1.0)
    sga_asset = capitalize_expense(income_records, field="sellingGeneralAdministrative", years=2, scale=0.30)
    rd_latest = pick_first(latest_income, "researchDevelopment")
    rd_ratio = (
        float(rd_latest / total_revenue)
        if rd_latest is not None and total_revenue and total_revenue > 0
        else None
    )
    sector_text = str(general.get("Sector") or "").strip().lower()
    intangible_adjustment_eligible = bool(
        (rd_ratio is not None and rd_ratio >= 0.02)
        or sector_text in INTANGIBLE_ADJUSTMENT_SECTORS
    )
    intangible_adjusted_equity = (
        base_equity + (rd_asset or 0.0) + (sga_asset or 0.0)
        if base_equity is not None and intangible_adjustment_eligible
        else None
    )
    intangible_adjusted_assets = (
        total_assets + (rd_asset or 0.0) + (sga_asset or 0.0)
        if total_assets is not None and intangible_adjustment_eligible
        else None
    )
    intangible_adjusted_book_to_market = (
        intangible_adjusted_equity / market_cap
        if intangible_adjusted_equity is not None and market_cap and market_cap > 0
        else None
    )
    intangible_adjusted_gross_profitability = (
        gross_profit / intangible_adjusted_assets
        if gross_profit is not None and intangible_adjusted_assets and intangible_adjusted_assets > 0
        else None
    )
    intangible_adjustment_applied = bool(
        getattr(config, "use_intangible_adjustments", False)
        and intangible_adjustment_eligible
        and (intangible_adjusted_book_to_market is not None or intangible_adjusted_gross_profitability is not None)
    )
    adjusted_book = intangible_adjusted_equity if intangible_adjusted_equity is not None else base_equity
    gross_profitability = (
        intangible_adjusted_gross_profitability
        if intangible_adjustment_applied and intangible_adjusted_gross_profitability is not None
        else reported_gross_profitability
    )
    adjusted_book_to_market = (
        intangible_adjusted_book_to_market
        if intangible_adjustment_applied and intangible_adjusted_book_to_market is not None
        else reported_book_to_market
    )
    net_income = pick_first(latest_income, "netIncome")
    total_debt = pick_first(latest_balance, "longTermDebt", "shortLongTermDebtTotal", "totalDebt")
    cash_like_assets = pick_first(
        latest_balance,
        "cashAndCashEquivalents",
        "cashAndShortTermInvestments",
        "cash",
        "cashAndEquivalents",
    )
    current_liabilities = pick_first(latest_balance, "totalCurrentLiabilities")
    reported_invested_capital = None
    if base_equity is not None or total_debt is not None:
        reported_invested_capital = float((base_equity or 0.0) + (total_debt or 0.0) - (cash_like_assets or 0.0))
    if reported_invested_capital is None or reported_invested_capital <= 0:
        if total_assets is not None:
            reported_invested_capital = float(total_assets - (current_liabilities or 0.0) - (cash_like_assets or 0.0))
    if reported_invested_capital is not None and reported_invested_capital <= 0:
        reported_invested_capital = None
    intangible_adjusted_invested_capital = (
        reported_invested_capital + (rd_asset or 0.0) + (sga_asset or 0.0)
        if reported_invested_capital is not None and intangible_adjustment_eligible
        else None
    )
    reported_return_on_assets = (
        net_income / total_assets if net_income is not None and total_assets and total_assets > 0 else None
    )
    intangible_adjusted_return_on_assets = (
        net_income / intangible_adjusted_assets
        if net_income is not None and intangible_adjusted_assets and intangible_adjusted_assets > 0
        else None
    )
    reported_return_on_invested_capital = (
        net_income / reported_invested_capital
        if net_income is not None and reported_invested_capital and reported_invested_capital > 0
        else None
    )
    intangible_adjusted_return_on_invested_capital = (
        net_income / intangible_adjusted_invested_capital
        if net_income is not None and intangible_adjusted_invested_capital and intangible_adjusted_invested_capital > 0
        else None
    )
    effective_return_on_assets = (
        intangible_adjusted_return_on_assets
        if intangible_adjustment_applied and intangible_adjusted_return_on_assets is not None
        else reported_return_on_assets
    )
    effective_return_on_invested_capital = (
        intangible_adjusted_return_on_invested_capital
        if intangible_adjustment_applied and intangible_adjusted_return_on_invested_capital is not None
        else reported_return_on_invested_capital
    )

    margin_source_records = income_statement_records(fundamentals, "quarterly")
    if len(margin_source_records) < 4:
        margin_source_records = income_records
    margin_values: list[float] = []
    for record in reversed(margin_source_records[:4]):
        revenue = pick_first(record, "totalRevenue")
        gross_profit_record = pick_first(record, "grossProfit")
        cost_of_revenue = pick_first(record, "costOfRevenue")
        if revenue is None or revenue <= 0:
            continue
        if gross_profit_record is None and cost_of_revenue is not None:
            gross_profit_record = revenue - cost_of_revenue
        if gross_profit_record is None:
            continue
        margin_values.append(float(gross_profit_record / revenue))
    peer_margin_trend_input = None
    if len(margin_values) >= 2:
        peer_margin_trend_input = float(margin_values[-1] - np.mean(margin_values[:-1]))

    previous_income = income_records[1] if len(income_records) >= 2 else {}
    previous_balance = balance_records[1] if len(balance_records) >= 2 else {}
    previous_revenue = pick_first(previous_income, "totalRevenue")
    revenue_growth_hint = (
        (float(total_revenue) - float(previous_revenue)) / max(abs(float(previous_revenue)), 1.0)
        if total_revenue is not None and previous_revenue not in (None, 0)
        else None
    )
    capex_hint = None
    latest_cashflow = cash_flow_records(fundamentals)[0] if cash_flow_records(fundamentals) else {}
    capex_value = pick_first(latest_cashflow, "capitalExpenditures", "capitalExpenditure")
    if capex_value is not None and total_assets and total_assets > 0:
        capex_hint = abs(float(capex_value)) / float(total_assets)
    peer_reinvestment_efficiency_input = None
    if revenue_growth_hint is not None:
        peer_reinvestment_efficiency_input = float(revenue_growth_hint - 0.75 * float(capex_hint or 0.0) - 0.25 * float(rd_ratio or 0.0))

    shares_outstanding = pick_first(latest_balance, "commonStockSharesOutstanding")
    if shares_outstanding is None:
        shares_outstanding = pick_first(shares_stats, "SharesOutstanding")
    previous_shares = pick_first(previous_balance, "commonStockSharesOutstanding")
    share_count_discipline_input = (
        (float(previous_shares) - float(shares_outstanding)) / max(abs(float(previous_shares)), 1.0)
        if shares_outstanding is not None and previous_shares not in (None, 0)
        else None
    )
    price_proxy = (
        market_cap / shares_outstanding
        if market_cap and shares_outstanding and shares_outstanding > 0
        else None
    )

    high_52 = pick_first(technicals, "52WeekHigh")
    ma200 = pick_first(technicals, "200DayMA")
    recency_ratio = (price_proxy / high_52) if price_proxy and high_52 and high_52 > 0 else None
    distance_from_high = recency_ratio - 1.0 if recency_ratio is not None else None
    price_to_200dma = (price_proxy / ma200) if price_proxy and ma200 and ma200 > 0 else None
    listing_exchange_codes = sorted(extract_listing_exchange_codes(general))

    if dividend_source == "trailing":
        trailing_cash_per_share = compute_trailing_dividend_cash_per_share(client, symbol)
        dividend_yield = (
            trailing_cash_per_share / price_proxy
            if trailing_cash_per_share and price_proxy and price_proxy > 0
            else None
        )
    elif dividend_source == "forward":
        dividend_yield = pick_first(splits_dividends, "ForwardAnnualDividendYield")
    else:
        dividend_yield = pick_first(splits_dividends, "ForwardAnnualDividendYield")
        if dividend_yield is None:
            trailing_cash_per_share = compute_trailing_dividend_cash_per_share(client, symbol)
            dividend_yield = (
                trailing_cash_per_share / price_proxy
                if trailing_cash_per_share and price_proxy and price_proxy > 0
                else None
            )

    buyback_yield = compute_buyback_yield(fundamentals)
    payout_ratio = pick_first(highlights, "PayoutRatio")
    piotroski_score = compute_piotroski_f_score(fundamentals)

    dividend_safety_pass = True
    if dividend_yield is not None and dividend_yield > 0:
        if payout_ratio is not None and payout_ratio > config.dividend_payout_cap:
            dividend_safety_pass = False
        if distance_from_high is not None and distance_from_high < -abs(config.max_distance_from_high):
            dividend_safety_pass = False
        if config.require_above_200dma and price_to_200dma is not None and price_to_200dma < 1.0:
            dividend_safety_pass = False

    safe_dividend_yield = dividend_yield if dividend_safety_pass else 0.0
    shareholder_yield = (
        (safe_dividend_yield or 0.0) + (buyback_yield or 0.0)
        if (safe_dividend_yield is not None or buyback_yield is not None)
        else None
    )

    metrics: Dict[str, Any] = {
        "alpha_factor_spec": alpha_factor_spec,
        "market_cap": market_cap,
        "dividend_yield": dividend_yield,
        "safe_dividend_yield": safe_dividend_yield,
        "buyback_yield": buyback_yield,
        "shareholder_yield": shareholder_yield,
        "gross_profitability": gross_profitability,
        "reported_gross_profitability": reported_gross_profitability,
        "intangible_adjusted_gross_profitability": intangible_adjusted_gross_profitability,
        "base_book_equity": base_equity,
        "reported_book_to_market": reported_book_to_market,
        "rd_asset_3y": rd_asset,
        "sga_asset_2y_30pct": sga_asset,
        "adjusted_book": adjusted_book,
        "adjusted_book_to_market": adjusted_book_to_market,
        "intangible_adjusted_book_to_market": intangible_adjusted_book_to_market,
        "reported_return_on_assets": reported_return_on_assets,
        "intangible_adjusted_return_on_assets": intangible_adjusted_return_on_assets,
        "return_on_assets": effective_return_on_assets,
        "reported_return_on_invested_capital": reported_return_on_invested_capital,
        "intangible_adjusted_return_on_invested_capital": intangible_adjusted_return_on_invested_capital,
        "return_on_invested_capital": effective_return_on_invested_capital,
        "rd_expense_ratio": rd_ratio,
        "intangible_adjustment_eligible": float(intangible_adjustment_eligible),
        "intangible_adjustment_applied": float(intangible_adjustment_applied),
        "sector": general.get("Sector"),
        "industry": general.get("Industry"),
        "exchange": general.get("Exchange"),
        "listing_exchanges": ",".join(listing_exchange_codes),
        "currency_code": general.get("CurrencyCode"),
        "currency_name": general.get("CurrencyName"),
        "company_name": general.get("Name"),
        "asset_type": general.get("Type"),
        "country": general.get("CountryName"),
        "country_iso": general.get("CountryISO"),
        "isin": general.get("ISIN"),
        "primary_ticker": general.get("PrimaryTicker"),
        "home_category": general.get("HomeCategory"),
        "international_domestic": general.get("InternationalDomestic"),
        "payout_ratio": payout_ratio,
        "shares_outstanding": shares_outstanding,
        "price_proxy": price_proxy,
        "52_week_high": high_52,
        "200_day_ma": ma200,
        "recency_ratio": recency_ratio,
        "distance_from_high": distance_from_high,
        "price_to_200dma": price_to_200dma,
        "dividend_safety_pass": float(dividend_safety_pass),
        "piotroski_score": piotroski_score,
        "total_revenue": total_revenue,
        "full_time_employees": full_time_employees,
        "revenue_per_employee": revenue_per_employee,
        "gross_profit_per_employee": gross_profit_per_employee,
        "peer_margin_trend_input": peer_margin_trend_input,
        "peer_reinvestment_efficiency_input": peer_reinvestment_efficiency_input,
        "peer_estimate_drift_input": None,
        "peer_dilution_discipline_input": share_count_discipline_input,
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
        "sue_signal": None,
        "sue_has_coverage": 0.0,
        "sue_surprise_raw": None,
        "sue_surprise_pct": None,
        "sue_std_error": None,
        "sue_report_date": None,
        "revision_impulse_signal": None,
        "revision_impulse_has_coverage": 0.0,
        "revision_impulse_analyst_count": None,
        "revision_impulse_drift_7d": None,
        "revision_impulse_drift_30d": None,
        "revision_impulse_breadth": None,
        "revision_impulse_growth_component": None,
        "revision_impulse_coverage_component": None,
        "revision_impulse_disagreement": None,
        "revision_impulse_disagreement_penalty": None,
        "estimate_term_structure_signal": None,
        "estimate_term_structure_has_coverage": 0.0,
        "estimate_term_structure_record_count": 0.0,
        "estimate_term_structure_persistence": None,
        "estimate_term_structure_improvement": None,
        "estimate_term_structure_disagreement_trend": None,
        "estimate_term_structure_coverage_component": None,
        "revenue_growth_yoy": None,
        "revenue_growth_yoy_prev": None,
        "revenue_acceleration": None,
        "revenue_growth_has_coverage": 0.0,
        "compounder_persistence_signal": None,
        "compounder_persistence_has_coverage": 0.0,
        "compounder_persistence_measure_count": 0.0,
        "compounder_persistence_level_component": None,
        "compounder_persistence_stability_component": None,
        "compounder_persistence_trend_component": None,
        "compounder_persistence_periodicity": None,
        "price_momentum_1m": None,
        "price_momentum_6m": None,
        "price_momentum_6m_ex_1m": None,
        "price_momentum_has_coverage": 0.0,
        "price_momentum_effective_signal": None,
        "price_momentum_signal_coverage": 0.0,
        "price_momentum_proxy_used": 0.0,
        "price_history_fetch_status": "not_requested",
        "price_history_fetch_error": 0.0,
        "price_history_fetch_error_type": None,
        "beneish_m_score": None,
        "beneish_data_status": None,
        "beneish_is_missing": 0.0,
        "beneish_is_pathological_clipped": 0.0,
        "beneish_hard_filter_pass": 1.0,
        "accrual_ratio": None,
        "accrual_volatility": None,
        "accrual_measure_count": None,
        "accrual_is_quarterly": None,
        "working_capital_stress_penalty": 0.0,
        "working_capital_stress_has_coverage": 0.0,
        "working_capital_receivables_stress": None,
        "working_capital_inventory_stress": None,
        "working_capital_payables_stress": None,
        "working_capital_cfo_stress": None,
        "investment_restraint_signal": None,
        "investment_restraint_has_coverage": 0.0,
        "investment_restraint_measure_count": 0.0,
        "investment_restraint_asset_growth": None,
        "investment_restraint_noa_growth": None,
        "investment_restraint_acquisition_intensity": None,
        "investment_restraint_capex_intensity": None,
        "investment_restraint_share_issuance": None,
        "investment_restraint_debt_funded_expansion": None,
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
        "capital_allocation_quality_signal": None,
        "capital_allocation_quality_has_coverage": 0.0,
        "capital_allocation_buyback_component": None,
        "capital_allocation_funding_component": None,
        "capital_allocation_debt_component": None,
        "capital_allocation_payout_component": None,
        "capital_allocation_reinvestment_component": None,
        "recovery_margin_inflection": None,
        "recovery_leverage_improvement": None,
        "recovery_accrual_improvement": None,
        "peer_relative_anomaly_signal": None,
        "peer_relative_anomaly_has_coverage": 0.0,
        "peer_relative_anomaly_peer_level": None,
        "peer_relative_margin_component": None,
        "peer_relative_reinvestment_component": None,
        "peer_relative_estimate_component": None,
        "peer_relative_dilution_component": None,
    }

    if config.use_pead:
        metrics.update(
            compute_pead_metrics_from_fundamentals(
                fundamentals,
                min_pead_analysts=config.min_pead_analysts,
                half_life_days=config.pead_half_life_days,
                max_abs_surprise_pct=config.pead_max_abs_surprise_pct,
                max_age_days=config.pead_max_age_days,
            )
        )
        metrics.update(compute_sue_metrics_from_fundamentals(fundamentals))

    if config.use_revision_impulse:
        metrics.update(
            compute_revision_impulse_metrics_from_fundamentals(
                fundamentals,
                min_revision_analysts=config.min_revision_analysts,
            )
        )

    if getattr(config, "use_estimate_term_structure", False):
        metrics.update(
            compute_estimate_term_structure_metrics_from_fundamentals(
                fundamentals,
                min_revision_analysts=config.min_revision_analysts,
                alpha_factor_spec=alpha_factor_spec,
            )
        )

    if getattr(config, "use_growth_acceleration", False):
        metrics.update(compute_revenue_growth_metrics_from_fundamentals(fundamentals))

    if getattr(config, "use_compounder_persistence", False):
        metrics.update(
            compute_compounder_persistence_metrics_from_fundamentals(
                fundamentals,
                alpha_factor_spec=alpha_factor_spec,
                use_intangible_adjustments=bool(getattr(config, "use_intangible_adjustments", False)),
            )
        )

    if getattr(config, "use_price_momentum", False):
        source_mode = str(getattr(config, "price_momentum_source_mode", "auto") or "auto").strip().lower()
        if source_mode != "trend_proxy":
            try:
                to_date = utc_today_ts().strftime("%Y-%m-%d")
                from_date = (utc_today_ts() - pd.Timedelta(days=420)).strftime("%Y-%m-%d")
                price_history = client.get_price_history(symbol, from_date=from_date, to_date=to_date)
                metrics.update(compute_price_momentum_metrics_from_history(price_history))
                metrics["price_history_fetch_status"] = (
                    "ok" if float(metrics.get("price_momentum_has_coverage", 0.0) or 0.0) >= 1.0 else "empty"
                )
            except Exception as exc:
                metrics["price_history_fetch_status"] = "error"
                metrics["price_history_fetch_error"] = 1.0
                metrics["price_history_fetch_error_type"] = type(exc).__name__

        if metrics.get("price_momentum_has_coverage", 0.0) >= 1.0:
            metrics["price_momentum_effective_signal"] = metrics.get("price_momentum_6m_ex_1m")
            metrics["price_momentum_signal_coverage"] = 1.0
            metrics["price_momentum_proxy_used"] = 0.0
        elif source_mode != "history_only":
            metrics.update(
                compute_price_momentum_proxy_metrics(
                    price_to_200dma=metrics.get("price_to_200dma"),
                    recency_ratio=metrics.get("recency_ratio"),
                    distance_from_high=metrics.get("distance_from_high"),
                )
            )
            if metrics["price_history_fetch_status"] == "not_requested":
                metrics["price_history_fetch_status"] = "proxy_only"
        else:
            metrics["price_momentum_effective_signal"] = None
            metrics["price_momentum_signal_coverage"] = 0.0
            metrics["price_momentum_proxy_used"] = 0.0
            if metrics["price_history_fetch_status"] == "not_requested":
                metrics["price_history_fetch_status"] = "history_only_unavailable"

    if config.use_beneish:
        beneish_metrics = compute_beneish_metrics(fundamentals)
        metrics.update(beneish_metrics)
        beneish_m_score = beneish_metrics.get("beneish_m_score")
        if beneish_m_score is not None:
            metrics["beneish_hard_filter_pass"] = 1.0 if beneish_m_score <= -1.20 else 0.0

    if config.use_accrual_volatility:
        metrics.update(compute_accrual_metrics(fundamentals))

    if getattr(config, "use_working_capital_stress", False):
        metrics.update(
            compute_working_capital_stress_metrics(
                fundamentals,
                alpha_factor_spec=alpha_factor_spec,
            )
        )

    if getattr(config, "use_investment_restraint", False):
        metrics.update(compute_investment_restraint_metrics(fundamentals))

    if getattr(config, "use_accrual_quality", False):
        metrics.update(compute_accrual_quality_metrics(fundamentals))

    if getattr(config, "use_capital_allocation_quality", False):
        metrics.update(
            compute_capital_allocation_quality_metrics(
                fundamentals,
                alpha_factor_spec=alpha_factor_spec,
            )
        )

    if getattr(config, "use_recovery_transition", False):
        metrics.update(
            compute_recovery_fundamental_metrics(
                fundamentals,
                alpha_factor_spec=alpha_factor_spec,
            )
        )

    estimate_drift_components = [
        float(value)
        for value in [
            metrics.get("revision_impulse_signal"),
            metrics.get("estimate_term_structure_signal"),
        ]
        if value is not None
    ]
    metrics["peer_estimate_drift_input"] = (
        float(sum(estimate_drift_components) / len(estimate_drift_components))
        if estimate_drift_components
        else None
    )
    if metrics.get("peer_reinvestment_efficiency_input") is None and metrics.get("revenue_growth_yoy") is not None:
        metrics["peer_reinvestment_efficiency_input"] = float(metrics["revenue_growth_yoy"])

    return metrics


def add_overlay_metrics(client: EODHDClient, row: Dict[str, Any], config: RankerConfig) -> Dict[str, Any]:
    out = dict(row)
    symbol = str(row.get("analysis_symbol") or row.get("symbol") or "")

    if config.use_pead and out.get("earnings_surprise_pct") is None and out.get("earnings_report_date") is None:
        fallback = compute_pead_metrics_from_calendar(
            client,
            symbol,
            config.pead_lookback_days,
            config.pead_half_life_days,
            config.pead_max_abs_surprise_pct,
            config.pead_max_age_days,
        )
        out["earnings_surprise_pct"] = fallback.get("earnings_surprise_pct")
        out["earnings_report_date"] = fallback.get("earnings_report_date")
        if out.get("pead_signal") is None:
            out["pead_signal"] = fallback.get("pead_signal_calendar")
            out["pead_signal_v2"] = fallback.get("pead_signal_calendar")
        out.setdefault("pead_filter_pass", 1.0)
        out.setdefault("pead_has_setup_coverage", 0.0)
    elif not config.use_pead:
        out.setdefault("earnings_surprise_pct", None)
        out.setdefault("earnings_report_date", None)
        out.setdefault("pead_signal", None)
        out.setdefault("pead_signal_v2", None)
        out.setdefault("pead_surprise_component", None)
        out.setdefault("pead_decay_component", None)
        out.setdefault("pead_breadth_component", None)
        out.setdefault("pead_revision_component", None)
        out.setdefault("pead_analyst_count", None)
        out.setdefault("pead_filter_pass", 1.0)
        out.setdefault("pead_has_setup_coverage", 0.0)

    if config.use_sentiment:
        out.update(compute_sentiment_metrics(client, symbol, config.sentiment_lookback_days))
    else:
        out.setdefault("sentiment_latest", None)
        out.setdefault("sentiment_speed", None)
        out.setdefault("sentiment_acceleration", None)
        out.setdefault("sentiment_count_days", None)
        out.setdefault("sentiment_article_count_recent", None)
        out.setdefault("sentiment_article_count_total", None)
        out.setdefault("sentiment_latest_count", None)
        out.setdefault("sentiment_fetch_status", "not_requested")
        out.setdefault("sentiment_fetch_error", 0.0)
        out.setdefault("sentiment_fetch_error_type", None)

    if config.use_news_events:
        out.update(compute_news_event_metrics(client, symbol, config.news_lookback_days))
    else:
        out.setdefault("news_event_signal", None)
        out.setdefault("news_event_breadth", None)
        out.setdefault("news_article_count_recent", None)
        out.setdefault("news_positive_article_share", None)
        out.setdefault("news_negative_article_share", None)
        out.setdefault("news_unique_title_ratio", None)
        out.setdefault("news_novelty_score", None)
        out.setdefault("news_saturation_score", None)
        out.setdefault("news_fetch_status", "not_requested")
        out.setdefault("news_fetch_error", 0.0)
        out.setdefault("news_fetch_error_type", None)

    if getattr(config, "use_news_theme_drift", False):
        out.update(
            compute_news_theme_drift_metrics(
                client,
                symbol,
                alpha_factor_spec=str(getattr(config, "alpha_factor_spec", "legacy") or "legacy"),
                revision_support=to_float(out.get("revision_impulse_signal")),
            )
        )
    else:
        out.setdefault("news_theme_drift_signal", None)
        out.setdefault("news_theme_drift_has_coverage", 0.0)
        out.setdefault("news_theme_drift_recent_intensity", None)
        out.setdefault("news_theme_drift_baseline_intensity", None)
        out.setdefault("news_theme_drift_fetch_status", "not_requested")
        out.setdefault("news_theme_drift_fetch_error", 0.0)
        out.setdefault("news_theme_drift_fetch_error_type", None)

    if getattr(config, "use_insider_conviction", False):
        out.update(
            compute_insider_conviction_metrics(
                client,
                symbol,
                alpha_factor_spec=str(getattr(config, "alpha_factor_spec", "legacy") or "legacy"),
                revision_support=to_float(out.get("revision_impulse_signal")),
            )
        )
    else:
        out.setdefault("insider_conviction_signal", None)
        out.setdefault("insider_conviction_has_coverage", 0.0)
        out.setdefault("insider_conviction_buy_cluster", 0.0)
        out.setdefault("insider_conviction_sell_pressure", 0.0)
        out.setdefault("insider_fetch_status", "not_requested")
        out.setdefault("insider_fetch_error", 0.0)
        out.setdefault("insider_fetch_error_type", None)

    out["sentiment_filter_pass"] = 1.0
    if config.use_sentiment:
        out["sentiment_filter_pass"] = (
            1.0
            if passes_sentiment_coverage_gate(
                sentiment_count_days=out.get("sentiment_count_days"),
                sentiment_article_count_recent=out.get("sentiment_article_count_recent"),
                sentiment_acceleration=out.get("sentiment_acceleration"),
                min_count_days=config.min_sentiment_days,
                min_article_count_recent=config.min_sentiment_articles_recent,
                min_sentiment_accel=config.min_sentiment_accel,
            )
            else 0.0
        )

    return out
