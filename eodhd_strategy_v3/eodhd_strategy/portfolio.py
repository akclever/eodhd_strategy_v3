from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd

from .config import PortfolioConfig


def _load_previous_symbols(path: Path | None) -> Set[str]:
    if path is None or not path.exists():
        return set()

    prev = pd.read_csv(path)
    if "symbol" not in prev.columns:
        return set()

    return set(prev["symbol"].dropna().astype(str).tolist())


def _force_sell_reason(row: pd.Series, cfg: PortfolioConfig) -> str:
    reasons: List[str] = []

    if cfg.force_sell_on_dividend_break and "dividend_safety_pass" in row and pd.notna(row.get("dividend_safety_pass")):
        if float(row["dividend_safety_pass"]) < 1.0:
            reasons.append("dividend_break")

    if cfg.force_sell_on_sentiment_break and "sentiment_filter_pass" in row and pd.notna(row.get("sentiment_filter_pass")):
        if float(row["sentiment_filter_pass"]) < 1.0:
            reasons.append("sentiment_break")

    if cfg.force_sell_on_trend_break and "price_to_200dma" in row and pd.notna(row.get("price_to_200dma")):
        if float(row["price_to_200dma"]) < float(cfg.force_sell_below_200dma):
            reasons.append("trend_break")

    return "|".join(reasons)


def _sector_slot_limit(cfg: PortfolioConfig) -> int:
    # For equal-weight portfolios, approximate max names per sector
    equal_weight = 1.0 / max(1, int(cfg.top_n_positions))
    return max(1, int(cfg.sector_cap / equal_weight + 1e-9))


def _allocate_equal_weights(selected: pd.DataFrame, cfg: PortfolioConfig) -> pd.DataFrame:
    if selected.empty:
        return selected.copy()

    n = len(selected)
    equal_weight = 1.0 / n

    if equal_weight > cfg.max_position_weight + 1e-12:
        raise ValueError(
            f"Portfolio cannot be fully invested with {n} positions and max_position_weight={cfg.max_position_weight:.2%}. "
            f"Need at least {equal_weight:.2%} max position weight."
        )

    out = selected.copy()
    out["target_weight"] = equal_weight
    return out


def build_target_portfolio(ranked: pd.DataFrame, cfg: PortfolioConfig) -> pd.DataFrame:
    if ranked.empty:
        return ranked.copy()

    ranked = ranked.copy().sort_values("rank").reset_index(drop=True)
    previous_symbols = _load_previous_symbols(cfg.previous_holdings_path)

    ranked["is_previous_holding"] = ranked["symbol"].astype(str).isin(previous_symbols)
    ranked["force_sell_reason"] = ranked.apply(lambda row: _force_sell_reason(row, cfg), axis=1)

    sector_slots = _sector_slot_limit(cfg)

    selected_rows: List[dict] = []
    selected_symbols: Set[str] = set()
    sector_counts: Dict[str, int] = defaultdict(int)

    def try_add(row: pd.Series, reason: str, enforce_sector_cap: bool = True) -> bool:
        symbol = str(row["symbol"])
        sector = str(row.get("sector", "Unknown"))

        if symbol in selected_symbols:
            return False

        if len(selected_rows) >= int(cfg.top_n_positions):
            return False

        if enforce_sector_cap and sector_counts[sector] >= sector_slots:
            return False

        item = row.to_dict()
        item["selection_reason"] = reason
        selected_rows.append(item)
        selected_symbols.add(symbol)
        sector_counts[sector] += 1
        return True

    # Anything with a force-sell reason is not eligible for selection.
    eligible_ranked = ranked[ranked["force_sell_reason"] == ""].copy()

    # 1) Keep prior holdings if still good enough
    previous_keepers = eligible_ranked[
        eligible_ranked["is_previous_holding"]
        & (pd.to_numeric(eligible_ranked["rank"], errors="coerce") <= int(cfg.hold_rank_buffer))
    ].copy()

    for _, row in previous_keepers.iterrows():
        try_add(row, "hold_buffer_keep")

    # 2) Add best new buys from inside buy buffer
    buy_buffer_rows = eligible_ranked[
        ~eligible_ranked["symbol"].isin(selected_symbols)
        & (pd.to_numeric(eligible_ranked["rank"], errors="coerce") <= int(cfg.buy_rank_buffer))
    ].copy()

    for _, row in buy_buffer_rows.iterrows():
        try_add(row, "buy_buffer_entry")

    # 3) Fill remaining slots by rank while respecting sector cap
    remaining_rows = eligible_ranked[~eligible_ranked["symbol"].isin(selected_symbols)].copy()
    for _, row in remaining_rows.iterrows():
        try_add(row, "rank_fill")

    # 4) Final fallback if sector cap blocked too many names
    if len(selected_rows) < int(cfg.top_n_positions):
        fallback_rows = eligible_ranked[~eligible_ranked["symbol"].isin(selected_symbols)].copy()
        for _, row in fallback_rows.iterrows():
            if len(selected_rows) >= int(cfg.top_n_positions):
                break
            try_add(row, "rank_fill_relaxed_sector_cap", enforce_sector_cap=False)

    selected = pd.DataFrame(selected_rows)
    if selected.empty:
        return selected

    selected = selected.sort_values("rank").reset_index(drop=True)
    selected = _allocate_equal_weights(selected, cfg)

    out_cols = [
        "symbol",
        "currency_code",
        "currency_name",
        "sector",
        "industry",
        "rank",
        "target_weight",
        "is_previous_holding",
        "selection_reason",
        "force_sell_reason",
        "composite_score",
        "shareholder_yield",
        "gross_profitability",
        "adjusted_book_to_market",
        "pead_signal",
        "sentiment_acceleration",
        "news_event_signal",
        "news_event_effective_signal",
        "news_peer_spillover_signal",
        "news_article_count_recent",
        "dividend_safety_pass",
        "price_to_200dma",
        "macro_regime",
        "macro_score",
    ]
    out_cols = [c for c in out_cols if c in selected.columns]

    return selected[out_cols].copy()
