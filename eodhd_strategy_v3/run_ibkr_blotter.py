#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a dry-run IBKR order blotter from target portfolio and live positions")
    parser.add_argument("--target-portfolio", required=True)
    parser.add_argument("--ibkr-positions", required=True)
    parser.add_argument("--account-summary", default="")
    parser.add_argument("--ranked-input", default="")

    parser.add_argument("--net-liq", type=float, default=0.0, help="Override total net liquidation value manually")
    parser.add_argument("--strategy-sleeve-usd", type=float, default=0.0, help="If set, only this USD sleeve is managed by the strategy")

    parser.add_argument("--managed-symbol-suffix", default=".US", help="Only positions with this symbol suffix are treated as strategy positions")
    parser.add_argument("--cash-buffer-pct", type=float, default=0.02, help="Keep this fraction in cash, e.g. 0.02 = 2%")

    parser.add_argument("--min-dollar-trade", type=float, default=250.0)
    parser.add_argument("--min-share-trade", type=float, default=1.0)

    parser.add_argument("--order-type", choices=["LMT", "MKT"], default="LMT")
    parser.add_argument("--limit-buy-bps", type=float, default=20.0, help="Limit buy offset in basis points above ref price")
    parser.add_argument("--limit-sell-bps", type=float, default=20.0, help="Limit sell offset in basis points below ref price")

    parser.add_argument("--output", default="ibkr_blotter.csv")
    parser.add_argument("--summary-output", default="ibkr_blotter_summary.csv")
    return parser.parse_args(argv)


def _to_float(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _safe_read_csv(path: str, empty_columns: list[str] | None = None) -> pd.DataFrame:
    p = Path(path)
    if not path or not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=empty_columns or [])

    try:
        return pd.read_csv(p)
    except EmptyDataError:
        return pd.DataFrame(columns=empty_columns or [])


def infer_net_liq(summary_df: pd.DataFrame) -> Optional[float]:
    if summary_df.empty:
        return None

    row = summary_df.iloc[0].to_dict()
    candidates = []

    for key, val in row.items():
        key_l = str(key).lower()
        if "netliquidation" in key_l:
            num = _to_float(val)
            if num is not None:
                priority = 0 if key_l.endswith("__usd") else 1
                candidates.append((priority, num))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def _round_limit_price(side: str, ref_price: float, buy_bps: float, sell_bps: float) -> float:
    if side == "BUY":
        px = ref_price * (1.0 + buy_bps / 10000.0)
    elif side == "SELL":
        px = ref_price * (1.0 - sell_bps / 10000.0)
    else:
        px = ref_price
    return round(px, 2)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    target = pd.read_csv(args.target_portfolio)

    positions = _safe_read_csv(
        args.ibkr_positions,
        empty_columns=["symbol", "position", "market_price", "market_value", "currency", "company_name"],
    )

    ranked = _safe_read_csv(
        args.ranked_input,
        empty_columns=["symbol", "price_proxy", "company_name", "sector", "rank"],
    ) if args.ranked_input else pd.DataFrame()

    summary = _safe_read_csv(args.account_summary) if args.account_summary else pd.DataFrame()

    total_net_liq = float(args.net_liq) if float(args.net_liq) > 0 else infer_net_liq(summary)
    if total_net_liq is None or total_net_liq <= 0:
        raise SystemExit("Could not infer net liquidation value. Pass --net-liq or provide a usable --account-summary file.")

    # Live-account safe mode: manage only a dedicated strategy sleeve if provided.
    strategy_gross_capital = float(args.strategy_sleeve_usd) if float(args.strategy_sleeve_usd) > 0 else float(total_net_liq)

    cash_buffer_pct = max(0.0, min(0.50, float(args.cash_buffer_pct)))
    investable_capital = strategy_gross_capital * (1.0 - cash_buffer_pct)

    target = target.copy()
    positions = positions.copy()

    if "symbol" not in target.columns:
        raise SystemExit("Target portfolio file must contain a 'symbol' column.")

    if "symbol" not in positions.columns:
        positions["symbol"] = pd.Series(dtype="object")

    target["symbol"] = target["symbol"].astype(str).str.upper()
    positions["symbol"] = positions["symbol"].astype(str).str.upper()

    managed_suffix = str(args.managed_symbol_suffix).upper().strip()
    strategy_positions = positions[positions["symbol"].astype(str).str.endswith(managed_suffix)].copy()
    ignored_positions = positions[~positions["symbol"].astype(str).str.endswith(managed_suffix)].copy()

    if not ranked.empty and "symbol" in ranked.columns:
        ranked = ranked.copy()
        ranked["symbol"] = ranked["symbol"].astype(str).str.upper()

    pos_cols = ["symbol", "position", "market_price", "market_value", "currency", "company_name"]
    for c in pos_cols:
        if c not in strategy_positions.columns:
            strategy_positions[c] = np.nan

    merged = target.merge(strategy_positions[pos_cols], on="symbol", how="outer", suffixes=("", "_ibkr"))

    if not ranked.empty:
        rank_cols = ["symbol", "price_proxy", "company_name", "sector", "rank"]
        rank_cols = [c for c in rank_cols if c in ranked.columns]
        merged = merged.merge(ranked[rank_cols], on="symbol", how="left", suffixes=("", "_rank"))

    merged["target_weight"] = pd.to_numeric(merged.get("target_weight"), errors="coerce").fillna(0.0)
    merged["current_shares"] = pd.to_numeric(merged.get("position"), errors="coerce").fillna(0.0)

    merged["ref_price"] = pd.to_numeric(merged.get("market_price"), errors="coerce")
    if "price_proxy" in merged.columns:
        merged["ref_price"] = merged["ref_price"].fillna(pd.to_numeric(merged.get("price_proxy"), errors="coerce"))

    merged["current_value"] = pd.to_numeric(merged.get("market_value"), errors="coerce")
    merged["current_value"] = merged["current_value"].fillna(merged["current_shares"] * merged["ref_price"])

    merged["target_value"] = merged["target_weight"] * investable_capital
    merged["delta_value"] = merged["target_value"] - merged["current_value"].fillna(0.0)

    merged["target_shares"] = np.where(
        merged["ref_price"].notna() & (merged["ref_price"] > 0),
        np.floor(merged["target_value"] / merged["ref_price"]),
        np.nan,
    )
    merged["delta_shares"] = merged["target_shares"] - merged["current_shares"]

    def side_from_delta(x):
        if pd.isna(x):
            return "REVIEW"
        if x > 0:
            return "BUY"
        if x < 0:
            return "SELL"
        return "HOLD"

    merged["side"] = merged["delta_shares"].apply(side_from_delta)
    merged["est_order_notional"] = (merged["delta_shares"].abs() * merged["ref_price"]).fillna(0.0)

    merged["trade_flag"] = (
        (merged["delta_value"].abs() >= float(args.min_dollar_trade))
        & (merged["delta_shares"].abs() >= float(args.min_share_trade))
    )

    merged["suggested_order_type"] = np.where(merged["trade_flag"], args.order_type, "")
    merged["suggested_tif"] = np.where(merged["trade_flag"], "DAY", "")

    if args.order_type == "LMT":
        merged["suggested_limit_price"] = np.where(
            merged["trade_flag"] & merged["ref_price"].notna(),
            merged.apply(
                lambda row: _round_limit_price(
                    str(row["side"]),
                    float(row["ref_price"]),
                    float(args.limit_buy_bps),
                    float(args.limit_sell_bps),
                ),
                axis=1,
            ),
            np.nan,
        )
    else:
        merged["suggested_limit_price"] = np.nan

    if "company_name_rank" in merged.columns:
        merged["company_name"] = merged["company_name"].fillna(merged["company_name_rank"])

    out_cols = [
        "symbol",
        "company_name",
        "sector",
        "rank",
        "target_weight",
        "current_shares",
        "target_shares",
        "delta_shares",
        "ref_price",
        "suggested_limit_price",
        "current_value",
        "target_value",
        "delta_value",
        "est_order_notional",
        "side",
        "trade_flag",
        "suggested_order_type",
        "suggested_tif",
        "currency",
    ]
    out_cols = [c for c in out_cols if c in merged.columns]
    out = merged[out_cols].copy()
    out.to_csv(args.output, index=False)

    ignored_count = int(len(ignored_positions))
    ignored_value = float(pd.to_numeric(ignored_positions.get("market_value"), errors="coerce").fillna(0.0).sum()) if not ignored_positions.empty else 0.0
    strategy_current_value = float(pd.to_numeric(merged.get("current_value"), errors="coerce").fillna(0.0).sum())

    summary_rows = [
        {"metric": "total_net_liq", "value": total_net_liq},
        {"metric": "strategy_gross_capital", "value": strategy_gross_capital},
        {"metric": "cash_buffer_pct", "value": cash_buffer_pct},
        {"metric": "investable_capital", "value": investable_capital},
        {"metric": "managed_symbol_suffix", "value": managed_suffix},
        {"metric": "current_strategy_value", "value": strategy_current_value},
        {"metric": "ignored_position_count", "value": ignored_count},
        {"metric": "ignored_position_value", "value": ignored_value},
    ]

    if not ignored_positions.empty and "currency" in ignored_positions.columns:
        for ccy, grp in ignored_positions.groupby(ignored_positions["currency"].fillna("UNKNOWN")):
            summary_rows.append(
                {
                    "metric": f"ignored_positions::{ccy}::count",
                    "value": int(len(grp)),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.summary_output, index=False)

    print(f"Total net liq used:    {total_net_liq:,.2f}")
    print(f"Strategy sleeve used:  {strategy_gross_capital:,.2f}")
    print(f"Cash buffer pct:       {cash_buffer_pct:.2%}")
    print(f"Investable capital:    {investable_capital:,.2f}")
    print(f"Managed suffix:        {managed_suffix}")
    print(f"Ignored live positions:{ignored_count}")
    print(f"Saved blotter to {args.output}")
    print(f"Saved summary to {args.summary_output}")

    print("\nTrade preview:")
    preview = out[out.get("trade_flag", False) == True].copy()
    if preview.empty:
        print("No trades passed the minimum trade thresholds.")
    else:
        print(preview.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())