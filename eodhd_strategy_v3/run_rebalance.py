#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

import pandas as pd

from eodhd_strategy.client import EODHDClient
from eodhd_strategy.config import PortfolioConfig
from eodhd_strategy.data_provider import DataProvider
from eodhd_strategy.portfolio import build_target_portfolio
from eodhd_strategy.regime import recommend_rebalance_date


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build target portfolio from ranked stocks with turnover buffers, force-sell rules, macro-aware rebalancing, and paper tracking"
    )
    parser.add_argument("--api-token", default=None)
    parser.add_argument("--ranked-input", required=True)
    parser.add_argument("--previous-holdings", default="")

    # 10-position default
    parser.add_argument("--top-n-positions", type=int, default=10)
    parser.add_argument("--max-position-weight", type=float, default=0.12)
    parser.add_argument("--sector-cap", type=float, default=0.30)

    parser.add_argument("--buy-rank-buffer", type=int, default=10)
    parser.add_argument("--hold-rank-buffer", type=int, default=18)

    parser.add_argument("--force-sell-below-200dma", type=float, default=0.95)

    parser.add_argument("--use-macro-rebalance", action="store_true")
    parser.add_argument("--macro-regime-override", choices=["", "neutral", "risk_on", "risk_off"], default="")

    parser.add_argument("--macro-top-n-risk-on", type=int, default=10)
    parser.add_argument("--macro-top-n-risk-off", type=int, default=12)

    parser.add_argument("--macro-max-position-risk-on", type=float, default=0.12)
    parser.add_argument("--macro-max-position-risk-off", type=float, default=0.10)

    parser.add_argument("--macro-sector-cap-risk-on", type=float, default=0.30)
    parser.add_argument("--macro-sector-cap-risk-off", type=float, default=0.25)

    parser.add_argument("--target-date", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--rebalance-country", default="US")
    parser.add_argument("--defer-if-macro-event-within-days", type=int, default=1)

    parser.add_argument("--output", default="target_portfolio.csv")

    # Paper tracking
    parser.add_argument("--history-summary-output", default="rebalance_history.csv")
    parser.add_argument("--history-holdings-output", default="rebalance_holdings_history.csv")
    return parser.parse_args(argv)


def resolve_macro_rebalance_settings(args, ranked: pd.DataFrame):
    regime = None

    if args.macro_regime_override:
        regime = args.macro_regime_override
    elif args.use_macro_rebalance and "macro_regime" in ranked.columns and ranked["macro_regime"].notna().any():
        regime = str(ranked["macro_regime"].dropna().iloc[0]).strip().lower()

    top_n_positions = int(args.top_n_positions)
    max_position_weight = float(args.max_position_weight)
    sector_cap = float(args.sector_cap)

    if regime == "risk_on":
        top_n_positions = int(args.macro_top_n_risk_on)
        max_position_weight = float(args.macro_max_position_risk_on)
        sector_cap = float(args.macro_sector_cap_risk_on)
    elif regime == "risk_off":
        top_n_positions = int(args.macro_top_n_risk_off)
        max_position_weight = float(args.macro_max_position_risk_off)
        sector_cap = float(args.macro_sector_cap_risk_off)

    return regime, top_n_positions, max_position_weight, sector_cap


def append_history(
    portfolio: pd.DataFrame,
    ranked: pd.DataFrame,
    args,
    regime: str | None,
    effective_top_n: int,
    effective_max_weight: float,
    effective_sector_cap: float,
) -> None:
    run_id = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    retained_count = int(pd.to_numeric(portfolio.get("is_previous_holding", 0), errors="coerce").fillna(0).sum())
    selected_count = int(len(portfolio))
    turnover_est = None
    if selected_count > 0:
        turnover_est = 1.0 - (retained_count / selected_count)

    summary_row = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "target_date": args.target_date,
                "ranked_input": args.ranked_input,
                "output": args.output,
                "macro_regime": regime or "",
                "top_n_positions": effective_top_n,
                "max_position_weight": effective_max_weight,
                "sector_cap": effective_sector_cap,
                "selected_count": selected_count,
                "retained_previous_count": retained_count,
                "turnover_estimate": turnover_est,
                "avg_rank": float(pd.to_numeric(portfolio.get("rank"), errors="coerce").mean()) if selected_count else None,
                "median_rank": float(pd.to_numeric(portfolio.get("rank"), errors="coerce").median()) if selected_count else None,
            }
        ]
    )

    summary_path = Path(args.history_summary_output)
    if summary_path.exists():
        old = pd.read_csv(summary_path)
        summary_row = pd.concat([old, summary_row], ignore_index=True)
    summary_row.to_csv(summary_path, index=False)

    holdings = portfolio.copy()
    holdings.insert(0, "run_id", run_id)
    holdings["target_date"] = args.target_date

    holdings_path = Path(args.history_holdings_output)
    if holdings_path.exists():
        old = pd.read_csv(holdings_path)
        holdings = pd.concat([old, holdings], ignore_index=True)
    holdings.to_csv(holdings_path, index=False)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    ranked = pd.read_csv(args.ranked_input)
    if ranked.empty:
        raise SystemExit("Ranked input is empty.")

    regime, eff_top_n, eff_max_weight, eff_sector_cap = resolve_macro_rebalance_settings(args, ranked)

    cfg = PortfolioConfig(
        top_n_positions=int(eff_top_n),
        max_position_weight=float(eff_max_weight),
        sector_cap=float(eff_sector_cap),
        buy_rank_buffer=int(args.buy_rank_buffer),
        hold_rank_buffer=int(args.hold_rank_buffer),
        defer_if_macro_event_within_days=int(args.defer_if_macro_event_within_days),
        rebalance_country=args.rebalance_country,
        previous_holdings_path=Path(args.previous_holdings) if args.previous_holdings else None,
        output=Path(args.output),
        force_sell_below_200dma=float(args.force_sell_below_200dma),
    )

    portfolio = build_target_portfolio(ranked, cfg)
    portfolio.to_csv(cfg.output, index=False)

    display_cols = [
        "symbol",
        "sector",
        "rank",
        "target_weight",
        "is_previous_holding",
        "selection_reason",
        "force_sell_reason",
    ]
    display_cols = [c for c in display_cols if c in portfolio.columns]
    print(portfolio[display_cols].to_string(index=False))
    print(f"\nSaved target portfolio to {cfg.output}")

    if regime:
        print("\nMacro rebalance settings:")
        print(f"  regime:              {regime}")
        print(f"  top_n_positions:     {eff_top_n}")
        print(f"  max_position_weight: {eff_max_weight:.2%}")
        print(f"  sector_cap:          {eff_sector_cap:.2%}")

    append_history(
        portfolio=portfolio,
        ranked=ranked,
        args=args,
        regime=regime,
        effective_top_n=eff_top_n,
        effective_max_weight=eff_max_weight,
        effective_sector_cap=eff_sector_cap,
    )
    print(f"Saved rebalance history to {args.history_summary_output}")
    print(f"Saved holdings history to {args.history_holdings_output}")

    api_token = args.api_token or os.getenv("EODHD_API_TOKEN")
    if api_token:
        eodhd_client = EODHDClient(api_token=api_token, cache_dir=Path(".eodhd_cache"))
        client = DataProvider(mode="eodhd", eodhd_client=eodhd_client)
        decision = recommend_rebalance_date(
            client=client,
            start_date=args.target_date,
            country=args.rebalance_country,
            defer_if_within_days=cfg.defer_if_macro_event_within_days,
        )
        print(f"\nTarget rebalance date:      {decision.target_date}")
        print(f"Recommended execution date: {decision.recommended_date}")
        if decision.blocking_events:
            print("Blocking macro events detected:")
            for event in decision.blocking_events[:10]:
                print(f"  - {event}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())