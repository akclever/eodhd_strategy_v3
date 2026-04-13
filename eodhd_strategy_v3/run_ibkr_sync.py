#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Any, Dict, List, Sequence

import pandas as pd

from eodhd_strategy.ibkr_client import IBGatewayClient


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync live positions and enhanced account summary from IB Gateway / TWS API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4002, help="IB Gateway paper is commonly 4002; live is commonly 4001")
    parser.add_argument("--client-id", type=int, default=101)
    parser.add_argument("--timeout", type=int, default=20)

    parser.add_argument("--account-id", default="")
    parser.add_argument("--positions-output", default="ibkr_positions.csv")
    parser.add_argument("--summary-output", default="ibkr_summary.csv")
    parser.add_argument("--summary-raw-output", default="ibkr_summary_raw.csv")
    parser.add_argument("--summary-compact-output", default="ibkr_summary_compact.csv")
    return parser.parse_args(argv)


def positions_to_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    base_cols = [
        "account",
        "conid",
        "symbol_ibkr",
        "symbol",
        "secType",
        "currency",
        "exchange",
        "primaryExchange",
        "localSymbol",
        "tradingClass",
        "position",
        "average_cost",
    ]

    if not rows:
        return pd.DataFrame(columns=base_cols)

    df = pd.DataFrame(rows)
    for c in base_cols:
        if c not in df.columns:
            df[c] = pd.NA

    df = df[df["symbol"].astype(str).str.len() > 0].copy()
    df = df.sort_values(["account", "symbol"]).reset_index(drop=True)
    return df[base_cols]


def summary_raw_to_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["reqId", "account", "tag", "value", "currency", "summary_bucket", "summary_tags_request"])
    df = pd.DataFrame(rows)
    return df.sort_values(["summary_bucket", "account", "tag", "currency"]).reset_index(drop=True)


def summary_wide_to_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw

    raw["summary_bucket"] = raw["summary_bucket"].fillna("unknown").astype(str)
    raw["account"] = raw["account"].fillna("unknown").astype(str)
    raw["currency"] = raw["currency"].fillna("").astype(str)
    raw["tag"] = raw["tag"].fillna("").astype(str)

    raw["wide_key"] = raw["summary_bucket"] + "__" + raw["tag"] + raw["currency"].map(lambda c: f"__{c}" if c else "")

    wide = (
        raw.pivot_table(index="account", columns="wide_key", values="value", aggfunc="last")
           .reset_index()
    )
    wide.columns.name = None
    return wide


def _pick_first_non_null(df: pd.DataFrame, account_preference: List[str], column_candidates: List[str]):
    for acct in account_preference:
        subset = df[df["account"] == acct]
        if subset.empty:
            continue
        row = subset.iloc[0]
        for col in column_candidates:
            if col in subset.columns and pd.notna(row.get(col)):
                return row.get(col)
    return None


def build_compact_summary(summary_wide: pd.DataFrame) -> pd.DataFrame:
    if summary_wide.empty:
        return pd.DataFrame()

    account_names = summary_wide["account"].astype(str).tolist()
    account_specific = [x for x in account_names if x != "All"]
    account_preference = account_specific + ["All"]

    compact = {
        "account": account_specific[0] if account_specific else "All",
        "account_type": _pick_first_non_null(summary_wide, account_preference, ["core__AccountType"]),
        "base_currency": _pick_first_non_null(summary_wide, ["All"] + account_preference, [
            "ledger_base__Currency__BASE",
            "ledger_all__Currency__BASE",
        ]),
        "reporting_currency": _pick_first_non_null(summary_wide, ["All"] + account_preference, [
            "ledger_all__Currency__EUR",
            "ledger_usd__Currency__USD",
        ]),
        "net_liquidation": _pick_first_non_null(summary_wide, account_preference, [
            "core__NetLiquidation__USD",
            "core__NetLiquidation__EUR",
        ]),
        "total_cash_value": _pick_first_non_null(summary_wide, account_preference, [
            "core__TotalCashValue__USD",
            "core__TotalCashValue__EUR",
        ]),
        "buying_power": _pick_first_non_null(summary_wide, account_preference, [
            "core__BuyingPower__USD",
            "core__BuyingPower__EUR",
        ]),
        "gross_position_value": _pick_first_non_null(summary_wide, account_preference, [
            "core__GrossPositionValue__USD",
            "core__GrossPositionValue__EUR",
        ]),
        "available_funds": _pick_first_non_null(summary_wide, account_preference, [
            "core__AvailableFunds__USD",
            "core__AvailableFunds__EUR",
        ]),
        "excess_liquidity": _pick_first_non_null(summary_wide, account_preference, [
            "core__ExcessLiquidity__USD",
            "core__ExcessLiquidity__EUR",
        ]),
        "equity_with_loan_value": _pick_first_non_null(summary_wide, account_preference, [
            "core__EquityWithLoanValue__USD",
            "core__EquityWithLoanValue__EUR",
        ]),
        "init_margin_req": _pick_first_non_null(summary_wide, account_preference, [
            "core__InitMarginReq__USD",
            "core__InitMarginReq__EUR",
        ]),
        "maint_margin_req": _pick_first_non_null(summary_wide, account_preference, [
            "core__MaintMarginReq__USD",
            "core__MaintMarginReq__EUR",
        ]),
        "cushion": _pick_first_non_null(summary_wide, account_preference, ["core__Cushion"]),
        "leverage": _pick_first_non_null(summary_wide, account_preference, ["core__Leverage"]),
        "ledger_base_cash_balance": _pick_first_non_null(summary_wide, ["All"] + account_preference, [
            "ledger_base__CashBalance__BASE",
            "ledger_base__TotalCashBalance__BASE",
        ]),
        "ledger_base_net_liq_by_currency": _pick_first_non_null(summary_wide, ["All"] + account_preference, [
            "ledger_base__NetLiquidationByCurrency__BASE",
        ]),
        "ledger_usd_cash_balance": _pick_first_non_null(summary_wide, ["All"] + account_preference, [
            "ledger_usd__CashBalance__USD",
            "ledger_usd__TotalCashBalance__USD",
        ]),
        "ledger_usd_net_liq_by_currency": _pick_first_non_null(summary_wide, ["All"] + account_preference, [
            "ledger_usd__NetLiquidationByCurrency__USD",
        ]),
        "ledger_all_base_cash_balance": _pick_first_non_null(summary_wide, ["All"] + account_preference, [
            "ledger_all__CashBalance__BASE",
            "ledger_all__TotalCashBalance__BASE",
        ]),
    }

    return pd.DataFrame([compact])


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    client = IBGatewayClient()
    try:
        client.connect_and_start(args.host, args.port, args.client_id, timeout=args.timeout)

        if client.connection_error:
            raise RuntimeError(
                f"IB Gateway connection failed: {client.connection_error}. "
                f"Check that the gateway is logged in and the socket port matches."
            )

        account_id = client.resolve_account_id(args.account_id or None)
        positions_raw = client.fetch_positions(timeout=args.timeout)
        summary_bundle_raw = client.fetch_account_summary_bundle(timeout=args.timeout)

    finally:
        client.disconnect_and_stop()

    positions_df = positions_to_dataframe(positions_raw)
    summary_raw_df = summary_raw_to_dataframe(summary_bundle_raw)
    summary_df = summary_wide_to_dataframe(summary_bundle_raw)
    summary_compact_df = build_compact_summary(summary_df)

    positions_df.to_csv(args.positions_output, index=False)
    summary_df.to_csv(args.summary_output, index=False)
    summary_raw_df.to_csv(args.summary_raw_output, index=False)
    summary_compact_df.to_csv(args.summary_compact_output, index=False)

    print(f"Connected to IB Gateway at {args.host}:{args.port}")
    print(f"Account ID: {account_id}")
    print(f"Saved positions to {args.positions_output}")
    print(f"Saved summary to {args.summary_output}")
    print(f"Saved raw summary to {args.summary_raw_output}")
    print(f"Saved compact summary to {args.summary_compact_output}")

    if not positions_df.empty:
        preview_cols = [c for c in ["symbol", "position", "average_cost", "currency", "secType", "primaryExchange"] if c in positions_df.columns]
        print("\nPositions preview:")
        print(positions_df[preview_cols].head(20).to_string(index=False))
    else:
        print("\nNo positions returned.")

    if not summary_compact_df.empty:
        print("\nCompact account summary:")
        print(summary_compact_df.to_string(index=False))
    elif not summary_df.empty:
        print("\nAccount summary preview:")
        print(summary_df.head(5).to_string(index=False))
    else:
        print("\nNo account summary returned.")

    if client.errors:
        print("\nIBKR API notices/errors seen during sync:")
        err_df = pd.DataFrame(client.errors)
        print(err_df.head(20).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())