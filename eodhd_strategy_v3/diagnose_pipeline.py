#!/usr/bin/env python3
"""Diagnostic: trace exactly where the pipeline loses data."""
import asyncio
import sys
from pathlib import Path

import pandas as pd

from eodhd_strategy.fmp_client import FMPClient, FMPConfig
from eodhd_strategy.fmp_mapper import (
    map_profile_data,
    map_financial_statements,
    map_beneish_components,
    map_analyst_estimates,
    map_institutional_ownership,
    map_insider_trading,
    merge_all_data,
    create_raw_fmp_dataframe,
)



API_KEY = sys.argv[1] if len(sys.argv) > 1 else "oYm8ALIS7e8s4Zl8SasXMWQJ7D4Zuntw"


async def main():
    config = FMPConfig(
        api_key=API_KEY,
        cache_dir=Path(".fmp_cache"),
        max_retries=3,
        retry_delay=1.0,
        request_timeout=30.0,
        batch_size=100,
    )

    async with FMPClient(config) as client:
        # --- Step 1: screener -------------------------------------------------
        screener = await client.fetch_screener(
            is_actively_trading=True,
            market_cap_more_than=2_000_000_000,
            limit=50,  # small for diag
        )
        print(f"[1] Screener: {len(screener)} rows, cols={list(screener.columns)[:10]}")
        if screener.empty:
            print("    *** SCREENER returned nothing. Pipeline dead here.")
            return

        symbols = screener["symbol"].tolist()[:10]
        print(f"    Using test symbols: {symbols}")

        # --- Step 2: profile-bulk -------------------------------------------
        profiles = await client.fetch_profile_bulk(market="us", part=0)
        print(f"\n[2] Profile-bulk: {len(profiles)} rows, cols={list(profiles.columns)[:10]}")

        # --- Step 3: income / balance / cashflow ----------------------------
        income = await client.fetch_income_statement_bulk(symbols=symbols, period="annual")
        print(f"\n[3a] Income: {len(income)} rows, cols={list(income.columns)[:10] if not income.empty else '(empty)'}")

        balance = await client.fetch_balance_sheet_bulk(symbols=symbols, period="annual")
        print(f"[3b] Balance: {len(balance)} rows, cols={list(balance.columns)[:10] if not balance.empty else '(empty)'}")

        cashflow = await client.fetch_cash_flow_bulk(symbols=symbols, period="annual")
        print(f"[3c] Cashflow: {len(cashflow)} rows, cols={list(cashflow.columns)[:10] if not cashflow.empty else '(empty)'}")

        # --- Step 4: estimates -----------------------------------------------
        estimates = await client.fetch_analyst_estimates_bulk(symbols=symbols, period="annual")
        print(f"\n[4] Estimates: {len(estimates)} rows, cols={list(estimates.columns)[:10] if not estimates.empty else '(empty)'}")

        # --- Step 5: insider -------------------------------------------------
        insider = await client.fetch_insider_trading_bulk(page=0, limit=100)
        print(f"\n[5] Insider: {len(insider)} rows, cols={list(insider.columns)[:10] if not insider.empty else '(empty)'}")

        # --- Step 6: institutional (expected to fail) -----------------------
        institutional = await client.fetch_institutional_ownership_bulk(symbols=symbols)
        print(f"\n[6] Institutional: {len(institutional)} rows (expected 0)")

        # --- Step 7: MAPPINGS -----------------------------------------------
        print("\n" + "="*60)
        print("MAPPER OUTPUTS")
        print("="*60)

        mapped_profiles = map_profile_data(screener)
        print(f"\n[M1] Mapped profiles: {len(mapped_profiles)} rows, cols={list(mapped_profiles.columns)}")

        mapped_financials = map_financial_statements(income, balance, cashflow, screener)
        print(f"[M2] Mapped financials: {len(mapped_financials)} rows")
        if not mapped_financials.empty:
            print(f"     cols: {list(mapped_financials.columns)[:15]}")
            print(f"     symbols: {mapped_financials['symbol'].unique()[:5].tolist()}")

        mapped_beneish = map_beneish_components(income, balance, cashflow)
        print(f"[M3] Mapped Beneish: {len(mapped_beneish)} rows")

        mapped_estimates = map_analyst_estimates(estimates)
        print(f"[M4] Mapped estimates: {len(mapped_estimates)} rows")

        mapped_institutional = map_institutional_ownership(institutional)
        print(f"[M5] Mapped institutional: {len(mapped_institutional)} rows")

        mapped_insider = map_insider_trading(insider)
        print(f"[M6] Mapped insider: {len(mapped_insider)} rows")

        # --- Step 8: MERGE --------------------------------------------------
        prices = await client.fetch_bulk_daily_prices(symbols=symbols)
        print(f"\n[7] Prices: {len(prices)} rows")

        merged = merge_all_data(
            mapped_profiles,
            prices,
            mapped_financials,
            mapped_beneish,
            mapped_estimates,
            mapped_institutional,
            mapped_insider,
        )
        print(f"\n[FINAL] Merged: {len(merged)} rows, {len(merged.columns)} cols")
        if not merged.empty:
            print(f"  Symbols: {merged['symbol'].unique()[:10].tolist()}")
            print(f"  Columns: {list(merged.columns)[:20]}")
            # Check key columns
            for col in ["market_cap", "sector", "gross_profitability", "shareholder_yield"]:
                if col in merged.columns:
                    non_null = merged[col].notna().sum()
                    print(f"  {col}: {non_null}/{len(merged)} non-null")
                else:
                    print(f"  {col}: MISSING")
        else:
            print("  *** MERGED IS EMPTY!")


asyncio.run(main())
