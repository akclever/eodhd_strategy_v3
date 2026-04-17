#!/usr/bin/env python3
"""
FMP Ranker Execution Script

This is the main execution script for the FMP Ultimate-powered ranker.
It fetches bulk data from FMP, maps it to ranker-expected columns, and
invokes the existing build_ranked_frame function.

Usage:
    python run_fmp_ranker.py --api-key YOUR_FMP_API_KEY --output ranked_stocks.csv
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from dataclasses import replace
from pathlib import Path

import pandas as pd

from eodhd_strategy.config import RankerConfig
from eodhd_strategy.fmp_client import FMPClient, FMPConfig
from eodhd_strategy.fmp_mapper import create_raw_fmp_dataframe
from eodhd_strategy.ranker import build_ranked_frame, print_error_summary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame has unique column labels by coalescing duplicate-name columns."""
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df
    if df.columns.is_unique:
        return df
    # Transpose-groupby-first coalesces duplicate columns, preferring first non-null values
    return df.T.groupby(level=0).first().T


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the FMP-powered quantitative factor ranking system"
    )
    
    # FMP API configuration
    parser.add_argument(
        "--api-key",
        required=True,
        help="FMP Ultimate API key"
    )
    parser.add_argument(
        "--cache-dir",
        default=".fmp_cache",
        help="Directory for caching FMP data"
    )
    
    # Data source configuration
    parser.add_argument(
        "--market",
        default="us",
        help="Market to fetch data for (e.g., us, global)"
    )
    parser.add_argument(
        "--financial-period",
        choices=["annual", "quarterly"],
        default="annual",
        help="Financial statement period to use"
    )
    
    # Ranker configuration (subset of full RankerConfig for MVP)
    parser.add_argument(
        "--min-market-cap",
        type=float,
        default=2_000_000_000,
        help="Minimum market cap in USD"
    )
    parser.add_argument(
        "--dividend-source",
        choices=["forward", "trailing", "hybrid"],
        default="hybrid",
        help="Dividend yield calculation method"
    )
    parser.add_argument(
        "--regime",
        choices=["neutral", "risk_on", "risk_off"],
        default="neutral",
        help="Market regime for factor weighting"
    )
    
    # Factor toggles
    parser.add_argument("--use-pead", action="store_true", help="Enable PEAD factor")
    parser.add_argument("--use-revision-impulse", action="store_true", help="Enable revision impulse factor")
    parser.add_argument("--use-estimate-term-structure", action="store_true", help="Enable estimate term structure factor")
    parser.add_argument("--use-growth-acceleration", action="store_true", help="Enable growth acceleration factor")
    parser.add_argument("--use-residual-valuation", action="store_true", help="Enable residual valuation factor")
    parser.add_argument("--use-compounder-persistence", action="store_true", help="Enable compounder persistence factor")
    parser.add_argument("--use-price-momentum", action="store_true", help="Enable price momentum factor")
    parser.add_argument("--use-life-cycle", action="store_true", help="Enable life cycle factor")
    parser.add_argument("--use-sentiment", action="store_true", help="Enable sentiment factor")
    parser.add_argument("--use-news-events", action="store_true", help="Enable news events factor")
    parser.add_argument("--use-beneish", action="store_true", help="Enable Beneish M-score factor")
    parser.add_argument("--use-accrual-volatility", action="store_true", help="Enable accrual volatility factor")
    parser.add_argument("--use-working-capital-stress", action="store_true", help="Enable working capital stress factor")
    parser.add_argument("--use-capital-allocation-quality", action="store_true", help="Enable capital allocation quality factor")
    parser.add_argument("--use-recovery-transition", action="store_true", help="Enable recovery transition factor")
    parser.add_argument("--use-insider-conviction", action="store_true", help="Enable insider conviction factor")
    
    # Output configuration
    parser.add_argument(
        "--output",
        default="ranked_stocks_fmp.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run verification checks after ranking"
    )
    parser.add_argument(
        "--trace-sparsity",
        action="store_true",
        help="Log raw->mapped sparsity trace for key factor families"
    )
    parser.add_argument(
        "--trace-sample-size",
        type=int,
        default=10,
        help="Number of symbols to include in sparsity trace sample"
    )
    parser.add_argument(
        "--symbols",
        default="",
        help="Comma-separated symbols to run (overrides screener-derived symbol list for per-ticker fetches)"
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=500,
        help="Maximum symbols to use for per-ticker endpoints when --symbols is not provided"
    )
    parser.add_argument(
        "--fast-debug-pead",
        action="store_true",
        help="Fetch PEAD-related data first and skip heavy optional endpoints for faster debug iteration"
    )
    
    return parser.parse_args()


def create_ranker_config(args) -> RankerConfig:
    """Create RankerConfig from command-line arguments."""
    config = RankerConfig(
        api_token=args.api_key,  # Using FMP API key as token
        cache_dir=Path(args.cache_dir),
        refresh=True,
        workers=4,
        min_market_cap=args.min_market_cap,
        dividend_source=args.dividend_source,
        regime=args.regime,
        
        # Factor toggles
        use_pead=args.use_pead,
        use_revision_impulse=args.use_revision_impulse,
        use_estimate_term_structure=args.use_estimate_term_structure,
        use_growth_acceleration=args.use_growth_acceleration,
        use_residual_valuation=args.use_residual_valuation,
        use_compounder_persistence=args.use_compounder_persistence,
        use_price_momentum=args.use_price_momentum,
        use_life_cycle=args.use_life_cycle,
        use_sentiment=args.use_sentiment,
        use_news_events=args.use_news_events,
        use_beneish=args.use_beneish,
        use_accrual_volatility=args.use_accrual_volatility,
        use_working_capital_stress=args.use_working_capital_stress,
        use_capital_allocation_quality=args.use_capital_allocation_quality,
        use_recovery_transition=args.use_recovery_transition,
        use_insider_conviction=args.use_insider_conviction,
        
        # Default values for other required config fields
        pead_lookback_days=120,
        pead_half_life_days=45,
        min_pead_analysts=3,
        min_revision_analysts=4,
        revision_impulse_weight=0.06,
        estimate_term_structure_weight=0.04,
        growth_weight=0.08,
        alpha_factor_spec="legacy",
        use_intangible_adjustments=True,
        require_real_momentum_coverage=False,
        momentum_weight=0.10,
        life_cycle_tilt_strength=0.50,
        sentiment_lookback_days=30,
        min_sentiment_accel=0.01,
        min_sentiment_articles_recent=5,
        news_lookback_days=30,
        min_news_articles=3,
        news_event_weight=0.04,
        use_news_peer_spillover=False,
        news_peer_spillover_weight=0.02,
        use_news_novelty_saturation=False,
        use_news_confirmation=False,
        news_confirmation_weight=0.02,
        use_news_macro_weighting=False,
        forensic_weight=0.05,
        missing_beneish_penalty=0.15,
        capital_allocation_weight=0.04,
        recovery_transition_weight=0.03,
        insider_conviction_weight=0.03,
        use_news_theme_drift=False,
        news_theme_drift_weight=0.03,
        use_peer_relative_anomalies=False,
        peer_relative_anomaly_weight=0.04,
        exclude_binary_biotech=False,
        binary_biotech_min_revenue=1_000_000_000,
        dividend_payout_cap=0.85,
        max_distance_from_high=0.15,
        require_above_200dma=False,
        neutralize_by="sector",
        min_group_size=5,
        overlay_top_n=250,
        output=Path(args.output),
        min_sentiment_days=7,
        min_piotroski_score=5,
        pead_max_abs_surprise_pct=100.0,
        pead_max_age_days=90,
        macro_state="auto",
        universe_size=1000,
        use_employee_efficiency=False,
        employee_efficiency_weight=0.03,
        data_provider="fmp",  # New mode for FMP
        alpha_vantage_api_key="",
        sec_edgar_email="",
        analysis_from_primary_ticker=False,
        exclude_special_situations=False,
        price_momentum_source_mode="auto",
        use_investment_restraint=False,
        investment_restraint_weight=0.04,
        use_accrual_quality=False,
        accrual_quality_weight=0.05,
        core_weight_floor=0.60,
        use_revision_jerk=False,
        revision_jerk_weight=0.04,
        use_news_shock=False,
        news_shock_weight=0.04,
        use_technical_momentum=False,
        technical_momentum_weight=0.05,
        
        # Structural thresholds
        shareholder_yield_range=(-0.25, 0.25),
        buyback_yield_range=(-0.20, 0.20),
        gross_profitability_range=(0.0, 2.0),
        adjusted_book_to_market_range=(0.0, 3.0),
        recency_ratio_range=(0.50, 1.20),
        price_to_200dma_range=(0.50, 2.50),
        zscore_clip=2.0,
        trend_penalty_slope=2.5,
        trend_penalty_cap=1.0,
        quality_penalty_cap=0.5,
        forensic_missing_penalty=0.15,
        core_impute_penalty=0.05,
        core_missing_penalty=0.15,
        momentum_missing_penalty=0.10,
        max_combined_news_share=0.04,
        earnings_momentum_base_weight=0.08,
        revision_jerk_persistence_min_days=14,
        life_cycle_min_confidence=0.15,
        huber_min_samples=16,
        residual_min_usable=12,
        peer_anomaly_min_inputs=2,
        peer_anomaly_min_group=12,
        growth_component_yoy_share=0.45,
        growth_component_accel_share=0.55,
        use_dynamic_weights=True,
        dynamic_weight_min_universe=30,
    )
    
    return config


async def run_fmp_ranker(args):
    """Main execution function for FMP ranker."""
    logger.info("=" * 80)
    logger.info("FMP Ultimate Quantitative Factor Ranking System")
    logger.info("=" * 80)
    
    # Create FMP config
    fmp_config = FMPConfig(
        api_key=args.api_key,
        cache_dir=Path(args.cache_dir),
        max_retries=5,
        retry_delay=1.0,
        request_timeout=30.0,
        batch_size=100
    )
    
    # Create ranker config
    ranker_config = create_ranker_config(args)
    
    logger.info(f"Market: {args.market}")
    logger.info(f"Financial period: {args.financial_period}")
    logger.info(f"Minimum market cap: ${args.min_market_cap:,.0f}")
    if args.fast_debug_pead:
        logger.info("Fast debug mode: PEAD-first fetch order with heavy endpoints skipped")
    
    # Fetch all bulk data from FMP
    async with FMPClient(fmp_config) as client:
        logger.info("Fetching bulk data from FMP Ultimate API...")
        
        # Step 1: Get universe from screener
        logger.info("Step 1: Fetching universe from screener...")
        screener_df = await client.fetch_screener(
            is_actively_trading=True,
            market_cap_more_than=args.min_market_cap,
            limit=5000,
        )
        logger.info(f"  Screener returned {len(screener_df)} symbols")
        
        if screener_df.empty or "symbol" not in screener_df.columns:
            logger.error("Screener returned no data. Cannot proceed.")
            sys.exit(1)
        
        # Use screener as profiles (it has symbol, market_cap, sector, industry, etc.)
        bulk_data = {
            "profiles": screener_df,
        }
        
        # Limit symbols for per-ticker endpoints to avoid rate limits
        # Sort by market cap descending to prioritize largest companies
        if "market_cap" in screener_df.columns:
            screener_sorted = screener_df.sort_values("market_cap", ascending=False)
        else:
            screener_sorted = screener_df
        if args.symbols:
            symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
            logger.info(f"  Using explicit symbol override: {symbols}")

            profile_subset = screener_df[screener_df["symbol"].astype(str).isin(symbols)].copy()
            if profile_subset.empty:
                logger.warning("  Explicit symbols not found in screener output; creating minimal profile rows")
                profile_subset = pd.DataFrame({"symbol": symbols})
            bulk_data["profiles"] = profile_subset
        else:
            max_symbols = max(1, int(args.max_symbols))
            symbols = screener_sorted["symbol"].tolist()[:max_symbols]
            logger.info(f"  Using top {len(symbols)} symbols for per-ticker data")

        async def fetch_earnings_surprises_bundle() -> tuple[pd.DataFrame, pd.DataFrame]:
            logger.info("Step 5: Fetching bulk earnings surprises (2025-2026)...")
            surprises_2026 = await client.fetch_earnings_surprises_bulk(year=2026)
            surprises_2025 = await client.fetch_earnings_surprises_bulk(year=2025)
            earnings_company_df = await client.fetch_earnings_company_bulk(symbols=symbols, limit=500)
            earnings_surprises_df = pd.concat(
                [surprises_2026, surprises_2025, earnings_company_df],
                ignore_index=True,
            ).drop_duplicates()
            logger.info(f"  Earnings surprises: {len(earnings_surprises_df)} records")
            logger.info(f"  Earnings company history: {len(earnings_company_df)} records")
            return earnings_surprises_df, earnings_company_df

        if args.fast_debug_pead:
            logger.info("Step 1b: Fast mode prefetch of PEAD inputs...")
            bulk_data["earnings_surprises"], _ = await fetch_earnings_surprises_bundle()
        
        # Step 2: Fetch per-ticker financial statements
        logger.info("Step 2: Fetching financial statements...")
        bulk_data["income_statements"] = await client.fetch_income_statement_bulk(
            symbols=symbols, period=args.financial_period
        )
        current_year = datetime.now().year
        income_bulk_current = await client.fetch_income_statement_bulk_by_year(
            year=current_year,
            period=args.financial_period,
            symbols=symbols,
        )
        income_bulk_prev = await client.fetch_income_statement_bulk_by_year(
            year=current_year - 1,
            period=args.financial_period,
            symbols=symbols,
        )
        bulk_data["income_statements"] = pd.concat(
            [
                _coalesce_duplicate_columns(bulk_data["income_statements"]),
                _coalesce_duplicate_columns(income_bulk_current),
                _coalesce_duplicate_columns(income_bulk_prev),
            ],
            ignore_index=True,
        ).drop_duplicates()
        logger.info(f"  Income statements: {len(bulk_data['income_statements'])} records")
        if not bulk_data["income_statements"].empty and "symbol" in bulk_data["income_statements"].columns:
            income_symbols = bulk_data["income_statements"]["symbol"].unique().tolist()
            logger.info(f"    Symbols with income data: {income_symbols}")
            logger.info(f"    Periods per symbol: {bulk_data['income_statements'].groupby('symbol').size().to_dict()}")
        
        bulk_data["balance_sheets"] = await client.fetch_balance_sheet_bulk(
            symbols=symbols, period=args.financial_period
        )
        balance_bulk_current = await client.fetch_balance_sheet_bulk_by_year(
            year=current_year,
            period=args.financial_period,
            symbols=symbols,
        )
        balance_bulk_prev = await client.fetch_balance_sheet_bulk_by_year(
            year=current_year - 1,
            period=args.financial_period,
            symbols=symbols,
        )
        bulk_data["balance_sheets"] = pd.concat(
            [
                _coalesce_duplicate_columns(bulk_data["balance_sheets"]),
                _coalesce_duplicate_columns(balance_bulk_current),
                _coalesce_duplicate_columns(balance_bulk_prev),
            ],
            ignore_index=True,
        ).drop_duplicates()
        logger.info(f"  Balance sheets: {len(bulk_data['balance_sheets'])} records")
        if not bulk_data["balance_sheets"].empty and "symbol" in bulk_data["balance_sheets"].columns:
            logger.info(f"    Balance sheet symbols: {bulk_data['balance_sheets']['symbol'].unique().tolist()}")
        
        bulk_data["cash_flows"] = await client.fetch_cash_flow_bulk(
            symbols=symbols, period=args.financial_period
        )
        cash_bulk_current = await client.fetch_cash_flow_bulk_by_year(
            year=current_year,
            period=args.financial_period,
            symbols=symbols,
        )
        cash_bulk_prev = await client.fetch_cash_flow_bulk_by_year(
            year=current_year - 1,
            period=args.financial_period,
            symbols=symbols,
        )
        bulk_data["cash_flows"] = pd.concat(
            [
                _coalesce_duplicate_columns(bulk_data["cash_flows"]),
                _coalesce_duplicate_columns(cash_bulk_current),
                _coalesce_duplicate_columns(cash_bulk_prev),
            ],
            ignore_index=True,
        ).drop_duplicates()
        logger.info(f"  Cash flows: {len(bulk_data['cash_flows'])} records")
        if not bulk_data["cash_flows"].empty and "symbol" in bulk_data["cash_flows"].columns:
            logger.info(f"    Cash flow symbols: {bulk_data['cash_flows']['symbol'].unique().tolist()}")

        bulk_data["income_growth"] = await client.fetch_income_statement_growth_bulk_by_year(
            year=current_year,
            period=args.financial_period,
            symbols=symbols,
        )
        logger.info(f"  Income growth bulk: {len(bulk_data['income_growth'])} records")
        
        # Step 3: Fetch prices
        logger.info("Step 3: Fetching prices...")
        bulk_data["prices"] = await client.fetch_bulk_daily_prices(symbols=symbols)
        logger.info(f"  Prices: {len(bulk_data['prices'])} records")

        if args.fast_debug_pead:
            bulk_data["price_history"] = pd.DataFrame()
            bulk_data["technical_indicators"] = pd.DataFrame()
            logger.info("  Fast mode: skipping price history and technical indicators")
        else:
            logger.info("  Step 3b: Fetching historical price EOD...")
            bulk_data["price_history"] = await client.fetch_historical_price_eod_bulk(
                symbols=symbols,
                mode="full",
                limit=500,
            )
            logger.info(f"  Price history: {len(bulk_data['price_history'])} records")

            logger.info("  Step 3c: Fetching technical indicators (RSI/ADX)...")
            technical_rsi = await client.fetch_technical_indicator_bulk(
                indicator="rsi", symbols=symbols, period_length=14, timeframe="1day", limit=300
            )
            technical_adx = await client.fetch_technical_indicator_bulk(
                indicator="adx", symbols=symbols, period_length=14, timeframe="1day", limit=300
            )
            bulk_data["technical_indicators"] = pd.concat([technical_rsi, technical_adx], ignore_index=True).drop_duplicates()
            logger.info(f"  Technical indicators: {len(bulk_data['technical_indicators'])} records")
        
        # Step 4: Fetch analyst estimates (now with historical time series)
        logger.info("Step 4: Fetching analyst estimates (current + historical)...")
        bulk_data["analyst_estimates"] = await client.fetch_analyst_estimates_bulk(
            symbols=symbols, period=args.financial_period, historical_limit=20
        )
        logger.info(f"  Analyst estimates: {len(bulk_data['analyst_estimates'])} records")

        logger.info("  Step 4b: Fetching financial estimates...")
        financial_estimates_df = await client.fetch_financial_estimates_bulk(
            symbols=symbols, period="quarterly", historical_limit=20
        )
        if not financial_estimates_df.empty:
            bulk_data["analyst_estimates"] = (
                pd.concat([bulk_data["analyst_estimates"], financial_estimates_df], ignore_index=True)
                .drop_duplicates()
            )
        logger.info(f"  Financial estimates: {len(financial_estimates_df)} records")
        logger.info(f"  Combined estimate history: {len(bulk_data['analyst_estimates'])} records")
        
        # Step 5: Fetch Earnings Surprises (BULK)
        # Fetch current and previous year to ensure we have context
        if not args.fast_debug_pead:
            bulk_data["earnings_surprises"], _ = await fetch_earnings_surprises_bundle()
        
        # Step 6: Fetch insider trading (bulk, not per-symbol)
        if args.fast_debug_pead:
            bulk_data["insider_trading"] = pd.DataFrame()
            bulk_data["insider_statistics"] = pd.DataFrame()
            bulk_data["employee_count"] = pd.DataFrame()
            logger.info("Step 6: Fast mode skipping insider/employee endpoints")
        else:
            logger.info("Step 6: Fetching insider trading...")
            bulk_data["insider_trading"] = await client.fetch_insider_trading_bulk(page=0, limit=500)
            logger.info(f"  Insider trading: {len(bulk_data['insider_trading'])} records")

            logger.info("  Step 6b: Fetching insider trading statistics...")
            bulk_data["insider_statistics"] = await client.fetch_insider_trading_statistics_bulk(
                symbols=symbols,
                limit=500,
            )
            logger.info(f"  Insider statistics: {len(bulk_data['insider_statistics'])} records")

            logger.info("  Step 6d: Fetching insider search trades...")
            insider_search_df = await client.fetch_search_insider_trades_bulk(symbols=symbols, limit=500)
            if not insider_search_df.empty:
                bulk_data["insider_trading"] = pd.concat(
                    [bulk_data["insider_trading"], insider_search_df],
                    ignore_index=True,
                ).drop_duplicates()
            logger.info(f"  Insider search trades: {len(insider_search_df)} records")

            logger.info("  Step 6c: Fetching employee counts...")
            bulk_data["employee_count"] = await client.fetch_employee_count_bulk(
                symbols=symbols,
                limit=500,
            )
            employee_hist_df = await client.fetch_historical_employee_count_bulk(symbols=symbols, limit=500)
            if not employee_hist_df.empty:
                bulk_data["employee_count"] = pd.concat(
                    [bulk_data["employee_count"], employee_hist_df],
                    ignore_index=True,
                ).drop_duplicates()
            logger.info(f"  Employee count: {len(bulk_data['employee_count'])} records")
        
        # Step 7: Institutional ownership (try recent quarter fallbacks)
        if args.fast_debug_pead:
            bulk_data["institutional_ownership"] = pd.DataFrame()
            bulk_data["scores"] = pd.DataFrame()
            bulk_data["peers"] = pd.DataFrame()
            logger.info("Step 7/8: Fast mode skipping institutional, scores, and peers")
        else:
            logger.info("Step 7: Fetching institutional ownership...")
            institutional_df = pd.DataFrame()
            inst_attempts = [(2024, 4), (2024, 3), (2023, 4)]
            for year, quarter in inst_attempts:
                logger.info(f"  Trying institutional ownership for {year} Q{quarter}...")
                institutional_df = await client.fetch_institutional_ownership_bulk(
                    symbols=symbols,
                    year=year,
                    quarter=quarter,
                )
                if not institutional_df.empty:
                    logger.info(f"  Institutional ownership: {len(institutional_df)} records ({year} Q{quarter})")
                    break

            if institutional_df.empty:
                logger.info("  Institutional ownership unavailable for tested quarters; continuing gracefully")

            bulk_data["institutional_ownership"] = institutional_df

            logger.info("Step 8: Fetching positions summary and scores/peers...")
            positions_summary_df = await client.fetch_positions_summary_bulk(symbols=symbols, limit=500)
            if not positions_summary_df.empty:
                bulk_data["institutional_ownership"] = pd.concat(
                    [bulk_data["institutional_ownership"], positions_summary_df],
                    ignore_index=True,
                ).drop_duplicates()
            bulk_data["scores"] = await client.fetch_scores_bulk(symbols=symbols)
            bulk_data["peers"] = await client.fetch_peers_bulk()
            if bulk_data["peers"].empty:
                bulk_data["peers"] = await client.fetch_peers_for_symbols(symbols=symbols, limit=500)
            logger.info(f"  Positions summary: {len(positions_summary_df)} records")
            logger.info(f"  Scores bulk: {len(bulk_data['scores'])} records")
            logger.info(f"  Peers: {len(bulk_data['peers'])} records")
        
        logger.info("Bulk data fetch completed")

    
    # Enable PEAD and Revision Jerk in the ranker config now that we have the data
    ranker_config.use_pead = not bulk_data["earnings_surprises"].empty
    ranker_config.use_revision_jerk = not bulk_data["analyst_estimates"].empty
    if ranker_config.use_pead:
        logger.info("  ✓ PEAD signal enabled")
    if ranker_config.use_revision_jerk:
        logger.info("  ✓ Analyst Revision Jerk enabled")

    # Map FMP data to ranker-expected format
    logger.info("Mapping FMP data to ranker format...")
    raw_df = create_raw_fmp_dataframe(
        bulk_data=bulk_data,
        market=args.market,
        financial_period=args.financial_period,
        historical_estimates_df=bulk_data["analyst_estimates"],
        surprises_df=bulk_data["earnings_surprises"]
    )

    if args.symbols and not raw_df.empty and "symbol" in raw_df.columns:
        requested_symbols = {s.strip().upper() for s in args.symbols.split(",") if s.strip()}
        before_count = len(raw_df)
        raw_df = raw_df[raw_df["symbol"].astype(str).str.upper().isin(requested_symbols)].copy()
        logger.info(f"Applied --symbols filter after mapping: {before_count} -> {len(raw_df)} rows")
    
    logger.info(f"Raw DataFrame: {len(raw_df)} symbols, {len(raw_df.columns)} columns")

    if args.trace_sparsity:
        trace_sparsity(raw_df, bulk_data, sample_size=args.trace_sample_size)
    
    if raw_df.empty:
        logger.error("No data available after mapping. Exiting.")
        sys.exit(1)
    
    # Run the ranker
    logger.info("Running ranker...")
    full_df, ranked_df, diagnostic_df = build_ranked_frame(raw_df, ranker_config)
    
    logger.info(f"Ranking completed: {len(ranked_df)} symbols ranked")
    
    # Filter for real errors in diagnostic_df (if any)
    if not diagnostic_df.empty and "error" in diagnostic_df.columns:
        real_errors = diagnostic_df[diagnostic_df["error"].notna() & (diagnostic_df["error"] != "")]
        if not real_errors.empty:
            print_error_summary(real_errors)
    
    # Save output
    output_path = Path(args.output)
    
    # Reorder columns to move composite_score and rank to the front
    cols = ranked_df.columns.tolist()
    front_cols = ["symbol", "composite_score", "rank"]
    actual_front = [c for c in front_cols if c in cols]
    remaining = [c for c in cols if c not in actual_front]
    ranked_df = ranked_df[actual_front + remaining]
    
    # Use ranked_df as the primary output
    ranked_df.to_csv(output_path, index=False)
    logger.info(f"Ranked stocks saved to {output_path}")
    
    # Save diagnostic comparison if available
    if not diagnostic_df.empty:
        diag_path = output_path.parent / f"{output_path.stem}_diagnostics.csv"
        diagnostic_df.to_csv(diag_path, index=False)
        logger.info(f"Diagnostics saved to {diag_path}")
    
    # Verification checks
    if args.verify:
        logger.info("Running verification checks...")
        verify_ranking(ranked_df, raw_df)
    
    logger.info("FMP ranker execution completed successfully")
    
    return ranked_df, diagnostic_df, pd.DataFrame()


def verify_ranking(ranked_df: pd.DataFrame, raw_df: pd.DataFrame):
    """
    Run verification checks on the ranking results.
    
    Args:
        ranked_df: Output from build_ranked_frame
        raw_df: Input DataFrame with raw metrics
    """
    logger.info("Verification Results:")
    
    # Check that we have ranked stocks
    if ranked_df.empty:
        logger.error("  ❌ No stocks ranked")
    else:
        logger.info(f"  ✓ {len(ranked_df)} stocks ranked")
    
    # Check for composite_score column
    if "composite_score" in ranked_df.columns:
        logger.info(f"  ✓ composite_score column exists")
        logger.info(f"    - Mean: {ranked_df['composite_score'].mean():.4f}")
        logger.info(f"    - Std: {ranked_df['composite_score'].std():.4f}")
    else:
        logger.error("  ❌ composite_score column missing")
    
    # Check for core factors
    core_factors = ["shareholder_yield", "gross_profitability", "adjusted_book_to_market"]
    for factor in core_factors:
        if factor in ranked_df.columns:
            non_null_count = ranked_df[factor].notna().sum()
            logger.info(f"  ✓ {factor}: {non_null_count}/{len(ranked_df)} non-null")
        else:
            logger.warning(f"  ⚠ {factor} column missing")
    
    # Check for factor_non_null_count if it exists
    if "factor_non_null_count" in ranked_df.columns:
        avg_factor_count = ranked_df["factor_non_null_count"].mean()
        logger.info(f"  ✓ Average factor_non_null_count: {avg_factor_count:.2f}")
        
        if avg_factor_count == 0:
            logger.error("  ❌ factor_non_null_count is zero - no factors computed!")
        else:
            logger.info(f"  ✓ Factors are being computed")
    
    # Check sector distribution if neutralization is enabled
    if "sector" in ranked_df.columns:
        sector_counts = ranked_df["sector"].value_counts()
        logger.info(f"  ✓ Sector distribution: {len(sector_counts)} sectors")
        logger.info(f"    Top 3 sectors: {sector_counts.head(3).to_dict()}")
    
    # Check for rank column
    if "rank" in ranked_df.columns:
        logger.info(f"  ✓ rank column exists")
        logger.info(f"    - Top rank: {ranked_df['rank'].min()}")
        logger.info(f"    - Bottom rank: {ranked_df['rank'].max()}")
    else:
        logger.warning(f"  ⚠ rank column missing")
    
    logger.info("Verification checks completed")


def trace_sparsity(raw_df: pd.DataFrame, bulk_data: dict[str, pd.DataFrame], sample_size: int = 10):
    """Log where sparse feature families lose coverage: payload, symbol-level, and merged non-null."""
    logger.info("Sparsity Trace (raw -> merged):")

    family_specs = {
        "revision": {
            "bulk_key": "analyst_estimates",
            "raw_cols": ["revision_impulse_signal", "revision_jerk_signal", "estimate_term_structure_signal"],
        },
        "earnings": {
            "bulk_key": "earnings_surprises",
            "raw_cols": ["pead_signal", "sue_signal", "earnings_momentum_signal"],
        },
        "momentum": {
            "bulk_key": "prices",
            "raw_cols": ["price_momentum_6m_ex_1m", "price_momentum_effective_signal", "price_momentum_has_coverage"],
        },
        "forensic": {
            "bulk_key": "income_statements",
            "raw_cols": ["beneish_m_score", "accrual_volatility", "working_capital_stress_penalty"],
        },
        "employee": {
            "bulk_key": "employee_count",
            "raw_cols": ["full_time_employees", "revenue_per_employee", "gross_profit_per_employee"],
        },
        "institutional": {
            "bulk_key": "institutional_ownership",
            "raw_cols": ["percent_institutions", "institution_holder_count_latest", "institutional_ownership_delta"],
        },
        "insider": {
            "bulk_key": "insider_statistics",
            "raw_cols": ["insider_conviction_trade_count", "insider_conviction_buy_person_count", "insider_conviction_sell_person_count"],
        },
    }

    for family, spec in family_specs.items():
        bulk_df = bulk_data.get(spec["bulk_key"], pd.DataFrame())
        payload_rows = len(bulk_df)
        payload_symbols = bulk_df["symbol"].nunique() if (not bulk_df.empty and "symbol" in bulk_df.columns) else 0
        logger.info(f"  [{family}] payload rows={payload_rows}, symbols={payload_symbols}")
        for col in spec["raw_cols"]:
            if col in raw_df.columns:
                nn = int(raw_df[col].notna().sum())
                logger.info(f"    merged {col}: {nn}/{len(raw_df)} non-null")
            else:
                logger.info(f"    merged {col}: MISSING_COLUMN")

    if raw_df.empty or "symbol" not in raw_df.columns:
        return

    sample_n = max(1, int(sample_size))
    sample_symbols = raw_df["symbol"].dropna().astype(str).head(sample_n).tolist()
    probe_cols = [
        "revision_impulse_signal",
        "revision_jerk_signal",
        "estimate_term_structure_signal",
        "revenue_growth_yoy",
        "revenue_acceleration",
        "price_momentum_6m_ex_1m",
        "price_momentum_effective_signal",
        "beneish_m_score",
        "accrual_volatility",
        "working_capital_stress_penalty",
        "insider_conviction_trade_count",
        "full_time_employees",
    ]
    available_probe_cols = [c for c in probe_cols if c in raw_df.columns]
    if not available_probe_cols:
        return

    sample_df = raw_df[raw_df["symbol"].astype(str).isin(sample_symbols)][["symbol"] + available_probe_cols].copy()
    for col in available_probe_cols:
        sample_df[col] = sample_df[col].notna().astype(int)
    logger.info("  Sample symbol non-null flags (1=present, 0=missing):")
    logger.info("\n%s", sample_df.to_string(index=False))


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        ranked_df, error_df, median_df = asyncio.run(run_fmp_ranker(args))
        sys.exit(0)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
