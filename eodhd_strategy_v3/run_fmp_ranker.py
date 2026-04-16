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
        symbols = screener_sorted["symbol"].tolist()[:500]
        logger.info(f"  Using top {len(symbols)} symbols for per-ticker data")
        
        # Step 2: Fetch per-ticker financial statements
        logger.info("Step 2: Fetching financial statements...")
        bulk_data["income_statements"] = await client.fetch_income_statement_bulk(
            symbols=symbols, period=args.financial_period
        )
        logger.info(f"  Income statements: {len(bulk_data['income_statements'])} records")
        
        bulk_data["balance_sheets"] = await client.fetch_balance_sheet_bulk(
            symbols=symbols, period=args.financial_period
        )
        logger.info(f"  Balance sheets: {len(bulk_data['balance_sheets'])} records")
        
        bulk_data["cash_flows"] = await client.fetch_cash_flow_bulk(
            symbols=symbols, period=args.financial_period
        )
        logger.info(f"  Cash flows: {len(bulk_data['cash_flows'])} records")
        
        # Step 3: Fetch prices
        logger.info("Step 3: Fetching prices...")
        bulk_data["prices"] = await client.fetch_bulk_daily_prices(symbols=symbols)
        logger.info(f"  Prices: {len(bulk_data['prices'])} records")
        
        # Step 4: Fetch analyst estimates (now with historical time series)
        logger.info("Step 4: Fetching analyst estimates (current + historical)...")
        bulk_data["analyst_estimates"] = await client.fetch_analyst_estimates_bulk(
            symbols=symbols, period=args.financial_period, historical_limit=20
        )
        logger.info(f"  Analyst estimates: {len(bulk_data['analyst_estimates'])} records")
        
        # Step 5: Fetch Earnings Surprises (BULK)
        # Fetch current and previous year to ensure we have context
        logger.info("Step 5: Fetching bulk earnings surprises (2025-2026)...")
        surprises_2026 = await client.fetch_earnings_surprises_bulk(year=2026)
        surprises_2025 = await client.fetch_earnings_surprises_bulk(year=2025)
        bulk_data["earnings_surprises"] = pd.concat([surprises_2026, surprises_2025]).drop_duplicates()
        logger.info(f"  Earnings surprises: {len(bulk_data['earnings_surprises'])} records")
        
        # Step 6: Fetch insider trading (bulk, not per-symbol)
        logger.info("Step 6: Fetching insider trading...")
        bulk_data["insider_trading"] = await client.fetch_insider_trading_bulk(page=0, limit=500)
        logger.info(f"  Insider trading: {len(bulk_data['insider_trading'])} records")
        
        # Step 7: Institutional ownership (currently unavailable on FMP stable API)
        logger.info("Step 7: Institutional ownership (degraded - endpoint unavailable)")
        bulk_data["institutional_ownership"] = pd.DataFrame()
        
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
    
    logger.info(f"Raw DataFrame: {len(raw_df)} symbols, {len(raw_df.columns)} columns")
    
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
