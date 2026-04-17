#!/usr/bin/env python3
"""
FMP API Endpoint Test Script

This script tests the FMP Ultimate API endpoints to verify they work correctly
and inspect the actual data structure returned. Use this to validate your API key
and identify any column name mismatches before running the full ranker.

Usage:
    python test_fmp_endpoints.py --api-key YOUR_FMP_API_KEY
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path

from eodhd_strategy.fmp_client import FMPClient, FMPConfig

# Configure logging to prevent garbled output from concurrent async operations
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    force=True  # Override any existing configuration
)


async def test_endpoint(client, endpoint_name, fetch_func, *args, **kwargs):
    """Test a single endpoint and report results."""
    print(f"\n{'='*80}")
    print(f"Testing: {endpoint_name}")
    print(f"{'='*80}")
    
    try:
        result = await fetch_func(*args, **kwargs)
        
        if result.empty:
            print(f"[FAILED] {endpoint_name}: No data returned")
            return False
        
        print(f"[OK] {endpoint_name}: {len(result)} records")
        print(f"  Columns: {list(result.columns)}")
        print(f"  Sample data:")
        print(result.head(2).to_string())
        
        return True
        
    except Exception as e:
        print(f"[FAILED] {endpoint_name}: Error - {e}")
        import traceback
        traceback.print_exc()
        return False


async def main(args):
    """Main test function."""
    print("="*80)
    print("FMP Stable API Endpoint Test")
    print("="*80)
    print(f"API Key: {args.api_key[:10]}...{args.api_key[-4:]}")
    print(f"Market: {args.market}")
    
    # Create FMP config
    config = FMPConfig(
        api_key=args.api_key,
        cache_dir=Path(args.cache_dir),
        max_retries=3,
        retry_delay=1.0,
        request_timeout=30.0,
        batch_size=100
    )
    
    # Create client
    async with FMPClient(config) as client:
        results = {}
        
        # Test screener first (this will give us symbols)
        results["screener"] = await test_endpoint(
            client,
            "Company Screener",
            client.fetch_screener,
            is_actively_trading=True,
            market_cap_more_than=1_000_000_000,  # $1B+
            limit=100
        )
        
        # Test profile bulk with part parameter
        results["profiles"] = await test_endpoint(
            client,
            "Profile Bulk",
            client.fetch_profile_bulk,
            market=args.market,
            part=0
        )
        
        # Use AAPL and MSFT for per-symbol tests
        test_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        
        results["prices"] = await test_endpoint(
            client,
            "Batch Quote (Prices)",
            client.fetch_bulk_daily_prices,
            symbols=test_symbols
        )
        
        results["income"] = await test_endpoint(
            client,
            "Income Statement (per-symbol)",
            client.fetch_income_statement_bulk,
            symbols=test_symbols,
            period="annual"
        )
        
        results["balance"] = await test_endpoint(
            client,
            "Balance Sheet (per-symbol)",
            client.fetch_balance_sheet_bulk,
            symbols=test_symbols,
            period="annual"
        )
        
        results["cashflow"] = await test_endpoint(
            client,
            "Cash Flow (per-symbol)",
            client.fetch_cash_flow_bulk,
            symbols=test_symbols,
            period="annual"
        )
        
        results["estimates"] = await test_endpoint(
            client,
            "Analyst Estimates (per-ticker)",
            client.fetch_analyst_estimates_bulk,
            symbols=test_symbols,
            period="annual"
        )
        
        results["institutional"] = await test_endpoint(
            client,
            "Institutional Ownership (per-ticker)",
            client.fetch_institutional_ownership_bulk,
            symbols=test_symbols
        )
        
        # Test corrected endpoints
        results["earnings_company"] = await test_endpoint(
            client,
            "Earnings Company (earnings-company endpoint)",
            client.fetch_earnings_company_bulk,
            symbols=["AAPL"],
            limit=1
        )
        
        results["insider_search"] = await test_endpoint(
            client,
            "Insider Trading Search (insider-trading/search endpoint)",
            client.fetch_search_insider_trades_bulk,
            symbols=["AAPL"],
            limit=1
        )
        
        results["stock_peers"] = await test_endpoint(
            client,
            "Stock Peers (stock-peers endpoint)",
            client.fetch_peers_for_symbols,
            symbols=["AAPL"],
            limit=1
        )
        
        results["positions_summary"] = await test_endpoint(
            client,
            "Positions Summary (institutional-ownership/symbol-positions-summary endpoint)",
            client.fetch_positions_summary_bulk,
            symbols=["AAPL"],
            limit=1
        )
        
        results["insider"] = await test_endpoint(
            client,
            "Insider Trading Latest",
            client.fetch_insider_trading_bulk,
            page=0,
            limit=100
        )
        
        # Summary
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        
        for endpoint, success in results.items():
            status = "[OK]" if success else "[FAILED]"
            print(f"{status} {endpoint}")
        
        print(f"\nPassed: {passed}/{total}")
        
        if passed == total:
            print("\n[OK] All endpoints working! Ready to run full ranker.")
            return 0
        else:
            print("\n[WARN] Some endpoints failed. Check errors above and adjust fmp_mapper.py if needed.")
            return 1


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Test FMP Ultimate API endpoints")
    parser.add_argument("--api-key", required=True, help="FMP Ultimate API key")
    parser.add_argument("--cache-dir", default=".fmp_cache", help="Cache directory")
    parser.add_argument("--market", default="us", help="Market to test")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    exit_code = asyncio.run(main(args))
    sys.exit(exit_code)
