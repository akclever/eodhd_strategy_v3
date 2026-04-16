#!/usr/bin/env python3
"""
Basic FMP API Key Test

Tests if the API key works with basic (non-Ultimate) endpoints.
"""

import asyncio
import argparse
import sys
from pathlib import Path

import aiohttp


async def test_basic_endpoints(api_key):
    """Test basic FMP endpoints that should work with any tier."""
    
    base_url = "https://financialmodelingprep.com/api/v3"
    
    print("="*80)
    print("Testing Basic FMP API Endpoints")
    print("="*80)
    
    async with aiohttp.ClientSession() as session:
        # Test 1: Company profile (single ticker)
        print("\nTest 1: Company Profile (AAPL)")
        url = f"{base_url}/profile/AAPL"
        params = {"apikey": api_key}
        
        try:
            async with session.get(url, params=params) as response:
                print(f"  Status: {response.status}")
                if response.status == 200:
                    data = await response.json()
                    print(f"  ✓ Success! Got {len(data)} records")
                    if data:
                        print(f"  Sample: {data[0].get('symbol', 'N/A')}")
                else:
                    text = await response.text()
                    print(f"  ❌ Failed: {text[:200]}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        # Test 2: Stock list (basic endpoint)
        print("\nTest 2: Stock List")
        url = f"{base_url}/stock/list"
        params = {"apikey": api_key}
        
        try:
            async with session.get(url, params=params) as response:
                print(f"  Status: {response.status}")
                if response.status == 200:
                    data = await response.json()
                    print(f"  ✓ Success! Got {len(data)} records if list, or data object")
                else:
                    text = await response.text()
                    print(f"  ❌ Failed: {text[:200]}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        # Test 3: Income statement (single ticker)
        print("\nTest 3: Income Statement (AAPL)")
        url = f"{base_url}/income-statement/AAPL"
        params = {"apikey": api_key}
        
        try:
            async with session.get(url, params=params) as response:
                print(f"  Status: {response.status}")
                if response.status == 200:
                    data = await response.json()
                    print(f"  ✓ Success! Got data")
                else:
                    text = await response.text()
                    print(f"  ❌ Failed: {text[:200]}")
        except Exception as e:
            print(f"  ❌ Error: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Test basic FMP API access")
    parser.add_argument("--api-key", required=True, help="FMP API key")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(test_basic_endpoints(args.api_key))
