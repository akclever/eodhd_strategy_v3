import asyncio
import pandas as pd
from eodhd_strategy.fmp_client import FMPClient, FMPConfig
from pathlib import Path

async def test():
    config = FMPConfig(
        api_key="oYm8ALIS7e8s4Zl8SasXMWQJ7D4Zuntw",
        cache_dir=Path(".fmp_test_cache")
    )
    async with FMPClient(config) as client:
        # Test earnings surprises
        print("Testing earnings-surprises for AAPL...")
        data = await client._get("earnings-surprises", {"symbol": "AAPL"})
        print(f"RAW DATA: {data[:1] if data else 'None'}")
        
        # Test historical estimates
        print("\nTesting analyst-estimates for AAPL...")
        est = await client._get("analyst-estimates", {"symbol": "AAPL", "limit": 5})
        print(f"RAW EST: {est[:1] if est else 'None'}")

asyncio.run(test())
