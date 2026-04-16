import asyncio
import aiohttp
import pandas as pd
from pathlib import Path

async def test():
    api_key = "oYm8ALIS7e8s4Zl8SasXMWQJ7D4Zuntw"
    urls = [
        # Stable
        f"https://financialmodelingprep.com/stable/earnings-surprises?symbol=AAPL&apikey={api_key}",
        f"https://financialmodelingprep.com/stable/analyst-estimates?symbol=AAPL&apikey={api_key}",
        # V3
        f"https://financialmodelingprep.com/api/v3/earnings-surprises/AAPL?apikey={api_key}",
        f"https://financialmodelingprep.com/api/v3/analyst-estimates/AAPL?apikey={api_key}",
    ]
    
    async with aiohttp.ClientSession() as session:
        for url in urls:
            print(f"\nTESTING: {url.replace(api_key, 'REDACTED')}")
            async with session.get(url) as response:
                print(f"STATUS: {response.status}")
                if response.status == 200:
                    data = await response.json()
                    print(f"COUNT: {len(data) if isinstance(data, list) else 1}")
                    print(f"SAMPLE: {str(data)[:200]}")
                else:
                    print(f"REASON: {await response.text()}")

asyncio.run(test())
