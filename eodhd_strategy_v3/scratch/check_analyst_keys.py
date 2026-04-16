import asyncio
import aiohttp
import pandas as pd

async def test():
    api_key = "oYm8ALIS7e8s4Zl8SasXMWQJ7D4Zuntw"
    # PATH for analyst estimates
    url = f"https://financialmodelingprep.com/stable/analyst-estimates?symbol=AAPL&period=annual&limit=20&apikey={api_key}"
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                if data:
                    print(f"KEYS: {data[0].keys()}")
                    print(f"SAMPLE DATES: {[d.get('date') for d in data[:5]]}")

asyncio.run(test())
