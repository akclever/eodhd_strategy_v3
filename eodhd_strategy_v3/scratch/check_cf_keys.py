import asyncio
import aiohttp

async def test():
    api_key = "oYm8ALIS7e8s4Zl8SasXMWQJ7D4Zuntw"
    url = f"https://financialmodelingprep.com/stable/cash-flow-statement/AAPL?period=annual&apikey={api_key}"
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                if data:
                    print(f"KEYS: {data[0].keys()}")
                    print(f"DIVIDENDS: {data[0].get('dividendsPaid')}")

asyncio.run(test())
