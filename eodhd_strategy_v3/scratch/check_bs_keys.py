import asyncio
import aiohttp

async def test():
    api_key = "oYm8ALIS7e8s4Zl8SasXMWQJ7D4Zuntw"
    url = f"https://financialmodelingprep.com/stable/balance-sheet-statement?symbol=AAPL&apikey={api_key}"
    
    async with aiohttp.ClientSession() as session:
        print(f"TESTING: {url.replace(api_key, 'REDACTED')}")
        async with session.get(url) as response:
            print(f"STATUS: {response.status}")
            if response.status == 200:
                data = await response.json()
                if data:
                    print(f"KEYS: {data[0].keys()}")

asyncio.run(test())
