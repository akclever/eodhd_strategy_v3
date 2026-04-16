import asyncio
import aiohttp

async def test():
    api_key = "oYm8ALIS7e8s4Zl8SasXMWQJ7D4Zuntw"
    urls = [
        f"https://financialmodelingprep.com/stable/cash-flow-statement/AAPL?apikey={api_key}",
        f"https://financialmodelingprep.com/stable/cash-flow-statement-bulk?year=2024&apikey={api_key}"
    ]
    
    async with aiohttp.ClientSession() as session:
        for url in urls:
            print(f"TESTING: {url.replace(api_key, 'REDACTED')}")
            async with session.get(url) as response:
                print(f"STATUS: {response.status}")
                text = await response.text()
                print(f"SAMPLE: {text[:200]}")

asyncio.run(test())
