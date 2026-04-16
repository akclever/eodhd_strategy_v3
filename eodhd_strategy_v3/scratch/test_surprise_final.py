import asyncio
import aiohttp

async def test():
    api_key = "oYm8ALIS7e8s4Zl8SasXMWQJ7D4Zuntw"
    # POSITIONAL SYMBOL
    url = f"https://financialmodelingprep.com/stable/earnings-surprises/AAPL?apikey={api_key}"
    
    async with aiohttp.ClientSession() as session:
        print(f"TESTING: {url.replace(api_key, 'REDACTED')}")
        async with session.get(url) as response:
            print(f"STATUS: {response.status}")
            text = await response.text()
            print(f"RAW: {text[:500]}")

asyncio.run(test())
