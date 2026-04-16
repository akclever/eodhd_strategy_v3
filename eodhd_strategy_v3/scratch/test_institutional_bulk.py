import asyncio
import aiohttp

async def test():
    api_key = "oYm8ALIS7e8s4Zl8SasXMWQJ7D4Zuntw"
    urls = [
        f"https://financialmodelingprep.com/stable/institutional-ownership-bulk?apikey={api_key}",
        f"https://financialmodelingprep.com/stable/institutional-ownership/all?apikey={api_key}"
    ]
    
    async with aiohttp.ClientSession() as session:
        for url in urls:
            print(f"\nTESTING: {url.replace(api_key, 'REDACTED')}")
            async with session.get(url) as response:
                print(f"STATUS: {response.status}")
                if response.status == 200:
                    text = await response.text()
                    print(f"SAMPLE: {text[:200]}")
                else:
                    print(f"REASON: {await response.text()}")

asyncio.run(test())
