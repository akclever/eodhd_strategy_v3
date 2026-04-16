import asyncio
import aiohttp

async def test():
    api_key = "oYm8ALIS7e8s4Zl8SasXMWQJ7D4Zuntw"
    urls = [
        f"https://financialmodelingprep.com/stable/sentiment-analysis-bulk?date=2026-04-15&apikey={api_key}",
        f"https://financialmodelingprep.com/stable/news?apikey={api_key}"
    ]
    
    async with aiohttp.ClientSession() as session:
        for url in urls:
            print(f"\nTESTING: {url.replace(api_key, 'REDACTED')}")
            async with session.get(url) as response:
                print(f"STATUS: {response.status}")
                if response.status == 200:
                    data = await response.json()
                    print(f"COUNT: {len(data) if isinstance(data, list) else 1}")
                    sample = data[0] if isinstance(data, list) and data else data
                    print(f"SAMPLE: {str(sample)[:300]}")
                else:
                    print(f"REASON: {await response.text()}")

asyncio.run(test())
