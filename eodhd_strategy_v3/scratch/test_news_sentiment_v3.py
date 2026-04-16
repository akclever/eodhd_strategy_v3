import asyncio
import aiohttp

async def test():
    api_key = "oYm8ALIS7e8s4Zl8SasXMWQJ7D4Zuntw"
    urls = [
        # Common v4 sentiment patterns
        f"https://financialmodelingprep.com/api/v4/sentiment/AAPL?apikey={api_key}",
        f"https://financialmodelingprep.com/api/v4/historical/social-sentiment?symbol=AAPL&apikey={api_key}",
        # V3 news
        f"https://financialmodelingprep.com/api/v3/stock_news?tickers=AAPL&limit=5&apikey={api_key}"
    ]
    
    async with aiohttp.ClientSession() as session:
        for url in urls:
            print(f"\nTESTING: {url.replace(api_key, 'REDACTED')}")
            async with session.get(url) as response:
                print(f"STATUS: {response.status}")
                if response.status == 200:
                    data = await response.json()
                    print(f"COUNT: {len(data) if isinstance(data, list) else 1}")
                    print(f"SAMPLE: {str(data)[:300]}")
                else:
                    print(f"REASON: {await response.text()}")

asyncio.run(test())
