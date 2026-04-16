import asyncio
import aiohttp

async def test():
    api_key = "oYm8ALIS7e8s4Zl8SasXMWQJ7D4Zuntw"
    urls = [
        f"https://financialmodelingprep.com/api/v4/stock-news-sentiments-rss-feed?page=0&apikey={api_key}",
        f"https://financialmodelingprep.com/stable/stock-news-sentiments-rss-feed?page=0&apikey={api_key}"
    ]
    
    async with aiohttp.ClientSession() as session:
        for url in urls:
            print(f"\nTESTING: {url.replace(api_key, 'REDACTED')}")
            async with session.get(url) as response:
                print(f"STATUS: {response.status}")
                if response.status == 200:
                    data = await response.json()
                    print(f"COUNT: {len(data)}")
                    if data:
                        print(f"KEYS: {data[0].keys()}")
                        print(f"SAMPLE: {data[0]}")
                else:
                    print(f"REASON: {await response.text()}")

asyncio.run(test())
