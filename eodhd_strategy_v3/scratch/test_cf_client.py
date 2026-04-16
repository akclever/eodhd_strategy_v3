import asyncio
import aiohttp

async def test():
    api_key = "oYm8ALIS7e8s4Zl8SasXMWQJ7D4Zuntw"
    # THE CLIENT USES THIS:
    url = f"https://financialmodelingprep.com/stable/cash-flow-statement?symbol=AAPL&apikey={api_key}"
    
    async with aiohttp.ClientSession() as session:
        print(f"TESTING: {url.replace(api_key, 'REDACTED')}")
        async with session.get(url) as response:
            print(f"STATUS: {response.status}")
            if response.status == 200:
                data = await response.json()
                if data:
                    print(f"COUNT: {len(data)}")
                    print(f"KEYS: {data[0].keys()}")
                else:
                    print("EMPTY DATA")
            else:
                print(f"ERROR: {await response.text()}")

asyncio.run(test())
