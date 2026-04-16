import asyncio
import aiohttp

async def main():
    api_key = "oYm8ALIS7e8s4Zl8SasXMWQJ7D4Zuntw"
    async with aiohttp.ClientSession(headers={"User-Agent": "eodhd_strategy_v3/2.0"}) as session:
        url = "https://financialmodelingprep.com/stable/company-screener"
        params = {"limit": 10, "apikey": api_key, "isActivelyTrading": "true"}
        async with session.get(url, params=params) as response:
            print("Status:", response.status)
            print("Headers:", response.headers)
            text = await response.text()
            print("Text:", text[:200])

asyncio.run(main())
