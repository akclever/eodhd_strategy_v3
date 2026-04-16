import asyncio
import aiohttp

async def test():
    api_key = "oYm8ALIS7e8s4Zl8SasXMWQJ7D4Zuntw"
    url = f"https://financialmodelingprep.com/stable/earnings-surprises-bulk?year=2026&apikey={api_key}"
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                text = await response.text()
                lines = text.splitlines()
                if lines:
                    print(f"HEADER: {lines[0]}")
                    print(f"SAMPLE: {lines[1] if len(lines) > 1 else 'None'}")

asyncio.run(test())
