import requests
import pandas as pd
import io

api_key = "oYm8ALIS7e8s4Zl8SasXMWQJ7D4Zuntw"
r = requests.get(f'https://financialmodelingprep.com/stable/profile-bulk?part=0&apikey={api_key}')
print("Status:", r.status_code)
text = r.text
print("Text length:", len(text))
if text.startswith("symbol,") or text.startswith('"symbol",'):
    df = pd.read_csv(io.StringIO(text))
    print("CSV parsed! shape:", df.shape)
else:
    print("Not starting with symbol:", repr(text[:50]))
