import pandas as pd
import numpy as np

# Mock symbols from screener
symbols = ["AAPL", "MSFT", "GOOGL"]
output = pd.DataFrame({"symbol": symbols})

# Mock surprises
surprises_df = pd.DataFrame({
    "symbol": ["AAPL", "AAPL", "MSFT", "MSFT"],
    "date": ["2026-01-01", "2026-04-01", "2026-01-01", "2026-04-01"],
    "surprise_percent": [5.0, 10.0, -2.0, 4.0]
})

# MATCHING LOGIC
sur_sorted = surprises_df.sort_values(["symbol", "date"])
latest_sur = sur_sorted.groupby("symbol").last().reset_index()
sur_map = latest_sur.set_index("symbol")["surprise_percent"]

print("Sur map:")
print(sur_map)

output["pead_signal"] = output["symbol"].map(sur_map) / 100.0
print("\nFinal output:")
print(output)
