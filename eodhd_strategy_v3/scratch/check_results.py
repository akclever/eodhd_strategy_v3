import pandas as pd
import sys
filename = sys.argv[1] if len(sys.argv) > 1 else 'ranked_deep_scan_fixed.csv'
df = pd.read_csv(filename)
print(f"File: {filename}")
print(f"Shape: {df.shape}")
print("\nUnique symbols count:", df['symbol'].nunique())

print("\nNon-null counts for key signals:")
signals = ['pead_signal', 'revision_impulse_signal', 'surprise_percent', 'composite_score']
for s in signals:
    if s in df.columns:
        print(f"{s}: {df[s].notna().sum()}")
    else:
        print(f"{s}: MISSING")

print("\nSample values for signals (first 10):")
if 'pead_signal' in df.columns:
    print(df[['symbol', 'pead_signal', 'revision_impulse_signal']].head(10))
