import csv

r = list(csv.DictReader(open('ranked_stocks_v3.csv')))
row = r[0]

print("=== KEY OVERLAY COLUMNS ===")
keys = ['technical_momentum_signal','technical_fetch_status','technical_rsi_14',
        'news_event_signal','news_fetch_status','news_has_coverage',
        'sentiment_latest','sentiment_fetch_status','sentiment_has_coverage',
        'insider_conviction_signal','insider_fetch_status',
        'pead_signal','sue_signal']
for k in keys:
    v = row.get(k, 'MISSING')
    print(f"{k}: {v}")

print("\n=== ALL NaN/EMPTY COLUMNS (first 30) ===")
nans = [k for k in row if row[k] in ('','None','nan','NaN')]
for k in sorted(nans)[:30]:
    print(f"  {k}")

print(f"\nTotal NaN/empty: {len(nans)}")
print(f"Total populated: {len([k for k in row if row[k] not in ('','None','nan','NaN','0','0.0')])}")
