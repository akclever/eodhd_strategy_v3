# EODHD Strategy v3

This refactor splits the workflow into three layers:

1. `run_rank.py` - builds a ranked stock list.
2. `run_rebalance.py` - converts the ranked list into target weights.
3. `eodhd_strategy/*` - shared modules for API access, feature engineering, ranking, portfolio construction, and macro event timing.

## Implemented improvements

- safer symbol normalization (`CODE.EXCHANGE`)
- staged pipeline: fundamentals for the full universe, sentiment overlays only for the shortlist
- dividend safety gate
- fundamentals-first PEAD v2 with surprise, decay, analyst breadth, and revision setup
- standardized earnings surprise (`SUE`) support inside the earnings-momentum layer
- revision-impulse overlay with estimate drift, revision breadth, coverage scaling, and disagreement penalty
- revenue growth / acceleration overlay from quarterly fundamentals
- 6m-ex-1m price momentum overlay from adjusted-price history
- entitlement-aware price momentum fallback to a trend proxy built from `price_to_200dma`, `recency_ratio`, and `distance_from_high`
- optional life-cycle conditioned tilts for `growth`, `mature`, and `recovery` profiles
- count-aware sentiment veto filter with article-count coverage gating
- optional Beneish M-Score hard filter, missing-data penalty, and accrual-volatility penalty
- sector- or industry-neutral scoring
- optional `sector` vs `none` neutralization comparison output
- ablation harness for comparing ranked outputs and holdings histories across strategy variants
- turnover buffer in portfolio construction
- sector caps and max position sizing
- event-aware rebalance timing using the economic events API
- diagnostics output for exposure checks, signal coverage, and per-factor contribution telemetry

## PowerShell example: ranking

```powershell
$env:EODHD_API_TOKEN = 'YOUR_TOKEN'

python run_rank.py `
  --exchanges US `
  --limit 2000 `
  --use-pead `
  --use-revision-impulse `
  --use-growth-acceleration `
  --growth-weight 0.10 `
  --use-price-momentum `
  --momentum-weight 0.10 `
  --use-life-cycle `
  --life-cycle-tilt-strength 0.35 `
  --use-sentiment `
  --use-beneish `
  --use-accrual-volatility `
  --missing-beneish-penalty 0.25 `
  --regime neutral `
  --dividend-source hybrid `
  --require-above-200dma `
  --neutralize-by sector `
  --compare-neutralization `
  --compare-revision-impulse-weights `
  --min-pead-analysts 3 `
  --min-revision-analysts 4 `
  --revision-impulse-weight 0.06 `
  --min-sentiment-articles-recent 3 `
  --overlay-top-n 150 `
  --top 25 `
  --output ranked_stocks_v3.csv `
  --diagnostics-output rank_diagnostics.csv
```

When `--compare-neutralization` is enabled, the ranker also writes a derived comparison CSV next to the ranked output, using the suffix `_neutralization_compare.csv`.
When `--compare-revision-impulse-weights` is enabled, the ranker also writes a derived comparison CSV using the suffix `_revision_impulse_compare.csv`; by default it compares weights `0.00, 0.04, 0.06, 0.08` on the same enriched dataset.
When `--use-beneish` is enabled, missing Beneish inputs get a small penalty, pathological-clipped Beneish cases are flagged separately in diagnostics/output, and the hard gate becomes slightly stricter only for large universes (`1000+` names).
When `--use-revision-impulse` is enabled, the ranker adds a small catalyst overlay built from EPS-estimate acceleration, revision breadth, analyst coverage, and estimate-dispersion control.
When `--use-growth-acceleration` is enabled, the ranker adds quarterly revenue growth and acceleration factors.
When `--use-price-momentum` is enabled, the ranker first tries to compute a medium-term `6m ex 1m` price momentum factor from adjusted closes. If the account does not appear to have historical price access, the ranker falls back automatically to a trend proxy built from `price_to_200dma`, `recency_ratio`, and `distance_from_high`, and prints a warning.
When `--use-life-cycle` is enabled, the ranker adds a small profile-aware tilt: `growth` names lean more on revenue acceleration / SUE / revision / momentum, `mature` names lean more on yield/profitability, and `recovery` names lean more on value plus timing momentum while paying a slightly higher forensic penalty.
The diagnostics file now includes signal coverage shares such as `share_price_momentum_has_coverage`, `share_price_momentum_proxy_used`, and `share_price_momentum_signal_coverage`, plus factor contribution medians and rank-correlation diagnostics like `ranked_contrib_corr::contrib_growth__contrib_revision_impulse`.
Ranked outputs now also include `currency_code` / `currency_name` and `listing_exchanges`, and you can write companion currency-specific lists with `--currency-list-output`, for example `EUR=ranked_stocks_eur.csv`.
When `--analysis-from-primary-ticker` is enabled, the ranker keeps the requested listing symbol as the output/buyable ticker but computes factors and overlays from the linked primary ticker when one is available. The ranked CSV then includes `analysis_symbol` / `analysis_exchange` / `analysis_currency_code` so you can verify which line was actually scored.

## Mixed USD + EUR example

To rank US and EUR-denominated listings in one run and also save a EUR-only companion list:

```powershell
python run_rank.py `
  --region US `
  --exchanges US,XETRA,PA,AS `
  --currency-list-output EUR=ranked_stocks_eur.csv `
  --top 25 `
  --output ranked_stocks_mixed.csv `
  --diagnostics-output rank_diagnostics_mixed.csv
```

That keeps the main ranked file as the full mixed universe and writes a second CSV containing only names with `currency_code = EUR`.

## US issuers on European exchanges only

To rank only European-listed tickers that belong to US issuers, and to exclude native European companies, use the European exchanges as the universe, keep `--region US --strict-issuer-country`, and require a US cross-listing:

```powershell
python run_rank.py `
  --region US `
  --strict-issuer-country `
  --exchanges XETRA,PA,AS `
  --require-crosslisting-exchanges US `
  --analysis-from-primary-ticker `
  --top 25 `
  --output ranked_us_issuers_in_europe.csv `
  --diagnostics-output rank_diagnostics_us_issuers_in_europe.csv
```

That keeps symbols like `APC.XETRA` or `QCI.XETRA` when EODHD shows the issuer is US and the issuer also has a US listing, while filtering out native European issuers such as `ASML.AS` or `SGO.PA`. With `--analysis-from-primary-ticker`, the scoring can come from `AAPL.US` / `QCOM.US` even though the output symbol remains the EUR buyable listing.

## PowerShell example: rebalance

```powershell
python run_rebalance.py `
  --ranked-input ranked_stocks_v3.csv `
  --previous-holdings previous_holdings.csv `
  --top-n-positions 20 `
  --max-position-weight 0.08 `
  --sector-cap 0.25 `
  --buy-rank-buffer 20 `
  --hold-rank-buffer 35 `
  --target-date 2026-03-17 `
  --rebalance-country US `
  --defer-if-macro-event-within-days 1 `
  --output target_portfolio.csv
```

## PowerShell example: ablation comparison

Use `run_ablation.py` to compare ranked outputs and holdings histories from multiple strategy variants against a baseline.

```powershell
python run_ablation.py `
  --ranked-scenario baseline=ranked_stocks_rev06.csv `
  --ranked-scenario lifecycle=ranked_stocks_rev06_lifecycle.csv `
  --ranked-scenario growth=ranked_stocks_rev06_lifecycle_growth.csv `
  --holdings-scenario baseline=rebalance_holdings_history_rev06.csv `
  --holdings-scenario lifecycle=rebalance_holdings_history_rev06_lifecycle.csv `
  --holdings-scenario growth=rebalance_holdings_history_rev06_lifecycle_growth.csv `
  --baseline baseline `
  --top 25 `
  --output-prefix ablation_rev06
```

This writes:

- `ablation_rev06_rank_summary.csv`
- `ablation_rev06_rank_pairs.csv`
- `ablation_rev06_holdings_pairs.csv`
