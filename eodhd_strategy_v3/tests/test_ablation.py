from __future__ import annotations

import pandas as pd

from eodhd_strategy.ablation import compare_holdings_scenarios, compare_rank_scenarios, summarize_rank_scenarios


def test_compare_rank_scenarios_reports_overlap_and_rank_shift() -> None:
    scenarios = {
        "baseline": pd.DataFrame(
            [
                {"symbol": "AAA", "rank": 1, "sector": "Tech", "composite_score": 1.0},
                {"symbol": "BBB", "rank": 2, "sector": "Tech", "composite_score": 0.9},
                {"symbol": "CCC", "rank": 3, "sector": "Health", "composite_score": 0.8},
            ]
        ),
        "growth": pd.DataFrame(
            [
                {"symbol": "AAA", "rank": 1, "sector": "Tech", "composite_score": 1.1},
                {"symbol": "CCC", "rank": 2, "sector": "Health", "composite_score": 0.95},
                {"symbol": "DDD", "rank": 3, "sector": "Industrials", "composite_score": 0.7},
            ]
        ),
    }

    summary = summarize_rank_scenarios(scenarios, top_n=2)
    pairs = compare_rank_scenarios(scenarios, baseline="baseline", top_n=2)

    assert set(summary["scenario"]) == {"baseline", "growth"}
    assert int(pairs.iloc[0]["top_n_overlap_count"]) == 1
    assert float(pairs.iloc[0]["mean_abs_rank_shift"]) >= 0.0


def test_compare_holdings_scenarios_uses_target_dates_when_present() -> None:
    scenarios = {
        "baseline": pd.DataFrame(
            [
                {"target_date": "2026-03-01", "symbol": "AAA"},
                {"target_date": "2026-03-01", "symbol": "BBB"},
                {"target_date": "2026-04-01", "symbol": "CCC"},
            ]
        ),
        "growth": pd.DataFrame(
            [
                {"target_date": "2026-03-01", "symbol": "AAA"},
                {"target_date": "2026-03-01", "symbol": "DDD"},
                {"target_date": "2026-04-01", "symbol": "CCC"},
            ]
        ),
    }

    pairs = compare_holdings_scenarios(scenarios, baseline="baseline")

    march = pairs.loc[pairs["target_date"] == "2026-03-01"].iloc[0]
    april = pairs.loc[pairs["target_date"] == "2026-04-01"].iloc[0]

    assert int(march["overlap_count"]) == 1
    assert march["added_symbols"] == "DDD"
    assert march["removed_symbols"] == "BBB"
    assert int(april["overlap_count"]) == 1
