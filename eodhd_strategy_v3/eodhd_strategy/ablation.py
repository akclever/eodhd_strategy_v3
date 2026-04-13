from __future__ import annotations

from typing import Dict

import pandas as pd


def summarize_rank_scenarios(scenarios: Dict[str, pd.DataFrame], top_n: int) -> pd.DataFrame:
    rows: list[dict] = []
    top_n = max(1, int(top_n))

    for scenario, frame in scenarios.items():
        ranked = frame.copy()
        top = ranked.head(top_n).copy()

        top_sector_share = float("nan")
        if "sector" in top.columns and not top.empty:
            top_sector_share = float(top["sector"].fillna("Unknown").value_counts(normalize=True).iloc[0])

        rows.append(
            {
                "scenario": scenario,
                "final_ranked_count": int(len(ranked)),
                "top_n": top_n,
                "top_n_count": int(len(top)),
                "top_n_median_composite_score": float(pd.to_numeric(top.get("composite_score"), errors="coerce").median())
                if "composite_score" in top.columns
                else float("nan"),
                "top_n_median_shareholder_yield": float(pd.to_numeric(top.get("shareholder_yield"), errors="coerce").median())
                if "shareholder_yield" in top.columns
                else float("nan"),
                "top_n_median_gross_profitability": float(pd.to_numeric(top.get("gross_profitability"), errors="coerce").median())
                if "gross_profitability" in top.columns
                else float("nan"),
                "top_n_median_adjusted_book_to_market": float(
                    pd.to_numeric(top.get("adjusted_book_to_market"), errors="coerce").median()
                )
                if "adjusted_book_to_market" in top.columns
                else float("nan"),
                "top_n_sector_top_share": top_sector_share,
            }
        )

    return pd.DataFrame(rows)


def compare_rank_scenarios(
    scenarios: Dict[str, pd.DataFrame],
    baseline: str,
    top_n: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    top_n = max(1, int(top_n))
    baseline_frame = scenarios.get(baseline, pd.DataFrame()).copy()
    baseline_top = baseline_frame.head(top_n).copy()
    baseline_symbols = set(baseline_top["symbol"]) if "symbol" in baseline_top.columns else set()

    for scenario, frame in scenarios.items():
        if scenario == baseline:
            continue

        top = frame.head(top_n).copy()
        top_symbols = set(top["symbol"]) if "symbol" in top.columns else set()
        overlap = baseline_symbols & top_symbols
        union = baseline_symbols | top_symbols

        row = {
            "baseline": baseline,
            "scenario": scenario,
            "top_n": top_n,
            "top_n_overlap_count": int(len(overlap)),
            "top_n_overlap_ratio": float(len(overlap) / len(union)) if union else float("nan"),
            "rank_spearman_corr": float("nan"),
            "mean_abs_rank_shift": float("nan"),
            "max_abs_rank_shift": float("nan"),
        }

        if {"symbol", "rank"}.issubset(baseline_frame.columns) and {"symbol", "rank"}.issubset(frame.columns):
            merged = (
                baseline_frame[["symbol", "rank"]]
                .rename(columns={"rank": "rank_baseline"})
                .merge(frame[["symbol", "rank"]].rename(columns={"rank": "rank_scenario"}), on="symbol", how="inner")
            )
            if len(merged) >= 2:
                row["rank_spearman_corr"] = float(
                    merged["rank_baseline"].corr(merged["rank_scenario"], method="spearman")
                )
            if not merged.empty:
                deltas = (merged["rank_baseline"] - merged["rank_scenario"]).abs()
                row["mean_abs_rank_shift"] = float(deltas.mean())
                row["max_abs_rank_shift"] = float(deltas.max())

        rows.append(row)

    return pd.DataFrame(rows)


def _normalize_holdings_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "target_date" not in out.columns:
        out["target_date"] = "snapshot"
    out["target_date"] = out["target_date"].fillna("snapshot").astype(str)
    if "symbol" not in out.columns:
        out["symbol"] = ""
    return out


def compare_holdings_scenarios(
    scenarios: Dict[str, pd.DataFrame],
    baseline: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    baseline_frame = _normalize_holdings_frame(scenarios.get(baseline, pd.DataFrame()))
    baseline_dates = sorted(baseline_frame["target_date"].dropna().unique().tolist())

    for scenario, frame in scenarios.items():
        if scenario == baseline:
            continue

        scenario_frame = _normalize_holdings_frame(frame)
        compare_dates = sorted(set(baseline_dates) & set(scenario_frame["target_date"].dropna().unique().tolist()))

        for target_date in compare_dates:
            base_symbols = set(
                baseline_frame.loc[baseline_frame["target_date"] == target_date, "symbol"].dropna().astype(str).tolist()
            )
            scenario_symbols = set(
                scenario_frame.loc[scenario_frame["target_date"] == target_date, "symbol"].dropna().astype(str).tolist()
            )
            overlap = base_symbols & scenario_symbols
            union = base_symbols | scenario_symbols
            added = sorted(scenario_symbols - base_symbols)
            removed = sorted(base_symbols - scenario_symbols)

            rows.append(
                {
                    "target_date": target_date,
                    "baseline": baseline,
                    "scenario": scenario,
                    "baseline_count": int(len(base_symbols)),
                    "scenario_count": int(len(scenario_symbols)),
                    "overlap_count": int(len(overlap)),
                    "overlap_ratio": float(len(overlap) / len(union)) if union else float("nan"),
                    "added_count": int(len(added)),
                    "removed_count": int(len(removed)),
                    "added_symbols": "|".join(added),
                    "removed_symbols": "|".join(removed),
                }
            )

    return pd.DataFrame(rows)
