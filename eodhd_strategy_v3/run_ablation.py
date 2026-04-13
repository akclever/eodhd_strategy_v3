#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Sequence

import pandas as pd

from eodhd_strategy.ablation import compare_holdings_scenarios, compare_rank_scenarios, summarize_rank_scenarios


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare ranked outputs and holdings histories across strategy scenarios."
    )
    parser.add_argument(
        "--ranked-scenario",
        action="append",
        default=[],
        help="Scenario input in the form name=ranked.csv. Repeat for multiple scenarios.",
    )
    parser.add_argument(
        "--holdings-scenario",
        action="append",
        default=[],
        help="Scenario input in the form name=holdings.csv. Repeat for multiple scenarios.",
    )
    parser.add_argument("--baseline", default="", help="Scenario name to treat as the anchor baseline.")
    parser.add_argument("--top", type=int, default=25, help="Top-N cutoff for ranked overlap comparisons.")
    parser.add_argument("--output-prefix", default="ablation", help="Prefix for comparison CSV outputs.")
    return parser.parse_args(argv)


def _parse_scenario_inputs(values: list[str]) -> Dict[str, Path]:
    parsed: Dict[str, Path] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"Invalid scenario '{raw}'. Expected name=path.csv")
        name, path = raw.split("=", 1)
        name = name.strip()
        csv_path = Path(path.strip())
        if not name:
            raise ValueError(f"Invalid scenario '{raw}'. Scenario name cannot be empty.")
        parsed[name] = csv_path
    return parsed


def _load_frames(scenarios: Dict[str, Path]) -> Dict[str, pd.DataFrame]:
    return {name: pd.read_csv(path) for name, path in scenarios.items()}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    ranked_inputs = _parse_scenario_inputs(args.ranked_scenario)
    holdings_inputs = _parse_scenario_inputs(args.holdings_scenario)

    scenario_names = list(ranked_inputs.keys() or holdings_inputs.keys())
    if not scenario_names:
        raise SystemExit("Provide at least one --ranked-scenario or --holdings-scenario input.")

    baseline = args.baseline.strip() or scenario_names[0]
    if baseline not in set(ranked_inputs.keys()) | set(holdings_inputs.keys()):
        raise SystemExit(f"Baseline scenario '{baseline}' was not found in the provided inputs.")

    output_prefix = Path(args.output_prefix)

    if ranked_inputs:
        ranked_frames = _load_frames(ranked_inputs)
        rank_summary = summarize_rank_scenarios(ranked_frames, top_n=args.top)
        rank_pairs = compare_rank_scenarios(ranked_frames, baseline=baseline, top_n=args.top)

        rank_summary_path = output_prefix.with_name(f"{output_prefix.name}_rank_summary.csv")
        rank_pairs_path = output_prefix.with_name(f"{output_prefix.name}_rank_pairs.csv")
        rank_summary.to_csv(rank_summary_path, index=False)
        rank_pairs.to_csv(rank_pairs_path, index=False)

        print(rank_summary.to_string(index=False))
        print(f"\nSaved rank summary to {rank_summary_path}")
        print(f"Saved rank comparison to {rank_pairs_path}")

    if holdings_inputs:
        holdings_frames = _load_frames(holdings_inputs)
        holdings_pairs = compare_holdings_scenarios(holdings_frames, baseline=baseline)
        holdings_pairs_path = output_prefix.with_name(f"{output_prefix.name}_holdings_pairs.csv")
        holdings_pairs.to_csv(holdings_pairs_path, index=False)

        if not holdings_pairs.empty:
            print("\nHoldings comparison:")
            print(holdings_pairs.to_string(index=False))
        print(f"\nSaved holdings comparison to {holdings_pairs_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
