#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import pandas as pd

from eodhd_strategy.client import EODHDClient
from eodhd_strategy.config import RankerConfig
from eodhd_strategy.exchanges import (
    extract_listing_symbols,
    infer_listing_region,
    normalize_exchange_code,
    requested_exchange_aliases,
)
from eodhd_strategy.features import add_overlay_metrics, compute_fundamental_metrics
from eodhd_strategy.macro import apply_macro_sector_tilts, infer_macro_decision
from eodhd_strategy.macro_states import classify_macro_state
from eodhd_strategy.ranker import (
    build_neutralization_comparison,
    build_revision_impulse_weight_comparison,
    build_ranked_frame,
    print_error_summary,
)
from eodhd_strategy.regions import apply_rank_defaults, get_region_preset, region_allows_listing
from eodhd_strategy.universe import collect_universe, normalize_symbol
from eodhd_strategy.valuation import DEFAULT_TOP_N as DEFAULT_VALUATION_TOP_N, build_valuation_report

MOMENTUM_OUTPUT_COLUMNS = [
    "price_momentum_1m",
    "price_momentum_6m",
    "price_momentum_6m_ex_1m",
    "price_momentum_has_coverage",
    "price_momentum_effective_signal",
    "price_momentum_signal_coverage",
    "price_momentum_proxy_used",
    "passes_momentum_gate",
    "contrib_momentum",
    "momentum_signal_confidence",
    "z_price_momentum_effective_signal",
]

SPECIAL_SITUATION_NAME_PATTERNS = {
    "when-issued": "when-issued security",
    "when issued": "when-issued security",
    "special purpose acquisition": "SPAC-like issuer",
    "blank check": "SPAC-like issuer",
    "acquisition corp": "acquisition shell / SPAC-like issuer",
    "acquisition corporation": "acquisition shell / SPAC-like issuer",
    "acquisition company": "acquisition shell / SPAC-like issuer",
    "acquisition holdings": "acquisition shell / SPAC-like issuer",
}

SPECIAL_SITUATION_HOME_CATEGORY_PATTERNS = {
    "special purpose acquisition": "SPAC-like issuer",
    "blank check": "SPAC-like issuer",
    "shell": "shell company",
}

SPECIAL_SITUATION_ASSET_TYPE_PATTERNS = {
    "warrant": "warrant",
    "right": "right",
    "unit": "unit",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Modular EODHD ranker with staged overlays, macro tilts, and regional presets"
    )
    parser.add_argument("--api-token", default=None)

    parser.add_argument("--region", choices=["US", "HK", "AU", "DE", "FR", "NL", "CH"], default="US")
    parser.add_argument("--strict-issuer-country", action="store_true")

    parser.add_argument("--exchanges", default="US")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--symbols-file", default="")
    parser.add_argument("--search-query", default="")
    parser.add_argument("--search-exchange", default="")
    parser.add_argument(
        "--require-crosslisting-exchanges",
        default="",
        help=(
            "Keep only securities whose fundamentals General.Listings include at least one of these "
            "exchange codes, e.g. US or XETRA,PA,AS."
        ),
    )
    parser.add_argument(
        "--analysis-from-primary-ticker",
        action="store_true",
        help=(
            "When available, compute factors and overlays from the linked primary ticker "
            "while keeping the requested listing symbol as the output/buyable ticker."
        ),
    )
    parser.add_argument(
        "--exclude-special-situations",
        action="store_true",
        help="Filter obvious special situations such as when-issued lines, SPAC-like shells, warrants, rights, and units using EODHD fundamentals metadata",
    )

    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache-dir", default=".eodhd_cache")
    parser.add_argument("--refresh", action="store_true")

    parser.add_argument("--dividend-source", choices=["forward", "trailing", "hybrid"], default="hybrid")
    parser.add_argument("--min-market-cap", type=float, default=2_000_000_000)

    parser.add_argument("--regime", choices=["neutral", "risk_on", "risk_off"], default="neutral")
    parser.add_argument("--use-macro", action="store_true")
    parser.add_argument("--macro-country", default="USA")
    parser.add_argument("--macro-as-of-date", default="")
    parser.add_argument("--macro-sector-tilt", type=float, default=0.10)
    parser.add_argument(
        "--macro-state",
        choices=["auto", "expansion", "neutral", "defensive", "inflation_stress"],
        default="auto",
        help="Use auto to classify macro state from macro indicators, or force a specific state",
    )

    parser.add_argument("--use-pead", action="store_true")
    parser.add_argument("--pead-lookback-days", type=int, default=120)
    parser.add_argument("--pead-half-life-days", type=int, default=45)
    parser.add_argument(
        "--min-pead-analysts",
        type=int,
        default=3,
        help="Analyst-count scaling anchor for PEAD breadth",
    )
    parser.add_argument("--use-revision-impulse", action="store_true")
    parser.add_argument(
        "--min-revision-analysts",
        type=int,
        default=4,
        help="Minimum analyst anchor for full-strength revision-impulse coverage",
    )
    parser.add_argument(
        "--revision-impulse-weight",
        type=float,
        default=0.06,
        help="Weight for revision-impulse overlay in composite score",
    )
    parser.add_argument(
        "--use-revision-jerk",
        action="store_true",
        help="Add a short-term revision-jerk overlay that rewards accelerating analyst estimate changes",
    )
    parser.add_argument(
        "--revision-jerk-weight",
        type=float,
        default=0.04,
        help="Weight for revision-jerk overlay in composite score",
    )
    parser.add_argument(
        "--use-estimate-term-structure",
        action="store_true",
        help="Add an estimate-term-structure overlay using recent Earnings.Trend persistence and disagreement compression",
    )
    parser.add_argument(
        "--estimate-term-structure-weight",
        type=float,
        default=0.04,
        help="Weight for estimate-term-structure overlay in composite score",
    )
    parser.add_argument(
        "--use-growth-acceleration",
        action="store_true",
        help="Add top-line growth and revenue-acceleration overlay factors",
    )
    parser.add_argument(
        "--alpha-factor-spec",
        choices=["legacy", "v2"],
        default="legacy",
        help="Keep the existing advanced-factor formulas with legacy, or enable the experimental v2 EODHD-only rework",
    )
    parser.add_argument(
        "--growth-weight",
        type=float,
        default=0.10,
        help="Weight for the revenue growth/acceleration overlay in composite score",
    )
    parser.add_argument(
        "--use-quality-acceleration",
        action="store_true",
        help="Add a quality-acceleration overlay rewarding improving margins, returns, cash conversion, and cleaner working-capital trends",
    )
    parser.add_argument(
        "--quality-acceleration-weight",
        type=float,
        default=0.05,
        help="Weight for the quality-acceleration overlay in composite score",
    )
    parser.add_argument(
        "--use-residual-valuation",
        action="store_true",
        help="Add a peer-relative residual value factor that prefers names cheap relative to quality/growth",
    )
    parser.add_argument(
        "--use-compounder-persistence",
        action="store_true",
        help="Add a core factor that rewards durable, stable, improving business quality across statement history",
    )
    parser.add_argument(
        "--use-intangible-adjustments",
        action="store_true",
        help="Use intangible-adjusted value and profitability inputs when R&D or SG&A capitalization is appropriate",
    )
    parser.add_argument(
        "--use-price-momentum",
        action="store_true",
        help="Add medium-term 6m-ex-1m price momentum overlay",
    )
    parser.add_argument(
        "--require-real-momentum-coverage",
        action="store_true",
        help="Require actual historical price momentum coverage and disable trend-proxy fallback",
    )
    parser.add_argument(
        "--momentum-weight",
        type=float,
        default=0.10,
        help="Weight for medium-term price momentum overlay in composite score",
    )
    parser.add_argument(
        "--use-life-cycle",
        action="store_true",
        help="Enable light life-cycle conditioned tilts for growth, mature, and recovery names",
    )
    parser.add_argument(
        "--life-cycle-tilt-strength",
        type=float,
        default=0.35,
        help="Strength of life-cycle weight tilts; 0 disables tilts and 1 is the strongest supported tilt",
    )

    parser.add_argument("--use-sentiment", action="store_true")
    parser.add_argument("--sentiment-lookback-days", type=int, default=14)
    parser.add_argument("--min-sentiment-accel", type=float, default=-0.02)
    parser.add_argument(
        "--min-sentiment-days",
        type=int,
        default=3,
        help="Minimum sentiment coverage days before sentiment can filter a stock",
    )
    parser.add_argument(
        "--min-sentiment-articles-recent",
        type=int,
        default=3,
        help="Minimum article count over the last 5 sentiment observations before sentiment can filter a stock",
    )
    parser.add_argument(
        "--use-news-events",
        action="store_true",
        help="Experimental: add a structured recent-news event overlay built from the News API",
    )
    parser.add_argument(
        "--news-lookback-days",
        type=int,
        default=10,
        help="Lookback window for the experimental news-event overlay",
    )
    parser.add_argument(
        "--min-news-articles",
        type=int,
        default=3,
        help="Minimum recent news articles before the experimental news-event overlay gets full confidence",
    )
    parser.add_argument(
        "--news-event-weight",
        type=float,
        default=0.06,
        help="Weight for the experimental structured-news event overlay in composite score",
    )
    parser.add_argument(
        "--use-news-shock",
        action="store_true",
        help="Add a short-term news-shock overlay using recent news intensity versus its own baseline and article-volume spikes",
    )
    parser.add_argument(
        "--news-shock-weight",
        type=float,
        default=0.04,
        help="Weight for the short-term news-shock overlay in composite score",
    )
    parser.add_argument(
        "--use-news-peer-spillover",
        action="store_true",
        help="Experimental: propagate recent news signals across industry/sector peer groups",
    )
    parser.add_argument(
        "--news-peer-spillover-weight",
        type=float,
        default=0.25,
        help="Additive weight applied to peer-spillover news signal before z-scoring",
    )
    parser.add_argument(
        "--use-news-novelty-saturation",
        action="store_true",
        help="Experimental: upweight novel news and damp crowded same-direction coverage",
    )
    parser.add_argument(
        "--use-news-confirmation",
        action="store_true",
        help="Experimental: boost news when revision impulse / PEAD confirms the event direction",
    )
    parser.add_argument(
        "--news-confirmation-weight",
        type=float,
        default=0.20,
        help="Additive weight applied to the news confirmation signal before z-scoring",
    )
    parser.add_argument(
        "--use-news-macro-weighting",
        action="store_true",
        help="Experimental: condition news-event strength on the current macro state",
    )

    parser.add_argument("--use-beneish", action="store_true", help="Enable Beneish M-Score hard-filter + penalty")
    parser.add_argument(
        "--use-accrual-volatility",
        action="store_true",
        help="Enable accrual-volatility forensic penalty",
    )
    parser.add_argument(
        "--use-working-capital-stress",
        action="store_true",
        help="Enable an additional forensic penalty for working-capital stress and cash-flow divergence",
    )
    parser.add_argument(
        "--use-investment-restraint",
        action="store_true",
        help="Add an investment-restraint overlay penalizing asset sprawl, acquisition-heavy growth, debt-funded expansion, and dilution",
    )
    parser.add_argument(
        "--investment-restraint-weight",
        type=float,
        default=0.04,
        help="Weight for investment-restraint overlay in composite score",
    )
    parser.add_argument(
        "--use-accrual-quality",
        action="store_true",
        help="Add an accrual-quality overlay rewarding cash conversion, low accrual intensity, and restrained working-capital stretch",
    )
    parser.add_argument(
        "--accrual-quality-weight",
        type=float,
        default=0.05,
        help="Weight for accrual-quality overlay in composite score",
    )
    parser.add_argument(
        "--forensic-weight",
        type=float,
        default=0.10,
        help="Weight deducted from composite score for forensic-quality risk",
    )
    parser.add_argument(
        "--missing-beneish-penalty",
        type=float,
        default=0.25,
        help="Small forensic-risk penalty applied when Beneish is missing but requested",
    )
    parser.add_argument(
        "--use-capital-allocation-quality",
        action="store_true",
        help="Add a capital-allocation overlay rewarding FCF-funded buybacks, debt discipline, and payout quality",
    )
    parser.add_argument(
        "--capital-allocation-weight",
        type=float,
        default=0.04,
        help="Weight for capital-allocation-quality overlay in composite score",
    )
    parser.add_argument(
        "--use-recovery-transition",
        action="store_true",
        help="Add a recovery-transition overlay for names with improving margins, leverage, revisions, and price confirmation",
    )
    parser.add_argument(
        "--recovery-transition-weight",
        type=float,
        default=0.03,
        help="Weight for recovery-transition overlay in composite score",
    )
    parser.add_argument(
        "--use-insider-conviction",
        action="store_true",
        help="Add an insider-conviction overlay from clustered insider buying or broad insider selling",
    )
    parser.add_argument(
        "--insider-conviction-weight",
        type=float,
        default=0.03,
        help="Weight for insider-conviction overlay in composite score",
    )
    parser.add_argument(
        "--use-news-theme-drift",
        action="store_true",
        help="Add a narrative-drift overlay comparing recent news-theme intensity to the trailing baseline",
    )
    parser.add_argument(
        "--news-theme-drift-weight",
        type=float,
        default=0.03,
        help="Weight for news-theme-drift overlay in composite score",
    )
    parser.add_argument(
        "--use-peer-relative-anomalies",
        action="store_true",
        help="Add a peer-relative anomaly overlay using industry/sector/global comparisons for margin trend, reinvestment efficiency, estimate drift, and dilution discipline",
    )
    parser.add_argument(
        "--peer-relative-anomaly-weight",
        type=float,
        default=0.04,
        help="Weight for peer-relative anomaly overlay in composite score",
    )
    parser.add_argument(
        "--exclude-binary-biotech",
        action="store_true",
        help="Exclude low-revenue biotechnology names that behave more like binary event trades than durable businesses",
    )
    parser.add_argument(
        "--binary-biotech-min-revenue",
        type=float,
        default=1_000_000_000,
        help="Minimum trailing revenue required for biotechnology names to avoid the binary-biotech exclusion",
    )

    parser.add_argument("--dividend-payout-cap", type=float, default=0.85)
    parser.add_argument("--max-distance-from-high", type=float, default=0.15)
    parser.add_argument("--require-above-200dma", action="store_true")
    parser.add_argument(
        "--core-weight-floor",
        type=float,
        default=0.60,
        help="Minimum stock-level share reserved for the core factor block after optional sleeves are coverage-adjusted",
    )

    parser.add_argument("--neutralize-by", choices=["none", "sector", "industry"], default="sector")
    parser.add_argument("--compare-neutralization", action="store_true")
    parser.add_argument(
        "--compare-revision-impulse-weights",
        nargs="?",
        const="0.00,0.04,0.06,0.08",
        default="",
        help="Compare comma-separated revision-impulse weights on the same enriched dataset; defaults to 0.00,0.04,0.06,0.08",
    )
    parser.add_argument("--min-group-size", type=int, default=5)
    parser.add_argument(
        "--overlay-top-n",
        type=int,
        default=250,
        help="Deprecated and kept only for CLI compatibility; Stage 2 now enriches all Stage 1 eligible rows.",
    )

    parser.add_argument(
        "--min-piotroski-score",
        type=int,
        default=5,
        help="Minimum Piotroski F-score required to remain eligible",
    )
    parser.add_argument(
        "--pead-max-abs-surprise-pct",
        type=float,
        default=100.0,
        help="Clip earnings surprise percentage at this absolute value before PEAD scaling",
    )
    parser.add_argument(
        "--pead-max-age-days",
        type=int,
        default=45,
        help="Set PEAD signal to zero if earnings are older than this many days",
    )

    parser.add_argument(
        "--use-employee-efficiency",
        action="store_true",
        help="Add a small employee-efficiency overlay using revenue/GP per employee",
    )
    parser.add_argument(
        "--employee-efficiency-weight",
        type=float,
        default=0.05,
        help="Weight for employee-efficiency overlay in composite score",
    )

    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--output", default="ranked_stocks_v3.csv")
    parser.add_argument("--diagnostics-output", default="rank_diagnostics.csv")
    parser.add_argument(
        "--valuation-top-n",
        type=int,
        default=0,
        help=(
            "Generate a separate valuation dashboard for the top N ranked names. "
            f"Use {DEFAULT_VALUATION_TOP_N} for a typical shortlist, or 0 to disable."
        ),
    )
    parser.add_argument(
        "--valuation-output",
        default="",
        help="Optional CSV path for the valuation dashboard. Defaults to <output stem>_valuation.csv when valuation is enabled.",
    )
    parser.add_argument(
        "--currency-list-output",
        action="append",
        default=[],
        metavar="CURRENCY=PATH",
        help="Write a companion ranked CSV filtered to a trading currency, e.g. EUR=ranked_stocks_eur.csv. Repeat for multiple currencies.",
    )

    return parser.parse_args(argv)


def _default_macro_state_from_regime(regime: str) -> str:
    regime = (regime or "neutral").lower()
    if regime == "risk_on":
        return "expansion"
    if regime == "risk_off":
        return "defensive"
    return "neutral"


def _extract_note_float(notes: List[str], label_prefix: str) -> Optional[float]:
    for note in notes or []:
        if not isinstance(note, str):
            continue
        if note.lower().startswith(label_prefix.lower()):
            match = re.search(r"(-?\d+(?:\.\d+)?)", note)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    return None
    return None


def _infer_macro_state_from_decision(macro_decision: Any, fallback_regime: str) -> str:
    notes = list(getattr(macro_decision, "notes", []) or [])

    gdp_growth = _extract_note_float(notes, "GDP growth:")
    inflation = _extract_note_float(notes, "Inflation:")
    unemployment = _extract_note_float(notes, "Unemployment:")
    real_interest_rate = _extract_note_float(notes, "Real interest rate:")
    recent_event_surprise_score = _extract_note_float(notes, "Recent event surprise score:")

    return classify_macro_state(
        gdp_growth=gdp_growth,
        inflation=inflation,
        unemployment=unemployment,
        real_interest_rate=real_interest_rate,
        recent_event_surprise_score=recent_event_surprise_score,
        fallback_regime=fallback_regime,
    )


def _neutralization_compare_output_path(output_path: str) -> Path:
    output = Path(output_path)
    suffix = output.suffix or ".csv"
    return output.with_name(f"{output.stem}_neutralization_compare{suffix}")


def _revision_impulse_compare_output_path(output_path: str) -> Path:
    output = Path(output_path)
    suffix = output.suffix or ".csv"
    return output.with_name(f"{output.stem}_revision_impulse_compare{suffix}")


def _valuation_output_path(output_path: str) -> Path:
    output = Path(output_path)
    suffix = output.suffix or ".csv"
    return output.with_name(f"{output.stem}_valuation{suffix}")


def _parse_currency_output_specs(raw_values: Sequence[str]) -> list[tuple[str, Path]]:
    specs: list[tuple[str, Path]] = []
    for raw_value in raw_values or []:
        text = str(raw_value or "").strip()
        if not text:
            continue
        currency_code, separator, path_text = text.partition("=")
        currency_code = currency_code.strip().upper()
        path_text = path_text.strip()
        if not separator or not currency_code or not path_text:
            raise ValueError(
                f"Invalid currency list output: {raw_value!r}. Use CURRENCY=PATH, for example EUR=ranked_stocks_eur.csv."
            )
        specs.append((currency_code, Path(path_text)))
    return specs


def _write_currency_filtered_outputs(
    ranked: pd.DataFrame,
    output_specs: Sequence[tuple[str, Path]],
) -> list[tuple[str, Path, int]]:
    written: list[tuple[str, Path, int]] = []
    for currency_code, output_path in output_specs:
        if "currency_code" not in ranked.columns:
            filtered = ranked.iloc[0:0].copy()
        else:
            filtered = ranked.loc[
                ranked["currency_code"].fillna("").astype(str).str.upper() == currency_code
            ].copy()
        filtered.to_csv(output_path, index=False)
        written.append((currency_code, output_path, int(len(filtered))))
    return written


def _canonical_exchange_codes(raw_exchanges: Sequence[str] | set[str] | str | None) -> list[str]:
    if raw_exchanges is None:
        return []

    if isinstance(raw_exchanges, str):
        items = [chunk.strip() for chunk in raw_exchanges.split(",") if chunk.strip()]
    else:
        items = [str(chunk).strip() for chunk in raw_exchanges if str(chunk).strip()]

    canonical = {
        normalize_exchange_code(item) or item.upper()
        for item in items
        if item
    }
    return sorted(code for code in canonical if code)


def _matches_required_crosslisting(
    metrics: Dict[str, Any],
    required_crosslisting_exchanges: set[str] | None,
) -> bool:
    if not required_crosslisting_exchanges:
        return True

    listing_aliases = requested_exchange_aliases(metrics.get("listing_exchanges"))
    primary_ticker_text = str(metrics.get("primary_ticker") or "").upper().strip()
    issuer_listing_symbol_text = str(metrics.get("issuer_listing_symbol") or "").upper().strip()

    for linked_symbol in [primary_ticker_text, issuer_listing_symbol_text]:
        if not linked_symbol or "." not in linked_symbol:
            continue
        _, suffix = linked_symbol.rsplit(".", 1)
        listing_aliases.update(requested_exchange_aliases([suffix]))

    return bool(listing_aliases & required_crosslisting_exchanges)


def _issuer_matches_region(metrics: Dict[str, Any], region: str) -> bool:
    preset = get_region_preset(region)
    country_text = str(metrics.get("country") or "").upper().strip()
    country_iso_text = str(metrics.get("country_iso") or "").upper().strip()
    isin_text = str(metrics.get("isin") or "").upper().strip()
    primary_ticker_text = str(metrics.get("primary_ticker") or "").upper().strip()

    if country_text in preset.country_aliases or country_iso_text in preset.country_aliases:
        return True

    if isin_text[:2] in preset.country_aliases:
        return True

    if primary_ticker_text and "." in primary_ticker_text:
        _, primary_suffix = primary_ticker_text.rsplit(".", 1)
        primary_suffix_norm = normalize_exchange_code(primary_suffix)
        if primary_suffix in preset.exchange_aliases or primary_suffix_norm in preset.exchange_aliases:
            return True

    return False


def _resolve_strict_issuer_identity(
    client: EODHDClient,
    symbol: str,
    fundamentals: Dict[str, Any],
    metrics: Dict[str, Any],
    region: str,
) -> Dict[str, Any]:
    out = dict(metrics)
    out.setdefault("strict_issuer_resolution_error_count", 0.0)
    out.setdefault("strict_issuer_resolution_error_message", None)

    if _issuer_matches_region(metrics, region):
        return out

    general = fundamentals.get("General") if isinstance(fundamentals, dict) else {}
    if not isinstance(general, dict):
        return out

    preset = get_region_preset(region)
    issuer_exchange_aliases = {
        str(alias).strip().upper()
        for alias in preset.exchange_aliases
        if str(alias).strip()
    }
    issuer_exchange_aliases.update(
        normalize_exchange_code(alias) for alias in list(issuer_exchange_aliases) if alias
    )

    resolution_error_count = 0
    resolution_error_message: str | None = None
    for code, exchange in extract_listing_symbols(general):
        exchange_norm = normalize_exchange_code(exchange)
        if exchange not in issuer_exchange_aliases and exchange_norm not in issuer_exchange_aliases:
            continue

        candidate_symbol = normalize_symbol(code, exchange_norm or exchange, region)
        if candidate_symbol.upper() == str(symbol or "").upper().strip():
            continue

        try:
            issuer_fundamentals = client.get_fundamentals(candidate_symbol)
        except Exception as exc:
            resolution_error_count += 1
            resolution_error_message = _summarize_exception(exc)
            continue

        issuer_general = issuer_fundamentals.get("General") if isinstance(issuer_fundamentals, dict) else {}
        if not isinstance(issuer_general, dict):
            continue

        issuer_metrics = dict(out)
        issuer_metrics["country"] = issuer_general.get("CountryName") or out.get("country")
        issuer_metrics["country_iso"] = issuer_general.get("CountryISO") or out.get("country_iso")
        issuer_metrics["isin"] = issuer_general.get("ISIN") or out.get("isin")
        issuer_metrics["primary_ticker"] = issuer_general.get("PrimaryTicker") or candidate_symbol
        issuer_metrics["issuer_listing_symbol"] = candidate_symbol

        if _issuer_matches_region(issuer_metrics, region):
            if resolution_error_count > 0:
                issuer_metrics["strict_issuer_resolution_error_count"] = float(resolution_error_count)
                issuer_metrics["strict_issuer_resolution_error_message"] = resolution_error_message
            return issuer_metrics

    if resolution_error_count > 0:
        out["strict_issuer_resolution_error_count"] = float(resolution_error_count)
        out["strict_issuer_resolution_error_message"] = resolution_error_message
    return out


def _resolve_analysis_symbol(symbol: str, metrics: Dict[str, Any], region: str) -> str:
    preset = get_region_preset(region)
    issuer_exchange_aliases = {
        str(alias).strip().upper()
        for alias in preset.exchange_aliases
        if str(alias).strip()
    }
    issuer_exchange_aliases.update(
        normalize_exchange_code(alias) for alias in list(issuer_exchange_aliases) if alias
    )

    seen: set[str] = set()
    candidates = [
        str(metrics.get("issuer_listing_symbol") or "").upper().strip(),
        str(metrics.get("primary_ticker") or "").upper().strip(),
    ]
    original_symbol = str(symbol or "").upper().strip()

    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if candidate == original_symbol or "." not in candidate:
            continue

        _, suffix = candidate.rsplit(".", 1)
        suffix_norm = normalize_exchange_code(suffix)
        if suffix in issuer_exchange_aliases or suffix_norm in issuer_exchange_aliases:
            return candidate

    return original_symbol or str(symbol or "")


def _merge_listing_and_analysis_metrics(
    listing_symbol: str,
    listing_metrics: Dict[str, Any],
    analysis_symbol: str,
    analysis_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    merged = dict(analysis_metrics)
    merged["listing_symbol"] = listing_symbol
    merged["listing_exchange"] = listing_metrics.get("exchange")
    merged["listing_currency_code"] = listing_metrics.get("currency_code")
    merged["listing_currency_name"] = listing_metrics.get("currency_name")
    merged["listing_country"] = listing_metrics.get("country")
    merged["listing_country_iso"] = listing_metrics.get("country_iso")
    merged["analysis_symbol"] = analysis_symbol
    merged["analysis_exchange"] = analysis_metrics.get("exchange")
    merged["analysis_currency_code"] = analysis_metrics.get("currency_code")
    merged["analysis_currency_name"] = analysis_metrics.get("currency_name")
    merged["analysis_country"] = analysis_metrics.get("country")
    merged["analysis_country_iso"] = analysis_metrics.get("country_iso")

    # Keep the displayed/buyable listing identity on the European line while
    # letting the scoring inputs come from the resolved analysis symbol.
    merged["exchange"] = listing_metrics.get("exchange")
    merged["listing_exchanges"] = listing_metrics.get("listing_exchanges")
    merged["currency_code"] = listing_metrics.get("currency_code")
    merged["currency_name"] = listing_metrics.get("currency_name")
    return merged


def _infer_price_history_feed_available(feeds: Sequence[str]) -> bool | None:
    normalized = [str(feed or "").strip().lower() for feed in feeds if str(feed or "").strip()]
    if not normalized:
        return None

    positive_markers = (
        "historical",
        "end-of-day",
        "end of day",
        "eod",
        "intraday",
        "real-time",
        "real time",
        "ohlcv",
        "prices",
        "market data",
    )
    if any(any(marker in feed for marker in positive_markers) for feed in normalized):
        return True

    return False


def _summarize_exception(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}".strip()
    return text[:240]


def _diagnostic_value(diagnostics: pd.DataFrame, metric: str) -> float | None:
    if diagnostics.empty or "metric" not in diagnostics.columns or "value" not in diagnostics.columns:
        return None
    matched = diagnostics.loc[diagnostics["metric"] == metric, "value"]
    if matched.empty:
        return None
    value = pd.to_numeric(matched.iloc[:1], errors="coerce").iloc[0]
    if pd.isna(value):
        return None
    return float(value)


def _special_situation_reason(metrics: dict[str, Any]) -> str | None:
    company_name = str(metrics.get("company_name") or "").strip().lower()
    asset_type = str(metrics.get("asset_type") or "").strip().lower()
    home_category = str(metrics.get("home_category") or "").strip().lower()

    for pattern, label in SPECIAL_SITUATION_NAME_PATTERNS.items():
        if pattern in company_name:
            return label

    for pattern, label in SPECIAL_SITUATION_HOME_CATEGORY_PATTERNS.items():
        if pattern in home_category:
            return label

    for pattern, label in SPECIAL_SITUATION_ASSET_TYPE_PATTERNS.items():
        if pattern in asset_type:
            return label

    return None


def _strip_inactive_ranked_output_columns(frame: pd.DataFrame, config: RankerConfig) -> pd.DataFrame:
    out = frame.copy()
    if not getattr(config, "use_price_momentum", False):
        cols_to_drop = [col for col in MOMENTUM_OUTPUT_COLUMNS if col in out.columns]
        if cols_to_drop:
            out = out.drop(columns=cols_to_drop)
    return out


def _print_empty_result_hints(config: RankerConfig, diagnostics: pd.DataFrame) -> None:
    if diagnostics.empty:
        return

    if getattr(config, "use_price_momentum", False):
        history_share = _diagnostic_value(diagnostics, "share_price_momentum_has_coverage")
        momentum_gate_share = _diagnostic_value(diagnostics, "share_passes_momentum_gate")
        if getattr(config, "require_real_momentum_coverage", False):
            if (momentum_gate_share or 0.0) <= 0.0:
                pct = 0.0 if history_share is None else 100.0 * history_share
                print(
                    f"Hint: the real-momentum-coverage gate removed the universe. Historical price momentum coverage was {pct:.1f}%.",
                    file=sys.stderr,
                )
                print(
                    "Hint: either add a real price-history source, or rerun without --require-real-momentum-coverage.",
                    file=sys.stderr,
                )

    if getattr(config, "exclude_binary_biotech", False):
        biotech_flag_share = _diagnostic_value(diagnostics, "share_binary_biotech_flag")
        biotech_gate_share = _diagnostic_value(diagnostics, "share_passes_biotech_gate")
        if biotech_flag_share is not None and biotech_gate_share is not None and biotech_gate_share < 1.0:
            print(
                f"Hint: the binary-biotech filter excluded about {(1.0 - biotech_gate_share):.1%} of stage-1 candidates.",
                file=sys.stderr,
            )


def _clone_ranker_config(config: RankerConfig, **overrides: Any) -> RankerConfig:
    return replace(config, **overrides)


def _make_client(config: RankerConfig) -> EODHDClient:
    return EODHDClient(
        api_token=config.api_token,
        cache_dir=config.cache_dir,
        refresh=config.refresh,
    )


def _thread_local_client_provider(config: RankerConfig) -> Callable[[], EODHDClient]:
    local = threading.local()

    def _get_client() -> EODHDClient:
        client = getattr(local, "client", None)
        if client is None:
            client = _make_client(config)
            local.client = client
        return client

    return _get_client


def _resolve_client(client_or_provider: EODHDClient | Callable[[], EODHDClient]) -> EODHDClient:
    if callable(client_or_provider):
        return client_or_provider()
    return client_or_provider


def _news_event_overlay_requested(config: RankerConfig) -> bool:
    return any(
        [
            bool(getattr(config, "use_news_events", False)),
            bool(getattr(config, "use_news_peer_spillover", False)),
            bool(getattr(config, "use_news_novelty_saturation", False)),
            bool(getattr(config, "use_news_confirmation", False)),
            bool(getattr(config, "use_news_macro_weighting", False)),
        ]
    )


def _news_overlay_requested(config: RankerConfig) -> bool:
    return any(
        [
            _news_event_overlay_requested(config),
            bool(getattr(config, "use_news_shock", False)),
            bool(getattr(config, "use_news_theme_drift", False)),
        ]
    )


def _stage2_overlay_requested(config: RankerConfig) -> bool:
    return bool(
        getattr(config, "use_pead", False)
        or getattr(config, "use_sentiment", False)
        or getattr(config, "use_insider_conviction", False)
        or _news_overlay_requested(config)
    )


def _normalize_overlay_dependencies(config: RankerConfig) -> list[str]:
    notes: list[str] = []
    if _news_event_overlay_requested(config) and not bool(getattr(config, "use_news_events", False)):
        config.use_news_events = True
        notes.append(
            "Auto-enabled --use-news-events because a dependent news overlay feature was requested."
        )
    return notes


def _stage1_core_config(config: RankerConfig) -> RankerConfig:
    return _clone_ranker_config(
        config,
        use_pead=False,
        use_revision_impulse=False,
        use_revision_jerk=False,
        use_estimate_term_structure=False,
        use_growth_acceleration=False,
        use_quality_acceleration=False,
        use_residual_valuation=False,
        use_compounder_persistence=False,
        use_intangible_adjustments=False,
        use_price_momentum=False,
        use_life_cycle=False,
        use_sentiment=False,
        use_news_events=False,
        use_news_shock=False,
        use_news_peer_spillover=False,
        use_news_novelty_saturation=False,
        use_news_confirmation=False,
        use_news_macro_weighting=False,
        use_capital_allocation_quality=False,
        use_recovery_transition=False,
        use_insider_conviction=False,
        use_news_theme_drift=False,
        use_peer_relative_anomalies=False,
        use_employee_efficiency=False,
        use_investment_restraint=False,
        use_accrual_quality=False,
    )


def _apply_analysis_identity_defaults(
    metrics: Dict[str, Any],
    *,
    listing_symbol: str,
    analysis_symbol: str,
) -> Dict[str, Any]:
    out = dict(metrics)
    out.setdefault("listing_symbol", listing_symbol)
    out.setdefault("analysis_symbol", analysis_symbol)
    out.setdefault("analysis_symbol_source", "listing_symbol")
    out.setdefault(
        "analysis_identity_mismatch",
        float(str(analysis_symbol).upper().strip() != str(listing_symbol or "").upper().strip()),
    )
    out.setdefault("analysis_resolution_error", 0.0)
    out.setdefault("analysis_resolution_error_message", None)
    return out


def _parse_weight_grid(raw_value: str) -> list[float]:
    weights: list[float] = []
    seen: set[float] = set()
    for chunk in str(raw_value or "").split(","):
        token = chunk.strip()
        if not token:
            continue
        try:
            weight = round(max(0.0, float(token)), 6)
        except ValueError as exc:
            raise ValueError(f"Invalid revision-impulse weight: {token}") from exc
        if weight not in seen:
            seen.add(weight)
            weights.append(weight)

    if not weights:
        raise ValueError("Revision-impulse comparison grid is empty")

    weights.sort()
    return weights


def fetch_core_symbol(
    client: EODHDClient | Callable[[], EODHDClient],
    symbol: str,
    config: RankerConfig,
    region: str,
    strict_issuer_country: bool,
    allowed_listing_exchanges: set[str] | None = None,
    required_crosslisting_exchanges: set[str] | None = None,
) -> Dict[str, Any]:
    try:
        client = _resolve_client(client)
        fundamentals = client.get_fundamentals(symbol)
        listing_metrics = compute_fundamental_metrics(client, symbol, fundamentals, config.dividend_source, config)
        if strict_issuer_country:
            listing_metrics = _resolve_strict_issuer_identity(client, symbol, fundamentals, listing_metrics, region)
        listing_special_situation_reason = _special_situation_reason(listing_metrics)

        if not region_allows_listing(
            symbol,
            listing_metrics,
            region,
            strict_issuer_country,
            allowed_exchange_aliases=allowed_listing_exchanges,
        ):
            return {
                "symbol": symbol,
                "error": (
                    f"Filtered outside region {region}: exchange={listing_metrics.get('exchange')} "
                    f"country={listing_metrics.get('country')}"
                ),
                "error_stage": "filter",
            }

        if not _matches_required_crosslisting(listing_metrics, required_crosslisting_exchanges):
            required_codes = ",".join(_canonical_exchange_codes(required_crosslisting_exchanges))
            listing_codes = ",".join(_canonical_exchange_codes(listing_metrics.get("listing_exchanges")))
            return {
                "symbol": symbol,
                "error": (
                    f"Missing required cross-listing exchange(s): required={required_codes or 'none'} "
                    f"listings={listing_codes or 'none'}"
                ),
                "error_stage": "filter",
            }

        analysis_symbol = str(symbol or "").upper().strip()
        metrics = _apply_analysis_identity_defaults(
            listing_metrics,
            listing_symbol=symbol,
            analysis_symbol=analysis_symbol,
        )
        if getattr(config, "analysis_from_primary_ticker", False):
            preferred_analysis_symbol = _resolve_analysis_symbol(symbol, listing_metrics, region)
            if preferred_analysis_symbol and preferred_analysis_symbol != analysis_symbol:
                try:
                    analysis_fundamentals = client.get_fundamentals(preferred_analysis_symbol)
                    analysis_metrics = compute_fundamental_metrics(
                        client,
                        preferred_analysis_symbol,
                        analysis_fundamentals,
                        config.dividend_source,
                        config,
                    )
                    analysis_symbol = preferred_analysis_symbol
                    metrics = _merge_listing_and_analysis_metrics(
                        symbol,
                        listing_metrics,
                        analysis_symbol,
                        analysis_metrics,
                    )
                    metrics = _apply_analysis_identity_defaults(
                        metrics,
                        listing_symbol=symbol,
                        analysis_symbol=analysis_symbol,
                    )
                    metrics["analysis_symbol_source"] = "primary_ticker"
                    metrics["analysis_identity_mismatch"] = float(
                        str(analysis_symbol).upper().strip() != str(symbol or "").upper().strip()
                    )
                    metrics["analysis_resolution_error"] = 0.0
                    metrics["analysis_resolution_error_message"] = None
                except Exception as exc:
                    analysis_symbol = str(symbol or "").upper().strip()
                    metrics = _apply_analysis_identity_defaults(
                        listing_metrics,
                        listing_symbol=symbol,
                        analysis_symbol=analysis_symbol,
                    )
                    metrics["analysis_symbol_source"] = "listing_symbol"
                    metrics["analysis_identity_mismatch"] = 0.0
                    metrics["analysis_resolution_error"] = 1.0
                    metrics["analysis_resolution_error_message"] = _summarize_exception(exc)

        metrics.setdefault("listing_symbol", symbol)
        metrics.setdefault("analysis_symbol", analysis_symbol)
        metrics.setdefault("listing_exchange", listing_metrics.get("exchange"))
        metrics.setdefault("listing_currency_code", listing_metrics.get("currency_code"))
        metrics.setdefault("listing_currency_name", listing_metrics.get("currency_name"))
        metrics.setdefault("listing_country", listing_metrics.get("country"))
        metrics.setdefault("listing_country_iso", listing_metrics.get("country_iso"))
        metrics.setdefault("analysis_exchange", metrics.get("exchange"))
        metrics.setdefault("analysis_currency_code", metrics.get("currency_code"))
        metrics.setdefault("analysis_currency_name", metrics.get("currency_name"))
        metrics.setdefault("analysis_country", metrics.get("country"))
        metrics.setdefault("analysis_country_iso", metrics.get("country_iso"))

        if metrics.get("sector") in {"Financial Services", "Financials", "Real Estate"}:
            return {
                "symbol": symbol,
                "error": f"Filtered sector: {metrics.get('sector')}",
                "error_stage": "filter",
            }

        if not metrics.get("sector") or not metrics.get("industry"):
            return {
                "symbol": symbol,
                "error": "Missing sector or industry",
                "error_stage": "filter",
            }

        if getattr(config, "exclude_special_situations", False):
            special_situation_reason = listing_special_situation_reason or _special_situation_reason(metrics)
            if special_situation_reason:
                return {
                    "symbol": symbol,
                    "error": f"Filtered special situation: {special_situation_reason}",
                    "error_stage": "filter",
                }

        row_region = infer_listing_region(symbol=symbol, exchange=metrics.get("exchange")) or region
        row = {"symbol": symbol, "region": row_region}
        row.update(metrics)
        return row

    except KeyboardInterrupt:
        raise
    except Exception as exc:
        return {"symbol": symbol, "region": region, "error": str(exc), "error_stage": "fundamentals"}


def enrich_overlay_symbol(
    client: EODHDClient | Callable[[], EODHDClient],
    row: Dict[str, Any],
    config: RankerConfig,
) -> Dict[str, Any]:
    try:
        client = _resolve_client(client)
        return add_overlay_metrics(client, row, config)
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        out = dict(row)
        out["overlay_error"] = str(exc)
        return out


def _apply_macro_if_needed(
    ranked: pd.DataFrame,
    diagnostics: pd.DataFrame,
    config: RankerConfig,
    macro_decision: Any,
    region: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_ranked = ranked.copy()
    out_diagnostics = diagnostics.copy()

    if macro_decision is not None and not out_ranked.empty:
        out_ranked = apply_macro_sector_tilts(out_ranked, macro_decision)
        out_diagnostics = pd.concat(
            [
                out_diagnostics,
                pd.DataFrame(
                    [
                        {"metric": "macro_regime", "value": macro_decision.regime},
                        {"metric": "macro_state", "value": config.macro_state},
                        {"metric": "macro_score", "value": macro_decision.macro_score},
                        {"metric": "region", "value": region},
                    ]
                ),
            ],
            ignore_index=True,
        )

    return out_ranked, out_diagnostics


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args = apply_rank_defaults(args)

    revision_impulse_compare_weights: list[float] = []
    if args.compare_revision_impulse_weights:
        try:
            revision_impulse_compare_weights = _parse_weight_grid(args.compare_revision_impulse_weights)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
        args.use_revision_impulse = True

    try:
        currency_output_specs = _parse_currency_output_specs(args.currency_list_output)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    api_token = args.api_token or os.getenv("EODHD_API_TOKEN")
    if not api_token:
        print("Error: missing API token. Use --api-token or set EODHD_API_TOKEN.", file=sys.stderr)
        return 2

    initial_macro_state = (
        args.macro_state if args.macro_state != "auto" else _default_macro_state_from_regime(args.regime)
    )

    config = RankerConfig(
        api_token=api_token,
        cache_dir=Path(args.cache_dir),
        refresh=bool(args.refresh),
        workers=max(1, int(args.workers)),
        min_market_cap=float(args.min_market_cap),
        dividend_source=args.dividend_source,
        regime=args.regime,
        use_pead=bool(args.use_pead),
        pead_lookback_days=int(args.pead_lookback_days),
        pead_half_life_days=int(args.pead_half_life_days),
        min_pead_analysts=max(1, int(args.min_pead_analysts)),
        use_revision_impulse=bool(args.use_revision_impulse),
        min_revision_analysts=max(1, int(args.min_revision_analysts)),
        revision_impulse_weight=float(args.revision_impulse_weight),
        use_revision_jerk=bool(args.use_revision_jerk),
        revision_jerk_weight=float(args.revision_jerk_weight),
        use_estimate_term_structure=bool(args.use_estimate_term_structure),
        estimate_term_structure_weight=float(args.estimate_term_structure_weight),
        use_growth_acceleration=bool(args.use_growth_acceleration),
        growth_weight=float(args.growth_weight),
        use_quality_acceleration=bool(args.use_quality_acceleration),
        quality_acceleration_weight=float(args.quality_acceleration_weight),
        alpha_factor_spec=str(args.alpha_factor_spec),
        use_residual_valuation=bool(args.use_residual_valuation),
        use_compounder_persistence=bool(args.use_compounder_persistence),
        use_intangible_adjustments=bool(args.use_intangible_adjustments),
        use_price_momentum=bool(args.use_price_momentum),
        require_real_momentum_coverage=bool(args.require_real_momentum_coverage),
        momentum_weight=float(args.momentum_weight),
        use_life_cycle=bool(args.use_life_cycle),
        life_cycle_tilt_strength=float(args.life_cycle_tilt_strength),
        use_sentiment=bool(args.use_sentiment),
        sentiment_lookback_days=int(args.sentiment_lookback_days),
        min_sentiment_accel=float(args.min_sentiment_accel),
        min_sentiment_articles_recent=max(1, int(args.min_sentiment_articles_recent)),
        use_news_events=bool(args.use_news_events),
        news_lookback_days=int(args.news_lookback_days),
        min_news_articles=max(1, int(args.min_news_articles)),
        news_event_weight=float(args.news_event_weight),
        use_news_shock=bool(args.use_news_shock),
        news_shock_weight=float(args.news_shock_weight),
        use_news_peer_spillover=bool(args.use_news_peer_spillover),
        news_peer_spillover_weight=float(args.news_peer_spillover_weight),
        use_news_novelty_saturation=bool(args.use_news_novelty_saturation),
        use_news_confirmation=bool(args.use_news_confirmation),
        news_confirmation_weight=float(args.news_confirmation_weight),
        use_news_macro_weighting=bool(args.use_news_macro_weighting),
        use_beneish=bool(args.use_beneish),
        use_accrual_volatility=bool(args.use_accrual_volatility),
        use_working_capital_stress=bool(args.use_working_capital_stress),
        forensic_weight=float(args.forensic_weight),
        missing_beneish_penalty=float(args.missing_beneish_penalty),
        use_investment_restraint=bool(args.use_investment_restraint),
        investment_restraint_weight=float(args.investment_restraint_weight),
        use_accrual_quality=bool(args.use_accrual_quality),
        accrual_quality_weight=float(args.accrual_quality_weight),
        use_capital_allocation_quality=bool(args.use_capital_allocation_quality),
        capital_allocation_weight=float(args.capital_allocation_weight),
        use_recovery_transition=bool(args.use_recovery_transition),
        recovery_transition_weight=float(args.recovery_transition_weight),
        use_insider_conviction=bool(args.use_insider_conviction),
        insider_conviction_weight=float(args.insider_conviction_weight),
        use_news_theme_drift=bool(args.use_news_theme_drift),
        news_theme_drift_weight=float(args.news_theme_drift_weight),
        use_peer_relative_anomalies=bool(args.use_peer_relative_anomalies),
        peer_relative_anomaly_weight=float(args.peer_relative_anomaly_weight),
        exclude_binary_biotech=bool(args.exclude_binary_biotech),
        binary_biotech_min_revenue=float(args.binary_biotech_min_revenue),
        min_sentiment_days=int(args.min_sentiment_days),
        min_piotroski_score=int(args.min_piotroski_score),
        pead_max_abs_surprise_pct=float(args.pead_max_abs_surprise_pct),
        pead_max_age_days=int(args.pead_max_age_days),
        dividend_payout_cap=float(args.dividend_payout_cap),
        max_distance_from_high=float(args.max_distance_from_high),
        require_above_200dma=bool(args.require_above_200dma),
        neutralize_by=args.neutralize_by,
        min_group_size=int(args.min_group_size),
        overlay_top_n=int(args.overlay_top_n),
        output=Path(args.output),
        macro_state=initial_macro_state,
        universe_size=0,
        use_employee_efficiency=bool(args.use_employee_efficiency),
        employee_efficiency_weight=float(args.employee_efficiency_weight),
        core_weight_floor=float(args.core_weight_floor),
        analysis_from_primary_ticker=bool(args.analysis_from_primary_ticker),
        exclude_special_situations=bool(args.exclude_special_situations),
        price_momentum_source_mode="history_only" if bool(args.require_real_momentum_coverage) else "auto",
    )

    dependency_notes = _normalize_overlay_dependencies(config)

    client = _make_client(config)
    worker_client = _thread_local_client_provider(config)

    macro_decision = None
    if args.use_macro:
        macro_decision = infer_macro_decision(
            client,
            country=args.macro_country,
            as_of_date=args.macro_as_of_date or None,
            tilt_strength=float(args.macro_sector_tilt),
        )
        config.regime = macro_decision.regime

        if args.macro_state == "auto":
            config.macro_state = _infer_macro_state_from_decision(macro_decision, fallback_regime=config.regime)
        else:
            config.macro_state = args.macro_state

        print(
            f"Macro regime: {macro_decision.regime} | score={macro_decision.macro_score:.2f} | country={macro_decision.country}",
            file=sys.stderr,
        )
        for note in macro_decision.notes:
            print(f"  - {note}", file=sys.stderr)
        print(f"Macro state: {config.macro_state}", file=sys.stderr)
    else:
        config.macro_state = initial_macro_state

    feed_probe_error: str | None = None
    try:
        user_info = client.get_user()
        feeds: list[str] = []
        if isinstance(user_info, dict):
            feeds.extend(user_info.get("availableDataFeeds", []) or [])
            feeds.extend(user_info.get("availableMarketplaceDataFeeds", []) or [])
        if feeds:
            print("Available data feeds:", file=sys.stderr)
            for feed in feeds:
                print(f"  - {feed}", file=sys.stderr)
        price_history_feed_available = _infer_price_history_feed_available(feeds)
        if config.use_price_momentum and price_history_feed_available is False:
            if config.require_real_momentum_coverage:
                config.price_momentum_source_mode = "history_only"
                print(
                    "Price momentum warning: no historical price feed detected in the account entitlements; proxy fallback is disabled, so requiring real momentum coverage may leave no eligible names.",
                    file=sys.stderr,
                )
            else:
                config.price_momentum_source_mode = "trend_proxy"
                print(
                    "Price momentum warning: no historical price feed detected in the account entitlements; using trend proxy fallback.",
                    file=sys.stderr,
                )
    except Exception as exc:
        feed_probe_error = _summarize_exception(exc)
        print(f"Warning: failed to inspect account feed entitlements ({feed_probe_error}).", file=sys.stderr)

    for note in dependency_notes:
        print(note, file=sys.stderr)

    universe = collect_universe(client, args)
    if not universe:
        print("No symbols found for the requested universe.", file=sys.stderr)
        return 1

    allowed_listing_exchanges = requested_exchange_aliases(args.exchanges)
    if not allowed_listing_exchanges:
        allowed_listing_exchanges = None
    required_crosslisting_exchanges = requested_exchange_aliases(args.require_crosslisting_exchanges)
    if not required_crosslisting_exchanges:
        required_crosslisting_exchanges = None

    config.universe_size = len(universe)

    print(f"Collected {len(universe)} symbols. Stage 1: core fundamentals...", file=sys.stderr)

    rows: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        futures = {
            executor.submit(
                fetch_core_symbol,
                worker_client,
                symbol,
                config,
                args.region,
                bool(args.strict_issuer_country),
                allowed_listing_exchanges,
                required_crosslisting_exchanges,
            ): symbol
            for symbol in universe
        }
        for idx, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if idx % 25 == 0 or idx == len(futures):
                print(f"Stage 1 processed {idx}/{len(futures)}", file=sys.stderr)

    raw_df = pd.DataFrame(rows)
    errors = raw_df[raw_df["error"].notna()].copy() if "error" in raw_df.columns else pd.DataFrame()
    usable_df = raw_df[raw_df["error"].isna()].copy() if "error" in raw_df.columns else raw_df.copy()

    if not errors.empty:
        print_error_summary(errors)

    if usable_df.empty:
        print("All symbols failed during data collection.", file=sys.stderr)
        return 1

    base_cfg = _stage1_core_config(config)
    stage1_rows, provisional, stage1_diagnostics = build_ranked_frame(usable_df, base_cfg)

    if provisional.empty:
        print("No eligible securities after core filters.", file=sys.stderr)
        _print_empty_result_hints(base_cfg, stage1_diagnostics)
        _strip_inactive_ranked_output_columns(stage1_rows, config).to_csv(config.output, index=False)
        stage1_diagnostics.to_csv(args.diagnostics_output, index=False)
        print(f"Saved stage-1 output to {config.output}", file=sys.stderr)
        print(f"Saved diagnostics to {args.diagnostics_output}", file=sys.stderr)
        return 1

    overlay_inputs = usable_df.to_dict(orient="records")
    overlay_requested = _stage2_overlay_requested(config)
    if overlay_requested:
        print(f"Stage 2: overlays for {len(overlay_inputs)} eligible names...", file=sys.stderr)
        enriched_rows: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=config.workers) as executor:
            futures = {
                executor.submit(enrich_overlay_symbol, worker_client, row, config): row["symbol"]
                for row in overlay_inputs
            }
            for idx, future in enumerate(as_completed(futures), start=1):
                enriched_rows.append(future.result())
                if idx % 25 == 0 or idx == len(futures):
                    print(f"Stage 2 processed {idx}/{len(futures)}", file=sys.stderr)
        enriched_df = pd.DataFrame(enriched_rows)
    else:
        print("Stage 2: no external overlays requested; reusing Stage 1 rows.", file=sys.stderr)
        enriched_df = usable_df.copy()
    if "overlay_error" in enriched_df.columns:
        enriched_df["overlay_error_flag"] = enriched_df["overlay_error"].notna().astype(float)

    all_rows, ranked, diagnostics = build_ranked_frame(enriched_df, config)
    diagnostics = pd.concat(
        [
            diagnostics,
            pd.DataFrame(
                [
                    {
                        "metric": "overlay_enrichment_scope",
                        "value": "all_stage1_rows" if overlay_requested else "none_requested",
                    },
                    {"metric": "overlay_enrichment_count", "value": float(len(enriched_df))},
                    {"metric": "overlay_top_n_deprecated", "value": float(config.overlay_top_n)},
                    {"metric": "feed_probe_failed", "value": float(feed_probe_error is not None)},
                    {"metric": "feed_probe_error", "value": feed_probe_error or ""},
                ]
            ),
        ],
        ignore_index=True,
    )
    ranked, diagnostics = _apply_macro_if_needed(ranked, diagnostics, config, macro_decision, args.region)

    if ranked.empty:
        print("No eligible securities after overlay filters.", file=sys.stderr)
        _print_empty_result_hints(config, diagnostics)
        _strip_inactive_ranked_output_columns(all_rows, config).to_csv(config.output, index=False)
        diagnostics.to_csv(args.diagnostics_output, index=False)
        print(f"Saved stage-2 output to {config.output}", file=sys.stderr)
        print(f"Saved diagnostics to {args.diagnostics_output}", file=sys.stderr)
        return 1

    ranked = _strip_inactive_ranked_output_columns(ranked, config)
    if "region" not in ranked.columns:
        ranked["region"] = args.region
    else:
        ranked["region"] = ranked["region"].fillna(args.region)
    ranked.to_csv(config.output, index=False)
    diagnostics.to_csv(args.diagnostics_output, index=False)

    valuation_output: Path | None = None
    valuation_rows = pd.DataFrame()
    if int(args.valuation_top_n) > 0:
        valuation_output = Path(args.valuation_output) if str(args.valuation_output or "").strip() else _valuation_output_path(
            args.output
        )
        valuation_rows = build_valuation_report(
            client,
            ranked,
            top_n=int(args.valuation_top_n),
        )
        valuation_rows.to_csv(valuation_output, index=False)

    currency_outputs = _write_currency_filtered_outputs(ranked, currency_output_specs)

    if config.use_price_momentum:
        price_history_share = _diagnostic_value(diagnostics, "share_price_momentum_has_coverage")
        proxy_share = _diagnostic_value(diagnostics, "share_price_momentum_proxy_used")
        signal_share = _diagnostic_value(diagnostics, "share_price_momentum_signal_coverage")
        if price_history_share is not None or proxy_share is not None:
            print("Price momentum coverage:", file=sys.stderr)
            if price_history_share is not None:
                print(f"  - historical coverage: {price_history_share:.1%}", file=sys.stderr)
            if proxy_share is not None:
                print(f"  - proxy fallback used: {proxy_share:.1%}", file=sys.stderr)
            if signal_share is not None:
                print(f"  - effective signal coverage: {signal_share:.1%}", file=sys.stderr)
        if (price_history_share or 0.0) <= 0.0 and (proxy_share or 0.0) <= 0.0:
            print(
                "Price momentum warning: no historical coverage and no proxy coverage were available, so the momentum overlay is effectively inactive.",
                file=sys.stderr,
            )

    if args.compare_neutralization:
        sector_cfg = _clone_ranker_config(config, neutralize_by="sector")
        none_cfg = _clone_ranker_config(config, neutralize_by="none")

        _, sector_ranked, sector_diag = build_ranked_frame(enriched_df, sector_cfg)
        _, none_ranked, none_diag = build_ranked_frame(enriched_df, none_cfg)
        sector_ranked, sector_diag = _apply_macro_if_needed(
            sector_ranked, sector_diag, sector_cfg, macro_decision, args.region
        )
        none_ranked, none_diag = _apply_macro_if_needed(none_ranked, none_diag, none_cfg, macro_decision, args.region)

        comparison = build_neutralization_comparison(sector_ranked, none_ranked, args.top)
        comparison_output = _neutralization_compare_output_path(args.output)
        comparison.to_csv(comparison_output, index=False)
        print(f"Saved neutralization comparison to {comparison_output}", file=sys.stderr)

    if revision_impulse_compare_weights:
        weight_ranked: dict[float, pd.DataFrame] = {}
        for weight in revision_impulse_compare_weights:
            weight_cfg = _clone_ranker_config(
                config,
                use_revision_impulse=True,
                revision_impulse_weight=float(weight),
            )
            _, ranked_for_weight, diag_for_weight = build_ranked_frame(enriched_df, weight_cfg)
            ranked_for_weight, diag_for_weight = _apply_macro_if_needed(
                ranked_for_weight,
                diag_for_weight,
                weight_cfg,
                macro_decision,
                args.region,
            )
            weight_ranked[float(weight)] = ranked_for_weight

        revision_comparison = build_revision_impulse_weight_comparison(weight_ranked, args.top)
        revision_comparison_output = _revision_impulse_compare_output_path(args.output)
        revision_comparison.to_csv(revision_comparison_output, index=False)
        print(f"Saved revision impulse comparison to {revision_comparison_output}", file=sys.stderr)

    display_cols = [
        "rank",
        "symbol",
        "analysis_symbol",
        "company_name",
        "currency_code",
        "analysis_currency_code",
        "sector",
        "industry",
        "exchange",
        "analysis_exchange",
        "region",
        "neutralize_by",
        "regime",
        "macro_state",
        "life_cycle_stage",
        "life_cycle_confidence",
        "composite_score",
        "shareholder_yield",
        "gross_profitability",
        "adjusted_book_to_market",
        "residual_value_signal",
        "residual_value_peer_level",
        "compounder_persistence_signal",
        "peer_relative_anomaly_signal",
        "peer_relative_anomaly_peer_level",
        "intangible_adjustment_applied",
        "pead_signal",
        "sue_signal",
        "pead_revision_component",
        "revision_impulse_signal",
        "estimate_term_structure_signal",
        "revenue_growth_yoy",
        "revenue_acceleration",
        "price_momentum_6m_ex_1m",
        "price_momentum_has_coverage",
        "price_momentum_effective_signal",
        "price_momentum_proxy_used",
        "passes_momentum_gate",
        "news_event_signal",
        "news_event_effective_signal",
        "news_event_breadth",
        "news_article_count_recent",
        "news_novelty_score",
        "news_saturation_score",
        "news_peer_spillover_signal",
        "news_confirmation_signal",
        "news_macro_multiplier",
        "news_theme_drift_signal",
        "news_signal_confidence",
        "capital_allocation_quality_signal",
        "recovery_transition_signal",
        "insider_conviction_signal",
        "binary_biotech_flag",
        "revision_impulse_drift_7d",
        "revision_impulse_drift_30d",
        "revision_impulse_breadth",
        "revision_impulse_disagreement_penalty",
        "beneish_m_score",
        "beneish_data_status",
        "accrual_volatility",
        "working_capital_stress_penalty",
        "investment_restraint_signal",
        "investment_restraint_measure_count",
        "accrual_quality_signal",
        "accrual_quality_measure_count",
        "beneish_missing_penalty_applied",
        "beneish_hard_filter_threshold",
        "forensic_penalty",
        "sentiment_acceleration",
        "sentiment_article_count_recent",
        "contrib_news_event",
        "contrib_residual_value",
        "contrib_compounder_persistence",
        "contrib_estimate_term_structure",
        "contrib_peer_relative_anomaly",
        "contrib_capital_allocation",
        "contrib_recovery_transition",
        "contrib_investment_restraint",
        "contrib_accrual_quality",
        "contrib_insider_conviction",
        "contrib_news_theme_drift",
        "market_cap",
        "macro_regime",
        "macro_score",
        "macro_sector_bonus",
    ]
    display_cols = [col for col in display_cols if col in ranked.columns]

    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", None)
    print(ranked[display_cols].head(args.top).to_string(index=False))

    print(f"\nSaved ranked output to {config.output}", file=sys.stderr)
    print(f"Saved diagnostics to {args.diagnostics_output}", file=sys.stderr)
    if valuation_output is not None:
        print(
            f"Saved valuation dashboard ({len(valuation_rows)} rows) to {valuation_output}",
            file=sys.stderr,
        )
    for currency_code, output_path, row_count in currency_outputs:
        print(f"Saved {currency_code} list ({row_count} rows) to {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
