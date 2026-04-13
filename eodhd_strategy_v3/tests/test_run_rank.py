from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from run_rank import (
    _issuer_matches_region,
    _merge_listing_and_analysis_metrics,
    _matches_required_crosslisting,
    _parse_currency_output_specs,
    _resolve_analysis_symbol,
    _special_situation_reason,
    _strip_inactive_ranked_output_columns,
    _write_currency_filtered_outputs,
)
from eodhd_strategy.config import RankerConfig


def test_parse_currency_output_specs_accepts_repeatable_entries() -> None:
    specs = _parse_currency_output_specs(["eur=ranked_eur.csv", "USD=ranked_usd.csv"])

    assert specs == [
        ("EUR", Path("ranked_eur.csv")),
        ("USD", Path("ranked_usd.csv")),
    ]


def test_parse_currency_output_specs_rejects_invalid_entry() -> None:
    with pytest.raises(ValueError):
        _parse_currency_output_specs(["EUR"])


def test_write_currency_filtered_outputs_writes_only_matching_rows(tmp_path: Path) -> None:
    ranked = pd.DataFrame(
        [
            {"symbol": "SAP.XETRA", "currency_code": "EUR", "rank": 1},
            {"symbol": "MSFT.US", "currency_code": "USD", "rank": 2},
            {"symbol": "AD.AS", "currency_code": "EUR", "rank": 3},
        ]
    )

    output_path = tmp_path / "ranked_eur.csv"
    written = _write_currency_filtered_outputs(ranked, [("EUR", output_path)])
    eur_frame = pd.read_csv(output_path)

    assert written == [("EUR", output_path, 2)]
    assert eur_frame["symbol"].tolist() == ["SAP.XETRA", "AD.AS"]


def test_matches_required_crosslisting_uses_listing_exchanges_column() -> None:
    metrics = {"listing_exchanges": "US,XETRA"}

    assert _matches_required_crosslisting(metrics, {"US"})
    assert _matches_required_crosslisting(metrics, {"PA", "XETRA"})
    assert not _matches_required_crosslisting(metrics, {"AS"})


def test_matches_required_crosslisting_accepts_primary_ticker_suffix() -> None:
    metrics = {"listing_exchanges": "XETRA", "primary_ticker": "AAPL.US"}

    assert _matches_required_crosslisting(metrics, {"US"})


def test_issuer_matches_region_accepts_primary_ticker_or_isin() -> None:
    assert _issuer_matches_region({"primary_ticker": "AAPL.US"}, "US")
    assert _issuer_matches_region({"isin": "US0378331005"}, "US")
    assert not _issuer_matches_region({"country": "Germany", "primary_ticker": "SAP.XETRA"}, "US")


def test_resolve_analysis_symbol_prefers_primary_us_ticker_for_us_region() -> None:
    metrics = {"primary_ticker": "AAPL.US"}

    assert _resolve_analysis_symbol("APC.XETRA", metrics, "US") == "AAPL.US"


def test_merge_listing_and_analysis_metrics_keeps_listing_identity() -> None:
    listing_metrics = {
        "exchange": "XETRA",
        "currency_code": "EUR",
        "currency_name": "Euro",
        "listing_exchanges": "US,XETRA",
        "country": "Germany",
        "country_iso": "DE",
    }
    analysis_metrics = {
        "exchange": "US",
        "currency_code": "USD",
        "currency_name": "US Dollar",
        "country": "United States",
        "country_iso": "US",
        "market_cap": 123.0,
        "revision_impulse_signal": 0.42,
    }

    merged = _merge_listing_and_analysis_metrics(
        "APC.XETRA",
        listing_metrics,
        "AAPL.US",
        analysis_metrics,
    )

    assert merged["listing_symbol"] == "APC.XETRA"
    assert merged["analysis_symbol"] == "AAPL.US"
    assert merged["exchange"] == "XETRA"
    assert merged["currency_code"] == "EUR"
    assert merged["analysis_exchange"] == "US"
    assert merged["analysis_currency_code"] == "USD"
    assert merged["market_cap"] == 123.0
    assert merged["revision_impulse_signal"] == 0.42


def test_special_situation_reason_flags_when_issued_and_spac_like_names() -> None:
    assert (
        _special_situation_reason({"company_name": "Versant Media Group, Inc. Class A Common Stock When-Issued"})
        == "when-issued security"
    )
    assert (
        _special_situation_reason({"company_name": "Example Special Purpose Acquisition Corp"})
        == "SPAC-like issuer"
    )
    assert _special_situation_reason({"asset_type": "Warrant"}) == "warrant"
    assert _special_situation_reason({"company_name": "Incyte Corporation", "asset_type": "Common Stock"}) is None


def test_strip_inactive_ranked_output_columns_hides_momentum_fields_when_disabled(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            {
                "symbol": "ABC.US",
                "rank": 1,
                "price_momentum_6m_ex_1m": 0.12,
                "price_momentum_has_coverage": 1.0,
                "passes_momentum_gate": True,
                "contrib_momentum": 0.03,
                "composite_score": 0.55,
            }
        ]
    )
    config = RankerConfig(
        api_token="test",
        cache_dir=tmp_path,
        refresh=False,
        workers=1,
        min_market_cap=100.0,
        dividend_source="hybrid",
        regime="neutral",
        use_pead=False,
        pead_lookback_days=120,
        pead_half_life_days=45,
        min_pead_analysts=3,
        use_revision_impulse=False,
        min_revision_analysts=4,
        revision_impulse_weight=0.06,
        use_estimate_term_structure=False,
        estimate_term_structure_weight=0.04,
        use_growth_acceleration=False,
        growth_weight=0.10,
        alpha_factor_spec="legacy",
        use_residual_valuation=False,
        use_compounder_persistence=False,
        use_intangible_adjustments=False,
        use_price_momentum=False,
        require_real_momentum_coverage=False,
        momentum_weight=0.10,
        use_life_cycle=False,
        life_cycle_tilt_strength=0.35,
        use_sentiment=False,
        sentiment_lookback_days=14,
        min_sentiment_accel=-0.02,
        min_sentiment_articles_recent=3,
        use_news_events=False,
        news_lookback_days=10,
        min_news_articles=3,
        news_event_weight=0.06,
        use_news_peer_spillover=False,
        news_peer_spillover_weight=0.25,
        use_news_novelty_saturation=False,
        use_news_confirmation=False,
        news_confirmation_weight=0.20,
        use_news_macro_weighting=False,
        use_beneish=False,
        use_accrual_volatility=False,
        use_working_capital_stress=False,
        forensic_weight=0.10,
        missing_beneish_penalty=0.25,
        use_capital_allocation_quality=False,
        capital_allocation_weight=0.04,
        use_recovery_transition=False,
        recovery_transition_weight=0.03,
        use_insider_conviction=False,
        insider_conviction_weight=0.03,
        use_news_theme_drift=False,
        news_theme_drift_weight=0.03,
        use_peer_relative_anomalies=False,
        peer_relative_anomaly_weight=0.04,
        exclude_binary_biotech=False,
        binary_biotech_min_revenue=1_000_000_000.0,
        dividend_payout_cap=0.85,
        max_distance_from_high=0.15,
        require_above_200dma=False,
        neutralize_by="sector",
        min_group_size=1,
        overlay_top_n=50,
        output=tmp_path / "ranked.csv",
        min_sentiment_days=3,
        min_piotroski_score=5,
        pead_max_abs_surprise_pct=100.0,
        pead_max_age_days=45,
        macro_state="neutral",
        universe_size=100,
        use_employee_efficiency=False,
        employee_efficiency_weight=0.05,
        analysis_from_primary_ticker=False,
    )

    stripped = _strip_inactive_ranked_output_columns(frame, config)

    assert "price_momentum_6m_ex_1m" not in stripped.columns
    assert "passes_momentum_gate" not in stripped.columns
    assert "contrib_momentum" not in stripped.columns
    assert stripped["symbol"].tolist() == ["ABC.US"]
