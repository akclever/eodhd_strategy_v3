from __future__ import annotations

from pathlib import Path

import pandas as pd

from eodhd_strategy.config import RankerConfig
from eodhd_strategy.ranker import (
    build_neutralization_comparison,
    build_ranked_frame,
    build_revision_impulse_weight_comparison,
)


def _config(**overrides) -> RankerConfig:
    base = {
        "api_token": "test",
        "cache_dir": Path("."),
        "refresh": False,
        "workers": 1,
        "min_market_cap": 100.0,
        "dividend_source": "hybrid",
        "regime": "neutral",
        "use_pead": True,
        "pead_lookback_days": 120,
        "pead_half_life_days": 45,
        "min_pead_analysts": 3,
        "use_revision_impulse": True,
        "min_revision_analysts": 4,
        "revision_impulse_weight": 0.06,
        "use_estimate_term_structure": False,
        "estimate_term_structure_weight": 0.04,
        "use_growth_acceleration": False,
        "growth_weight": 0.10,
        "alpha_factor_spec": "legacy",
        "use_residual_valuation": False,
        "use_compounder_persistence": False,
        "use_intangible_adjustments": False,
        "use_price_momentum": False,
        "require_real_momentum_coverage": False,
        "momentum_weight": 0.10,
        "use_life_cycle": False,
        "life_cycle_tilt_strength": 0.35,
        "use_sentiment": True,
        "sentiment_lookback_days": 14,
        "min_sentiment_accel": -0.02,
        "min_sentiment_articles_recent": 3,
        "use_news_events": False,
        "news_lookback_days": 10,
        "min_news_articles": 3,
        "news_event_weight": 0.06,
        "use_news_peer_spillover": False,
        "news_peer_spillover_weight": 0.25,
        "use_news_novelty_saturation": False,
        "use_news_confirmation": False,
        "news_confirmation_weight": 0.20,
        "use_news_macro_weighting": False,
        "use_beneish": True,
        "use_accrual_volatility": True,
        "use_working_capital_stress": False,
        "forensic_weight": 0.10,
        "missing_beneish_penalty": 0.25,
        "use_capital_allocation_quality": False,
        "capital_allocation_weight": 0.04,
        "use_recovery_transition": False,
        "recovery_transition_weight": 0.03,
        "use_insider_conviction": False,
        "insider_conviction_weight": 0.03,
        "use_news_theme_drift": False,
        "news_theme_drift_weight": 0.03,
        "use_peer_relative_anomalies": False,
        "peer_relative_anomaly_weight": 0.04,
        "exclude_binary_biotech": False,
        "binary_biotech_min_revenue": 1_000_000_000.0,
        "dividend_payout_cap": 0.85,
        "max_distance_from_high": 0.15,
        "require_above_200dma": False,
        "neutralize_by": "sector",
        "min_group_size": 1,
        "overlay_top_n": 50,
        "output": Path("ranked.csv"),
        "min_sentiment_days": 3,
        "min_piotroski_score": 5,
        "pead_max_abs_surprise_pct": 100.0,
        "pead_max_age_days": 45,
        "macro_state": "neutral",
        "universe_size": 200,
        "use_employee_efficiency": False,
        "employee_efficiency_weight": 0.05,
        "analysis_from_primary_ticker": False,
    }
    base.update(overrides)
    return RankerConfig(**base)


def _base_row(symbol: str, *, sector: str = "Technology", industry: str = "Software") -> dict:
    return {
        "symbol": symbol,
        "sector": sector,
        "industry": industry,
        "market_cap": 1000.0,
        "shareholder_yield": 0.08,
        "gross_profitability": 0.40,
        "adjusted_book_to_market": 0.60,
        "buyback_yield": 0.02,
        "recency_ratio": 0.90,
        "price_to_200dma": 1.10,
        "piotroski_score": 7,
        "pead_signal": 0.00,
        "pead_filter_pass": 1.0,
        "pead_has_setup_coverage": 0.0,
        "sentiment_count_days": 0,
        "sentiment_article_count_recent": 0.0,
        "sentiment_filter_pass": 1.0,
        "beneish_hard_filter_pass": 1.0,
        "beneish_m_score": -2.5,
        "accrual_volatility": 0.03,
    }


def test_build_ranked_frame_applies_forensic_and_pead_filters() -> None:
    df = pd.DataFrame(
        [
            {
                "symbol": "GOOD",
                "sector": "Technology",
                "industry": "Software",
                "market_cap": 1000.0,
                "shareholder_yield": 0.08,
                "gross_profitability": 0.40,
                "adjusted_book_to_market": 0.60,
                "buyback_yield": 0.02,
                "recency_ratio": 0.90,
                "price_to_200dma": 1.10,
                "piotroski_score": 7,
                "pead_signal": 0.40,
                "pead_filter_pass": 1.0,
                "pead_has_setup_coverage": 1.0,
                "sentiment_count_days": 4,
                "sentiment_article_count_recent": 5.0,
                "sentiment_filter_pass": 1.0,
                "beneish_hard_filter_pass": 1.0,
                "beneish_m_score": -2.5,
                "accrual_volatility": 0.03,
                "revenue_per_employee": 10.0,
                "gross_profit_per_employee": 5.0,
            },
            {
                "symbol": "BADPEAD",
                "sector": "Technology",
                "industry": "Software",
                "market_cap": 1000.0,
                "shareholder_yield": 0.07,
                "gross_profitability": 0.39,
                "adjusted_book_to_market": 0.58,
                "buyback_yield": 0.01,
                "recency_ratio": 0.88,
                "price_to_200dma": 1.05,
                "piotroski_score": 7,
                "pead_signal": -0.20,
                "pead_filter_pass": 0.0,
                "pead_has_setup_coverage": 1.0,
                "sentiment_count_days": 4,
                "sentiment_article_count_recent": 5.0,
                "sentiment_filter_pass": 1.0,
                "beneish_hard_filter_pass": 1.0,
                "beneish_m_score": -2.0,
                "accrual_volatility": 0.04,
                "revenue_per_employee": 9.0,
                "gross_profit_per_employee": 4.5,
            },
            {
                "symbol": "BADBEN",
                "sector": "Technology",
                "industry": "Software",
                "market_cap": 1000.0,
                "shareholder_yield": 0.06,
                "gross_profitability": 0.35,
                "adjusted_book_to_market": 0.55,
                "buyback_yield": 0.01,
                "recency_ratio": 0.87,
                "price_to_200dma": 1.03,
                "piotroski_score": 7,
                "pead_signal": 0.10,
                "pead_filter_pass": 1.0,
                "pead_has_setup_coverage": 1.0,
                "sentiment_count_days": 4,
                "sentiment_article_count_recent": 5.0,
                "sentiment_filter_pass": 1.0,
                "beneish_hard_filter_pass": 0.0,
                "beneish_m_score": -0.5,
                "accrual_volatility": 0.10,
                "revenue_per_employee": 8.0,
                "gross_profit_per_employee": 4.0,
            },
        ]
    )

    _, ranked, diagnostics = build_ranked_frame(df, _config())

    assert ranked["symbol"].tolist() == ["GOOD"]
    assert "median_beneish_m_score" in diagnostics["metric"].tolist()
    assert "share_beneish_is_missing" in diagnostics["metric"].tolist()
    assert "share_beneish_is_pathological_clipped" in diagnostics["metric"].tolist()
    assert "median_revision_impulse_signal" in diagnostics["metric"].tolist()


def test_revision_impulse_overlay_can_break_tie_between_similar_names() -> None:
    df = pd.DataFrame(
        [
            {
                "symbol": "FASTREV",
                "sector": "Technology",
                "industry": "Software",
                "market_cap": 1000.0,
                "shareholder_yield": 0.08,
                "gross_profitability": 0.40,
                "adjusted_book_to_market": 0.60,
                "buyback_yield": 0.02,
                "recency_ratio": 0.90,
                "price_to_200dma": 1.10,
                "piotroski_score": 7,
                "pead_signal": 0.00,
                "pead_filter_pass": 1.0,
                "pead_has_setup_coverage": 0.0,
                "revision_impulse_signal": 0.45,
                "revision_impulse_has_coverage": 1.0,
                "sentiment_count_days": 0,
                "sentiment_article_count_recent": 0.0,
                "sentiment_filter_pass": 1.0,
                "beneish_hard_filter_pass": 1.0,
                "beneish_m_score": -2.5,
                "accrual_volatility": 0.03,
            },
            {
                "symbol": "SLOWREV",
                "sector": "Technology",
                "industry": "Software",
                "market_cap": 1000.0,
                "shareholder_yield": 0.08,
                "gross_profitability": 0.40,
                "adjusted_book_to_market": 0.60,
                "buyback_yield": 0.02,
                "recency_ratio": 0.90,
                "price_to_200dma": 1.10,
                "piotroski_score": 7,
                "pead_signal": 0.00,
                "pead_filter_pass": 1.0,
                "pead_has_setup_coverage": 0.0,
                "revision_impulse_signal": -0.10,
                "revision_impulse_has_coverage": 1.0,
                "sentiment_count_days": 0,
                "sentiment_article_count_recent": 0.0,
                "sentiment_filter_pass": 1.0,
                "beneish_hard_filter_pass": 1.0,
                "beneish_m_score": -2.5,
                "accrual_volatility": 0.03,
            },
        ]
    )

    _, ranked, _ = build_ranked_frame(df, _config())

    assert ranked["symbol"].tolist() == ["FASTREV", "SLOWREV"]


def test_news_event_overlay_can_break_tie_between_similar_names() -> None:
    df = pd.DataFrame(
        [
            {
                "symbol": "GOODNEWS",
                "sector": "Technology",
                "industry": "Software",
                "market_cap": 1000.0,
                "shareholder_yield": 0.08,
                "gross_profitability": 0.40,
                "adjusted_book_to_market": 0.60,
                "buyback_yield": 0.02,
                "recency_ratio": 0.90,
                "price_to_200dma": 1.10,
                "piotroski_score": 7,
                "pead_has_setup_coverage": 0.0,
                "sentiment_count_days": 0,
                "sentiment_article_count_recent": 0.0,
                "sentiment_filter_pass": 1.0,
                "news_event_signal": 0.50,
                "news_event_breadth": 2.0,
                "news_article_count_recent": 5.0,
                "beneish_hard_filter_pass": 1.0,
                "beneish_m_score": -2.5,
                "accrual_volatility": 0.03,
            },
            {
                "symbol": "BADNEWS",
                "sector": "Technology",
                "industry": "Software",
                "market_cap": 1000.0,
                "shareholder_yield": 0.08,
                "gross_profitability": 0.40,
                "adjusted_book_to_market": 0.60,
                "buyback_yield": 0.02,
                "recency_ratio": 0.90,
                "price_to_200dma": 1.10,
                "piotroski_score": 7,
                "pead_has_setup_coverage": 0.0,
                "sentiment_count_days": 0,
                "sentiment_article_count_recent": 0.0,
                "sentiment_filter_pass": 1.0,
                "news_event_signal": -0.20,
                "news_event_breadth": 1.0,
                "news_article_count_recent": 5.0,
                "beneish_hard_filter_pass": 1.0,
                "beneish_m_score": -2.5,
                "accrual_volatility": 0.03,
            },
        ]
    )

    _, ranked, _ = build_ranked_frame(
        df,
        _config(
            use_pead=False,
            use_revision_impulse=False,
            use_sentiment=False,
            use_news_events=True,
            news_event_weight=0.08,
        ),
    )

    assert ranked["symbol"].tolist() == ["GOODNEWS", "BADNEWS"]


def test_investment_and_accrual_sleeves_can_break_tie_between_similar_names() -> None:
    df = pd.DataFrame(
        [
            {
                **_base_row("DISCIPLINED"),
                "investment_restraint_signal": 0.80,
                "investment_restraint_has_coverage": 1.0,
                "investment_restraint_measure_count": 6.0,
                "accrual_quality_signal": 0.70,
                "accrual_quality_has_coverage": 1.0,
                "accrual_quality_measure_count": 8.0,
                "accrual_quality_periodicity": 1.0,
            },
            {
                **_base_row("SPRAWLER"),
                "investment_restraint_signal": -0.30,
                "investment_restraint_has_coverage": 1.0,
                "investment_restraint_measure_count": 6.0,
                "accrual_quality_signal": -0.40,
                "accrual_quality_has_coverage": 1.0,
                "accrual_quality_measure_count": 8.0,
                "accrual_quality_periodicity": 1.0,
            },
        ]
    )

    _, ranked, _ = build_ranked_frame(
        df,
        _config(
            use_pead=False,
            use_revision_impulse=False,
            use_sentiment=False,
            use_beneish=False,
            use_accrual_volatility=False,
            use_investment_restraint=True,
            investment_restraint_weight=0.05,
            use_accrual_quality=True,
            accrual_quality_weight=0.05,
        ),
    )

    assert ranked["symbol"].tolist() == ["DISCIPLINED", "SPRAWLER"]


def test_quality_acceleration_sleeve_can_break_tie_between_similar_names() -> None:
    df = pd.DataFrame(
        [
            {
                **_base_row("INFLECTING"),
                "quality_acceleration_signal": 0.85,
                "quality_acceleration_has_coverage": 1.0,
                "quality_acceleration_measure_count": 6.0,
                "quality_acceleration_periodicity": 1.0,
            },
            {
                **_base_row("STALLING"),
                "quality_acceleration_signal": -0.20,
                "quality_acceleration_has_coverage": 1.0,
                "quality_acceleration_measure_count": 6.0,
                "quality_acceleration_periodicity": 1.0,
            },
        ]
    )

    _, ranked, _ = build_ranked_frame(
        df,
        _config(
            use_pead=False,
            use_revision_impulse=False,
            use_sentiment=False,
            use_beneish=False,
            use_accrual_volatility=False,
            use_quality_acceleration=True,
            quality_acceleration_weight=0.05,
        ),
    )

    assert ranked["symbol"].tolist() == ["INFLECTING", "STALLING"]


def test_revision_jerk_sleeve_can_break_tie_between_similar_names() -> None:
    df = pd.DataFrame(
        [
            {
                **_base_row("ACCEL"),
                "revision_jerk_signal": 0.75,
                "revision_jerk_has_coverage": 1.0,
            },
            {
                **_base_row("FLAT"),
                "revision_jerk_signal": -0.20,
                "revision_jerk_has_coverage": 1.0,
            },
        ]
    )

    _, ranked, _ = build_ranked_frame(
        df,
        _config(
            use_pead=False,
            use_revision_impulse=False,
            use_sentiment=False,
            use_beneish=False,
            use_accrual_volatility=False,
            use_revision_jerk=True,
            revision_jerk_weight=0.05,
        ),
    )

    assert ranked["symbol"].tolist() == ["ACCEL", "FLAT"]


def test_news_shock_sleeve_can_break_tie_between_similar_names() -> None:
    df = pd.DataFrame(
        [
            {
                **_base_row("SHOCKUP"),
                "news_shock_signal": 0.65,
                "news_shock_has_coverage": 1.0,
                "news_article_volume_spike": 2.0,
                "news_novelty_score": 0.90,
            },
            {
                **_base_row("SHOCKDOWN"),
                "news_shock_signal": -0.15,
                "news_shock_has_coverage": 1.0,
                "news_article_volume_spike": 0.8,
                "news_novelty_score": 0.40,
            },
        ]
    )

    _, ranked, _ = build_ranked_frame(
        df,
        _config(
            use_pead=False,
            use_revision_impulse=False,
            use_sentiment=False,
            use_beneish=False,
            use_accrual_volatility=False,
            use_news_shock=True,
            news_shock_weight=0.05,
        ),
    )

    assert ranked["symbol"].tolist() == ["SHOCKUP", "SHOCKDOWN"]


def test_stock_level_weight_renormalization_returns_unused_optional_budget_to_core() -> None:
    df = pd.DataFrame(
        [
            {
                **_base_row("COVERED"),
                "investment_restraint_signal": 0.60,
                "investment_restraint_has_coverage": 1.0,
                "investment_restraint_measure_count": 6.0,
            },
            {
                **_base_row("UNCOVERED"),
                "investment_restraint_signal": None,
                "investment_restraint_has_coverage": 0.0,
                "investment_restraint_measure_count": 0.0,
            },
        ]
    )

    all_rows, _, _ = build_ranked_frame(
        df,
        _config(
            use_pead=False,
            use_revision_impulse=False,
            use_sentiment=False,
            use_beneish=False,
            use_accrual_volatility=False,
            use_investment_restraint=True,
            investment_restraint_weight=0.20,
            core_weight_floor=0.60,
        ),
    )

    covered = all_rows.loc[all_rows["symbol"] == "COVERED"].iloc[0]
    uncovered = all_rows.loc[all_rows["symbol"] == "UNCOVERED"].iloc[0]

    assert covered["effective_optional_share"] > 0.19
    assert covered["effective_core_share"] < 0.81
    assert uncovered["effective_optional_share"] == 0.0
    assert uncovered["effective_core_share"] == 1.0


def test_estimate_term_structure_overlap_control_residualizes_against_revision_family() -> None:
    rows = []
    for idx in range(14):
        row = _base_row(f"PAIR{idx}")
        signal = 0.70 - idx * 0.08
        row["revision_impulse_signal"] = signal
        row["revision_impulse_has_coverage"] = 1.0
        row["revision_impulse_coverage_component"] = 1.0
        row["estimate_term_structure_signal"] = signal
        row["estimate_term_structure_has_coverage"] = 1.0
        row["estimate_term_structure_coverage_component"] = 1.0
        rows.append(row)

    all_rows, ranked, _ = build_ranked_frame(
        pd.DataFrame(rows),
        _config(
            use_pead=False,
            use_sentiment=False,
            use_beneish=False,
            use_accrual_volatility=False,
            use_revision_impulse=True,
            use_estimate_term_structure=True,
        ),
    )

    assert all_rows["estimate_term_structure_overlap_penalty"].max() > 0.9
    assert ranked["contrib_revision_impulse"].abs().max() > ranked["contrib_estimate_term_structure"].abs().max()
    assert ranked["estimate_term_structure_signal_confidence"].max() < 1.0


def test_build_ranked_frame_shrinks_residual_and_compounder_on_thinner_coverage() -> None:
    df = pd.DataFrame(
        [
            {
                **_base_row("THICK"),
                "residual_value_signal": 1.0,
                "residual_value_has_coverage": 1.0,
                "residual_value_peer_level": "industry",
                "compounder_persistence_signal": 1.0,
                "compounder_persistence_has_coverage": 1.0,
                "compounder_persistence_measure_count": 8.0,
                "compounder_persistence_periodicity": 1.0,
            },
            {
                **_base_row("THIN"),
                "residual_value_signal": 1.0,
                "residual_value_has_coverage": 1.0,
                "residual_value_peer_level": "global",
                "compounder_persistence_signal": 1.0,
                "compounder_persistence_has_coverage": 1.0,
                "compounder_persistence_measure_count": 4.0,
                "compounder_persistence_periodicity": 0.0,
            },
            {
                **_base_row("WEAK"),
                "residual_value_signal": -1.0,
                "residual_value_has_coverage": 1.0,
                "residual_value_peer_level": "industry",
                "compounder_persistence_signal": -1.0,
                "compounder_persistence_has_coverage": 1.0,
                "compounder_persistence_measure_count": 8.0,
                "compounder_persistence_periodicity": 1.0,
            },
        ]
    )

    all_rows, ranked, _ = build_ranked_frame(
        df,
        _config(
            use_pead=False,
            use_revision_impulse=False,
            use_sentiment=False,
            use_beneish=False,
            use_accrual_volatility=False,
            use_residual_valuation=True,
            use_compounder_persistence=True,
        ),
    )

    thick = all_rows.loc[all_rows["symbol"] == "THICK"].iloc[0]
    thin = all_rows.loc[all_rows["symbol"] == "THIN"].iloc[0]

    assert thick["residual_value_signal_confidence"] > thin["residual_value_signal_confidence"]
    assert thick["compounder_persistence_signal_confidence"] > thin["compounder_persistence_signal_confidence"]
    assert thick["contrib_residual_value"] > thin["contrib_residual_value"]
    assert thick["contrib_compounder_persistence"] > thin["contrib_compounder_persistence"]
    assert ranked["symbol"].tolist()[0] == "THICK"


def test_news_peer_spillover_can_lift_peer_without_direct_articles() -> None:
    df = pd.DataFrame(
        [
            {
                "symbol": "NEWSLEADER",
                "sector": "Technology",
                "industry": "Software",
                "market_cap": 1000.0,
                "shareholder_yield": 0.08,
                "gross_profitability": 0.40,
                "adjusted_book_to_market": 0.60,
                "buyback_yield": 0.02,
                "recency_ratio": 0.90,
                "price_to_200dma": 1.10,
                "piotroski_score": 7,
                "pead_has_setup_coverage": 0.0,
                "sentiment_count_days": 0,
                "sentiment_article_count_recent": 0.0,
                "sentiment_filter_pass": 1.0,
                "news_event_signal": 0.70,
                "news_event_breadth": 2.0,
                "news_article_count_recent": 6.0,
                "news_novelty_score": 0.90,
                "news_saturation_score": 0.10,
                "beneish_hard_filter_pass": 1.0,
                "beneish_m_score": -2.5,
                "accrual_volatility": 0.03,
            },
            {
                "symbol": "FOLLOWER",
                "sector": "Technology",
                "industry": "Software",
                "market_cap": 1000.0,
                "shareholder_yield": 0.08,
                "gross_profitability": 0.40,
                "adjusted_book_to_market": 0.60,
                "buyback_yield": 0.02,
                "recency_ratio": 0.90,
                "price_to_200dma": 1.10,
                "piotroski_score": 7,
                "pead_has_setup_coverage": 0.0,
                "sentiment_count_days": 0,
                "sentiment_article_count_recent": 0.0,
                "sentiment_filter_pass": 1.0,
                "news_event_signal": None,
                "news_event_breadth": 0.0,
                "news_article_count_recent": 0.0,
                "news_novelty_score": 0.0,
                "news_saturation_score": 0.0,
                "beneish_hard_filter_pass": 1.0,
                "beneish_m_score": -2.5,
                "accrual_volatility": 0.03,
            },
        ]
    )

    all_rows, ranked, _ = build_ranked_frame(
        df,
        _config(
            use_pead=False,
            use_revision_impulse=False,
            use_sentiment=False,
            use_news_events=True,
            use_news_peer_spillover=True,
            news_peer_spillover_weight=0.40,
        ),
    )

    follower = all_rows.loc[all_rows["symbol"] == "FOLLOWER"].iloc[0]
    assert follower["news_peer_spillover_signal"] > 0
    assert follower["news_event_effective_signal"] > 0
    assert follower["news_signal_confidence"] > 0
    assert "FOLLOWER" in ranked["symbol"].tolist()


def test_news_novelty_and_confirmation_improve_effective_signal_quality() -> None:
    df = pd.DataFrame(
        [
            {
                "symbol": "FRESHCONF",
                "sector": "Technology",
                "industry": "Software",
                "market_cap": 1000.0,
                "shareholder_yield": 0.08,
                "gross_profitability": 0.40,
                "adjusted_book_to_market": 0.60,
                "buyback_yield": 0.02,
                "recency_ratio": 0.90,
                "price_to_200dma": 1.10,
                "piotroski_score": 7,
                "pead_signal": 0.25,
                "pead_has_setup_coverage": 1.0,
                "revision_impulse_signal": 0.35,
                "revision_impulse_has_coverage": 1.0,
                "sentiment_count_days": 0,
                "sentiment_article_count_recent": 0.0,
                "sentiment_filter_pass": 1.0,
                "news_event_signal": 0.50,
                "news_event_breadth": 2.0,
                "news_article_count_recent": 4.0,
                "news_novelty_score": 1.0,
                "news_saturation_score": 0.0,
                "beneish_hard_filter_pass": 1.0,
                "beneish_m_score": -2.5,
                "accrual_volatility": 0.03,
            },
            {
                "symbol": "STALECONFLICT",
                "sector": "Technology",
                "industry": "Software",
                "market_cap": 1000.0,
                "shareholder_yield": 0.08,
                "gross_profitability": 0.40,
                "adjusted_book_to_market": 0.60,
                "buyback_yield": 0.02,
                "recency_ratio": 0.90,
                "price_to_200dma": 1.10,
                "piotroski_score": 7,
                "pead_signal": -0.25,
                "pead_has_setup_coverage": 1.0,
                "revision_impulse_signal": -0.35,
                "revision_impulse_has_coverage": 1.0,
                "sentiment_count_days": 0,
                "sentiment_article_count_recent": 0.0,
                "sentiment_filter_pass": 1.0,
                "news_event_signal": 0.50,
                "news_event_breadth": 2.0,
                "news_article_count_recent": 6.0,
                "news_novelty_score": 0.20,
                "news_saturation_score": 1.0,
                "beneish_hard_filter_pass": 1.0,
                "beneish_m_score": -2.5,
                "accrual_volatility": 0.03,
            },
        ]
    )

    all_rows, ranked, _ = build_ranked_frame(
        df,
        _config(
            use_pead=True,
            use_revision_impulse=True,
            use_sentiment=False,
            use_news_events=True,
            use_news_novelty_saturation=True,
            use_news_confirmation=True,
            news_confirmation_weight=0.30,
            news_event_weight=0.08,
        ),
    )

    fresh = all_rows.loc[all_rows["symbol"] == "FRESHCONF"].iloc[0]
    stale = all_rows.loc[all_rows["symbol"] == "STALECONFLICT"].iloc[0]
    assert fresh["news_event_effective_signal"] > stale["news_event_effective_signal"]
    assert fresh["news_confirmation_signal"] > 0
    assert stale["news_confirmation_signal"] < 0
    assert ranked["symbol"].tolist() == ["FRESHCONF", "STALECONFLICT"]


def test_news_macro_weighting_is_sign_sensitive() -> None:
    df = pd.DataFrame(
        [
            {
                "symbol": "POSNEWS",
                "sector": "Technology",
                "industry": "Software",
                "market_cap": 1000.0,
                "shareholder_yield": 0.08,
                "gross_profitability": 0.40,
                "adjusted_book_to_market": 0.60,
                "buyback_yield": 0.02,
                "recency_ratio": 0.90,
                "price_to_200dma": 1.10,
                "piotroski_score": 7,
                "pead_has_setup_coverage": 0.0,
                "sentiment_count_days": 0,
                "sentiment_article_count_recent": 0.0,
                "sentiment_filter_pass": 1.0,
                "news_event_signal": 0.35,
                "news_event_breadth": 1.0,
                "news_article_count_recent": 4.0,
                "news_novelty_score": 0.6,
                "news_saturation_score": 0.2,
                "beneish_hard_filter_pass": 1.0,
                "beneish_m_score": -2.5,
                "accrual_volatility": 0.03,
            },
            {
                "symbol": "NEGNEWS",
                "sector": "Technology",
                "industry": "Software",
                "market_cap": 1000.0,
                "shareholder_yield": 0.08,
                "gross_profitability": 0.40,
                "adjusted_book_to_market": 0.60,
                "buyback_yield": 0.02,
                "recency_ratio": 0.90,
                "price_to_200dma": 1.10,
                "piotroski_score": 7,
                "pead_has_setup_coverage": 0.0,
                "sentiment_count_days": 0,
                "sentiment_article_count_recent": 0.0,
                "sentiment_filter_pass": 1.0,
                "news_event_signal": -0.35,
                "news_event_breadth": 1.0,
                "news_article_count_recent": 4.0,
                "news_novelty_score": 0.6,
                "news_saturation_score": 0.2,
                "beneish_hard_filter_pass": 1.0,
                "beneish_m_score": -2.5,
                "accrual_volatility": 0.03,
            },
        ]
    )

    all_rows, _, _ = build_ranked_frame(
        df,
        _config(
            use_pead=False,
            use_revision_impulse=False,
            use_sentiment=False,
            use_news_events=True,
            use_news_macro_weighting=True,
            macro_state="defensive",
        ),
    )

    pos = all_rows.loc[all_rows["symbol"] == "POSNEWS"].iloc[0]
    neg = all_rows.loc[all_rows["symbol"] == "NEGNEWS"].iloc[0]
    assert neg["news_macro_multiplier"] > pos["news_macro_multiplier"]
    assert neg["news_event_effective_signal"] < neg["news_event_signal"]
    assert pos["news_event_effective_signal"] < pos["news_event_signal"]


def test_build_ranked_frame_handles_revision_columns_when_overlay_disabled() -> None:
    df = pd.DataFrame(
        [
            {
                "symbol": "BASEPASS",
                "sector": "Technology",
                "industry": "Software",
                "market_cap": 1000.0,
                "shareholder_yield": 0.08,
                "gross_profitability": 0.40,
                "adjusted_book_to_market": 0.60,
                "buyback_yield": 0.02,
                "recency_ratio": 0.90,
                "price_to_200dma": 1.10,
                "piotroski_score": 7,
                "pead_signal": 0.00,
                "pead_filter_pass": 1.0,
                "pead_has_setup_coverage": 0.0,
                "revision_impulse_signal": 0.30,
                "revision_impulse_has_coverage": 1.0,
                "sentiment_count_days": 0,
                "sentiment_article_count_recent": 0.0,
                "sentiment_filter_pass": 1.0,
                "beneish_hard_filter_pass": 1.0,
                "beneish_m_score": -2.5,
                "accrual_volatility": 0.03,
            }
        ]
    )

    _, ranked, _ = build_ranked_frame(df, _config(use_revision_impulse=False))

    assert ranked["symbol"].tolist() == ["BASEPASS"]


def test_life_cycle_assigns_expected_stages_and_conditioned_weights() -> None:
    df = pd.DataFrame(
        [
            {
                "symbol": "GROW",
                "sector": "Technology",
                "industry": "Software",
                "market_cap": 1000.0,
                "shareholder_yield": 0.01,
                "gross_profitability": 0.62,
                "adjusted_book_to_market": 0.10,
                "buyback_yield": 0.00,
                "recency_ratio": 0.92,
                "price_to_200dma": 1.20,
                "piotroski_score": 8,
                "pead_signal": 0.08,
                "pead_filter_pass": 1.0,
                "pead_has_setup_coverage": 1.0,
                "sue_signal": 1.20,
                "revision_impulse_signal": 0.55,
                "revision_impulse_has_coverage": 1.0,
                "revenue_growth_yoy": 0.32,
                "revenue_acceleration": 0.11,
                "price_momentum_6m_ex_1m": 0.24,
                "price_momentum_effective_signal": 0.24,
                "price_momentum_has_coverage": 1.0,
                "price_momentum_signal_coverage": 1.0,
                "price_momentum_proxy_used": 0.0,
                "sentiment_count_days": 0,
                "sentiment_article_count_recent": 0.0,
                "sentiment_filter_pass": 1.0,
                "beneish_hard_filter_pass": 1.0,
                "beneish_m_score": -2.8,
            },
            {
                "symbol": "MATURE",
                "sector": "Utilities",
                "industry": "Utilities - Regulated Electric",
                "market_cap": 1000.0,
                "shareholder_yield": 0.09,
                "gross_profitability": 0.42,
                "adjusted_book_to_market": 0.28,
                "buyback_yield": 0.01,
                "recency_ratio": 0.95,
                "price_to_200dma": 1.08,
                "piotroski_score": 7,
                "pead_signal": 0.00,
                "pead_filter_pass": 1.0,
                "pead_has_setup_coverage": 1.0,
                "sue_signal": 0.10,
                "revision_impulse_signal": 0.05,
                "revision_impulse_has_coverage": 1.0,
                "revenue_growth_yoy": 0.05,
                "revenue_acceleration": -0.01,
                "price_momentum_6m_ex_1m": 0.06,
                "price_momentum_effective_signal": 0.06,
                "price_momentum_has_coverage": 1.0,
                "price_momentum_signal_coverage": 1.0,
                "price_momentum_proxy_used": 0.0,
                "sentiment_count_days": 0,
                "sentiment_article_count_recent": 0.0,
                "sentiment_filter_pass": 1.0,
                "beneish_hard_filter_pass": 1.0,
                "beneish_m_score": -2.8,
            },
            {
                "symbol": "RECOVER",
                "sector": "Consumer Cyclical",
                "industry": "Apparel Retail",
                "market_cap": 1000.0,
                "shareholder_yield": 0.02,
                "gross_profitability": 0.09,
                "adjusted_book_to_market": 0.92,
                "buyback_yield": 0.00,
                "recency_ratio": 0.90,
                "price_to_200dma": 1.04,
                "piotroski_score": 5,
                "pead_signal": -0.03,
                "pead_filter_pass": 1.0,
                "pead_has_setup_coverage": 1.0,
                "sue_signal": -0.30,
                "revision_impulse_signal": -0.18,
                "revision_impulse_has_coverage": 1.0,
                "revenue_growth_yoy": 0.02,
                "revenue_acceleration": 0.04,
                "price_momentum_6m_ex_1m": 0.18,
                "price_momentum_effective_signal": 0.18,
                "price_momentum_has_coverage": 1.0,
                "price_momentum_signal_coverage": 1.0,
                "price_momentum_proxy_used": 0.0,
                "sentiment_count_days": 0,
                "sentiment_article_count_recent": 0.0,
                "sentiment_filter_pass": 1.0,
                "beneish_hard_filter_pass": 1.0,
                "beneish_m_score": -2.8,
            },
        ]
    )

    _, ranked, diagnostics = build_ranked_frame(
        df,
        _config(
            use_life_cycle=True,
            use_beneish=False,
            use_accrual_volatility=False,
            use_sentiment=False,
            use_growth_acceleration=True,
            use_price_momentum=True,
            life_cycle_tilt_strength=0.80,
        ),
    )

    stage_map = ranked.set_index("symbol")["life_cycle_stage"].to_dict()

    assert stage_map["GROW"] == "growth"
    assert stage_map["MATURE"] == "mature"
    assert stage_map["RECOVER"] == "recovery"

    grow = ranked.loc[ranked["symbol"] == "GROW"].iloc[0]
    mature = ranked.loc[ranked["symbol"] == "MATURE"].iloc[0]
    recover = ranked.loc[ranked["symbol"] == "RECOVER"].iloc[0]

    assert grow["life_cycle_core_weight_shareholder_yield"] < mature["life_cycle_core_weight_shareholder_yield"]
    assert grow["life_cycle_revision_impulse_multiplier"] > mature["life_cycle_revision_impulse_multiplier"]
    assert recover["life_cycle_core_weight_adjusted_book_to_market"] > mature["life_cycle_core_weight_adjusted_book_to_market"]
    assert recover["life_cycle_forensic_multiplier"] > mature["life_cycle_forensic_multiplier"]
    assert "final_life_cycle_share::growth" in diagnostics["metric"].tolist()
    assert "final_life_cycle_share::recovery" in diagnostics["metric"].tolist()


def test_growth_and_momentum_overlays_promote_accelerating_trend_name() -> None:
    df = pd.DataFrame(
        [
            {
                "symbol": "ACCEL",
                "sector": "Technology",
                "industry": "Software",
                "market_cap": 1000.0,
                "shareholder_yield": 0.05,
                "gross_profitability": 0.39,
                "adjusted_book_to_market": 0.25,
                "buyback_yield": 0.01,
                "recency_ratio": 0.90,
                "price_to_200dma": 1.15,
                "piotroski_score": 7,
                "pead_signal": 0.06,
                "sue_signal": 0.80,
                "pead_filter_pass": 1.0,
                "pead_has_setup_coverage": 1.0,
                "revision_impulse_signal": 0.20,
                "revision_impulse_has_coverage": 1.0,
                "revenue_growth_yoy": 0.28,
                "revenue_acceleration": 0.09,
                "price_momentum_6m_ex_1m": 0.18,
                "price_momentum_effective_signal": 0.18,
                "price_momentum_has_coverage": 1.0,
                "price_momentum_signal_coverage": 1.0,
                "price_momentum_proxy_used": 0.0,
                "sentiment_count_days": 0,
                "sentiment_article_count_recent": 0.0,
                "sentiment_filter_pass": 1.0,
                "beneish_hard_filter_pass": 1.0,
                "beneish_m_score": -2.5,
            },
            {
                "symbol": "STATIC",
                "sector": "Technology",
                "industry": "Software",
                "market_cap": 1000.0,
                "shareholder_yield": 0.08,
                "gross_profitability": 0.37,
                "adjusted_book_to_market": 0.27,
                "buyback_yield": 0.01,
                "recency_ratio": 0.90,
                "price_to_200dma": 1.15,
                "piotroski_score": 7,
                "pead_signal": 0.02,
                "sue_signal": 0.05,
                "pead_filter_pass": 1.0,
                "pead_has_setup_coverage": 1.0,
                "revision_impulse_signal": 0.02,
                "revision_impulse_has_coverage": 1.0,
                "revenue_growth_yoy": 0.03,
                "revenue_acceleration": -0.02,
                "price_momentum_6m_ex_1m": 0.01,
                "price_momentum_effective_signal": 0.01,
                "price_momentum_has_coverage": 1.0,
                "price_momentum_signal_coverage": 1.0,
                "price_momentum_proxy_used": 0.0,
                "sentiment_count_days": 0,
                "sentiment_article_count_recent": 0.0,
                "sentiment_filter_pass": 1.0,
                "beneish_hard_filter_pass": 1.0,
                "beneish_m_score": -2.5,
            },
            {
                "symbol": "LAG",
                "sector": "Technology",
                "industry": "Software",
                "market_cap": 1000.0,
                "shareholder_yield": 0.01,
                "gross_profitability": 0.20,
                "adjusted_book_to_market": 0.18,
                "buyback_yield": 0.0,
                "recency_ratio": 0.90,
                "price_to_200dma": 1.05,
                "piotroski_score": 5,
                "pead_signal": -0.03,
                "sue_signal": -0.25,
                "pead_filter_pass": 1.0,
                "pead_has_setup_coverage": 1.0,
                "revision_impulse_signal": -0.04,
                "revision_impulse_has_coverage": 1.0,
                "revenue_growth_yoy": -0.04,
                "revenue_acceleration": -0.05,
                "price_momentum_6m_ex_1m": -0.08,
                "price_momentum_effective_signal": -0.08,
                "price_momentum_has_coverage": 1.0,
                "price_momentum_signal_coverage": 1.0,
                "price_momentum_proxy_used": 0.0,
                "sentiment_count_days": 0,
                "sentiment_article_count_recent": 0.0,
                "sentiment_filter_pass": 1.0,
                "beneish_hard_filter_pass": 1.0,
                "beneish_m_score": -2.5,
            },
        ]
    )

    _, ranked, diagnostics = build_ranked_frame(
        df,
        _config(
            use_growth_acceleration=True,
            use_price_momentum=True,
            use_beneish=False,
            use_accrual_volatility=False,
            use_sentiment=False,
        ),
    )

    ranked_by_symbol = ranked.set_index("symbol")
    assert ranked_by_symbol.loc["ACCEL", "rank"] < ranked_by_symbol.loc["LAG", "rank"]
    assert ranked_by_symbol.loc["ACCEL", "contrib_growth"] > ranked_by_symbol.loc["STATIC", "contrib_growth"]
    assert ranked_by_symbol.loc["ACCEL", "contrib_momentum"] > ranked_by_symbol.loc["STATIC", "contrib_momentum"]
    assert "share_revenue_growth_has_coverage" in diagnostics["metric"].tolist()
    assert "share_price_momentum_has_coverage" in diagnostics["metric"].tolist()


def test_proxy_momentum_and_signal_confidence_are_reflected_in_ranking() -> None:
    df = pd.DataFrame(
        [
            {
                "symbol": "HISTORY",
                "sector": "Technology",
                "industry": "Software",
                "market_cap": 1000.0,
                "shareholder_yield": 0.04,
                "gross_profitability": 0.32,
                "adjusted_book_to_market": 0.22,
                "buyback_yield": 0.01,
                "recency_ratio": 0.92,
                "price_to_200dma": 1.12,
                "piotroski_score": 7,
                "pead_signal": 0.01,
                "sue_signal": 0.20,
                "pead_filter_pass": 1.0,
                "pead_has_setup_coverage": 1.0,
                "revision_impulse_signal": 0.12,
                "revision_impulse_has_coverage": 1.0,
                "revision_impulse_coverage_component": 1.0,
                "revenue_growth_yoy": 0.12,
                "revenue_acceleration": 0.04,
                "revenue_growth_has_coverage": 1.0,
                "price_momentum_effective_signal": 0.16,
                "price_momentum_has_coverage": 1.0,
                "price_momentum_signal_coverage": 1.0,
                "price_momentum_proxy_used": 0.0,
                "sentiment_count_days": 0,
                "sentiment_article_count_recent": 0.0,
                "sentiment_filter_pass": 1.0,
                "beneish_hard_filter_pass": 1.0,
                "beneish_m_score": -2.5,
            },
            {
                "symbol": "PROXY",
                "sector": "Technology",
                "industry": "Software",
                "market_cap": 1000.0,
                "shareholder_yield": 0.04,
                "gross_profitability": 0.32,
                "adjusted_book_to_market": 0.22,
                "buyback_yield": 0.01,
                "recency_ratio": 0.92,
                "price_to_200dma": 1.12,
                "piotroski_score": 7,
                "pead_signal": 0.01,
                "sue_signal": 0.20,
                "pead_filter_pass": 1.0,
                "pead_has_setup_coverage": 1.0,
                "revision_impulse_signal": 0.12,
                "revision_impulse_has_coverage": 1.0,
                "revision_impulse_coverage_component": 1.0,
                "revenue_growth_yoy": 0.12,
                "revenue_acceleration": 0.04,
                "revenue_growth_has_coverage": 1.0,
                "price_momentum_effective_signal": 0.16,
                "price_momentum_has_coverage": 0.0,
                "price_momentum_signal_coverage": 1.0,
                "price_momentum_proxy_used": 1.0,
                "sentiment_count_days": 0,
                "sentiment_article_count_recent": 0.0,
                "sentiment_filter_pass": 1.0,
                "beneish_hard_filter_pass": 1.0,
                "beneish_m_score": -2.5,
            },
        ]
    )

    _, ranked, _ = build_ranked_frame(
        df,
        _config(
            use_growth_acceleration=True,
            use_price_momentum=True,
            use_beneish=False,
            use_accrual_volatility=False,
            use_sentiment=False,
        ),
    )

    ranks = ranked.set_index("symbol")["rank"].to_dict()
    assert ranks["HISTORY"] < ranks["PROXY"]


def test_real_momentum_coverage_gate_excludes_proxy_only_names() -> None:
    df = pd.DataFrame(
        [
            {
                "symbol": "HISTORY",
                "sector": "Technology",
                "industry": "Software",
                "market_cap": 1000.0,
                "shareholder_yield": 0.04,
                "gross_profitability": 0.32,
                "adjusted_book_to_market": 0.22,
                "buyback_yield": 0.01,
                "recency_ratio": 0.92,
                "price_to_200dma": 1.12,
                "piotroski_score": 7,
                "pead_signal": 0.01,
                "pead_filter_pass": 1.0,
                "pead_has_setup_coverage": 1.0,
                "price_momentum_effective_signal": 0.16,
                "price_momentum_has_coverage": 1.0,
                "price_momentum_signal_coverage": 1.0,
                "price_momentum_proxy_used": 0.0,
                "sentiment_count_days": 0,
                "sentiment_article_count_recent": 0.0,
                "sentiment_filter_pass": 1.0,
                "beneish_hard_filter_pass": 1.0,
                "beneish_m_score": -2.5,
            },
            {
                "symbol": "PROXY",
                "sector": "Technology",
                "industry": "Software",
                "market_cap": 1000.0,
                "shareholder_yield": 0.04,
                "gross_profitability": 0.32,
                "adjusted_book_to_market": 0.22,
                "buyback_yield": 0.01,
                "recency_ratio": 0.92,
                "price_to_200dma": 1.12,
                "piotroski_score": 7,
                "pead_signal": 0.01,
                "pead_filter_pass": 1.0,
                "pead_has_setup_coverage": 1.0,
                "price_momentum_effective_signal": 0.16,
                "price_momentum_has_coverage": 0.0,
                "price_momentum_signal_coverage": 1.0,
                "price_momentum_proxy_used": 1.0,
                "sentiment_count_days": 0,
                "sentiment_article_count_recent": 0.0,
                "sentiment_filter_pass": 1.0,
                "beneish_hard_filter_pass": 1.0,
                "beneish_m_score": -2.5,
            },
        ]
    )

    _, ranked, diagnostics = build_ranked_frame(
        df,
        _config(
            use_price_momentum=True,
            require_real_momentum_coverage=True,
            use_beneish=False,
            use_accrual_volatility=False,
            use_sentiment=False,
        ),
    )

    assert ranked["symbol"].tolist() == ["HISTORY"]
    assert "share_passes_momentum_gate" in diagnostics["metric"].tolist()


def test_binary_biotech_filter_excludes_low_revenue_biotech_only() -> None:
    df = pd.DataFrame(
        [
            {
                "symbol": "BIOLOW",
                "sector": "Healthcare",
                "industry": "Biotechnology",
                "market_cap": 3000.0,
                "total_revenue": 250_000_000.0,
                "shareholder_yield": 0.03,
                "gross_profitability": 0.20,
                "adjusted_book_to_market": 0.40,
                "buyback_yield": 0.00,
                "recency_ratio": 0.90,
                "price_to_200dma": 1.05,
                "piotroski_score": 6,
                "pead_signal": 0.01,
                "pead_filter_pass": 1.0,
                "pead_has_setup_coverage": 0.0,
                "sentiment_count_days": 0,
                "sentiment_article_count_recent": 0.0,
                "sentiment_filter_pass": 1.0,
                "beneish_hard_filter_pass": 1.0,
                "beneish_m_score": -2.5,
            },
            {
                "symbol": "BIOBIG",
                "sector": "Healthcare",
                "industry": "Biotechnology",
                "market_cap": 12000.0,
                "total_revenue": 2_500_000_000.0,
                "shareholder_yield": 0.03,
                "gross_profitability": 0.20,
                "adjusted_book_to_market": 0.40,
                "buyback_yield": 0.00,
                "recency_ratio": 0.90,
                "price_to_200dma": 1.05,
                "piotroski_score": 6,
                "pead_signal": 0.01,
                "pead_filter_pass": 1.0,
                "pead_has_setup_coverage": 0.0,
                "sentiment_count_days": 0,
                "sentiment_article_count_recent": 0.0,
                "sentiment_filter_pass": 1.0,
                "beneish_hard_filter_pass": 1.0,
                "beneish_m_score": -2.5,
            },
            {
                "symbol": "SOFT",
                "sector": "Technology",
                "industry": "Software",
                "market_cap": 12000.0,
                "total_revenue": 250_000_000.0,
                "shareholder_yield": 0.03,
                "gross_profitability": 0.20,
                "adjusted_book_to_market": 0.40,
                "buyback_yield": 0.00,
                "recency_ratio": 0.90,
                "price_to_200dma": 1.05,
                "piotroski_score": 6,
                "pead_signal": 0.01,
                "pead_filter_pass": 1.0,
                "pead_has_setup_coverage": 0.0,
                "sentiment_count_days": 0,
                "sentiment_article_count_recent": 0.0,
                "sentiment_filter_pass": 1.0,
                "beneish_hard_filter_pass": 1.0,
                "beneish_m_score": -2.5,
            },
        ]
    )

    _, ranked, diagnostics = build_ranked_frame(
        df,
        _config(
            exclude_binary_biotech=True,
            binary_biotech_min_revenue=1_000_000_000.0,
            use_beneish=False,
            use_accrual_volatility=False,
            use_sentiment=False,
        ),
    )

    assert set(ranked["symbol"].tolist()) == {"BIOBIG", "SOFT"}
    assert "share_binary_biotech_flag" in diagnostics["metric"].tolist()


def test_missing_beneish_gets_small_penalty_and_pathological_is_tracked() -> None:
    df = pd.DataFrame(
        [
            {
                "symbol": "MISSBEN",
                "sector": "Technology",
                "industry": "Software",
                "market_cap": 1000.0,
                "shareholder_yield": 0.08,
                "gross_profitability": 0.40,
                "adjusted_book_to_market": 0.60,
                "buyback_yield": 0.02,
                "recency_ratio": 0.90,
                "price_to_200dma": 1.10,
                "piotroski_score": 7,
                "pead_signal": 0.40,
                "pead_filter_pass": 1.0,
                "pead_has_setup_coverage": 0.0,
                "sentiment_count_days": 0,
                "sentiment_article_count_recent": 0.0,
                "sentiment_filter_pass": 1.0,
                "beneish_m_score": None,
                "beneish_data_status": "missing",
                "beneish_is_missing": 1.0,
                "beneish_is_pathological_clipped": 0.0,
                "beneish_hard_filter_pass": 1.0,
                "accrual_volatility": 0.03,
            },
            {
                "symbol": "PATHBEN",
                "sector": "Technology",
                "industry": "Software",
                "market_cap": 1000.0,
                "shareholder_yield": 0.07,
                "gross_profitability": 0.39,
                "adjusted_book_to_market": 0.58,
                "buyback_yield": 0.01,
                "recency_ratio": 0.88,
                "price_to_200dma": 1.05,
                "piotroski_score": 7,
                "pead_signal": 0.20,
                "pead_filter_pass": 1.0,
                "pead_has_setup_coverage": 0.0,
                "sentiment_count_days": 0,
                "sentiment_article_count_recent": 0.0,
                "sentiment_filter_pass": 1.0,
                "beneish_m_score": None,
                "beneish_data_status": "pathological_clipped",
                "beneish_is_missing": 0.0,
                "beneish_is_pathological_clipped": 1.0,
                "beneish_hard_filter_pass": 1.0,
                "accrual_volatility": 0.04,
            },
        ]
    )

    _, ranked, diagnostics = build_ranked_frame(df, _config())

    missing_penalty = ranked.loc[ranked["symbol"] == "MISSBEN", "beneish_missing_penalty_applied"].iloc[0]
    pathological_penalty = ranked.loc[ranked["symbol"] == "PATHBEN", "beneish_missing_penalty_applied"].iloc[0]
    pathological_flag_share = diagnostics.loc[
        diagnostics["metric"] == "share_beneish_is_pathological_clipped",
        "value",
    ].iloc[0]

    assert missing_penalty == 0.25
    assert pathological_penalty == 0.0
    assert pathological_flag_share == 0.5


def test_large_universe_uses_stricter_beneish_gate() -> None:
    df = pd.DataFrame(
        [
            {
                "symbol": "EDGE",
                "sector": "Technology",
                "industry": "Software",
                "market_cap": 1000.0,
                "shareholder_yield": 0.08,
                "gross_profitability": 0.40,
                "adjusted_book_to_market": 0.60,
                "buyback_yield": 0.02,
                "recency_ratio": 0.90,
                "price_to_200dma": 1.10,
                "piotroski_score": 7,
                "pead_signal": 0.40,
                "pead_filter_pass": 1.0,
                "pead_has_setup_coverage": 0.0,
                "sentiment_count_days": 0,
                "sentiment_article_count_recent": 0.0,
                "sentiment_filter_pass": 1.0,
                "beneish_m_score": -1.30,
                "beneish_data_status": "ok",
                "beneish_is_missing": 0.0,
                "beneish_is_pathological_clipped": 0.0,
                "beneish_hard_filter_pass": 1.0,
                "accrual_volatility": 0.03,
            }
        ]
    )

    _, small_ranked, small_diagnostics = build_ranked_frame(df, _config(universe_size=200))
    _, large_ranked, large_diagnostics = build_ranked_frame(df, _config(universe_size=1000))

    large_threshold = large_diagnostics.loc[
        large_diagnostics["metric"] == "forensic_gate_beneish_threshold",
        "value",
    ].iloc[0]
    large_mode = large_diagnostics.loc[
        large_diagnostics["metric"] == "large_universe_forensic_mode",
        "value",
    ].iloc[0]

    assert small_ranked["symbol"].tolist() == ["EDGE"]
    assert large_ranked.empty
    assert large_threshold == -1.40
    assert large_mode == 1.0


def test_build_neutralization_comparison_reports_overlap() -> None:
    sector_ranked = pd.DataFrame(
        [
            {"symbol": "AAA", "rank": 1, "sector": "Technology", "composite_score": 1.0},
            {"symbol": "BBB", "rank": 2, "sector": "Healthcare", "composite_score": 0.9},
            {"symbol": "CCC", "rank": 3, "sector": "Utilities", "composite_score": 0.8},
        ]
    )
    none_ranked = pd.DataFrame(
        [
            {"symbol": "AAA", "rank": 1, "sector": "Technology", "composite_score": 1.0},
            {"symbol": "CCC", "rank": 2, "sector": "Utilities", "composite_score": 0.85},
            {"symbol": "DDD", "rank": 3, "sector": "Financials", "composite_score": 0.7},
        ]
    )

    comparison = build_neutralization_comparison(sector_ranked, none_ranked, top_n=2)

    overlap = comparison.loc[comparison["metric"] == "top_2_overlap_count", "value"].iloc[0]
    assert overlap == 1.0
    assert "rank_spearman_corr" in comparison["metric"].tolist()


def test_build_revision_impulse_weight_comparison_reports_anchor_overlap() -> None:
    baseline = pd.DataFrame(
        [
            {"symbol": "AAA", "rank": 1, "sector": "Technology", "composite_score": 1.0, "revision_impulse_signal": 0.0},
            {"symbol": "BBB", "rank": 2, "sector": "Healthcare", "composite_score": 0.9, "revision_impulse_signal": 0.0},
            {"symbol": "CCC", "rank": 3, "sector": "Utilities", "composite_score": 0.8, "revision_impulse_signal": 0.0},
        ]
    )
    tilted = pd.DataFrame(
        [
            {"symbol": "AAA", "rank": 1, "sector": "Technology", "composite_score": 1.1, "revision_impulse_signal": 0.4},
            {"symbol": "CCC", "rank": 2, "sector": "Utilities", "composite_score": 0.95, "revision_impulse_signal": 0.3},
            {"symbol": "DDD", "rank": 3, "sector": "Industrials", "composite_score": 0.7, "revision_impulse_signal": 0.2},
        ]
    )

    comparison = build_revision_impulse_weight_comparison({0.0: baseline, 0.06: tilted}, top_n=2)

    overlap = comparison.loc[
        (comparison["comparison"] == "anchor_vs_weight") & (comparison["metric"] == "top_2_overlap_count"),
        "value",
    ].iloc[0]
    assert overlap == 1.0
    assert "rank_spearman_corr" in comparison["metric"].tolist()


def test_residual_value_falls_back_from_industry_to_sector_to_global() -> None:
    rows = []
    for idx in range(11):
        row = _base_row(f"SOFT{idx}")
        row["gross_profitability"] = 0.30 + idx * 0.01
        row["adjusted_book_to_market"] = 0.45 + idx * 0.015
        row["revenue_growth_yoy"] = 0.08 + idx * 0.005
        row["revenue_acceleration"] = 0.01 + idx * 0.002
        row["revision_impulse_signal"] = 0.05 + idx * 0.01
        rows.append(row)

    sector_fallback = _base_row("SECTORFALL", industry="TinyIndustry")
    sector_fallback["gross_profitability"] = 0.55
    sector_fallback["adjusted_book_to_market"] = 0.95
    sector_fallback["revenue_growth_yoy"] = 0.16
    sector_fallback["revenue_acceleration"] = 0.05
    sector_fallback["revision_impulse_signal"] = 0.12
    rows.append(sector_fallback)

    global_fallback = _base_row("GLOBALFALL", sector="UniqueSector", industry="UniqueIndustry")
    global_fallback["gross_profitability"] = 0.50
    global_fallback["adjusted_book_to_market"] = 0.98
    global_fallback["revenue_growth_yoy"] = 0.14
    global_fallback["revenue_acceleration"] = 0.04
    global_fallback["revision_impulse_signal"] = 0.11
    rows.append(global_fallback)

    all_rows, ranked, _ = build_ranked_frame(
        pd.DataFrame(rows),
        _config(
            alpha_factor_spec="v2",
            use_pead=False,
            use_sentiment=False,
            use_residual_valuation=True,
            use_revision_impulse=True,
        ),
    )

    assert not ranked.empty
    assert all_rows.loc[all_rows["symbol"] == "SECTORFALL", "residual_value_peer_level"].iloc[0] == "sector"
    assert all_rows.loc[all_rows["symbol"] == "GLOBALFALL", "residual_value_peer_level"].iloc[0] == "global"


def test_build_ranked_frame_includes_new_contributions_when_enabled() -> None:
    rows = []
    symbols = [f"RECOVERY{idx}" for idx in range(12)] + ["STEADY"]
    for idx, symbol in enumerate(symbols):
        row = _base_row(symbol, industry="Hardware" if idx < 11 else "Software")
        row["revision_impulse_signal"] = 0.35 - idx * 0.20
        row["revision_impulse_has_coverage"] = 1.0
        row["estimate_term_structure_signal"] = 0.25 - idx * 0.10
        row["estimate_term_structure_has_coverage"] = 1.0
        row["estimate_term_structure_coverage_component"] = 1.0
        row["revenue_growth_yoy"] = 0.10 + idx * 0.01
        row["revenue_growth_has_coverage"] = 1.0
        row["revenue_acceleration"] = 0.04 - idx * 0.02
        row["compounder_persistence_signal"] = 0.30 - idx * 0.15
        row["compounder_persistence_has_coverage"] = 1.0
        row["capital_allocation_quality_signal"] = 0.20 - idx * 0.10
        row["capital_allocation_quality_has_coverage"] = 1.0
        row["insider_conviction_signal"] = 0.18 - idx * 0.20
        row["insider_conviction_has_coverage"] = 1.0
        row["news_theme_drift_signal"] = 0.22 - idx * 0.15
        row["news_theme_drift_has_coverage"] = 1.0
        row["price_momentum_effective_signal"] = 0.25 - idx * 0.25
        row["price_momentum_signal_coverage"] = 1.0
        row["working_capital_stress_penalty"] = 0.01 * idx
        row["recovery_margin_inflection"] = 0.08 - idx * 0.12
        row["recovery_leverage_improvement"] = 0.06 - idx * 0.08
        row["recovery_accrual_improvement"] = 0.05 - idx * 0.05
        row["peer_margin_trend_input"] = 0.10 - idx * 0.02
        row["peer_reinvestment_efficiency_input"] = 0.08 - idx * 0.015
        row["peer_estimate_drift_input"] = 0.12 - idx * 0.018
        row["peer_dilution_discipline_input"] = 0.05 - idx * 0.01
        rows.append(row)

    all_rows, ranked, diagnostics = build_ranked_frame(
        pd.DataFrame(rows),
        _config(
            alpha_factor_spec="v2",
            use_pead=False,
            use_sentiment=False,
            use_revision_impulse=True,
            use_estimate_term_structure=True,
            use_growth_acceleration=True,
            use_compounder_persistence=True,
            use_price_momentum=True,
            use_working_capital_stress=True,
            use_peer_relative_anomalies=True,
            use_capital_allocation_quality=True,
            use_recovery_transition=True,
            use_insider_conviction=True,
            use_news_theme_drift=True,
        ),
    )

    assert not ranked.empty
    assert "contrib_estimate_term_structure" in ranked.columns
    assert "contrib_capital_allocation" in ranked.columns
    assert "contrib_insider_conviction" in ranked.columns
    assert "contrib_news_theme_drift" in ranked.columns
    assert "contrib_compounder_persistence" in ranked.columns
    assert "contrib_peer_relative_anomaly" in ranked.columns
    assert "contrib_recovery_transition" in ranked.columns
    assert "median_working_capital_stress_penalty" in diagnostics["metric"].tolist()


def test_build_ranked_frame_zeroes_overlay_confidence_without_coverage() -> None:
    df = pd.DataFrame(
        [
            {
                **_base_row("AAA"),
                "capital_allocation_quality_signal": None,
                "capital_allocation_quality_has_coverage": 0.0,
                "peer_relative_anomaly_signal": None,
                "peer_relative_anomaly_has_coverage": 0.0,
                "recovery_transition_signal": None,
                "recovery_transition_has_coverage": 0.0,
                "insider_conviction_signal": None,
                "insider_conviction_has_coverage": 0.0,
                "news_theme_drift_signal": None,
                "news_theme_drift_has_coverage": 0.0,
            },
            {
                **_base_row("BBB"),
                "capital_allocation_quality_signal": None,
                "capital_allocation_quality_has_coverage": 0.0,
                "peer_relative_anomaly_signal": None,
                "peer_relative_anomaly_has_coverage": 0.0,
                "recovery_transition_signal": None,
                "recovery_transition_has_coverage": 0.0,
                "insider_conviction_signal": None,
                "insider_conviction_has_coverage": 0.0,
                "news_theme_drift_signal": None,
                "news_theme_drift_has_coverage": 0.0,
            },
        ]
    )

    _, ranked, _ = build_ranked_frame(
        df,
        _config(
            use_capital_allocation_quality=True,
            use_peer_relative_anomalies=True,
            use_recovery_transition=True,
            use_insider_conviction=True,
            use_news_theme_drift=True,
        ),
    )

    assert (ranked["capital_allocation_signal_confidence"] == 0.0).all()
    assert (ranked["peer_relative_anomaly_signal_confidence"] == 0.0).all()
    assert (ranked["recovery_transition_signal_confidence"] == 0.0).all()
    assert (ranked["insider_conviction_signal_confidence"] == 0.0).all()
    assert (ranked["news_theme_drift_signal_confidence"] == 0.0).all()


def test_build_ranked_frame_recalibrates_working_capital_forensic_penalty() -> None:
    df = pd.DataFrame(
        [
            {
                **_base_row("AAA"),
                "working_capital_stress_penalty": 0.03,
                "working_capital_stress_has_coverage": 1.0,
            },
            {
                **_base_row("BBB"),
                "working_capital_stress_penalty": 0.00,
                "working_capital_stress_has_coverage": 1.0,
            },
        ]
    )

    _, ranked, _ = build_ranked_frame(
        df,
        _config(
            use_beneish=False,
            use_accrual_volatility=False,
            use_working_capital_stress=True,
            forensic_weight=0.10,
        ),
    )

    stressed = ranked.loc[ranked["symbol"] == "AAA"].iloc[0]
    calm = ranked.loc[ranked["symbol"] == "BBB"].iloc[0]

    assert stressed["forensic_penalty"] <= 0.5
    assert stressed["forensic_penalty"] > calm["forensic_penalty"]
