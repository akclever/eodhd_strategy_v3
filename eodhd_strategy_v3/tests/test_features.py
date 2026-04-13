from __future__ import annotations

from pathlib import Path

import pandas as pd

from eodhd_strategy.config import RankerConfig
from eodhd_strategy.features import (
    add_overlay_metrics,
    compute_fundamental_metrics,
    compute_insider_conviction_metrics,
    compute_news_theme_drift_metrics,
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
        "use_revision_impulse": False,
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
        "use_news_events": True,
        "news_lookback_days": 10,
        "min_news_articles": 3,
        "news_event_weight": 0.06,
        "use_news_peer_spillover": False,
        "news_peer_spillover_weight": 0.25,
        "use_news_novelty_saturation": False,
        "use_news_confirmation": False,
        "news_confirmation_weight": 0.20,
        "use_news_macro_weighting": False,
        "use_beneish": False,
        "use_accrual_volatility": False,
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


def _fundamentals_for_intangible_adjustments() -> dict:
    return {
        "General": {
            "Sector": "Technology",
            "Exchange": "US",
            "CurrencyCode": "USD",
            "CurrencyName": "US Dollar",
            "Name": "Acme Software",
            "Type": "Common Stock",
            "CountryName": "United States",
            "CountryISO": "US",
            "ISIN": "US0000000001",
            "PrimaryTicker": "ACME.US",
        },
        "Highlights": {
            "MarketCapitalization": 1000.0,
            "PayoutRatio": 0.25,
        },
        "Technicals": {
            "52WeekHigh": 100.0,
            "200DayMA": 80.0,
        },
        "SharesStats": {
            "SharesOutstanding": 10.0,
        },
        "Financials": {
            "Income_Statement": {
                "yearly": {
                    "2025-12-31": {
                        "date": "2025-12-31",
                        "grossProfit": 400.0,
                        "netIncome": 120.0,
                        "totalRevenue": 1000.0,
                        "researchDevelopment": 60.0,
                        "sellingGeneralAdministrative": 120.0,
                    },
                    "2024-12-31": {
                        "date": "2024-12-31",
                        "grossProfit": 360.0,
                        "netIncome": 108.0,
                        "totalRevenue": 920.0,
                        "researchDevelopment": 50.0,
                        "sellingGeneralAdministrative": 110.0,
                    },
                    "2023-12-31": {
                        "date": "2023-12-31",
                        "grossProfit": 320.0,
                        "netIncome": 96.0,
                        "totalRevenue": 850.0,
                        "researchDevelopment": 42.0,
                        "sellingGeneralAdministrative": 100.0,
                    },
                }
            },
            "Balance_Sheet": {
                "yearly": {
                    "2025-12-31": {
                        "date": "2025-12-31",
                        "totalAssets": 1500.0,
                        "totalCurrentLiabilities": 250.0,
                        "totalStockholderEquity": 700.0,
                        "longTermDebt": 200.0,
                        "commonStockSharesOutstanding": 10.0,
                    },
                    "2024-12-31": {
                        "date": "2024-12-31",
                        "totalAssets": 1450.0,
                        "totalCurrentLiabilities": 240.0,
                        "totalStockholderEquity": 660.0,
                        "longTermDebt": 205.0,
                        "commonStockSharesOutstanding": 10.5,
                    }
                }
            },
            "Cash_Flow": {
                "yearly": {
                    "2025-12-31": {"date": "2025-12-31", "totalCashFromOperatingActivities": 140.0, "capitalExpenditures": -30.0},
                    "2024-12-31": {"date": "2024-12-31", "totalCashFromOperatingActivities": 125.0, "capitalExpenditures": -28.0},
                }
            },
        },
        "SplitsDividends": {
            "ForwardAnnualDividendYield": 0.01,
        },
    }


def test_add_overlay_metrics_prefers_analysis_symbol(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_pead(client, symbol, *args, **kwargs):
        calls.append(("pead", symbol))
        return {"earnings_surprise_pct": 5.0, "earnings_report_date": "2026-03-01", "pead_signal_calendar": 0.25}

    def fake_sentiment(client, symbol, *args):
        calls.append(("sentiment", symbol))
        return {
            "sentiment_latest": 0.1,
            "sentiment_speed": 0.02,
            "sentiment_acceleration": 0.01,
            "sentiment_count_days": 5,
            "sentiment_article_count_recent": 6.0,
            "sentiment_article_count_total": 8.0,
            "sentiment_latest_count": 2.0,
        }

    def fake_news(client, symbol, *args, **kwargs):
        calls.append(("news", symbol))
        return {
            "news_event_signal": 0.3,
            "news_event_breadth": 1.0,
            "news_article_count_recent": 4.0,
            "news_positive_article_share": 0.75,
            "news_negative_article_share": 0.0,
            "news_unique_title_ratio": 1.0,
            "news_novelty_score": 0.8,
            "news_saturation_score": 0.2,
        }

    monkeypatch.setattr("eodhd_strategy.features.compute_pead_metrics_from_calendar", fake_pead)
    monkeypatch.setattr("eodhd_strategy.features.compute_sentiment_metrics", fake_sentiment)
    monkeypatch.setattr("eodhd_strategy.features.compute_news_event_metrics", fake_news)

    row = {
        "symbol": "APC.XETRA",
        "analysis_symbol": "AAPL.US",
        "earnings_surprise_pct": None,
        "earnings_report_date": None,
    }

    out = add_overlay_metrics(object(), row, _config())

    assert calls == [
        ("pead", "AAPL.US"),
        ("sentiment", "AAPL.US"),
        ("news", "AAPL.US"),
    ]
    assert out["pead_signal"] == 0.25
    assert out["sentiment_latest"] == 0.1
    assert out["news_event_signal"] == 0.3


def test_compute_fundamental_metrics_only_applies_intangible_adjustments_when_enabled() -> None:
    fundamentals = _fundamentals_for_intangible_adjustments()

    plain = compute_fundamental_metrics(object(), "ACME.US", fundamentals, "forward", _config())
    adjusted = compute_fundamental_metrics(
        object(),
        "ACME.US",
        fundamentals,
        "forward",
        _config(use_intangible_adjustments=True),
    )

    assert plain["intangible_adjustment_eligible"] == 1.0
    assert plain["intangible_adjustment_applied"] == 0.0
    assert adjusted["intangible_adjustment_applied"] == 1.0
    assert adjusted["adjusted_book_to_market"] > plain["adjusted_book_to_market"]
    assert adjusted["gross_profitability"] < plain["gross_profitability"]
    assert adjusted["return_on_assets"] is not None
    assert adjusted["return_on_invested_capital"] is not None


def test_compute_fundamental_metrics_wires_investment_and_accrual_quality_when_enabled() -> None:
    fundamentals = _fundamentals_for_intangible_adjustments()
    fundamentals["Financials"]["Income_Statement"]["yearly"]["2025-12-31"]["operatingIncome"] = 150.0
    fundamentals["Financials"]["Income_Statement"]["yearly"]["2024-12-31"]["operatingIncome"] = 138.0
    fundamentals["Financials"]["Income_Statement"]["yearly"]["2023-12-31"]["operatingIncome"] = 126.0
    fundamentals["Financials"]["Income_Statement"]["yearly"]["2022-12-31"] = {
        "date": "2022-12-31",
        "grossProfit": 280.0,
        "netIncome": 84.0,
        "totalRevenue": 780.0,
        "researchDevelopment": 36.0,
        "sellingGeneralAdministrative": 92.0,
        "operatingIncome": 110.0,
    }
    fundamentals["Financials"]["Balance_Sheet"]["yearly"]["2023-12-31"] = {
        "date": "2023-12-31",
        "totalAssets": 1415.0,
        "totalCurrentLiabilities": 235.0,
        "totalStockholderEquity": 640.0,
        "longTermDebt": 208.0,
        "netReceivables": 155.0,
        "inventory": 0.0,
        "commonStockSharesOutstanding": 10.8,
    }
    fundamentals["Financials"]["Balance_Sheet"]["yearly"]["2022-12-31"] = {
        "date": "2022-12-31",
        "totalAssets": 1380.0,
        "totalCurrentLiabilities": 230.0,
        "totalStockholderEquity": 620.0,
        "longTermDebt": 210.0,
        "netReceivables": 145.0,
        "inventory": 0.0,
        "commonStockSharesOutstanding": 11.0,
    }
    fundamentals["Financials"]["Balance_Sheet"]["yearly"]["2025-12-31"]["netReceivables"] = 180.0
    fundamentals["Financials"]["Balance_Sheet"]["yearly"]["2025-12-31"]["inventory"] = 0.0
    fundamentals["Financials"]["Balance_Sheet"]["yearly"]["2024-12-31"]["netReceivables"] = 168.0
    fundamentals["Financials"]["Balance_Sheet"]["yearly"]["2024-12-31"]["inventory"] = 0.0
    fundamentals["Financials"]["Cash_Flow"]["yearly"]["2023-12-31"] = {
        "date": "2023-12-31",
        "totalCashFromOperatingActivities": 120.0,
        "capitalExpenditures": -27.0,
    }
    fundamentals["Financials"]["Cash_Flow"]["yearly"]["2022-12-31"] = {
        "date": "2022-12-31",
        "totalCashFromOperatingActivities": 115.0,
        "capitalExpenditures": -26.0,
    }

    metrics = compute_fundamental_metrics(
        object(),
        "ACME.US",
        fundamentals,
        "forward",
        _config(use_investment_restraint=True, use_accrual_quality=True),
    )

    assert metrics["investment_restraint_signal"] is not None
    assert metrics["investment_restraint_has_coverage"] > 0
    assert metrics["accrual_quality_signal"] is not None
    assert metrics["accrual_quality_has_coverage"] > 0


def test_compute_fundamental_metrics_wires_quality_acceleration_when_enabled() -> None:
    fundamentals = _fundamentals_for_intangible_adjustments()
    fundamentals["Financials"]["Balance_Sheet"]["yearly"]["2023-12-31"] = {
        "date": "2023-12-31",
        "totalAssets": 1390.0,
        "totalCurrentLiabilities": 232.0,
        "totalStockholderEquity": 630.0,
        "longTermDebt": 212.0,
        "netReceivables": 150.0,
        "inventory": 0.0,
        "commonStockSharesOutstanding": 10.9,
    }
    fundamentals["Financials"]["Cash_Flow"]["yearly"]["2023-12-31"] = {
        "date": "2023-12-31",
        "totalCashFromOperatingActivities": 118.0,
        "capitalExpenditures": -26.0,
    }
    fundamentals["Financials"]["Balance_Sheet"]["yearly"]["2025-12-31"]["netReceivables"] = 175.0
    fundamentals["Financials"]["Balance_Sheet"]["yearly"]["2025-12-31"]["inventory"] = 0.0
    fundamentals["Financials"]["Balance_Sheet"]["yearly"]["2024-12-31"]["netReceivables"] = 162.0
    fundamentals["Financials"]["Balance_Sheet"]["yearly"]["2024-12-31"]["inventory"] = 0.0

    metrics = compute_fundamental_metrics(
        object(),
        "ACME.US",
        fundamentals,
        "forward",
        _config(use_quality_acceleration=True),
    )

    assert metrics["quality_acceleration_signal"] is not None
    assert metrics["quality_acceleration_has_coverage"] > 0


def test_compute_fundamental_metrics_wires_revision_jerk_when_enabled() -> None:
    fundamentals = _fundamentals_for_intangible_adjustments()
    fundamentals["Earnings"] = {
        "Trend": {
            "2025-12-31": {
                "date": "2025-12-31",
                "earningsEstimateAvg": 1.00,
                "earningsEstimateNumberOfAnalysts": 6,
                "epsTrendCurrent": 1.18,
                "epsTrend7daysAgo": 1.06,
                "epsTrend30daysAgo": 0.97,
                "epsRevisionsUpLast7days": 4,
                "epsRevisionsDownLast30days": 1,
                "earningsEstimateHigh": 1.24,
                "earningsEstimateLow": 0.98,
            }
        }
    }

    metrics = compute_fundamental_metrics(
        object(),
        "ACME.US",
        fundamentals,
        "forward",
        _config(use_revision_jerk=True),
    )

    assert metrics["revision_jerk_signal"] is not None
    assert metrics["revision_jerk_has_coverage"] == 1.0


def test_news_theme_drift_and_insider_metrics_stay_neutral_without_coverage() -> None:
    class EmptyClient:
        def get_news(self, **kwargs):
            return []

        def get_insider_transactions(self, **kwargs):
            return []

    news_metrics = compute_news_theme_drift_metrics(EmptyClient(), "ACME.US")
    insider_metrics = compute_insider_conviction_metrics(EmptyClient(), "ACME.US")

    assert news_metrics["news_theme_drift_signal"] is None
    assert news_metrics["news_theme_drift_has_coverage"] == 0.0
    assert insider_metrics["insider_conviction_signal"] is None
    assert insider_metrics["insider_conviction_has_coverage"] == 0.0


def test_news_theme_drift_detects_positive_narrative_shift() -> None:
    today = pd.Timestamp.now(tz="UTC").normalize()

    class FakeClient:
        def get_news(self, **kwargs):
            return [
                {
                    "date": (today - pd.Timedelta(days=5)).isoformat(),
                    "title": "Acme raises guidance after contract win and upgrade",
                    "tags": ["upgrade", "contract win"],
                    "symbols": ["ACME.US"],
                    "sentiment": {"polarity": 0.6},
                },
                {
                    "date": (today - pd.Timedelta(days=12)).isoformat(),
                    "title": "Acme beats estimates and raises guidance",
                    "tags": ["estimate revisions"],
                    "symbols": ["ACME.US"],
                    "sentiment": {"polarity": 0.5},
                },
                {
                    "date": (today - pd.Timedelta(days=55)).isoformat(),
                    "title": "Acme faces weak outlook and downgrade concerns",
                    "tags": ["downgrade"],
                    "symbols": ["ACME.US"],
                    "sentiment": {"polarity": -0.4},
                },
                {
                    "date": (today - pd.Timedelta(days=70)).isoformat(),
                    "title": "Acme warning on restructuring costs",
                    "tags": ["warning"],
                    "symbols": ["ACME.US"],
                    "sentiment": {"polarity": -0.3},
                },
            ]

    metrics = compute_news_theme_drift_metrics(FakeClient(), "ACME.US")

    assert metrics["news_theme_drift_has_coverage"] == 1.0
    assert metrics["news_theme_drift_signal"] is not None
    assert metrics["news_theme_drift_signal"] > 0


def test_news_theme_drift_v2_and_insider_v2_use_revision_support() -> None:
    today = pd.Timestamp.now(tz="UTC").normalize()

    class FakeClient:
        def get_news(self, **kwargs):
            return [
                {
                    "date": (today - pd.Timedelta(days=3)).isoformat(),
                    "title": "Acme raises guidance and announces buyback authorization",
                    "symbols": ["ACME.US"],
                    "sentiment": {"polarity": 0.5},
                },
                {
                    "date": (today - pd.Timedelta(days=40)).isoformat(),
                    "title": "Acme faces dilution concerns after offering",
                    "symbols": ["ACME.US"],
                    "sentiment": {"polarity": -0.4},
                },
            ]

        def get_insider_transactions(self, **kwargs):
            return [
                {
                    "date": (today - pd.Timedelta(days=7)).isoformat(),
                    "transactionCode": "P",
                    "transactionAmount": 10000,
                    "transactionPrice": 10.0,
                    "ownerName": "Jane Doe",
                    "officerTitle": "Chief Executive Officer",
                },
                {
                    "date": (today - pd.Timedelta(days=14)).isoformat(),
                    "transactionCode": "P",
                    "transactionAmount": 5000,
                    "transactionPrice": 10.0,
                    "ownerName": "John Doe",
                    "officerTitle": "Chief Financial Officer",
                },
            ]

    news_metrics = compute_news_theme_drift_metrics(
        FakeClient(),
        "ACME.US",
        alpha_factor_spec="v2",
        revision_support=0.4,
    )
    insider_metrics = compute_insider_conviction_metrics(
        FakeClient(),
        "ACME.US",
        alpha_factor_spec="v2",
        revision_support=0.4,
    )

    assert news_metrics["news_theme_drift_signal"] is not None
    assert news_metrics["news_theme_drift_signal"] > 0
    assert insider_metrics["insider_conviction_signal"] is not None
    assert insider_metrics["insider_conviction_signal"] > 0
