from __future__ import annotations

from copy import deepcopy

import pandas as pd

from eodhd_strategy.advanced_factors import (
    compute_accrual_metrics,
    compute_accrual_quality_metrics,
    compute_beneish_metrics,
    compute_beneish_m_score,
    compute_capital_allocation_quality_metrics,
    compute_compounder_persistence_metrics_from_fundamentals,
    compute_estimate_term_structure_metrics_from_fundamentals,
    compute_investment_restraint_metrics,
    compute_pead_metrics_from_fundamentals,
    compute_price_momentum_metrics_from_history,
    compute_price_momentum_proxy_metrics,
    compute_quality_acceleration_metrics_from_fundamentals,
    compute_recovery_fundamental_metrics,
    compute_revenue_growth_metrics_from_fundamentals,
    compute_revision_impulse_metrics_from_fundamentals,
    compute_sue_metrics_from_fundamentals,
    compute_turnover_day_metrics_from_fundamentals,
    compute_working_capital_stress_metrics,
    passes_sentiment_coverage_gate,
)
from eodhd_strategy.features import compute_news_event_metrics


def _make_earnings_fundamentals() -> dict:
    return {
        "Earnings": {
            "History": {
                "2025-12-31": {
                    "date": "2025-12-31",
                    "reportDate": "2026-02-15",
                    "epsActual": 1.20,
                    "epsEstimate": 1.00,
                    "epsDifference": 0.20,
                    "surprisePercent": 20.0,
                },
                "2025-09-30": {
                    "date": "2025-09-30",
                    "reportDate": "2025-11-15",
                    "epsActual": 1.00,
                    "epsEstimate": 0.92,
                },
                "2025-06-30": {
                    "date": "2025-06-30",
                    "reportDate": "2025-08-15",
                    "epsActual": 0.96,
                    "epsEstimate": 0.90,
                },
                "2025-03-31": {
                    "date": "2025-03-31",
                    "reportDate": "2025-05-15",
                    "epsActual": 0.91,
                    "epsEstimate": 0.87,
                },
                "2024-12-31": {
                    "date": "2024-12-31",
                    "reportDate": "2025-02-15",
                    "epsActual": 0.88,
                    "epsEstimate": 0.84,
                },
                "2024-09-30": {
                    "date": "2024-09-30",
                    "reportDate": "2024-11-15",
                    "epsActual": 0.83,
                    "epsEstimate": 0.80,
                }
            },
            "Trend": {
                "2025-12-31": {
                    "date": "2025-12-31",
                    "earningsEstimateAvg": 1.00,
                    "earningsEstimateNumberOfAnalysts": 5,
                    "epsTrendCurrent": 1.10,
                    "epsTrend7daysAgo": 1.02,
                    "epsTrend30daysAgo": 0.90,
                    "epsRevisionsUpLast7days": 3,
                    "epsRevisionsDownLast7days": 1,
                    "epsRevisionsUpLast30days": 4,
                    "epsRevisionsDownLast30days": 1,
                }
            },
        }
    }


def _make_statement_fundamentals() -> dict:
    return {
        "Financials": {
            "Income_Statement": {
                "yearly": {
                    "2025-12-31": {
                        "date": "2025-12-31",
                        "netIncome": 120.0,
                        "netIncomeFromContinuingOps": 120.0,
                        "totalRevenue": 1000.0,
                        "costOfRevenue": 600.0,
                        "grossProfit": 400.0,
                        "researchDevelopment": 40.0,
                        "sellingGeneralAdministrative": 120.0,
                        "depreciationAndAmortization": 45.0,
                    },
                    "2024-12-31": {
                        "date": "2024-12-31",
                        "netIncome": 90.0,
                        "netIncomeFromContinuingOps": 90.0,
                        "totalRevenue": 900.0,
                        "costOfRevenue": 560.0,
                        "grossProfit": 340.0,
                        "researchDevelopment": 34.0,
                        "sellingGeneralAdministrative": 100.0,
                        "depreciationAndAmortization": 40.0,
                    },
                },
                "quarterly": {
                    "2025-12-31": {"date": "2025-12-31", "netIncome": 40.0, "totalRevenue": 330.0, "costOfRevenue": 190.0, "grossProfit": 140.0},
                    "2025-09-30": {"date": "2025-09-30", "netIncome": 35.0, "totalRevenue": 295.0, "costOfRevenue": 173.0, "grossProfit": 122.0},
                    "2025-06-30": {"date": "2025-06-30", "netIncome": 30.0, "totalRevenue": 270.0, "costOfRevenue": 162.0, "grossProfit": 108.0},
                    "2025-03-31": {"date": "2025-03-31", "netIncome": 25.0, "totalRevenue": 245.0, "costOfRevenue": 151.0, "grossProfit": 94.0},
                    "2024-12-31": {"date": "2024-12-31", "netIncome": 20.0, "totalRevenue": 250.0, "costOfRevenue": 160.0, "grossProfit": 90.0},
                    "2024-09-30": {"date": "2024-09-30", "netIncome": 18.0, "totalRevenue": 235.0, "costOfRevenue": 152.0, "grossProfit": 83.0},
                    "2024-06-30": {"date": "2024-06-30", "netIncome": 16.0, "totalRevenue": 225.0, "costOfRevenue": 147.0, "grossProfit": 78.0},
                    "2024-03-31": {"date": "2024-03-31", "netIncome": 15.0, "totalRevenue": 220.0, "costOfRevenue": 145.0, "grossProfit": 75.0},
                },
            },
            "Balance_Sheet": {
                "yearly": {
                    "2025-12-31": {
                        "date": "2025-12-31",
                        "totalAssets": 1500.0,
                        "totalCurrentAssets": 600.0,
                        "propertyPlantAndEquipmentNet": 320.0,
                        "shortTermInvestments": 40.0,
                        "longTermInvestments": 20.0,
                        "netReceivables": 180.0,
                        "inventory": 92.0,
                        "accountsPayable": 118.0,
                        "totalCurrentLiabilities": 280.0,
                        "totalStockholderEquity": 820.0,
                        "longTermDebt": 200.0,
                        "commonStockSharesOutstanding": 95.0,
                    },
                    "2024-12-31": {
                        "date": "2024-12-31",
                        "totalAssets": 1400.0,
                        "totalCurrentAssets": 580.0,
                        "propertyPlantAndEquipmentNet": 310.0,
                        "shortTermInvestments": 35.0,
                        "longTermInvestments": 15.0,
                        "netReceivables": 150.0,
                        "inventory": 74.0,
                        "accountsPayable": 98.0,
                        "totalCurrentLiabilities": 260.0,
                        "totalStockholderEquity": 770.0,
                        "longTermDebt": 190.0,
                        "commonStockSharesOutstanding": 100.0,
                    },
                },
                "quarterly": {
                    "2025-12-31": {"date": "2025-12-31", "totalAssets": 1500.0, "longTermDebt": 200.0, "commonStockSharesOutstanding": 95.0},
                    "2025-09-30": {"date": "2025-09-30", "totalAssets": 1470.0, "longTermDebt": 205.0, "commonStockSharesOutstanding": 96.0},
                    "2025-06-30": {"date": "2025-06-30", "totalAssets": 1450.0, "longTermDebt": 210.0, "commonStockSharesOutstanding": 97.0},
                    "2025-03-31": {"date": "2025-03-31", "totalAssets": 1430.0, "longTermDebt": 214.0, "commonStockSharesOutstanding": 98.0},
                    "2024-12-31": {"date": "2024-12-31", "totalAssets": 1400.0, "longTermDebt": 220.0, "commonStockSharesOutstanding": 100.0},
                    "2024-09-30": {"date": "2024-09-30", "totalAssets": 1385.0, "longTermDebt": 222.0, "commonStockSharesOutstanding": 101.0},
                    "2024-06-30": {"date": "2024-06-30", "totalAssets": 1370.0, "longTermDebt": 225.0, "commonStockSharesOutstanding": 102.0},
                    "2024-03-31": {"date": "2024-03-31", "totalAssets": 1350.0, "longTermDebt": 230.0, "commonStockSharesOutstanding": 103.0},
                },
            },
            "Cash_Flow": {
                "yearly": {
                    "2025-12-31": {"date": "2025-12-31", "totalCashFromOperatingActivities": 100.0, "capitalExpenditures": -28.0},
                    "2024-12-31": {"date": "2024-12-31", "totalCashFromOperatingActivities": 80.0, "capitalExpenditures": -30.0},
                },
                "quarterly": {
                    "2025-12-31": {"date": "2025-12-31", "totalCashFromOperatingActivities": 30.0, "capitalExpenditures": -8.0},
                    "2025-09-30": {"date": "2025-09-30", "totalCashFromOperatingActivities": 28.0, "capitalExpenditures": -8.0},
                    "2025-06-30": {"date": "2025-06-30", "totalCashFromOperatingActivities": 26.0, "capitalExpenditures": -7.0},
                    "2025-03-31": {"date": "2025-03-31", "totalCashFromOperatingActivities": 24.0, "capitalExpenditures": -7.0},
                    "2024-12-31": {"date": "2024-12-31", "totalCashFromOperatingActivities": 22.0, "capitalExpenditures": -8.0},
                    "2024-09-30": {"date": "2024-09-30", "totalCashFromOperatingActivities": 21.0, "capitalExpenditures": -8.0},
                    "2024-06-30": {"date": "2024-06-30", "totalCashFromOperatingActivities": 20.0, "capitalExpenditures": -7.0},
                },
            },
        }
    }


def test_pead_v2_positive_setup_produces_signal() -> None:
    metrics = compute_pead_metrics_from_fundamentals(
        _make_earnings_fundamentals(),
        min_pead_analysts=3,
        half_life_days=45,
        max_abs_surprise_pct=100.0,
        max_age_days=365,
    )

    assert metrics["pead_has_setup_coverage"] == 1.0
    assert metrics["pead_filter_pass"] == 1.0
    assert metrics["pead_signal"] is not None
    assert metrics["pead_signal"] > 0


def test_pead_v2_negative_revision_fails_gate() -> None:
    fundamentals = _make_earnings_fundamentals()
    trend = fundamentals["Earnings"]["Trend"]["2025-12-31"]
    trend["epsTrendCurrent"] = 0.80
    trend["epsTrend30daysAgo"] = 1.00
    trend["epsRevisionsUpLast7days"] = 0
    trend["epsRevisionsDownLast30days"] = 4

    metrics = compute_pead_metrics_from_fundamentals(
        fundamentals,
        min_pead_analysts=3,
        half_life_days=45,
        max_abs_surprise_pct=100.0,
        max_age_days=365,
    )

    assert metrics["pead_has_setup_coverage"] == 1.0
    assert metrics["pead_filter_pass"] == 0.0
    assert metrics["pead_signal"] is not None
    assert metrics["pead_signal"] < 0


def test_pead_v2_missing_trend_data_stays_neutral() -> None:
    fundamentals = _make_earnings_fundamentals()
    fundamentals["Earnings"]["Trend"] = {}

    metrics = compute_pead_metrics_from_fundamentals(
        fundamentals,
        min_pead_analysts=3,
        half_life_days=45,
        max_abs_surprise_pct=100.0,
        max_age_days=365,
    )

    assert metrics["earnings_surprise_pct"] == 20.0
    assert metrics["pead_has_setup_coverage"] == 0.0
    assert metrics["pead_filter_pass"] == 1.0
    assert metrics["pead_signal"] is None


def test_revision_impulse_positive_setup_produces_signal() -> None:
    metrics = compute_revision_impulse_metrics_from_fundamentals(
        _make_earnings_fundamentals(),
        min_revision_analysts=4,
    )

    assert metrics["revision_impulse_has_coverage"] == 1.0
    assert metrics["revision_impulse_signal"] is not None
    assert metrics["revision_impulse_signal"] > 0
    assert metrics["revision_impulse_breadth"] is not None


def test_revision_impulse_low_coverage_scales_down_signal() -> None:
    fundamentals = _make_earnings_fundamentals()
    fundamentals["Earnings"]["Trend"]["2025-12-31"]["earningsEstimateNumberOfAnalysts"] = 1

    high_coverage = compute_revision_impulse_metrics_from_fundamentals(
        _make_earnings_fundamentals(),
        min_revision_analysts=4,
    )
    low_coverage = compute_revision_impulse_metrics_from_fundamentals(
        fundamentals,
        min_revision_analysts=4,
    )

    assert high_coverage["revision_impulse_signal"] is not None
    assert low_coverage["revision_impulse_signal"] is not None
    assert low_coverage["revision_impulse_signal"] < high_coverage["revision_impulse_signal"]


def test_revision_jerk_rewards_accelerating_estimate_change() -> None:
    accelerating = _make_earnings_fundamentals()
    fading = deepcopy(_make_earnings_fundamentals())

    accelerating["Earnings"]["Trend"]["2025-12-31"].update(
        {
            "epsTrendCurrent": 1.16,
            "epsTrend7daysAgo": 1.06,
            "epsTrend30daysAgo": 0.99,
            "earningsEstimateHigh": 1.22,
            "earningsEstimateLow": 0.98,
        }
    )
    fading["Earnings"]["Trend"]["2025-12-31"].update(
        {
            "epsTrendCurrent": 1.16,
            "epsTrend7daysAgo": 1.14,
            "epsTrend30daysAgo": 1.00,
            "earningsEstimateHigh": 1.20,
            "earningsEstimateLow": 0.99,
        }
    )

    accelerating_metrics = compute_revision_impulse_metrics_from_fundamentals(
        accelerating,
        min_revision_analysts=4,
    )
    fading_metrics = compute_revision_impulse_metrics_from_fundamentals(
        fading,
        min_revision_analysts=4,
    )

    assert accelerating_metrics["revision_jerk_signal"] is not None
    assert fading_metrics["revision_jerk_signal"] is not None
    assert accelerating_metrics["revision_jerk_signal"] > fading_metrics["revision_jerk_signal"]
    assert accelerating_metrics["revision_jerk_has_coverage"] == 1.0


def test_revision_impulse_v2_incorporates_short_interest_divergence_without_changing_legacy() -> None:
    bullish = _make_earnings_fundamentals()
    bullish["SharesStats"] = {
        "SharesOutstanding": 100.0,
        "SharesFloat": 80.0,
        "ShortPercentFloat": 0.01,
    }
    bullish["Technicals"] = {
        "SharesShort": 1.0,
        "SharesShortPriorMonth": 2.0,
        "ShortRatio": 1.0,
        "ShortPercent": 0.01,
    }
    bullish["Earnings"]["Trend"]["2025-12-31"].update(
        {
            "earningsEstimateGrowth": 0.18,
            "epsTrendCurrent": 1.16,
            "epsTrend7daysAgo": 1.06,
            "epsTrend30daysAgo": 0.95,
            "epsRevisionsUpLast7days": 4,
            "epsRevisionsDownLast30days": 0,
        }
    )

    bearish = deepcopy(_make_earnings_fundamentals())
    bearish["SharesStats"] = {
        "SharesOutstanding": 100.0,
        "SharesFloat": 80.0,
        "ShortPercentFloat": 0.20,
    }
    bearish["Technicals"] = {
        "SharesShort": 20.0,
        "SharesShortPriorMonth": 12.0,
        "ShortRatio": 10.0,
        "ShortPercent": 0.20,
    }
    bearish["Earnings"]["Trend"]["2025-12-31"].update(
        {
            "earningsEstimateGrowth": -0.18,
            "epsTrendCurrent": 0.82,
            "epsTrend7daysAgo": 0.92,
            "epsTrend30daysAgo": 1.05,
            "epsRevisionsUpLast7days": 0,
            "epsRevisionsDownLast30days": 4,
        }
    )

    bullish_v2 = compute_revision_impulse_metrics_from_fundamentals(
        bullish,
        min_revision_analysts=4,
        alpha_factor_spec="v2",
    )
    bearish_v2 = compute_revision_impulse_metrics_from_fundamentals(
        bearish,
        min_revision_analysts=4,
        alpha_factor_spec="v2",
    )
    bullish_legacy = compute_revision_impulse_metrics_from_fundamentals(
        bullish,
        min_revision_analysts=4,
        alpha_factor_spec="legacy",
    )
    bearish_legacy = compute_revision_impulse_metrics_from_fundamentals(
        bearish,
        min_revision_analysts=4,
        alpha_factor_spec="legacy",
    )

    assert bullish_v2["revision_short_divergence_component"] is not None
    assert bearish_v2["revision_short_divergence_component"] is not None
    assert bullish_v2["revision_short_divergence_component"] > 0
    assert bearish_v2["revision_short_divergence_component"] < 0
    assert bullish_v2["revision_impulse_signal"] is not None
    assert bearish_v2["revision_impulse_signal"] is not None
    assert bullish_v2["revision_impulse_signal"] > bearish_v2["revision_impulse_signal"]
    assert bullish_legacy["revision_short_divergence_component"] is None
    assert bearish_legacy["revision_short_divergence_component"] is None
    assert bullish_legacy["revision_impulse_signal"] is not None
    assert bearish_legacy["revision_impulse_signal"] is not None
    assert bullish_legacy["revision_impulse_signal"] > bearish_legacy["revision_impulse_signal"]


def test_revision_impulse_v2_uses_breadth_acceleration_without_changing_legacy() -> None:
    positive = _make_earnings_fundamentals()
    negative = deepcopy(_make_earnings_fundamentals())

    positive_trend = positive["Earnings"]["Trend"]["2025-12-31"]
    negative_trend = negative["Earnings"]["Trend"]["2025-12-31"]
    for trend in [positive_trend, negative_trend]:
        trend.update(
            {
                "earningsEstimateGrowth": 0.12,
                "epsTrendCurrent": 1.12,
                "epsTrend7daysAgo": 1.06,
                "epsTrend30daysAgo": 0.98,
                "epsRevisionsUpLast7days": 2,
                "epsRevisionsDownLast30days": 1,
            }
        )

    positive_trend.update(
        {
            "epsRevisionsDownLast7days": 0,
            "epsRevisionsUpLast30days": 1,
        }
    )
    negative_trend.update(
        {
            "epsRevisionsDownLast7days": 2,
            "epsRevisionsUpLast30days": 5,
        }
    )

    positive_v2 = compute_revision_impulse_metrics_from_fundamentals(
        positive,
        min_revision_analysts=4,
        alpha_factor_spec="v2",
    )
    negative_v2 = compute_revision_impulse_metrics_from_fundamentals(
        negative,
        min_revision_analysts=4,
        alpha_factor_spec="v2",
    )
    positive_legacy = compute_revision_impulse_metrics_from_fundamentals(
        positive,
        min_revision_analysts=4,
        alpha_factor_spec="legacy",
    )
    negative_legacy = compute_revision_impulse_metrics_from_fundamentals(
        negative,
        min_revision_analysts=4,
        alpha_factor_spec="legacy",
    )

    assert positive_v2["revision_breadth_7d"] is not None
    assert negative_v2["revision_breadth_7d"] is not None
    assert positive_v2["revision_breadth_acceleration"] is not None
    assert negative_v2["revision_breadth_acceleration"] is not None
    assert positive_v2["revision_breadth_acceleration"] > 0
    assert negative_v2["revision_breadth_acceleration"] < 0
    assert positive_v2["revision_impulse_signal"] is not None
    assert negative_v2["revision_impulse_signal"] is not None
    assert positive_v2["revision_impulse_signal"] > negative_v2["revision_impulse_signal"]
    assert positive_legacy["revision_impulse_signal"] == negative_legacy["revision_impulse_signal"]


def test_revision_impulse_v2_builds_squeeze_convexity_only_with_positive_revision_catalyst() -> None:
    bullish = _make_statement_fundamentals()
    bullish.update(_make_earnings_fundamentals())
    bullish["SharesStats"] = {
        "SharesOutstanding": 95.0,
        "SharesFloat": 75.0,
        "ShortPercentFloat": 0.22,
    }
    bullish["Technicals"] = {
        "ShortRatio": 9.0,
        "SharesShort": 16.5,
        "SharesShortPriorMonth": 14.0,
    }
    bullish["Holders"] = {
        "Institutions": {
            "2025-12-31_a": {"date": "2025-12-31", "totalShares": 18.0},
            "2025-12-31_b": {"date": "2025-12-31", "totalShares": 17.0},
            "2025-12-31_c": {"date": "2025-12-31", "totalShares": 15.0},
            "2025-09-30_a": {"date": "2025-09-30", "totalShares": 9.0},
        }
    }
    bullish["Earnings"]["Trend"]["2025-12-31"].update(
        {
            "earningsEstimateGrowth": 0.16,
            "epsTrendCurrent": 1.18,
            "epsTrend7daysAgo": 1.08,
            "epsTrend30daysAgo": 0.98,
            "epsRevisionsUpLast7days": 4,
            "epsRevisionsDownLast7days": 0,
            "epsRevisionsUpLast30days": 5,
            "epsRevisionsDownLast30days": 1,
        }
    )

    no_catalyst = deepcopy(bullish)
    no_catalyst["Earnings"]["Trend"]["2025-12-31"].update(
        {
            "earningsEstimateGrowth": -0.02,
            "epsTrendCurrent": 0.98,
            "epsTrend7daysAgo": 1.08,
            "epsTrend30daysAgo": 1.00,
            "epsRevisionsUpLast7days": 0,
            "epsRevisionsDownLast7days": 3,
            "epsRevisionsUpLast30days": 1,
            "epsRevisionsDownLast30days": 4,
        }
    )

    bullish_metrics = compute_revision_impulse_metrics_from_fundamentals(
        bullish,
        min_revision_analysts=4,
        alpha_factor_spec="v2",
    )
    no_catalyst_metrics = compute_revision_impulse_metrics_from_fundamentals(
        no_catalyst,
        min_revision_analysts=4,
        alpha_factor_spec="v2",
    )

    assert bullish_metrics["float_absorption_signal"] is not None
    assert no_catalyst_metrics["float_absorption_signal"] is not None
    assert bullish_metrics["float_absorption_signal"] > 0
    assert no_catalyst_metrics["float_absorption_signal"] > 0
    assert bullish_metrics["squeeze_convexity_signal"] is not None
    assert no_catalyst_metrics["squeeze_convexity_signal"] == 0.0
    assert bullish_metrics["squeeze_convexity_signal"] > 0
    assert bullish_metrics["revision_impulse_signal"] is not None
    assert no_catalyst_metrics["revision_impulse_signal"] is not None
    assert bullish_metrics["revision_impulse_signal"] > no_catalyst_metrics["revision_impulse_signal"]


def test_estimate_term_structure_rewards_persistent_positive_revisions() -> None:
    fundamentals = _make_earnings_fundamentals()
    fundamentals["Earnings"]["Trend"]["2025-09-30"] = {
        "date": "2025-09-30",
        "earningsEstimateAvg": 0.95,
        "earningsEstimateNumberOfAnalysts": 5,
        "epsTrendCurrent": 1.00,
        "epsTrend7daysAgo": 0.98,
        "epsTrend30daysAgo": 0.90,
        "epsRevisionsUpLast7days": 3,
        "epsRevisionsDownLast30days": 1,
        "earningsEstimateHigh": 1.06,
        "earningsEstimateLow": 0.90,
    }
    fundamentals["Earnings"]["Trend"]["2025-06-30"] = {
        "date": "2025-06-30",
        "earningsEstimateAvg": 0.90,
        "earningsEstimateNumberOfAnalysts": 4,
        "epsTrendCurrent": 0.92,
        "epsTrend7daysAgo": 0.91,
        "epsTrend30daysAgo": 0.84,
        "epsRevisionsUpLast7days": 2,
        "epsRevisionsDownLast30days": 1,
        "earningsEstimateHigh": 0.98,
        "earningsEstimateLow": 0.86,
    }

    metrics = compute_estimate_term_structure_metrics_from_fundamentals(
        fundamentals,
        min_revision_analysts=4,
    )

    assert metrics["estimate_term_structure_has_coverage"] == 1.0
    assert metrics["estimate_term_structure_signal"] is not None
    assert metrics["estimate_term_structure_signal"] > 0


def test_sue_metrics_standardize_latest_surprise() -> None:
    metrics = compute_sue_metrics_from_fundamentals(_make_earnings_fundamentals())

    assert metrics["sue_has_coverage"] == 1.0
    assert metrics["sue_signal"] is not None
    assert metrics["sue_signal"] > 0
    assert metrics["sue_std_error"] is not None


def test_revenue_growth_metrics_capture_acceleration() -> None:
    metrics = compute_revenue_growth_metrics_from_fundamentals(_make_statement_fundamentals())

    assert metrics["revenue_growth_has_coverage"] == 1.0
    assert metrics["revenue_growth_yoy"] is not None
    assert metrics["revenue_growth_yoy_prev"] is not None
    assert metrics["revenue_acceleration"] is not None
    assert metrics["revenue_acceleration"] > 0


def test_price_momentum_metrics_compute_6m_ex_1m() -> None:
    price_history = [
        {"date": f"2025-01-{day:02d}", "adjusted_close": 100.0 + day}
        for day in range(1, 32)
    ]
    price_history += [
        {"date": f"2025-02-{day:02d}", "adjusted_close": 131.0 + day}
        for day in range(1, 29)
    ]
    price_history += [
        {"date": f"2025-03-{day:02d}", "adjusted_close": 159.0 + day}
        for day in range(1, 32)
    ]
    price_history += [
        {"date": f"2025-04-{day:02d}", "adjusted_close": 190.0 + day}
        for day in range(1, 31)
    ]
    price_history += [
        {"date": f"2025-05-{day:02d}", "adjusted_close": 220.0 + day}
        for day in range(1, 32)
    ]

    metrics = compute_price_momentum_metrics_from_history(price_history)

    assert metrics["price_momentum_has_coverage"] == 1.0
    assert metrics["price_momentum_1m"] is not None
    assert metrics["price_momentum_6m_ex_1m"] is not None
    assert metrics["price_momentum_6m"] is not None


def test_price_momentum_proxy_metrics_build_signal_from_trend_posture() -> None:
    metrics = compute_price_momentum_proxy_metrics(
        price_to_200dma=1.18,
        recency_ratio=0.91,
        distance_from_high=-0.09,
    )

    assert metrics["price_momentum_signal_coverage"] == 1.0
    assert metrics["price_momentum_proxy_used"] == 1.0
    assert metrics["price_momentum_effective_signal"] is not None
    assert metrics["price_momentum_effective_signal"] > 0


def test_sentiment_gate_requires_recent_article_coverage() -> None:
    assert passes_sentiment_coverage_gate(4, 1, -0.50, 3, 3, -0.02) is True
    assert passes_sentiment_coverage_gate(4, 4, -0.50, 3, 3, -0.02) is False


def test_news_event_metrics_extract_structured_signal_from_recent_articles() -> None:
    today = pd.Timestamp.now(tz="UTC").normalize()

    class FakeClient:
        def get_news(self, symbol: str, start_date: str, end_date: str, limit: int):
            return [
                {
                    "date": (today - pd.Timedelta(days=1)).isoformat(),
                    "title": "Acme beats estimates and raises guidance after contract win",
                    "tags": ["EARNINGS SURPRISE", "ESTIMATE REVISIONS"],
                    "symbols": [symbol],
                    "sentiment": {"polarity": 0.7},
                },
                {
                    "date": (today - pd.Timedelta(days=2)).isoformat(),
                    "title": "Acme faces class action lawsuit after product recall",
                    "tags": ["CLASS ACTION"],
                    "symbols": [symbol],
                    "sentiment": {"polarity": -0.5},
                },
            ]

    metrics = compute_news_event_metrics(FakeClient(), "ACME.US", lookback_days=10)

    assert metrics["news_event_signal"] is not None
    assert metrics["news_event_signal"] > 0
    assert metrics["news_article_count_recent"] == 2.0
    assert metrics["news_event_breadth"] >= 2.0
    assert 0.0 <= metrics["news_novelty_score"] <= 1.0
    assert 0.0 <= metrics["news_saturation_score"] <= 1.0


def test_news_event_metrics_compute_positive_shock_against_baseline() -> None:
    today = pd.Timestamp.now(tz="UTC").normalize()

    class FakeClient:
        def get_news(self, symbol: str, start_date: str, end_date: str, limit: int):
            return [
                {
                    "date": (today - pd.Timedelta(days=1)).isoformat(),
                    "title": "Acme wins major contract and raises guidance",
                    "tags": ["contract win", "estimate revisions"],
                    "symbols": [symbol],
                    "sentiment": {"polarity": 0.8},
                },
                {
                    "date": (today - pd.Timedelta(days=3)).isoformat(),
                    "title": "Acme upgraded after product launch momentum",
                    "tags": ["upgrade"],
                    "symbols": [symbol],
                    "sentiment": {"polarity": 0.7},
                },
                {
                    "date": (today - pd.Timedelta(days=15)).isoformat(),
                    "title": "Acme faces soft demand concerns",
                    "tags": ["warning"],
                    "symbols": [symbol],
                    "sentiment": {"polarity": -0.4},
                },
                {
                    "date": (today - pd.Timedelta(days=22)).isoformat(),
                    "title": "Acme sees muted order intake",
                    "tags": ["downgrade"],
                    "symbols": [symbol],
                    "sentiment": {"polarity": -0.3},
                },
            ]

    metrics = compute_news_event_metrics(FakeClient(), "ACME.US", lookback_days=7, alpha_factor_spec="v2")

    assert metrics["news_shock_signal"] is not None
    assert metrics["news_shock_signal"] > 0
    assert metrics["news_shock_has_coverage"] > 0
    assert metrics["news_article_volume_spike"] > 0
    assert metrics["news_baseline_article_count"] > 0


def test_compounder_persistence_signal_handles_partial_statement_subcomponents() -> None:
    metrics = compute_compounder_persistence_metrics_from_fundamentals(_make_statement_fundamentals())

    assert metrics["compounder_persistence_has_coverage"] > 0
    assert metrics["compounder_persistence_signal"] is not None


def test_compounder_persistence_v2_uses_broader_family_coverage() -> None:
    fundamentals = deepcopy(_make_statement_fundamentals())
    fundamentals["General"] = {"Sector": "Technology"}

    metrics = compute_compounder_persistence_metrics_from_fundamentals(
        fundamentals,
        alpha_factor_spec="v2",
        use_intangible_adjustments=True,
    )

    assert metrics["compounder_persistence_has_coverage"] > 0
    assert metrics["compounder_persistence_signal"] is not None
    assert metrics["compounder_persistence_periodicity"] == 1.0


def test_working_capital_stress_penalty_is_capped() -> None:
    fundamentals = deepcopy(_make_statement_fundamentals())
    fundamentals["Financials"]["Income_Statement"]["yearly"]["2025-12-31"]["totalRevenue"] = 1020.0
    fundamentals["Financials"]["Income_Statement"]["yearly"]["2024-12-31"]["totalRevenue"] = 1000.0
    fundamentals["Financials"]["Balance_Sheet"]["yearly"]["2025-12-31"]["netReceivables"] = 380.0
    fundamentals["Financials"]["Balance_Sheet"]["yearly"]["2024-12-31"]["netReceivables"] = 120.0
    fundamentals["Financials"]["Balance_Sheet"]["yearly"]["2025-12-31"]["inventory"] = 220.0
    fundamentals["Financials"]["Balance_Sheet"]["yearly"]["2024-12-31"]["inventory"] = 60.0
    fundamentals["Financials"]["Balance_Sheet"]["yearly"]["2025-12-31"]["accountsPayable"] = 260.0
    fundamentals["Financials"]["Balance_Sheet"]["yearly"]["2024-12-31"]["accountsPayable"] = 70.0
    fundamentals["Financials"]["Cash_Flow"]["yearly"]["2025-12-31"]["totalCashFromOperatingActivities"] = 10.0

    metrics = compute_working_capital_stress_metrics(fundamentals)

    assert metrics["working_capital_stress_has_coverage"] > 0
    assert metrics["working_capital_stress_penalty"] == 0.06


def test_working_capital_stress_v2_penalty_is_capped() -> None:
    fundamentals = deepcopy(_make_statement_fundamentals())
    quarterly_income = fundamentals["Financials"]["Income_Statement"]["quarterly"]
    quarterly_balance = fundamentals["Financials"]["Balance_Sheet"]["quarterly"]
    quarterly_cashflow = fundamentals["Financials"]["Cash_Flow"]["quarterly"]

    quarterly_income["2025-12-31"]["totalRevenue"] = 255.0
    quarterly_income["2025-09-30"]["totalRevenue"] = 250.0
    quarterly_income["2025-06-30"]["totalRevenue"] = 245.0
    quarterly_income["2025-03-31"]["totalRevenue"] = 240.0

    for key, receivables, inventory, payables in [
        ("2025-12-31", 240.0, 180.0, 170.0),
        ("2025-09-30", 210.0, 150.0, 150.0),
        ("2025-06-30", 180.0, 130.0, 135.0),
        ("2025-03-31", 150.0, 115.0, 120.0),
    ]:
        quarterly_balance[key]["netReceivables"] = receivables
        quarterly_balance[key]["inventory"] = inventory
        quarterly_balance[key]["accountsPayable"] = payables
        quarterly_cashflow[key]["totalCashFromOperatingActivities"] = 5.0

    metrics = compute_working_capital_stress_metrics(fundamentals, alpha_factor_spec="v2")

    assert metrics["working_capital_stress_has_coverage"] > 0
    assert 0.0 < metrics["working_capital_stress_penalty"] <= 0.06


def test_investment_restraint_penalizes_empire_building_and_dilution() -> None:
    clean = _make_statement_fundamentals()
    bloated = deepcopy(_make_statement_fundamentals())

    bloated["Financials"]["Balance_Sheet"]["yearly"]["2025-12-31"].update(
        {
            "totalAssets": 1900.0,
            "longTermDebt": 320.0,
            "commonStockSharesOutstanding": 120.0,
            "goodWillAndOtherIntangibleAssets": 260.0,
        }
    )
    bloated["Financials"]["Balance_Sheet"]["yearly"]["2024-12-31"]["goodWillAndOtherIntangibleAssets"] = 40.0
    bloated["Financials"]["Cash_Flow"]["yearly"]["2025-12-31"]["capitalExpenditures"] = -90.0

    clean_metrics = compute_investment_restraint_metrics(clean)
    bloated_metrics = compute_investment_restraint_metrics(bloated)

    assert clean_metrics["investment_restraint_signal"] is not None
    assert bloated_metrics["investment_restraint_signal"] is not None
    assert bloated_metrics["investment_restraint_signal"] < clean_metrics["investment_restraint_signal"]
    assert bloated_metrics["investment_restraint_share_issuance"] is not None


def test_accrual_quality_distinguishes_cash_backed_from_accrual_heavy_earnings() -> None:
    clean = _make_statement_fundamentals()
    dirty = deepcopy(_make_statement_fundamentals())

    dirty["Financials"]["Cash_Flow"]["quarterly"]["2025-12-31"]["totalCashFromOperatingActivities"] = 5.0
    dirty["Financials"]["Cash_Flow"]["quarterly"]["2025-09-30"]["totalCashFromOperatingActivities"] = 4.0
    dirty["Financials"]["Cash_Flow"]["quarterly"]["2025-06-30"]["totalCashFromOperatingActivities"] = 3.0
    dirty["Financials"]["Cash_Flow"]["quarterly"]["2025-03-31"]["totalCashFromOperatingActivities"] = 2.0

    clean_metrics = compute_accrual_quality_metrics(clean)
    dirty_metrics = compute_accrual_quality_metrics(dirty)

    assert clean_metrics["accrual_quality_signal"] is not None
    assert dirty_metrics["accrual_quality_signal"] is not None
    assert dirty_metrics["accrual_quality_signal"] < clean_metrics["accrual_quality_signal"]
    assert dirty_metrics["accrual_quality_cash_conversion"] is not None


def test_quality_acceleration_rewards_improving_business_quality() -> None:
    improving = _make_statement_fundamentals()
    stalling = deepcopy(_make_statement_fundamentals())

    stalling["Financials"]["Income_Statement"]["quarterly"]["2025-12-31"].update(
        {
            "netIncome": 20.0,
            "costOfRevenue": 210.0,
            "grossProfit": 120.0,
        }
    )
    stalling["Financials"]["Cash_Flow"]["quarterly"]["2025-12-31"]["totalCashFromOperatingActivities"] = 10.0
    stalling["Financials"]["Balance_Sheet"]["quarterly"]["2025-12-31"]["netReceivables"] = 145.0
    stalling["Financials"]["Balance_Sheet"]["quarterly"]["2025-12-31"]["inventory"] = 98.0
    stalling["Financials"]["Balance_Sheet"]["quarterly"]["2025-09-30"]["netReceivables"] = 118.0
    stalling["Financials"]["Balance_Sheet"]["quarterly"]["2025-09-30"]["inventory"] = 76.0

    improving_metrics = compute_quality_acceleration_metrics_from_fundamentals(improving)
    stalling_metrics = compute_quality_acceleration_metrics_from_fundamentals(stalling)

    assert improving_metrics["quality_acceleration_signal"] is not None
    assert stalling_metrics["quality_acceleration_signal"] is not None
    assert improving_metrics["quality_acceleration_signal"] > stalling_metrics["quality_acceleration_signal"]
    assert improving_metrics["quality_acceleration_measure_count"] >= 3


def test_v2_turnover_day_signals_penalize_working_capital_deterioration_and_reward_improvement() -> None:
    improving = _make_statement_fundamentals()
    deteriorating = deepcopy(_make_statement_fundamentals())

    improving_quarterly = improving["Financials"]["Balance_Sheet"]["quarterly"]
    deteriorating_quarterly = deteriorating["Financials"]["Balance_Sheet"]["quarterly"]
    for date, receivables, inventory, payables in [
        ("2025-12-31", 52.0, 32.0, 70.0),
        ("2025-09-30", 55.0, 34.0, 69.0),
        ("2025-06-30", 58.0, 36.0, 68.0),
        ("2025-03-31", 60.0, 38.0, 67.0),
        ("2024-12-31", 62.0, 40.0, 66.0),
    ]:
        improving_quarterly[date].update(
            {
                "netReceivables": receivables,
                "inventory": inventory,
                "accountsPayable": payables,
            }
        )
    for date, receivables, inventory, payables in [
        ("2025-12-31", 105.0, 90.0, 95.0),
        ("2025-09-30", 80.0, 65.0, 78.0),
        ("2025-06-30", 65.0, 55.0, 70.0),
        ("2025-03-31", 60.0, 50.0, 65.0),
        ("2024-12-31", 58.0, 48.0, 60.0),
    ]:
        deteriorating_quarterly[date].update(
            {
                "netReceivables": receivables,
                "inventory": inventory,
                "accountsPayable": payables,
            }
        )
    deteriorating["Financials"]["Cash_Flow"]["quarterly"]["2025-12-31"]["totalCashFromOperatingActivities"] = 12.0
    deteriorating["Financials"]["Cash_Flow"]["quarterly"]["2025-09-30"]["totalCashFromOperatingActivities"] = 14.0
    deteriorating["Financials"]["Cash_Flow"]["quarterly"]["2025-06-30"]["totalCashFromOperatingActivities"] = 16.0
    deteriorating["Financials"]["Cash_Flow"]["quarterly"]["2025-03-31"]["totalCashFromOperatingActivities"] = 18.0

    improving_stress = compute_working_capital_stress_metrics(
        improving,
        alpha_factor_spec="v2",
    )
    deteriorating_stress = compute_working_capital_stress_metrics(
        deteriorating,
        alpha_factor_spec="v2",
    )
    improving_accrual = compute_accrual_quality_metrics(
        improving,
        alpha_factor_spec="v2",
    )
    deteriorating_accrual = compute_accrual_quality_metrics(
        deteriorating,
        alpha_factor_spec="v2",
    )
    improving_acceleration = compute_quality_acceleration_metrics_from_fundamentals(
        improving,
        alpha_factor_spec="v2",
    )
    deteriorating_acceleration = compute_quality_acceleration_metrics_from_fundamentals(
        deteriorating,
        alpha_factor_spec="v2",
    )
    deteriorating_turnover = compute_turnover_day_metrics_from_fundamentals(deteriorating)

    assert improving_stress["working_capital_stress_penalty"] is not None
    assert deteriorating_stress["working_capital_stress_penalty"] is not None
    assert deteriorating_stress["working_capital_stress_penalty"] > improving_stress["working_capital_stress_penalty"]
    assert deteriorating_stress["working_capital_receivables_stress"] is not None
    assert deteriorating_stress["working_capital_inventory_stress"] is not None
    assert deteriorating_turnover["cash_conversion_cycle_days"] is not None
    assert deteriorating_turnover["cash_conversion_cycle_days_delta"] is not None
    assert deteriorating_turnover["cash_conversion_cycle_convexity"] is not None
    assert deteriorating_stress["working_capital_cycle_stress"] is not None
    assert improving_stress["working_capital_cycle_stress"] is not None
    assert deteriorating_stress["working_capital_cycle_stress"] > improving_stress["working_capital_cycle_stress"]
    assert improving_accrual["accrual_quality_signal"] is not None
    assert deteriorating_accrual["accrual_quality_signal"] is not None
    assert improving_accrual["accrual_quality_signal"] > deteriorating_accrual["accrual_quality_signal"]
    assert improving_accrual["accrual_quality_cycle_convexity"] is not None
    assert deteriorating_accrual["accrual_quality_cycle_convexity"] is not None
    assert improving_accrual["accrual_quality_cycle_convexity"] < deteriorating_accrual["accrual_quality_cycle_convexity"]
    assert improving_acceleration["quality_acceleration_signal"] is not None
    assert deteriorating_acceleration["quality_acceleration_signal"] is not None
    assert improving_acceleration["quality_acceleration_signal"] > deteriorating_acceleration["quality_acceleration_signal"]


def test_capital_allocation_quality_rewards_fcf_funded_buybacks_and_deleveraging() -> None:
    metrics = compute_capital_allocation_quality_metrics(_make_statement_fundamentals())

    assert metrics["capital_allocation_quality_has_coverage"] > 0
    assert metrics["capital_allocation_quality_signal"] is not None
    assert metrics["capital_allocation_quality_signal"] > 0


def test_capital_allocation_quality_v2_penalizes_debt_funded_buybacks() -> None:
    good_metrics = compute_capital_allocation_quality_metrics(
        _make_statement_fundamentals(),
        alpha_factor_spec="v2",
    )

    bad_fundamentals = deepcopy(_make_statement_fundamentals())
    bad_fundamentals["Financials"]["Balance_Sheet"]["yearly"]["2025-12-31"]["longTermDebt"] = 340.0
    bad_fundamentals["Financials"]["Cash_Flow"]["yearly"]["2025-12-31"]["totalCashFromOperatingActivities"] = 10.0
    bad_fundamentals["Financials"]["Cash_Flow"]["yearly"]["2025-12-31"]["capitalExpenditures"] = -40.0
    bad_fundamentals["Financials"]["Balance_Sheet"]["yearly"]["2025-12-31"]["commonStockSharesOutstanding"] = 92.0

    bad_metrics = compute_capital_allocation_quality_metrics(
        bad_fundamentals,
        alpha_factor_spec="v2",
    )

    assert good_metrics["capital_allocation_quality_signal"] is not None
    assert bad_metrics["capital_allocation_quality_signal"] is not None
    assert bad_metrics["capital_allocation_quality_signal"] < good_metrics["capital_allocation_quality_signal"]


def test_capital_allocation_quality_v2_penalizes_reversal_heavy_buybacks() -> None:
    good_metrics = compute_capital_allocation_quality_metrics(
        _make_statement_fundamentals(),
        alpha_factor_spec="v2",
    )

    reversal_heavy = deepcopy(_make_statement_fundamentals())
    reversal_heavy["Financials"]["Balance_Sheet"]["quarterly"]["2025-12-31"]["commonStockSharesOutstanding"] = 101.0
    reversal_heavy["Financials"]["Balance_Sheet"]["quarterly"]["2025-09-30"]["commonStockSharesOutstanding"] = 98.0
    reversal_heavy["Financials"]["Balance_Sheet"]["quarterly"]["2025-06-30"]["commonStockSharesOutstanding"] = 96.0
    reversal_heavy["Financials"]["Balance_Sheet"]["quarterly"]["2025-03-31"]["commonStockSharesOutstanding"] = 94.0
    reversal_heavy["Financials"]["Balance_Sheet"]["quarterly"]["2024-12-31"]["commonStockSharesOutstanding"] = 100.0
    reversal_heavy["Financials"]["Balance_Sheet"]["yearly"]["2025-12-31"]["longTermDebt"] = 340.0
    reversal_heavy["Financials"]["Cash_Flow"]["yearly"]["2025-12-31"]["totalCashFromOperatingActivities"] = 10.0
    reversal_heavy["Financials"]["Cash_Flow"]["yearly"]["2025-12-31"]["capitalExpenditures"] = -40.0

    reversal_metrics = compute_capital_allocation_quality_metrics(
        reversal_heavy,
        alpha_factor_spec="v2",
    )

    assert good_metrics["capital_allocation_quality_signal"] is not None
    assert reversal_metrics["capital_allocation_quality_signal"] is not None
    assert good_metrics["capital_allocation_buyback_component"] is not None
    assert reversal_metrics["capital_allocation_buyback_component"] is not None
    assert good_metrics["capital_allocation_buyback_component"] > reversal_metrics["capital_allocation_buyback_component"]
    assert good_metrics["capital_allocation_quality_signal"] > reversal_metrics["capital_allocation_quality_signal"]


def test_capital_allocation_quality_v2_penalizes_financing_dependency_stress() -> None:
    healthy = _make_statement_fundamentals()
    healthy_metrics = compute_capital_allocation_quality_metrics(
        healthy,
        alpha_factor_spec="v2",
    )

    stressed = deepcopy(_make_statement_fundamentals())
    stressed["Financials"]["Balance_Sheet"]["yearly"]["2025-12-31"].update(
        {
            "cash": 8.0,
            "shortTermInvestments": 2.0,
            "longTermDebt": 260.0,
            "commonStockSharesOutstanding": 110.0,
        }
    )
    stressed["Financials"]["Balance_Sheet"]["yearly"]["2024-12-31"].update(
        {
            "cash": 18.0,
            "shortTermInvestments": 4.0,
            "longTermDebt": 220.0,
            "commonStockSharesOutstanding": 100.0,
        }
    )
    for date, debt, shares in [
        ("2025-12-31", 280.0, 110.0),
        ("2025-09-30", 270.0, 107.0),
        ("2025-06-30", 260.0, 104.0),
        ("2025-03-31", 250.0, 102.0),
    ]:
        stressed["Financials"]["Balance_Sheet"]["quarterly"][date].update(
            {
                "cash": 8.0,
                "shortTermInvestments": 2.0,
                "longTermDebt": debt,
                "commonStockSharesOutstanding": shares,
            }
        )
    for date, cfo, capex in [
        ("2025-12-31", -6.0, -6.0),
        ("2025-09-30", -5.0, -6.0),
        ("2025-06-30", -4.0, -6.0),
        ("2025-03-31", -3.0, -6.0),
    ]:
        stressed["Financials"]["Cash_Flow"]["quarterly"][date].update(
            {
                "totalCashFromOperatingActivities": cfo,
                "capitalExpenditures": capex,
            }
        )
    stressed["Earnings"] = {
        "Trend": {
            "2025-12-31": {
                "date": "2025-12-31",
                "earningsEstimateAvg": 1.0,
                "earningsEstimateNumberOfAnalysts": 4,
                "epsTrendCurrent": 0.82,
                "epsTrend7daysAgo": 0.90,
                "epsTrend30daysAgo": 1.00,
                "epsRevisionsUpLast7days": 0,
                "epsRevisionsDownLast7days": 2,
                "epsRevisionsUpLast30days": 1,
                "epsRevisionsDownLast30days": 4,
            }
        }
    }

    stressed_metrics = compute_capital_allocation_quality_metrics(
        stressed,
        alpha_factor_spec="v2",
    )

    assert healthy_metrics["financing_dependency_stress"] == 0.0
    assert stressed_metrics["financing_dependency_stress"] is not None
    assert stressed_metrics["financing_dependency_stress"] > 0
    assert stressed_metrics["capital_allocation_financing_dependency_component"] is not None
    assert stressed_metrics["capital_allocation_financing_dependency_component"] < 0
    assert healthy_metrics["capital_allocation_quality_signal"] is not None
    assert stressed_metrics["capital_allocation_quality_signal"] is not None
    assert stressed_metrics["capital_allocation_quality_signal"] < healthy_metrics["capital_allocation_quality_signal"]


def test_recovery_fundamentals_capture_margin_and_leverage_improvement() -> None:
    metrics = compute_recovery_fundamental_metrics(_make_statement_fundamentals())

    assert metrics["recovery_margin_inflection"] is not None
    assert metrics["recovery_leverage_improvement"] is not None


def test_recovery_fundamentals_v2_capture_accrual_improvement() -> None:
    metrics = compute_recovery_fundamental_metrics(
        _make_statement_fundamentals(),
        alpha_factor_spec="v2",
    )

    assert metrics["recovery_margin_inflection"] is not None
    assert metrics["recovery_leverage_improvement"] is not None
    assert metrics["recovery_accrual_improvement"] is not None


def test_beneish_and_accrual_metrics_compute() -> None:
    fundamentals = _make_statement_fundamentals()

    beneish = compute_beneish_m_score(fundamentals)
    accruals = compute_accrual_metrics(fundamentals)

    assert beneish is not None
    assert accruals["accrual_ratio"] is not None
    assert accruals["accrual_volatility"] is not None
    assert accruals["accrual_is_quarterly"] == 1.0


def test_beneish_returns_none_for_pathological_outlier_inputs() -> None:
    fundamentals = deepcopy(_make_statement_fundamentals())
    fundamentals["Financials"]["Income_Statement"]["yearly"]["2025-12-31"]["depreciationAndAmortization"] = -5000.0

    beneish = compute_beneish_m_score(fundamentals)
    metrics = compute_beneish_metrics(fundamentals)

    assert beneish is None
    assert metrics["beneish_data_status"] == "pathological_clipped"
    assert metrics["beneish_is_missing"] == 0.0
    assert metrics["beneish_is_pathological_clipped"] == 1.0


def test_beneish_metrics_mark_missing_history_separately() -> None:
    metrics = compute_beneish_metrics({"Financials": {}})

    assert metrics["beneish_m_score"] is None
    assert metrics["beneish_data_status"] == "missing"
    assert metrics["beneish_is_missing"] == 1.0
    assert metrics["beneish_is_pathological_clipped"] == 0.0


def test_accrual_metrics_fall_back_to_yearly() -> None:
    fundamentals = deepcopy(_make_statement_fundamentals())
    fundamentals["Financials"]["Income_Statement"]["quarterly"] = {
        "2025-12-31": {"date": "2025-12-31", "netIncome": 40.0},
        "2025-09-30": {"date": "2025-09-30", "netIncome": 35.0},
        "2025-06-30": {"date": "2025-06-30", "netIncome": 30.0},
        "2025-03-31": {"date": "2025-03-31", "netIncome": 25.0},
    }
    fundamentals["Financials"]["Income_Statement"]["yearly"].update(
        {
            "2023-12-31": {
                "date": "2023-12-31",
                "netIncome": 80.0,
                "netIncomeFromContinuingOps": 80.0,
                "totalRevenue": 840.0,
                "costOfRevenue": 525.0,
                "sellingGeneralAdministrative": 95.0,
                "depreciationAndAmortization": 36.0,
            },
            "2022-12-31": {
                "date": "2022-12-31",
                "netIncome": 70.0,
                "netIncomeFromContinuingOps": 70.0,
                "totalRevenue": 780.0,
                "costOfRevenue": 500.0,
                "sellingGeneralAdministrative": 90.0,
                "depreciationAndAmortization": 32.0,
            },
            "2021-12-31": {
                "date": "2021-12-31",
                "netIncome": 62.0,
                "netIncomeFromContinuingOps": 62.0,
                "totalRevenue": 730.0,
                "costOfRevenue": 470.0,
                "sellingGeneralAdministrative": 86.0,
                "depreciationAndAmortization": 30.0,
            },
        }
    )
    fundamentals["Financials"]["Balance_Sheet"]["yearly"].update(
        {
            "2023-12-31": {
                "date": "2023-12-31",
                "totalAssets": 1320.0,
                "totalCurrentAssets": 555.0,
                "propertyPlantAndEquipmentNet": 300.0,
                "shortTermInvestments": 30.0,
                "longTermInvestments": 12.0,
                "netReceivables": 140.0,
                "totalCurrentLiabilities": 245.0,
                "longTermDebt": 185.0,
            },
            "2022-12-31": {
                "date": "2022-12-31",
                "totalAssets": 1250.0,
                "totalCurrentAssets": 530.0,
                "propertyPlantAndEquipmentNet": 290.0,
                "shortTermInvestments": 25.0,
                "longTermInvestments": 10.0,
                "netReceivables": 128.0,
                "totalCurrentLiabilities": 230.0,
                "longTermDebt": 175.0,
            },
            "2021-12-31": {
                "date": "2021-12-31",
                "totalAssets": 1190.0,
                "totalCurrentAssets": 505.0,
                "propertyPlantAndEquipmentNet": 275.0,
                "shortTermInvestments": 22.0,
                "longTermInvestments": 8.0,
                "netReceivables": 120.0,
                "totalCurrentLiabilities": 220.0,
                "longTermDebt": 168.0,
            },
        }
    )
    fundamentals["Financials"]["Cash_Flow"]["yearly"].update(
        {
            "2023-12-31": {"date": "2023-12-31", "totalCashFromOperatingActivities": 72.0},
            "2022-12-31": {"date": "2022-12-31", "totalCashFromOperatingActivities": 64.0},
            "2021-12-31": {"date": "2021-12-31", "totalCashFromOperatingActivities": 58.0},
        }
    )

    accruals = compute_accrual_metrics(fundamentals)

    assert accruals["accrual_ratio"] is not None
    assert accruals["accrual_volatility"] is not None
    assert accruals["accrual_is_quarterly"] == 0.0
