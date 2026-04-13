from __future__ import annotations

import pandas as pd

from eodhd_strategy.valuation import build_valuation_report


def _statement_map(rows: list[dict]) -> dict[str, dict]:
    return {row["date"]: row for row in rows}


def _fundamentals_from_rows(
    *,
    income_rows: list[dict],
    balance_rows: list[dict],
    cashflow_rows: list[dict],
) -> dict:
    return {
        "Financials": {
            "Income_Statement": {"yearly": _statement_map(income_rows)},
            "Balance_Sheet": {"yearly": _statement_map(balance_rows)},
            "Cash_Flow": {"yearly": _statement_map(cashflow_rows)},
        }
    }


class FakeClient:
    def __init__(self, payloads: dict[str, dict], price_history: dict[str, list[dict]] | None = None):
        self.payloads = payloads
        self.price_history = price_history or {}
        self.requested: list[str] = []
        self.price_requested: list[str] = []

    def get_fundamentals(self, symbol: str):
        self.requested.append(symbol)
        return self.payloads[symbol]

    def get_price_history(self, symbol: str, from_date: str | None = None, to_date: str | None = None, period: str = "d"):
        self.price_requested.append(symbol)
        return self.price_history.get(symbol, [])


def test_build_valuation_report_produces_band_for_stable_company() -> None:
    stable = _fundamentals_from_rows(
        income_rows=[
            {"date": "2024-12-31", "totalRevenue": 1200, "grossProfit": 720, "operatingIncome": 210, "netIncome": 180},
            {"date": "2023-12-31", "totalRevenue": 1120, "grossProfit": 660, "operatingIncome": 190, "netIncome": 165},
            {"date": "2022-12-31", "totalRevenue": 1040, "grossProfit": 600, "operatingIncome": 170, "netIncome": 150},
            {"date": "2021-12-31", "totalRevenue": 960, "grossProfit": 548, "operatingIncome": 150, "netIncome": 132},
            {"date": "2020-12-31", "totalRevenue": 900, "grossProfit": 504, "operatingIncome": 138, "netIncome": 120},
        ],
        balance_rows=[
            {"date": "2024-12-31", "totalAssets": 1500, "commonStockSharesOutstanding": 100, "totalDebt": 260},
            {"date": "2023-12-31", "totalAssets": 1420, "commonStockSharesOutstanding": 101, "totalDebt": 265},
            {"date": "2022-12-31", "totalAssets": 1360, "commonStockSharesOutstanding": 102, "totalDebt": 270},
            {"date": "2021-12-31", "totalAssets": 1300, "commonStockSharesOutstanding": 103, "totalDebt": 275},
            {"date": "2020-12-31", "totalAssets": 1260, "commonStockSharesOutstanding": 104, "totalDebt": 280},
        ],
        cashflow_rows=[
            {"date": "2024-12-31", "totalCashFromOperatingActivities": 230, "capitalExpenditures": -40},
            {"date": "2023-12-31", "totalCashFromOperatingActivities": 214, "capitalExpenditures": -38},
            {"date": "2022-12-31", "totalCashFromOperatingActivities": 198, "capitalExpenditures": -34},
            {"date": "2021-12-31", "totalCashFromOperatingActivities": 181, "capitalExpenditures": -32},
            {"date": "2020-12-31", "totalCashFromOperatingActivities": 168, "capitalExpenditures": -30},
        ],
    )
    ranked = pd.DataFrame(
        [
            {
                "rank": 1,
                "symbol": "STABLE.US",
                "analysis_symbol": "STABLE.US",
                "company_name": "Stable Co",
                "sector": "Technology",
                "industry": "Software",
                "price_proxy": 30.0,
                "shareholder_yield": 0.03,
                "revenue_growth_yoy": 0.12,
                "revision_impulse_signal": 0.25,
                "estimate_term_structure_signal": 0.18,
                "compounder_persistence_signal": 0.55,
                "accrual_quality_signal": 0.35,
                "capital_allocation_quality_signal": 0.28,
                "investment_restraint_signal": 0.20,
                "return_on_invested_capital": 0.18,
                "forensic_penalty": 0.08,
                "life_cycle_stage": "mature",
                "analysis_currency_code": "USD",
            }
        ]
    )

    client = FakeClient(
        {"STABLE.US": stable},
        price_history={
            "STABLE.US": [
                {"date": "2025-03-28", "adjusted_close": 31.5},
                {"date": "2025-03-31", "adjusted_close": 32.0},
            ]
        },
    )
    report = build_valuation_report(client, ranked, top_n=1)
    row = report.iloc[0]

    assert float(row["fair_value_low"]) < float(row["fair_value_base"]) < float(row["fair_value_high"])
    assert float(row["fair_value_base"]) > float(row["current_price"])
    assert float(row["upside_to_base"]) > 0.0
    assert float(row["valuation_confidence"]) >= 0.5
    assert float(row["current_price"]) == 32.0
    assert row["current_price_source"] == "history_adjusted_close"
    assert row["current_price_as_of_date"] == "2025-03-31"
    assert bool(row["cheap_to_base"]) is True
    assert float(row["valuation_gap_pct"]) < 0.0
    assert float(row["reversion_dependency"]) >= 0.0
    assert float(row["rank_percentile"]) == 1.0
    assert float(row["valuation_percentile"]) == 1.0
    assert float(row["valuation_tension"]) == 0.0
    assert bool(row["valuation_actionable"]) is True
    assert row["valuation_method"] in {
        "normalized_eps_multiple",
        "normalized_fcf_multiple",
        "blended_fcf_eps",
    }
    assert row["valuation_reason_code"] in {
        "eps_anchor",
        "fcf_anchor",
        "blended_medium",
        "blended_high",
    }


def test_build_valuation_report_returns_unavailable_when_history_is_too_thin() -> None:
    thin = _fundamentals_from_rows(
        income_rows=[
            {"date": "2024-12-31", "totalRevenue": 200, "grossProfit": 80, "operatingIncome": 10, "netIncome": 5},
            {"date": "2023-12-31", "totalRevenue": 180, "grossProfit": 70, "operatingIncome": 4, "netIncome": -2},
        ],
        balance_rows=[
            {"date": "2024-12-31", "totalAssets": 400, "commonStockSharesOutstanding": 100, "totalDebt": 150},
            {"date": "2023-12-31", "totalAssets": 380, "commonStockSharesOutstanding": 100, "totalDebt": 145},
        ],
        cashflow_rows=[
            {"date": "2024-12-31", "totalCashFromOperatingActivities": 8, "capitalExpenditures": -12},
            {"date": "2023-12-31", "totalCashFromOperatingActivities": 5, "capitalExpenditures": -10},
        ],
    )
    ranked = pd.DataFrame(
        [
            {
                "rank": 1,
                "symbol": "THIN.US",
                "analysis_symbol": "THIN.US",
                "company_name": "Thin Co",
                "sector": "Healthcare",
                "industry": "Biotechnology",
                "price_proxy": 25.0,
                "shareholder_yield": 0.0,
                "life_cycle_stage": "recovery",
            }
        ]
    )

    report = build_valuation_report(
        FakeClient(
            {"THIN.US": thin},
            price_history={"THIN.US": [{"date": "2025-03-31", "adjusted_close": 25.0}]},
        ),
        ranked,
        top_n=1,
    )
    row = report.iloc[0]

    assert row["valuation_method"] == "unavailable"
    assert row["valuation_reason_code"] == "insufficient_history"
    assert pd.isna(row["fair_value_base"])
    assert float(row["valuation_confidence"]) == 0.0


def test_build_valuation_report_fetches_analysis_symbol_for_cross_listings() -> None:
    fundamentals = _fundamentals_from_rows(
        income_rows=[
            {"date": "2024-12-31", "totalRevenue": 500, "grossProfit": 250, "operatingIncome": 80, "netIncome": 60},
            {"date": "2023-12-31", "totalRevenue": 470, "grossProfit": 232, "operatingIncome": 74, "netIncome": 56},
            {"date": "2022-12-31", "totalRevenue": 440, "grossProfit": 216, "operatingIncome": 68, "netIncome": 50},
        ],
        balance_rows=[
            {"date": "2024-12-31", "totalAssets": 700, "commonStockSharesOutstanding": 50, "totalDebt": 90},
            {"date": "2023-12-31", "totalAssets": 680, "commonStockSharesOutstanding": 50, "totalDebt": 95},
            {"date": "2022-12-31", "totalAssets": 660, "commonStockSharesOutstanding": 50, "totalDebt": 100},
        ],
        cashflow_rows=[
            {"date": "2024-12-31", "totalCashFromOperatingActivities": 88, "capitalExpenditures": -14},
            {"date": "2023-12-31", "totalCashFromOperatingActivities": 82, "capitalExpenditures": -13},
            {"date": "2022-12-31", "totalCashFromOperatingActivities": 76, "capitalExpenditures": -12},
        ],
    )
    client = FakeClient({"AAA.US": fundamentals})
    ranked = pd.DataFrame(
        [
            {
                "rank": 1,
                "symbol": "AAA.XETRA",
                "analysis_symbol": "AAA.US",
                "company_name": "AAA",
                "sector": "Industrials",
                "industry": "Industrial Distribution",
                "price_proxy": 30.0,
                "shareholder_yield": 0.02,
                "analysis_currency_code": "USD",
            }
        ]
    )

    report = build_valuation_report(client, ranked, top_n=1)

    assert client.requested == ["AAA.US"]
    assert client.price_requested == ["AAA.US"]
    assert report.iloc[0]["valuation_symbol"] == "AAA.US"


def test_build_valuation_report_flags_anchor_multiple_outlier_as_unavailable() -> None:
    outlier = _fundamentals_from_rows(
        income_rows=[
            {"date": "2024-12-31", "totalRevenue": 420, "grossProfit": 220, "operatingIncome": 50, "netIncome": 42},
            {"date": "2023-12-31", "totalRevenue": 405, "grossProfit": 210, "operatingIncome": 47, "netIncome": 39},
            {"date": "2022-12-31", "totalRevenue": 390, "grossProfit": 200, "operatingIncome": 44, "netIncome": 36},
            {"date": "2021-12-31", "totalRevenue": 372, "grossProfit": 191, "operatingIncome": 41, "netIncome": 34},
        ],
        balance_rows=[
            {"date": "2024-12-31", "totalAssets": 620, "commonStockSharesOutstanding": 120, "totalDebt": 90},
            {"date": "2023-12-31", "totalAssets": 605, "commonStockSharesOutstanding": 120, "totalDebt": 92},
            {"date": "2022-12-31", "totalAssets": 592, "commonStockSharesOutstanding": 120, "totalDebt": 95},
            {"date": "2021-12-31", "totalAssets": 580, "commonStockSharesOutstanding": 120, "totalDebt": 98},
        ],
        cashflow_rows=[
            {"date": "2024-12-31", "totalCashFromOperatingActivities": 60, "capitalExpenditures": -12},
            {"date": "2023-12-31", "totalCashFromOperatingActivities": 57, "capitalExpenditures": -12},
            {"date": "2022-12-31", "totalCashFromOperatingActivities": 54, "capitalExpenditures": -11},
            {"date": "2021-12-31", "totalCashFromOperatingActivities": 52, "capitalExpenditures": -11},
        ],
    )
    ranked = pd.DataFrame(
        [
            {
                "rank": 1,
                "symbol": "OUTLIER.US",
                "analysis_symbol": "OUTLIER.US",
                "company_name": "Outlier Co",
                "sector": "Technology",
                "industry": "Software",
                "price_proxy": 40.0,
                "shareholder_yield": 0.01,
                "revenue_growth_yoy": 0.08,
                "compounder_persistence_signal": 0.25,
                "accrual_quality_signal": 0.20,
                "capital_allocation_quality_signal": 0.10,
                "investment_restraint_signal": 0.15,
                "return_on_invested_capital": 0.11,
                "forensic_penalty": 0.04,
                "life_cycle_stage": "mature",
                "analysis_currency_code": "USD",
            }
        ]
    )

    report = build_valuation_report(
        FakeClient(
            {"OUTLIER.US": outlier},
            price_history={"OUTLIER.US": [{"date": "2025-03-31", "adjusted_close": 500.0}]},
        ),
        ranked,
        top_n=1,
    )
    row = report.iloc[0]

    assert row["valuation_method"] == "unavailable"
    assert row["valuation_reason_code"] == "anchor_multiple_outlier"
    assert pd.isna(row["fair_value_base"])
    assert float(row["current_anchor_multiple"]) > 80.0
    assert bool(row["extreme_anchor_flag"]) is True
    assert bool(row["valuation_actionable"]) is False
