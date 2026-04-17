"""
FMP Data Mapper

This module translates FMP's raw JSON keys into the exact column names
expected by the ranker. It handles missing keys gracefully with NaN filling
and computes derived metrics like shareholder_yield, gross_profitability,
and adjusted_book_to_market.

The mapper ensures backward compatibility with the existing ranker by
producing DataFrames with the exact column structure that build_ranked_frame
expects.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _safe_divide(numerator: pd.Series, denominator: pd.Series, default: float = np.nan) -> pd.Series:
    """
    Safely divide two series, preserving NaN unless an explicit non-NaN default is supplied.
    """
    result = pd.Series(np.nan, index=numerator.index, dtype=float)
    valid_mask = (denominator.notna()) & (denominator != 0) & (numerator.notna())
    result.loc[valid_mask] = numerator.loc[valid_mask] / denominator.loc[valid_mask]
    if pd.isna(default):
        return result
    return result.fillna(default)


def map_profile_data(profiles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map FMP profile data to ranker-expected columns.
    """
    if profiles_df.empty:
        return pd.DataFrame()

    df = profiles_df.copy()

    required_cols = {
        "symbol": "symbol",
        "market_cap": "market_cap",
        "sector": "sector",
        "industry": "industry",
        "is_actively_trading": "is_actively_trading",
    }

    output = pd.DataFrame()
    for fmp_col, ranker_col in required_cols.items():
        if fmp_col in df.columns:
            output[ranker_col] = df[fmp_col]
        else:
            output[ranker_col] = pd.NA

    optional_cols = {
        "company_name": "company_name",
        "exchange": "exchange",
        "country": "country",
    }

    for fmp_col, ranker_col in optional_cols.items():
        if fmp_col in df.columns:
            output[ranker_col] = df[fmp_col]
        else:
            output[ranker_col] = pd.NA

    return output


def map_employee_count(employee_df: pd.DataFrame) -> pd.DataFrame:
    """Map employee count feed to ranker employee coverage fields."""
    if employee_df.empty or "symbol" not in employee_df.columns:
        return pd.DataFrame()

    df = employee_df.copy()
    if "date" in df.columns:
        df = df.sort_values(["symbol", "date"]).groupby("symbol").last().reset_index()
    else:
        df = df.sort_values("symbol").groupby("symbol").last().reset_index()

    output = pd.DataFrame()
    output["symbol"] = df["symbol"]
    if "full_time_employees" in df.columns:
        output["full_time_employees"] = pd.to_numeric(df["full_time_employees"], errors="coerce")
    else:
        output["full_time_employees"] = np.nan
    return output


def map_scores_data(scores_df: pd.DataFrame) -> pd.DataFrame:
    """Map FMP scores bulk data to ranker score fields (e.g. Piotroski)."""
    if scores_df.empty or "symbol" not in scores_df.columns:
        return pd.DataFrame()

    df = scores_df.copy()
    output = pd.DataFrame()
    output["symbol"] = df["symbol"]

    output["piotroski_score"] = np.nan
    for col in ["piotroskiScore", "piotroski", "piotroski_score"]:
        if col in df.columns:
            output["piotroski_score"] = pd.to_numeric(df[col], errors="coerce")
            break

    return output.sort_values("symbol").groupby("symbol").last().reset_index()


def map_price_history(price_history_df: pd.DataFrame) -> pd.DataFrame:
    """Map historical EOD prices to momentum and trend fields."""
    if price_history_df.empty or "symbol" not in price_history_df.columns:
        return pd.DataFrame()

    df = price_history_df.copy()
    if "date" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()]

    price_col = "adj_close" if "adj_close" in df.columns else "close" if "close" in df.columns else None
    if price_col is None:
        return pd.DataFrame()

    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df[df[price_col].notna()]
    if df.empty:
        return pd.DataFrame()

    rows = []
    for symbol, g in df.sort_values("date").groupby("symbol"):
        prices = g[price_col].astype(float).reset_index(drop=True)
        latest = prices.iloc[-1] if len(prices) >= 1 else np.nan
        lag_21 = prices.iloc[-22] if len(prices) >= 22 else np.nan
        lag_126 = prices.iloc[-127] if len(prices) >= 127 else np.nan
        latest_200dma = prices.iloc[-200:].mean() if len(prices) >= 200 else np.nan
        high_52w = prices.iloc[-252:].max() if len(prices) >= 252 else prices.max()

        mom_1m = (latest / lag_21 - 1.0) if pd.notna(latest) and pd.notna(lag_21) and lag_21 != 0 else np.nan
        mom_6m = (latest / lag_126 - 1.0) if pd.notna(latest) and pd.notna(lag_126) and lag_126 != 0 else np.nan
        mom_6m_ex_1m = mom_6m - mom_1m if pd.notna(mom_6m) and pd.notna(mom_1m) else np.nan

        rows.append(
            {
                "symbol": symbol,
                "price_momentum_1m": mom_1m,
                "price_momentum_6m": mom_6m,
                "price_momentum_6m_ex_1m": mom_6m_ex_1m,
                "price_momentum_effective_signal": mom_6m_ex_1m,
                "price_momentum_has_coverage": float(pd.notna(mom_6m_ex_1m)),
                "price_momentum_signal_coverage": float(pd.notna(mom_6m_ex_1m)),
                "price_momentum_proxy_used": 0.0,
                "52_week_high": high_52w,
                "200_day_ma": latest_200dma,
            }
        )

    return pd.DataFrame(rows)


def map_financial_statements(
    income_df: pd.DataFrame,
    balance_df: pd.DataFrame,
    cashflow_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    financial_period: str = "annual",
) -> pd.DataFrame:
    """
    Map FMP financial statements to ranker-expected fundamental columns.
    """
    if all(df.empty for df in [income_df, balance_df, cashflow_df]):
        logger.warning("All financial statement DataFrames are empty")
        return pd.DataFrame()

    def get_latest_records(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        if "date" in df.columns:
            df_local = df.copy()
            date_series = df_local["date"]
            if isinstance(date_series, pd.DataFrame):
                date_series = date_series.iloc[:, 0]
            df_local["date"] = pd.to_datetime(date_series, errors="coerce")
            return df_local.sort_values("date").groupby("symbol").last().reset_index()
        return df

    income_latest = get_latest_records(income_df)
    balance_latest = get_latest_records(balance_df)
    cashflow_latest = get_latest_records(cashflow_df)

    def safe_get_column(df: pd.DataFrame, col: str) -> pd.Series:
        """Extract column as Series, handling duplicate column names."""
        if col not in df.columns:
            return pd.Series([np.nan] * len(df), index=df.index)
        result = df[col]
        if isinstance(result, pd.DataFrame):
            result = result.iloc[:, 0]
        return result

    if not income_latest.empty:
        df = income_latest.copy()
    else:
        df = pd.DataFrame(columns=["symbol"])

    if not balance_latest.empty:
        df = df.merge(balance_latest, on="symbol", how="outer", suffixes=("", "_balance"))

    if not cashflow_latest.empty:
        df = df.merge(cashflow_latest, on="symbol", how="outer", suffixes=("", "_cashflow"))

    profile_cols = ["symbol"]
    for col in ["market_cap", "lastAnnualDividend", "lastDividend", "price"]:
        if not profiles_df.empty and col in profiles_df.columns:
            profile_cols.append(col)
    if not profiles_df.empty:
        df = df.merge(profiles_df[list(dict.fromkeys(profile_cols))], on="symbol", how="left")

    output = pd.DataFrame(index=df.index)
    output["symbol"] = df["symbol"]

    income_mapping = {
        "total_revenue": ["total_revenue", "revenue", "totalRevenue"],
        "gross_profit": ["gross_profit", "grossProfit"],
        "operating_income": ["operating_income", "operatingIncome"],
        "net_income": ["net_income", "netIncome"],
        "ebitda": ["ebitda", "ebitdaValue"],
        "research_development": ["research_development", "researchAndDevelopmentExpenses", "rd_expense"],
        "rd_expenses": ["rd_expenses", "rd_expense", "researchAndDevelopmentExpenses"],
        "sga_expenses": ["sga_expenses", "sellingGeneralAndAdministrativeExpenses"],
    }

    for ranker_col, candidates in income_mapping.items():
        series = None
        for candidate in candidates:
            candidate_series = pd.to_numeric(safe_get_column(df, candidate), errors="coerce")
            if candidate_series.notna().any():
                series = candidate_series
                break
        output[ranker_col] = series if series is not None else np.nan

    balance_mapping = {
        "total_assets": ["total_assets", "totalAssets"],
        "total_liabilities": ["total_liabilities", "totalLiabilities"],
        "total_stockholders_equity": ["total_stockholders_equity", "totalStockholdersEquity"],
        "shareholders_equity": ["shareholders_equity", "total_stockholders_equity", "totalStockholdersEquity"],
        "shares_outstanding": ["shares_outstanding", "commonStockSharesOutstanding", "weightedAverageShsOut"],
        "net_receivables": ["net_receivables", "netReceivables"],
        "inventory": ["inventory"],
        "account_payables": ["account_payables", "accountPayables"],
        "cash_and_equivalents": ["cash_and_equivalents", "cashAndCashEquivalents"],
        "intangible_assets": ["intangible_assets", "intangibleAssets"],
        "goodwill": ["goodwill", "goodWill"],
    }

    for ranker_col, candidates in balance_mapping.items():
        series = None
        for candidate in candidates:
            candidate_series = pd.to_numeric(safe_get_column(df, candidate), errors="coerce")
            if candidate_series.notna().any():
                series = candidate_series
                break
        output[ranker_col] = series if series is not None else np.nan

    cashflow_mapping = {
        "operating_cash_flow": ["operating_cash_flow", "operatingCashFlow", "netCashProvidedByOperatingActivities"],
        "capital_expenditure": ["capital_expenditure", "capitalExpenditure"],
        "free_cash_flow": ["free_cash_flow", "freeCashFlow"],
        "dividends_paid": ["dividends_paid", "dividendsPaid", "commonDividendsPaid", "netDividendsPaid"],
        "stock_issued": ["stock_issued", "stockIssued", "commonStockIssued", "netCommonStockIssuance"],
        "stock_repurchased": ["stock_repurchased", "stockRepurchased", "commonStockRepurchased"],
    }

    for ranker_col, candidates in cashflow_mapping.items():
        series = None
        for candidate in candidates:
            candidate_series = pd.to_numeric(safe_get_column(df, candidate), errors="coerce")
            if candidate_series.notna().any():
                series = candidate_series
                break
        output[ranker_col] = series if series is not None else np.nan

    market_cap_col = safe_get_column(df, "market_cap")
    if market_cap_col.notna().any():
        output["market_cap"] = pd.to_numeric(market_cap_col, errors="coerce")
    else:
        output["market_cap"] = np.nan

    output["gross_profitability"] = _safe_divide(
        output["gross_profit"],
        output["total_assets"],
        default=np.nan,
    )
    output["reported_gross_profitability"] = output["gross_profitability"]

    equity = output["shareholders_equity"].fillna(output["total_stockholders_equity"])
    output["reported_book_to_market"] = _safe_divide(
        equity,
        output["market_cap"],
        default=np.nan,
    )

    intangibles = pd.concat([output["intangible_assets"], output["goodwill"]], axis=1).sum(axis=1, min_count=1)
    adjusted_equity = equity - intangibles
    output["adjusted_book_to_market"] = _safe_divide(
        adjusted_equity,
        output["market_cap"],
        default=np.nan,
    )

    last_div = safe_get_column(df, "lastAnnualDividend").fillna(
        safe_get_column(df, "lastDividend")
    )
    price_for_yield = safe_get_column(df, "price").replace(0, np.nan)
    computed_div_yield = _safe_divide(
        pd.to_numeric(last_div, errors="coerce"),
        pd.to_numeric(price_for_yield, errors="coerce"),
        default=np.nan,
    )
    output["dividend_yield"] = computed_div_yield
    output["safe_dividend_yield"] = computed_div_yield

    stock_repurchased_abs = output["stock_repurchased"].abs()
    output["buyback_yield"] = _safe_divide(
        stock_repurchased_abs,
        output["market_cap"],
        default=np.nan,
    )

    output["shareholder_yield"] = pd.concat(
        [output["dividend_yield"], output["buyback_yield"]],
        axis=1,
    ).sum(axis=1, min_count=1)

    estimated_dividends_paid = pd.to_numeric(last_div, errors="coerce") * pd.to_numeric(
        output["shares_outstanding"], errors="coerce"
    )
    dividends_for_payout = output["dividends_paid"].abs().fillna(estimated_dividends_paid.abs())
    output["payout_ratio"] = _safe_divide(
        dividends_for_payout,
        output["net_income"],
        default=np.nan,
    )
    output["dividend_safety_pass"] = (
        (output["payout_ratio"].notna()) & (output["payout_ratio"] <= 0.85)
    ).astype(float)

    output["reported_return_on_assets"] = _safe_divide(
        output["net_income"],
        output["total_assets"],
        default=np.nan,
    )

    output["reported_return_on_invested_capital"] = output["reported_return_on_assets"]

    output["rd_expense_ratio"] = _safe_divide(
        output["research_development"].fillna(output["rd_expenses"]),
        output["total_revenue"],
        default=np.nan,
    )

    output["intangible_adjustment_eligible"] = (
        (output["rd_expense_ratio"].fillna(0) >= 0.02) |
        (output["intangible_assets"].fillna(0) > 0)
    ).astype(float)

    rd_capitalized = pd.to_numeric(
        output["research_development"].fillna(output["rd_expenses"]),
        errors="coerce",
    )
    sga_capitalized = pd.to_numeric(output["sga_expenses"], errors="coerce") * 0.30

    intangible_adjusted_assets = (
        pd.to_numeric(output["total_assets"], errors="coerce")
        + rd_capitalized.fillna(0.0)
        + sga_capitalized.fillna(0.0)
    )

    output["intangible_adjusted_gross_profitability"] = _safe_divide(
        output["gross_profit"],
        intangible_adjusted_assets,
        default=np.nan,
    )

    intangible_adjusted_equity = (
        pd.to_numeric(equity, errors="coerce")
        + rd_capitalized.fillna(0.0)
        + sga_capitalized.fillna(0.0)
    )

    output["intangible_adjusted_book_to_market"] = _safe_divide(
        intangible_adjusted_equity,
        output["market_cap"],
        default=np.nan,
    )

    output["intangible_adjusted_return_on_assets"] = _safe_divide(
        output["net_income"],
        intangible_adjusted_assets,
        default=np.nan,
    )

    output["intangible_adjusted_return_on_invested_capital"] = output["intangible_adjusted_return_on_assets"]

    output["intangible_adjustment_applied"] = (
        (output["intangible_adjustment_eligible"].fillna(0).astype(bool)) &
        (
            output["intangible_adjusted_gross_profitability"].notna() |
            output["intangible_adjusted_book_to_market"].notna()
        )
    ).astype(float)

    output["gross_profitability"] = np.where(
        output["intangible_adjustment_applied"] == 1,
        output["intangible_adjusted_gross_profitability"],
        output["gross_profitability"],
    )

    output["adjusted_book_to_market"] = np.where(
        output["intangible_adjustment_applied"] == 1,
        output["intangible_adjusted_book_to_market"],
        output["adjusted_book_to_market"],
    )

    output["return_on_assets"] = np.where(
        output["intangible_adjustment_applied"] == 1,
        output["intangible_adjusted_return_on_assets"],
        output["reported_return_on_assets"],
    )

    output["return_on_invested_capital"] = np.where(
        output["intangible_adjustment_applied"] == 1,
        output["intangible_adjusted_return_on_invested_capital"],
        output["reported_return_on_invested_capital"],
    )

    output["revenue_growth_yoy"] = np.nan
    output["revenue_growth_yoy_prev"] = np.nan
    output["revenue_acceleration"] = np.nan
    output["revenue_growth_has_coverage"] = np.nan

    turnover_cols = [
        "receivables_days", "receivables_days_prev", "receivables_days_delta",
        "inventory_days", "inventory_days_prev", "inventory_days_delta",
        "payables_days", "payables_days_prev", "payables_days_delta",
        "cash_conversion_cycle_days", "cash_conversion_cycle_days_prev",
        "cash_conversion_cycle_days_delta", "cash_conversion_cycle_convexity",
    ]
    for col in turnover_cols:
        output[col] = np.nan

    output["accrual_ratio"] = np.nan
    output["accrual_volatility"] = np.nan
    output["accrual_measure_count"] = np.nan

    if not income_df.empty and "symbol" in income_df.columns and "date" in income_df.columns:
        inc_hist = income_df.copy()
        inc_hist["date"] = pd.to_datetime(inc_hist["date"], errors="coerce")
        rev_col = None
        cogs_col = None
        for candidate in ["total_revenue", "revenue", "totalRevenue"]:
            if candidate in inc_hist.columns:
                rev_col = candidate
                break
        for candidate in ["cost_of_revenue", "costOfRevenue"]:
            if candidate in inc_hist.columns:
                cogs_col = candidate
                break

        if rev_col is not None:
            inc_hist["revenue_val"] = pd.to_numeric(inc_hist[rev_col], errors="coerce")
            if cogs_col is not None:
                inc_hist["cogs_val"] = pd.to_numeric(inc_hist[cogs_col], errors="coerce")
            else:
                inc_hist["cogs_val"] = np.nan

            growth_rows = []
            period_key = str(financial_period).lower()
            for symbol, g in inc_hist.sort_values("date").groupby("symbol"):
                vals = g["revenue_val"].dropna().astype(float).tolist()
                yoy = np.nan
                yoy_prev = np.nan

                if period_key == "annual":
                    if len(vals) >= 2:
                        latest = vals[-1]
                        prev = vals[-2]
                        yoy = (latest - prev) / abs(prev) if prev != 0 else np.nan
                    if len(vals) >= 3:
                        prev = vals[-2]
                        prev2 = vals[-3]
                        yoy_prev = (prev - prev2) / abs(prev2) if prev2 != 0 else np.nan
                else:
                    if len(vals) >= 5:
                        latest = vals[-1]
                        lag4 = vals[-5]
                        yoy = (latest - lag4) / abs(lag4) if lag4 != 0 else np.nan
                    if len(vals) >= 6:
                        prev_q = vals[-2]
                        prev_q_lag4 = vals[-6]
                        yoy_prev = (prev_q - prev_q_lag4) / abs(prev_q_lag4) if prev_q_lag4 != 0 else np.nan

                accel = yoy - yoy_prev if pd.notna(yoy) and pd.notna(yoy_prev) else np.nan
                growth_rows.append((symbol, yoy, yoy_prev, accel, float(pd.notna(yoy))))

            growth_df = pd.DataFrame(
                growth_rows,
                columns=[
                    "symbol",
                    "revenue_growth_yoy",
                    "revenue_growth_yoy_prev",
                    "revenue_acceleration",
                    "revenue_growth_has_coverage",
                ],
            )
            output = output.merge(growth_df, on="symbol", how="left", suffixes=("", "_growth"))
            for col in [
                "revenue_growth_yoy",
                "revenue_growth_yoy_prev",
                "revenue_acceleration",
                "revenue_growth_has_coverage",
            ]:
                gcol = f"{col}_growth"
                if gcol in output.columns:
                    output[col] = output[col].combine_first(output[gcol])
                    output.drop(columns=[gcol], inplace=True)

        if not balance_df.empty and "symbol" in balance_df.columns and "date" in balance_df.columns:
            bal_hist = balance_df.copy()
            bal_hist["date"] = pd.to_datetime(bal_hist["date"], errors="coerce")

            recv_col = "net_receivables" if "net_receivables" in bal_hist.columns else "netReceivables" if "netReceivables" in bal_hist.columns else None
            inv_col = "inventory" if "inventory" in bal_hist.columns else None
            pay_col = "account_payables" if "account_payables" in bal_hist.columns else "accountPayables" if "accountPayables" in bal_hist.columns else None

            period_days = 365.0 if str(financial_period).lower() == "annual" else 90.0

            turnover_rows = []
            for symbol, g in inc_hist.sort_values("date").groupby("symbol"):
                g2 = g.sort_values("date")
                if len(g2) < 2:
                    continue
                latest_row = g2.iloc[-1]
                prev_row = g2.iloc[-2]

                b = bal_hist[bal_hist["symbol"] == symbol].sort_values("date")
                if len(b) < 2 or recv_col is None:
                    continue
                b_latest = b.iloc[-1]
                b_prev = b.iloc[-2]

                latest_rev = pd.to_numeric(latest_row.get("revenue_val", np.nan), errors="coerce")
                prev_rev = pd.to_numeric(prev_row.get("revenue_val", np.nan), errors="coerce")
                latest_cogs = pd.to_numeric(latest_row.get("cogs_val", np.nan), errors="coerce")
                prev_cogs = pd.to_numeric(prev_row.get("cogs_val", np.nan), errors="coerce")

                recv_latest = pd.to_numeric(b_latest.get(recv_col, np.nan), errors="coerce")
                recv_prev = pd.to_numeric(b_prev.get(recv_col, np.nan), errors="coerce")
                inv_latest = pd.to_numeric(b_latest.get(inv_col, np.nan), errors="coerce") if inv_col else np.nan
                inv_prev = pd.to_numeric(b_prev.get(inv_col, np.nan), errors="coerce") if inv_col else np.nan
                pay_latest = pd.to_numeric(b_latest.get(pay_col, np.nan), errors="coerce") if pay_col else np.nan
                pay_prev = pd.to_numeric(b_prev.get(pay_col, np.nan), errors="coerce") if pay_col else np.nan

                rec_days = period_days * recv_latest / latest_rev if pd.notna(recv_latest) and pd.notna(latest_rev) and latest_rev != 0 else np.nan
                rec_days_prev = period_days * recv_prev / prev_rev if pd.notna(recv_prev) and pd.notna(prev_rev) and prev_rev != 0 else np.nan
                inv_days = period_days * inv_latest / latest_cogs if pd.notna(inv_latest) and pd.notna(latest_cogs) and latest_cogs != 0 else np.nan
                inv_days_prev = period_days * inv_prev / prev_cogs if pd.notna(inv_prev) and pd.notna(prev_cogs) and prev_cogs != 0 else np.nan
                pay_days = period_days * pay_latest / latest_cogs if pd.notna(pay_latest) and pd.notna(latest_cogs) and latest_cogs != 0 else np.nan
                pay_days_prev = period_days * pay_prev / prev_cogs if pd.notna(pay_prev) and pd.notna(prev_cogs) and prev_cogs != 0 else np.nan

                ccc = rec_days + inv_days - pay_days if pd.notna(rec_days) and pd.notna(inv_days) and pd.notna(pay_days) else np.nan
                ccc_prev = rec_days_prev + inv_days_prev - pay_days_prev if pd.notna(rec_days_prev) and pd.notna(inv_days_prev) and pd.notna(pay_days_prev) else np.nan

                turnover_rows.append(
                    {
                        "symbol": symbol,
                        "receivables_days": rec_days,
                        "receivables_days_prev": rec_days_prev,
                        "receivables_days_delta": rec_days - rec_days_prev if pd.notna(rec_days) and pd.notna(rec_days_prev) else np.nan,
                        "inventory_days": inv_days,
                        "inventory_days_prev": inv_days_prev,
                        "inventory_days_delta": inv_days - inv_days_prev if pd.notna(inv_days) and pd.notna(inv_days_prev) else np.nan,
                        "payables_days": pay_days,
                        "payables_days_prev": pay_days_prev,
                        "payables_days_delta": pay_days - pay_days_prev if pd.notna(pay_days) and pd.notna(pay_days_prev) else np.nan,
                        "cash_conversion_cycle_days": ccc,
                        "cash_conversion_cycle_days_prev": ccc_prev,
                        "cash_conversion_cycle_days_delta": ccc - ccc_prev if pd.notna(ccc) and pd.notna(ccc_prev) else np.nan,
                        "cash_conversion_cycle_convexity": np.nan,
                    }
                )

            if turnover_rows:
                turnover_df = pd.DataFrame(turnover_rows)
                output = output.merge(turnover_df, on="symbol", how="left", suffixes=("", "_turn"))
                for col in turnover_cols:
                    tcol = f"{col}_turn"
                    if tcol in output.columns:
                        output[col] = output[col].combine_first(output[tcol])
                        output.drop(columns=[tcol], inplace=True)

    if not cashflow_df.empty and "symbol" in cashflow_df.columns and "date" in cashflow_df.columns and not income_df.empty:
        cf_hist = cashflow_df.copy()
        cf_hist["date"] = pd.to_datetime(cf_hist["date"], errors="coerce")
        cfo_col = "operating_cash_flow" if "operating_cash_flow" in cf_hist.columns else "operatingCashFlow" if "operatingCashFlow" in cf_hist.columns else "netCashProvidedByOperatingActivities" if "netCashProvidedByOperatingActivities" in cf_hist.columns else None
        ni_col = "net_income" if "net_income" in income_df.columns else "netIncome" if "netIncome" in income_df.columns else None
        ta_col = "total_assets" if "total_assets" in balance_df.columns else "totalAssets" if "totalAssets" in balance_df.columns else None
        if cfo_col and ni_col and ta_col and not balance_df.empty:
            inc_hist = income_df.copy()
            inc_hist["date"] = pd.to_datetime(inc_hist["date"], errors="coerce")
            bal_hist = balance_df.copy()
            bal_hist["date"] = pd.to_datetime(bal_hist["date"], errors="coerce")
            accr_rows = []
            for symbol in inc_hist["symbol"].dropna().unique():
                i = inc_hist[inc_hist["symbol"] == symbol].sort_values("date")
                c = cf_hist[cf_hist["symbol"] == symbol].sort_values("date")
                b = bal_hist[bal_hist["symbol"] == symbol].sort_values("date")
                i_sel = i[["symbol", "date", ni_col]].rename(columns={ni_col: "__ni__"})
                c_sel = c[["symbol", "date", cfo_col]].rename(columns={cfo_col: "__cfo__"})
                b_sel = b[["symbol", "date", ta_col]].rename(columns={ta_col: "__ta__"})
                merged_hist = i_sel.merge(c_sel, on=["symbol", "date"], how="inner")
                merged_hist = merged_hist.merge(b_sel, on=["symbol", "date"], how="inner")
                if merged_hist.empty:
                    continue

                def _to_1d_numeric(val):
                    if isinstance(val, pd.DataFrame):
                        val = val.iloc[:, 0]
                    return pd.to_numeric(val, errors="coerce")

                ni = _to_1d_numeric(merged_hist["__ni__"])
                cfo = _to_1d_numeric(merged_hist["__cfo__"])
                ta = _to_1d_numeric(merged_hist["__ta__"]).replace(0, np.nan)
                accrual_ratio_series = (ni - cfo) / ta
                accr_rows.append(
                    {
                        "symbol": symbol,
                        "accrual_ratio": accrual_ratio_series.iloc[-1] if len(accrual_ratio_series) else np.nan,
                        "accrual_volatility": accrual_ratio_series.std() if len(accrual_ratio_series) >= 2 else np.nan,
                        "accrual_measure_count": float(accrual_ratio_series.notna().sum()),
                    }
                )
            if accr_rows:
                accr_df = pd.DataFrame(accr_rows)
                output = output.merge(accr_df, on="symbol", how="left", suffixes=("", "_accr"))
                for col in ["accrual_ratio", "accrual_volatility", "accrual_measure_count"]:
                    acol = f"{col}_accr"
                    if acol in output.columns:
                        output[col] = output[col].combine_first(output[acol])
                        output.drop(columns=[acol], inplace=True)

    return output


def map_beneish_components(
    income_df: pd.DataFrame,
    balance_df: pd.DataFrame,
    cashflow_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Map Beneish M-Score components from financial statement data.
    """
    if all(df.empty for df in [income_df, balance_df, cashflow_df]):
        return pd.DataFrame()

    def get_last_n_periods_with_rank(df: pd.DataFrame, n: int = 2) -> pd.DataFrame:
        if df.empty or "date" not in df.columns:
            return df
        df_local = df.copy()
        date_series = df_local["date"]
        if isinstance(date_series, pd.DataFrame):
            date_series = date_series.iloc[:, 0]
        df_local["date"] = pd.to_datetime(date_series, errors="coerce")
        df_sorted = df_local.sort_values("date").groupby("symbol").tail(n).reset_index()
        df_sorted["period_rank"] = df_sorted.groupby("symbol").cumcount(ascending=False) + 1
        return df_sorted

    income_2p = get_last_n_periods_with_rank(income_df, 2)
    balance_2p = get_last_n_periods_with_rank(balance_df, 2)
    cashflow_2p = get_last_n_periods_with_rank(cashflow_df, 2)

    df = income_2p.merge(balance_2p, on=["symbol", "period_rank"], how="outer", suffixes=("", "_balance"))
    df = df.merge(cashflow_2p, on=["symbol", "period_rank"], how="outer", suffixes=("", "_cashflow"))

    if df.empty:
        return pd.DataFrame()

    current = df[df["period_rank"] == 1].copy() if 1 in df["period_rank"].values else pd.DataFrame()
    prior = df[df["period_rank"] == 2].copy() if 2 in df["period_rank"].values else pd.DataFrame()

    output = pd.DataFrame()
    output["symbol"] = current["symbol"]

    beneish_components = ["dsri", "gmi", "aqi", "sgi", "depi", "sgai", "lvgi", "tata"]
    for comp in beneish_components:
        output[comp] = np.nan

    def _safe_col(df_: pd.DataFrame, col: str, suffix: str = "") -> pd.Series:
        candidates = [
            f"{col}{suffix}",
            f"{col}_cashflow{suffix}",
            f"{col}_balance{suffix}",
        ]
        if suffix:
            candidates += [f"{col}_cashflow", f"{col}_balance", col]
        for name in candidates:
            if name in df_.columns:
                val = df_[name]
                if isinstance(val, pd.DataFrame):
                    val = val.iloc[:, 0]
                return pd.to_numeric(val, errors="coerce")
        return pd.Series(np.nan, index=df_.index, dtype=float)

    if not prior.empty:
        merged = current.merge(prior, on="symbol", how="left", suffixes=("_current", "_prior"))

        def mc(col): return _safe_col(merged, col, "_current")
        def mp(col): return _safe_col(merged, col, "_prior")

        receivables_current = mc("net_receivables")
        receivables_prior = mp("net_receivables")
        sales_current = mc("revenue")
        sales_prior = mp("revenue")
        dsr_current = _safe_divide(receivables_current, sales_current, default=np.nan)
        dsr_prior = _safe_divide(receivables_prior, sales_prior, default=np.nan)
        output["dsri"] = _safe_divide(dsr_current, dsr_prior, default=np.nan)

        gross_profit_current = mc("gross_profit")
        gross_profit_prior = mp("gross_profit")
        gm_current = _safe_divide(gross_profit_current, sales_current, default=np.nan)
        gm_prior = _safe_divide(gross_profit_prior, sales_prior, default=np.nan)
        output["gmi"] = _safe_divide(gm_prior, gm_current, default=np.nan)

        output["sgi"] = _safe_divide(sales_current, sales_prior, default=np.nan)

        capex_current = mc("capital_expenditure").abs()
        capex_prior = mp("capital_expenditure").abs()
        ppe_current = mc("total_assets")
        ppe_prior = mp("total_assets")
        depi_current = _safe_divide(capex_current, ppe_current + capex_current, default=np.nan)
        depi_prior = _safe_divide(capex_prior, ppe_prior + capex_prior, default=np.nan)
        output["depi"] = _safe_divide(depi_prior, depi_current, default=np.nan)

        sga_current = mc("sga_expenses")
        sga_prior = mp("sga_expenses")
        sga_ratio_current = _safe_divide(sga_current, sales_current, default=np.nan)
        sga_ratio_prior = _safe_divide(sga_prior, sales_prior, default=np.nan)
        output["sgai"] = _safe_divide(sga_ratio_current, sga_ratio_prior, default=np.nan)

        liabilities_current = mc("total_liabilities")
        liabilities_prior = mp("total_liabilities")
        assets_current = mc("total_assets")
        assets_prior = mp("total_assets")
        leverage_current = _safe_divide(liabilities_current, assets_current, default=np.nan)
        leverage_prior = _safe_divide(liabilities_prior, assets_prior, default=np.nan)
        output["lvgi"] = _safe_divide(leverage_current, leverage_prior, default=np.nan)

        cfo_current = mc("operating_cash_flow")
        income_current = mc("net_income")
        accruals = income_current - cfo_current
        output["tata"] = _safe_divide(accruals, assets_current, default=np.nan)

        cash_current = mc("cash_and_equivalents")
        cash_prior = mp("cash_and_equivalents")
        inventory_current = mc("inventory")
        inventory_prior = mp("inventory")
        current_assets_proxy = cash_current + receivables_current + inventory_current
        current_assets_proxy_prior = cash_prior + receivables_prior + inventory_prior
        asset_quality_current = 1 - _safe_divide(current_assets_proxy, assets_current, default=np.nan)
        asset_quality_prior = 1 - _safe_divide(current_assets_proxy_prior, assets_prior, default=np.nan)
        output["aqi"] = _safe_divide(asset_quality_current, asset_quality_prior, default=np.nan)

    output["beneish_m_score"] = (
        -4.84
        + 0.92 * output["dsri"]
        + 0.528 * output["gmi"]
        + 0.404 * output["aqi"]
        + 0.892 * output["sgi"]
        + 0.115 * output["depi"]
        - 0.172 * output["sgai"]
        + 4.679 * output["tata"]
        - 0.327 * output["lvgi"]
    )

    output["beneish_is_missing"] = output[beneish_components].isna().all(axis=1).astype(float)
    return output


def map_analyst_estimates(
    estimates_df: pd.DataFrame,
    historical_estimates_df: Optional[pd.DataFrame] = None,
    surprises_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Map FMP analyst estimates to ranker revision impulse metrics.
    """
    if estimates_df.empty:
        return pd.DataFrame()

    df_sorted = estimates_df.sort_values(["symbol", "date"] if "date" in estimates_df.columns else ["symbol"])
    df = df_sorted.groupby("symbol").last().reset_index()

    output = pd.DataFrame()
    output["symbol"] = df["symbol"]

    def _col_or_nan(df_: pd.DataFrame, *candidates):
        for col in candidates:
            if col in df_.columns:
                return pd.to_numeric(df_[col], errors="coerce")
        return pd.Series(np.nan, index=df_.index, dtype=float)

    output["revision_impulse_analyst_count"] = _col_or_nan(
        df,
        "numberAnalystEstimateRevenue",
        "numberAnalystEstimatedRevenue",
        "numberAnalystEstimateEps",
        "numberAnalystEstimatedEps",
        "number_of_analysts",
        "analystCount",
        "analyst_count",
        "analyst_count_revenue",
    )

    output["estimated_eps"] = _col_or_nan(df, "epsAvg", "estimated_eps", "eps")
    output["estimated_revenue"] = _col_or_nan(df, "revenueAvg", "estimated_revenue", "revenue")

    output["revision_impulse_signal"] = np.nan
    output["revision_impulse_has_coverage"] = np.nan
    output["revision_jerk_signal"] = np.nan
    output["revision_jerk_has_coverage"] = np.nan
    output["revision_jerk_recent_velocity"] = np.nan
    output["revision_jerk_prior_velocity"] = np.nan
    output["estimate_term_structure_signal"] = np.nan
    output["estimate_term_structure_has_coverage"] = np.nan
    output["revision_impulse_disagreement"] = np.nan
    output["revision_impulse_disagreement_penalty"] = np.nan

    output["pead_signal"] = np.nan
    output["pead_signal_v2"] = np.nan
    output["sue_signal"] = np.nan
    output["sue_has_coverage"] = np.nan
    output["earnings_report_date"] = pd.NaT

    if historical_estimates_df is not None and not historical_estimates_df.empty:
        logger.info("Computing revision impulse from historical estimates")

        hist_df = historical_estimates_df.copy()
        if "date" in hist_df.columns:
            hist_df["date"] = pd.to_datetime(hist_df["date"], errors="coerce")
        if "estimated_eps" in hist_df.columns:
            hist_df["estimated_eps"] = pd.to_numeric(hist_df["estimated_eps"], errors="coerce")
        if "estimated_revenue" in hist_df.columns:
            hist_df["estimated_revenue"] = pd.to_numeric(hist_df["estimated_revenue"], errors="coerce")
        if "period" in hist_df.columns:
            hist_df["period"] = hist_df["period"].astype(str).str.lower()
        hist_df = hist_df.drop_duplicates(
            subset=[c for c in ["symbol", "period", "date", "estimated_eps", "estimated_revenue"] if c in hist_df.columns]
        )

        if "date" in hist_df.columns and "estimated_eps" in hist_df.columns:
            hist_df = hist_df.sort_values(["symbol", "date"])

            hist_df["eps_prev"] = hist_df.groupby("symbol")["estimated_eps"].shift(periods=1, fill_value=None)

            hist_df["revision_impulse_raw"] = _safe_divide(
                (hist_df["estimated_eps"] - hist_df["eps_prev"]),
                hist_df["eps_prev"].abs(),
                default=np.nan,
            )

            latest_revision = hist_df.groupby("symbol").last().reset_index()

            output = output.merge(
                latest_revision[["symbol", "revision_impulse_raw"]],
                on="symbol",
                how="left",
            )

            revision_col = "revision_impulse_raw"
            if revision_col in output.columns:
                median_rev = output[revision_col].median()
                mad = (output[revision_col] - median_rev).abs().median()
                if mad > 0:
                    output["revision_impulse_signal"] = (
                        (output[revision_col] - median_rev) / (mad * 1.4826)
                    ).clip(-2.0, 2.0) / 2.0
                output["revision_impulse_has_coverage"] = output[revision_col].notna().astype(float)

            if "date" in hist_df.columns:
                hist_df["revision_jerk_raw"] = hist_df.groupby("symbol")["revision_impulse_raw"].diff()
                hist_df["revision_jerk_prior"] = hist_df.groupby("symbol")["revision_jerk_raw"].shift(1)

                latest_jerk = hist_df.groupby("symbol").last().reset_index()
                output = output.merge(
                    latest_jerk[["symbol", "revision_jerk_raw", "revision_jerk_prior"]],
                    on="symbol",
                    how="left",
                )

                if "revision_jerk_raw" in output.columns:
                    output["revision_jerk_recent_velocity"] = output["revision_jerk_raw"]
                    output["revision_jerk_prior_velocity"] = output.get("revision_jerk_prior", np.nan)
                    output["revision_jerk_has_coverage"] = output["revision_jerk_raw"].notna().astype(float)

            if "period" in hist_df.columns:
                q_df = hist_df[hist_df["period"].isin(["quarter", "quarterly", "q", "q1", "q2", "q3", "q4"])].copy()
                a_df = hist_df[hist_df["period"].isin(["annual", "year", "fy"])].copy()
                if not q_df.empty and not a_df.empty:
                    q_latest = q_df.sort_values(["symbol", "date"]).groupby("symbol").last().reset_index()
                    a_latest = a_df.sort_values(["symbol", "date"]).groupby("symbol").last().reset_index()

                    q_cols = ["symbol"]
                    a_cols = ["symbol"]
                    if "estimated_eps" in q_latest.columns:
                        q_cols.append("estimated_eps")
                    if "estimated_revenue" in q_latest.columns:
                        q_cols.append("estimated_revenue")
                    if "estimated_eps" in a_latest.columns:
                        a_cols.append("estimated_eps")
                    if "estimated_revenue" in a_latest.columns:
                        a_cols.append("estimated_revenue")

                    q_latest = q_latest[q_cols].rename(
                        columns={
                            "estimated_eps": "estimated_eps_quarter",
                            "estimated_revenue": "estimated_revenue_quarter",
                        }
                    )
                    a_latest = a_latest[a_cols].rename(
                        columns={
                            "estimated_eps": "estimated_eps_annual",
                            "estimated_revenue": "estimated_revenue_annual",
                        }
                    )

                    term_df = q_latest.merge(a_latest, on="symbol", how="inner")
                    if not term_df.empty:
                        eps_term = pd.Series(np.nan, index=term_df.index, dtype=float)
                        rev_term = pd.Series(np.nan, index=term_df.index, dtype=float)

                        if "estimated_eps_annual" in term_df.columns and "estimated_eps_quarter" in term_df.columns:
                            eps_term = _safe_divide(
                                term_df["estimated_eps_annual"] - 4.0 * term_df["estimated_eps_quarter"],
                                (4.0 * term_df["estimated_eps_quarter"]).abs(),
                                default=np.nan,
                            )
                        if "estimated_revenue_annual" in term_df.columns and "estimated_revenue_quarter" in term_df.columns:
                            rev_term = _safe_divide(
                                term_df["estimated_revenue_annual"] - 4.0 * term_df["estimated_revenue_quarter"],
                                (4.0 * term_df["estimated_revenue_quarter"]).abs(),
                                default=np.nan,
                            )

                        term_df["estimate_term_structure_signal"] = pd.concat([eps_term, rev_term], axis=1).mean(axis=1).clip(-1.0, 1.0)
                        term_df["estimate_term_structure_has_coverage"] = term_df["estimate_term_structure_signal"].notna().astype(float)
                        output = output.merge(
                            term_df[["symbol", "estimate_term_structure_signal", "estimate_term_structure_has_coverage"]],
                            on="symbol",
                            how="left",
                            suffixes=("", "_term"),
                        )
                        if "estimate_term_structure_signal_term" in output.columns:
                            output["estimate_term_structure_signal"] = output["estimate_term_structure_signal"].combine_first(output["estimate_term_structure_signal_term"])
                            output.drop(columns=["estimate_term_structure_signal_term"], inplace=True)
                        if "estimate_term_structure_has_coverage_term" in output.columns:
                            output["estimate_term_structure_has_coverage"] = output["estimate_term_structure_has_coverage"].combine_first(output["estimate_term_structure_has_coverage_term"])
                            output.drop(columns=["estimate_term_structure_has_coverage_term"], inplace=True)

            if "estimated_eps_high" in hist_df.columns and "estimated_eps_low" in hist_df.columns:
                hist_df["estimate_dispersion"] = _safe_divide(
                    pd.to_numeric(hist_df["estimated_eps_high"], errors="coerce") -
                    pd.to_numeric(hist_df["estimated_eps_low"], errors="coerce"),
                    pd.to_numeric(hist_df["estimated_eps"], errors="coerce").abs(),
                    default=np.nan,
                )
                latest_dispersion = hist_df.groupby("symbol").last().reset_index()
                output = output.merge(
                    latest_dispersion[["symbol", "estimate_dispersion"]],
                    on="symbol",
                    how="left",
                )
                output["revision_impulse_disagreement"] = output["revision_impulse_disagreement"].combine_first(
                    output.get("estimate_dispersion", np.nan)
                )
                output.drop(columns=[c for c in ["estimate_dispersion"] if c in output.columns], inplace=True)

    if surprises_df is not None and not surprises_df.empty:
        logger.info("Computing revision jerk and PEAD from earnings surprises")

        surprises_df = surprises_df.copy()

        if "surprise_percent" not in surprises_df.columns and "actual_eps" in surprises_df.columns and "estimated_eps" in surprises_df.columns:
            surprises_df["actual_eps"] = pd.to_numeric(surprises_df["actual_eps"], errors="coerce")
            surprises_df["estimated_eps"] = pd.to_numeric(surprises_df["estimated_eps"], errors="coerce")
            surprises_df["surprise_percent"] = (
                (surprises_df["actual_eps"] - surprises_df["estimated_eps"]) /
                surprises_df["estimated_eps"].abs().replace(0, np.nan)
            ) * 100.0

        if "surprise_percent" in surprises_df.columns:
            sur_sorted = surprises_df.sort_values(["symbol", "date"])
            latest_sur = sur_sorted.groupby("symbol").last().reset_index()

            sur_map = latest_sur.set_index("symbol")["surprise_percent"]
            output["pead_signal"] = output["symbol"].map(sur_map) / 100.0
            output["pead_signal_v2"] = output["pead_signal"]
            output["sue_signal"] = output["pead_signal"]
            output["sue_has_coverage"] = output["pead_signal"].notna().astype(float)
            output["earnings_surprise_pct"] = output["symbol"].map(sur_map)

            if "date" in latest_sur.columns:
                report_dates = latest_sur.set_index("symbol")["date"]
                output["earnings_report_date"] = output["symbol"].map(report_dates)

            surprise_agg = surprises_df.groupby("symbol").agg({
                "surprise_percent": ["mean", "std", "count"]
            }).reset_index()

            surprise_agg.columns = ["symbol", "surprise_mean", "surprise_std", "surprise_count"]

            output = output.merge(surprise_agg, on="symbol", how="left")

            if "surprise_std" in output.columns:
                output["revision_impulse_disagreement"] = pd.to_numeric(
                    output["surprise_std"],
                    errors="coerce",
                )

                output["revision_impulse_disagreement_penalty"] = (
                    output["revision_impulse_disagreement"] / 10.0
                ).clip(0.0, 1.0)

    if "revision_jerk_recent_velocity" in output.columns:
        output["revision_jerk_signal"] = output["revision_jerk_recent_velocity"]

    output["pead_analyst_count"] = output["revision_impulse_analyst_count"]
    output["pead_filter_pass"] = output["pead_signal"].notna().astype(float)
    output["pead_has_setup_coverage"] = output["pead_signal"].notna().astype(float)
    output["pead_surprise_component"] = output["pead_signal"]
    output["pead_decay_component"] = output["pead_signal"].notna().astype(float)
    output["pead_breadth_component"] = output["pead_signal"].notna().astype(float)
    output["pead_revision_component"] = output["revision_impulse_signal"]

    output["sue_surprise_raw"] = output["pead_signal"]
    output["sue_surprise_pct"] = output.get("earnings_surprise_pct", np.nan)
    output["sue_std_error"] = output.get("surprise_std", np.nan)

    output["revision_jerk_signal_raw"] = output["revision_jerk_signal"]
    output["earnings_momentum_signal"] = output["pead_signal"]
    output["earnings_momentum_coverage"] = output["pead_signal"].notna().astype(float)

    return output


def map_institutional_ownership(institutional_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map FMP institutional ownership data to ranker metrics.
    """
    if institutional_df.empty:
        return pd.DataFrame()

    df = institutional_df.copy()

    alias_candidates = {
        "ownership_percent": ["ownership_percent", "ownershipPercent", "percentageHeld", "institutionalOwnershipPercentage"],
        "ownership_percent_prev": ["ownershipPercentPrev", "previousOwnershipPercent", "previousPercentageHeld"],
        "value": ["value", "marketValue", "totalValue", "positionValue"],
        "holder_count": ["holder_count", "holderCount", "numberOfHolders", "holdersCount", "institutionsHolding"],
        "holder_count_prev": ["holderCountPrev", "numberOfHoldersPrev", "holdersCountPrev", "institutionsHoldingPrev"],
        "max_ownership_pct": ["max_ownership_pct", "topHolderPercentage", "largestHolderPercentage"],
        "as_of_date": ["date", "asOfDate", "reportDate", "filingDate"],
    }

    normalized = {}
    for normalized_col, candidates in alias_candidates.items():
        for candidate in candidates:
            if candidate in df.columns:
                if normalized_col == "as_of_date":
                    normalized[normalized_col] = pd.to_datetime(df[candidate], errors="coerce")
                else:
                    normalized[normalized_col] = pd.to_numeric(df[candidate], errors="coerce")
                break

    if "symbol" not in df.columns:
        return pd.DataFrame()

    if "ownership_percent" in normalized:
        df["ownership_percent"] = normalized["ownership_percent"]
    if "value" in normalized:
        df["value"] = normalized["value"]
    if "holder_count" in normalized:
        df["holder_count"] = normalized["holder_count"]
    if "holder_count_prev" in normalized:
        df["holder_count_prev"] = normalized["holder_count_prev"]
    if "ownership_percent_prev" in normalized:
        df["ownership_percent_prev"] = normalized["ownership_percent_prev"]
    if "max_ownership_pct" in normalized:
        df["max_ownership_pct"] = normalized["max_ownership_pct"]
    if "as_of_date" in normalized:
        df["as_of_date"] = normalized["as_of_date"]

    if "ownership_percent" in df.columns:
        ownership_sum = df.groupby("symbol")["ownership_percent"].sum(min_count=1)
        ownership_max = df.groupby("symbol")["ownership_percent"].max()
    else:
        ownership_sum = pd.Series(np.nan, index=df["symbol"].dropna().unique())
        ownership_max = pd.Series(np.nan, index=df["symbol"].dropna().unique())

    if "holder_count" in df.columns:
        holder_count = df.groupby("symbol")["holder_count"].max()
    else:
        holder_count = pd.Series(np.nan, index=df["symbol"].dropna().unique(), dtype=float)

    if "holder_count_prev" in df.columns:
        holder_count_prev = df.groupby("symbol")["holder_count_prev"].max()
    else:
        holder_count_prev = pd.Series(np.nan, index=df["symbol"].dropna().unique(), dtype=float)

    if "max_ownership_pct" in df.columns:
        max_holder_pct = df.groupby("symbol")["max_ownership_pct"].max()
    else:
        max_holder_pct = ownership_max

    if "value" in df.columns:
        total_value = df.groupby("symbol")["value"].sum(min_count=1)
    else:
        total_value = pd.Series(np.nan, index=df["symbol"].dropna().unique())

    if "ownership_percent_prev" in df.columns:
        ownership_prev = df.groupby("symbol")["ownership_percent_prev"].max()
    else:
        ownership_prev = pd.Series(np.nan, index=df["symbol"].dropna().unique(), dtype=float)

    if "as_of_date" in df.columns and df["as_of_date"].notna().any():
        dated = df.sort_values(["symbol", "as_of_date"]).copy()
        if "holder_count" in dated.columns:
            derived_prev_holder = dated.groupby("symbol")["holder_count"].nth(-2)
            holder_count_prev = holder_count_prev.combine_first(derived_prev_holder)
        if "ownership_percent" in dated.columns:
            derived_prev_own = dated.groupby("symbol")["ownership_percent"].nth(-2)
            ownership_prev = ownership_prev.combine_first(derived_prev_own)

    agg = pd.DataFrame({
        "symbol": ownership_sum.index,
        "total_ownership_pct": ownership_sum.values,
        "holder_count": holder_count.reindex(ownership_sum.index).values,
        "max_ownership_pct": max_holder_pct.reindex(ownership_sum.index).values,
        "total_value": total_value.reindex(ownership_sum.index).values,
    }).reset_index(drop=True)

    output = pd.DataFrame()
    output["symbol"] = agg["symbol"]
    output["percent_institutions"] = pd.to_numeric(agg["total_ownership_pct"], errors="coerce")
    output["institution_holder_count_latest"] = pd.to_numeric(agg["holder_count"], errors="coerce")
    output["institutional_ownership_from_holders"] = output["percent_institutions"]

    output["institutional_top5_concentration_delta"] = pd.to_numeric(
        agg["max_ownership_pct"],
        errors="coerce",
    )

    output["institution_holder_count_prev"] = output["symbol"].map(holder_count_prev)
    output["institutional_ownership_prev"] = output["symbol"].map(ownership_prev)
    output["institutional_breadth_delta"] = output["institution_holder_count_latest"] - output["institution_holder_count_prev"]
    output["institutional_ownership_delta"] = output["percent_institutions"] - output["institutional_ownership_prev"]
    output.drop(columns=["institutional_ownership_prev"], inplace=True)

    return output


def map_insider_trading(
    insider_df: pd.DataFrame,
    insider_statistics_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Map FMP insider trading data to ranker conviction metrics.
    """
    if insider_df.empty:
        return pd.DataFrame()

    df = insider_df.copy()

    col_aliases = {
        "reportingName": "insider_name",
        "insiderName": "insider_name",
        "securitiesOwned": "shares",
        "securitiesTransacted": "shares",
        "acquisitionOrDisposition": "acquisition_or_disposition",
    }
    for raw_col, norm_col in col_aliases.items():
        if raw_col in df.columns and norm_col not in df.columns:
            df[norm_col] = df[raw_col]

    if "shares" in df.columns:
        df["shares"] = pd.to_numeric(df["shares"], errors="coerce")
    else:
        df["shares"] = np.nan

    if "insider_name" not in df.columns:
        df["insider_name"] = "unknown"

    if "transaction_type" in df.columns:
        buy_mask = df["transaction_type"].str.contains("P", case=False, na=False)
        sell_mask = df["transaction_type"].str.contains("S", case=False, na=False)
        exclude_mask = df["transaction_type"].str.contains("10b5", case=False, na=False)
        open_market_buys = df[buy_mask & ~exclude_mask]
        open_market_sells = df[sell_mask & ~exclude_mask]
    else:
        open_market_buys = df
        open_market_sells = pd.DataFrame()

    output = pd.DataFrame()

    if not open_market_buys.empty and "symbol" in open_market_buys.columns:
        buy_agg = open_market_buys.groupby("symbol").agg(
            total_shares_bought=("shares", lambda s: s.sum(min_count=1)),
            unique_buyers=("insider_name", "nunique"),
            trade_count=("symbol", "size"),
        ).reset_index()

        output["symbol"] = buy_agg["symbol"]
        output["insider_conviction_buy_cluster"] = pd.to_numeric(
            buy_agg["total_shares_bought"], errors="coerce"
        )
        output["insider_conviction_buy_person_count"] = pd.to_numeric(
            buy_agg["unique_buyers"], errors="coerce"
        )
        output["insider_conviction_trade_count"] = pd.to_numeric(
            buy_agg["trade_count"], errors="coerce"
        )
    else:
        output = pd.DataFrame(columns=["symbol"])
        output["insider_conviction_buy_cluster"] = np.nan
        output["insider_conviction_buy_person_count"] = np.nan
        output["insider_conviction_trade_count"] = np.nan

    if not open_market_sells.empty and "symbol" in open_market_sells.columns:
        sell_agg = open_market_sells.groupby("symbol").agg(
            total_shares_sold=("shares", lambda s: s.sum(min_count=1)),
            unique_sellers=("insider_name", "nunique"),
        ).reset_index()
        output = output.merge(sell_agg[["symbol", "total_shares_sold", "unique_sellers"]], on="symbol", how="outer")
        output["insider_conviction_sell_pressure"] = pd.to_numeric(
            output["total_shares_sold"], errors="coerce"
        )
        output["insider_conviction_sell_person_count"] = pd.to_numeric(
            output["unique_sellers"], errors="coerce"
        )
        output.drop(columns=["total_shares_sold", "unique_sellers"], inplace=True, errors="ignore")
    else:
        output["insider_conviction_sell_pressure"] = np.nan
        output["insider_conviction_sell_person_count"] = np.nan

    if insider_statistics_df is not None and not insider_statistics_df.empty and "symbol" in insider_statistics_df.columns:
        stats_cols = [
            c for c in ["symbol", "trade_count", "buy_count", "sell_count", "total_buy", "total_sell", "net_activity"]
            if c in insider_statistics_df.columns
        ]
        if len(stats_cols) > 1:
            stats_latest = insider_statistics_df[stats_cols].copy()
            stats_latest = stats_latest.sort_values("symbol").groupby("symbol").last().reset_index()
            output = output.merge(stats_latest, on="symbol", how="outer")
            if "trade_count" in output.columns:
                output["insider_conviction_trade_count"] = output["insider_conviction_trade_count"].fillna(
                    pd.to_numeric(output["trade_count"], errors="coerce")
                )
            if "buy_count" in output.columns and "insider_conviction_buy_person_count" in output.columns:
                output["insider_conviction_buy_person_count"] = output["insider_conviction_buy_person_count"].fillna(
                    pd.to_numeric(output["buy_count"], errors="coerce")
                )
            if "sell_count" in output.columns and "insider_conviction_sell_person_count" in output.columns:
                output["insider_conviction_sell_person_count"] = output["insider_conviction_sell_person_count"].fillna(
                    pd.to_numeric(output["sell_count"], errors="coerce")
                )

    output["insider_conviction_signal"] = np.nan
    output["insider_conviction_has_coverage"] = np.where(
        output["insider_conviction_buy_cluster"].notna() | output["insider_conviction_sell_pressure"].notna(),
        1.0, 0.0
    )
    output["insider_short_crowding_penalty"] = 0.0

    for col in ["insider_conviction_trade_count", "insider_conviction_buy_cluster", "insider_conviction_buy_person_count"]:
        if col not in output.columns:
            output[col] = np.nan

    return output


def merge_all_data(
    profiles_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    price_history_mapped_df: pd.DataFrame,
    financials_df: pd.DataFrame,
    beneish_df: pd.DataFrame,
    estimates_df: pd.DataFrame,
    institutional_df: pd.DataFrame,
    insider_df: pd.DataFrame,
    employee_df: pd.DataFrame,
    scores_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge all mapped data into a single DataFrame for the ranker.
    """
    def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "symbol" not in df.columns:
            return df
        return df.sort_values("symbol").groupby("symbol").last().reset_index()

    profiles_df = _dedupe(profiles_df)
    financials_df = _dedupe(financials_df)
    beneish_df = _dedupe(beneish_df)
    estimates_df = _dedupe(estimates_df)
    institutional_df = _dedupe(institutional_df)
    insider_df = _dedupe(insider_df)
    employee_df = _dedupe(employee_df)
    price_history_mapped_df = _dedupe(price_history_mapped_df)
    scores_df = _dedupe(scores_df)

    if not profiles_df.empty:
        merged = profiles_df.copy()
    else:
        merged = pd.DataFrame(columns=["symbol"])

    if not financials_df.empty:
        merged = merged.merge(financials_df, on="symbol", how="outer")

    if not beneish_df.empty:
        merged = merged.merge(beneish_df, on="symbol", how="outer")

    if not estimates_df.empty:
        merged = merged.merge(estimates_df, on="symbol", how="outer")

    if not institutional_df.empty:
        merged = merged.merge(institutional_df, on="symbol", how="outer")

    if not insider_df.empty:
        merged = merged.merge(insider_df, on="symbol", how="outer")

    if not employee_df.empty:
        merged = merged.merge(employee_df, on="symbol", how="outer")

    if not scores_df.empty:
        merged = merged.merge(scores_df, on="symbol", how="outer")

    if not price_history_mapped_df.empty:
        merged = merged.merge(price_history_mapped_df, on="symbol", how="outer")

    if not prices_df.empty:
        if "date" in prices_df.columns:
            latest_prices = prices_df.sort_values("date").groupby("symbol").last().reset_index()
        else:
            latest_prices = prices_df

        price_mapping = {
            "yearHigh": "52_week_high",
            "priceAvg200": "200_day_ma",
            "priceAvg50": "50_day_ma",
        }
        for fmp_col, internal_col in price_mapping.items():
            if fmp_col in latest_prices.columns and internal_col not in latest_prices.columns:
                latest_prices[internal_col] = latest_prices[fmp_col]

        merged = merged.merge(latest_prices, on="symbol", how="left")

    if "price" in merged.columns:
        price_proxy = pd.to_numeric(merged["price"], errors="coerce")
    elif "close" in merged.columns:
        price_proxy = pd.to_numeric(merged["close"], errors="coerce")
    else:
        price_proxy = pd.Series(np.nan, index=merged.index, dtype=float)

    merged["price_proxy"] = price_proxy
    if "200_day_ma" in merged.columns:
        merged["price_to_200dma"] = price_proxy / pd.to_numeric(merged["200_day_ma"], errors="coerce").replace(0, np.nan)
    if "52_week_high" in merged.columns:
        merged["distance_from_high"] = price_proxy / pd.to_numeric(merged["52_week_high"], errors="coerce").replace(0, np.nan)

    if "full_time_employees" in merged.columns:
        emp = pd.to_numeric(merged["full_time_employees"], errors="coerce").replace(0, np.nan)
        if "total_revenue" in merged.columns:
            merged["revenue_per_employee"] = pd.to_numeric(merged["total_revenue"], errors="coerce") / emp
        if "gross_profit" in merged.columns:
            merged["gross_profit_per_employee"] = pd.to_numeric(merged["gross_profit"], errors="coerce") / emp

    x_cols = [c for c in merged.columns if c.endswith("_x")]
    for xcol in x_cols:
        base = xcol[:-2]
        ycol = f"{base}_y"
        if ycol in merged.columns:
            merged[base] = merged[xcol].combine_first(merged[ycol])
            merged.drop(columns=[xcol, ycol], inplace=True)
        else:
            merged.rename(columns={xcol: base}, inplace=True)

    if "symbol" in merged.columns:
        cols = ["symbol"] + [col for col in merged.columns if col != "symbol"]
        merged = merged[cols]

    return merged


def create_raw_fmp_dataframe(
    bulk_data: Dict[str, pd.DataFrame],
    market: str = "us",
    financial_period: str = "annual",
    historical_estimates_df: Optional[pd.DataFrame] = None,
    surprises_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Create the raw DataFrame from FMP bulk data that the ranker expects.
    """
    profiles_df = bulk_data.get("profiles", pd.DataFrame())
    prices_df = bulk_data.get("prices", pd.DataFrame())
    income_df = bulk_data.get("income_statements", pd.DataFrame())
    balance_df = bulk_data.get("balance_sheets", pd.DataFrame())
    cashflow_df = bulk_data.get("cash_flows", pd.DataFrame())
    estimates_df = bulk_data.get("analyst_estimates", pd.DataFrame())
    institutional_df = bulk_data.get("institutional_ownership", pd.DataFrame())
    insider_df = bulk_data.get("insider_trading", pd.DataFrame())
    insider_statistics_df = bulk_data.get("insider_statistics", pd.DataFrame())
    employee_df = bulk_data.get("employee_count", pd.DataFrame())
    scores_df = bulk_data.get("scores", pd.DataFrame())
    price_history_df = bulk_data.get("price_history", pd.DataFrame())

    if historical_estimates_df is None:
        historical_estimates_df = bulk_data.get("historical_estimates", estimates_df)
    if surprises_df is None:
        surprises_df = bulk_data.get("earnings_surprises", pd.DataFrame())

    logger.info("Mapping FMP profile data")
    mapped_profiles = map_profile_data(profiles_df)

    logger.info("Mapping FMP financial statements")
    mapped_financials = map_financial_statements(
        income_df,
        balance_df,
        cashflow_df,
        profiles_df,
        financial_period=financial_period,
    )

    logger.info("Mapping Beneish M-Score components")
    mapped_beneish = map_beneish_components(income_df, balance_df, cashflow_df)

    logger.info("Mapping analyst estimates")
    mapped_estimates = map_analyst_estimates(
        estimates_df,
        historical_estimates_df=historical_estimates_df,
        surprises_df=surprises_df,
    )

    logger.info("Mapping institutional ownership")
    mapped_institutional = map_institutional_ownership(institutional_df)

    logger.info("Mapping insider trading")
    mapped_insider = map_insider_trading(insider_df, insider_statistics_df)

    logger.info("Mapping employee count")
    mapped_employee = map_employee_count(employee_df)

    logger.info("Mapping score data")
    mapped_scores = map_scores_data(scores_df)

    logger.info("Mapping historical price momentum")
    mapped_price_history = map_price_history(price_history_df)

    logger.info("Merging all mapped data")
    raw_df = merge_all_data(
        mapped_profiles,
        prices_df,
        mapped_price_history,
        mapped_financials,
        mapped_beneish,
        mapped_estimates,
        mapped_institutional,
        mapped_insider,
        mapped_employee,
        mapped_scores,
    )

    logger.info(f"Created raw DataFrame with {len(raw_df)} symbols and {len(raw_df.columns)} columns")

    return raw_df