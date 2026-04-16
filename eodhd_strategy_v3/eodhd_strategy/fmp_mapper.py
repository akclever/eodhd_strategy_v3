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
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _safe_divide(numerator: pd.Series, denominator: pd.Series, default: float = 0.0) -> pd.Series:
    """
    Safely divide two series, returning default when denominator is zero or NaN.
    
    Args:
        numerator: Numerator series
        denominator: Denominator series
        default: Default value to return on division by zero
        
    Returns:
        Series with division results
    """
    result = pd.Series(np.nan, index=numerator.index, dtype=float)
    valid_mask = (denominator.notna()) & (denominator != 0) & (numerator.notna())
    result.loc[valid_mask] = numerator.loc[valid_mask] / denominator.loc[valid_mask]
    result = result.fillna(default)
    return result


def map_profile_data(profiles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map FMP profile data to ranker-expected columns.
    
    Args:
        profiles_df: Raw profile data from FMP
        
    Returns:
        DataFrame with columns: symbol, market_cap, sector, industry, 
        is_actively_trading, company_name, exchange, country
    """
    if profiles_df.empty:
        return pd.DataFrame()
    
    df = profiles_df.copy()
    
    # Ensure required columns exist
    required_cols = {
        "symbol": "symbol",
        "market_cap": "market_cap",
        "sector": "sector",
        "industry": "industry",
        "is_actively_trading": "is_actively_trading",
    }
    
    # Create output DataFrame with required columns
    output = pd.DataFrame()
    for fmp_col, ranker_col in required_cols.items():
        if fmp_col in df.columns:
            output[ranker_col] = df[fmp_col]
        else:
            output[ranker_col] = pd.NA
    
    # Add optional columns if available
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


def map_financial_statements(
    income_df: pd.DataFrame,
    balance_df: pd.DataFrame,
    cashflow_df: pd.DataFrame,
    profiles_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Map FMP financial statements to ranker-expected fundamental columns.
    
    This function merges income statement, balance sheet, and cash flow data,
    then computes derived metrics like shareholder_yield, gross_profitability,
    and adjusted_book_to_market.
    
    Args:
        income_df: Income statement data from FMP
        balance_df: Balance sheet data from FMP
        cashflow_df: Cash flow data from FMP
        profiles_df: Profile data (for market cap)
        
    Returns:
        DataFrame with fundamental columns matching ranker expectations
    """
    if all(df.empty for df in [income_df, balance_df, cashflow_df]):
        logger.warning("All financial statement DataFrames are empty")
        return pd.DataFrame()
    
    # Get the most recent annual/quarterly data for each symbol
    def get_latest_records(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        if "date" in df.columns:
            return df.sort_values("date").groupby("symbol").last().reset_index()
        return df
    
    income_latest = get_latest_records(income_df)
    balance_latest = get_latest_records(balance_df)
    cashflow_latest = get_latest_records(cashflow_df)
    
    # Helper to safely get column as Series (handles duplicate column names)
    def safe_get_column(df: pd.DataFrame, col: str) -> pd.Series:
        """Extract column as Series, handling duplicate column names."""
        if col not in df.columns:
            return pd.Series([np.nan] * len(df), index=df.index)
        result = df[col]
        if isinstance(result, pd.DataFrame):
            # Duplicate column names - take the first one
            result = result.iloc[:, 0]
        return result
    
    # Start with income statement
    if not income_latest.empty:
        df = income_latest.copy()
    else:
        # Create empty DataFrame with symbol column
        df = pd.DataFrame(columns=["symbol"])
    
    # Merge balance sheet
    if not balance_latest.empty:
        df = df.merge(balance_latest, on="symbol", how="outer", suffixes=("", "_balance"))
    
    # Merge cash flow
    if not cashflow_latest.empty:
        df = df.merge(cashflow_latest, on="symbol", how="outer", suffixes=("", "_cashflow"))
    
    # Merge profiles for market cap AND dividend data
    profile_cols = ["symbol", "market_cap"]
    for col in ["lastAnnualDividend", "lastDividend", "price"]:
        if not profiles_df.empty and col in profiles_df.columns:
            profile_cols.append(col)
    if not profiles_df.empty:
        df = df.merge(profiles_df[list(dict.fromkeys(profile_cols))], on="symbol", how="left")
    
    # Initialize output columns with NaN
    output = pd.DataFrame(index=df.index)
    output["symbol"] = df["symbol"]
    
    # Map income statement fields
    income_mapping = {
        "total_revenue": "total_revenue",
        "gross_profit": "gross_profit",
        "operating_income": "operating_income",
        "net_income": "net_income",
        "ebitda": "ebitda",
        "research_development": "research_development",
        "rd_expenses": "rd_expenses",
        "sga_expenses": "sga_expenses",
    }
    
    for fmp_col, ranker_col in income_mapping.items():
        col_data = safe_get_column(df, fmp_col)
        if col_data.notna().any():
            output[ranker_col] = pd.to_numeric(col_data, errors="coerce")
        else:
            output[ranker_col] = np.nan
    
    # Map balance sheet fields
    balance_mapping = {
        "total_assets": "total_assets",
        "total_liabilities": "total_liabilities",
        "total_stockholders_equity": "total_stockholders_equity",
        "shareholders_equity": "shareholders_equity",
        "shares_outstanding": "shares_outstanding",
        "net_receivables": "net_receivables",
        "inventory": "inventory",
        "account_payables": "account_payables",
        "cash_and_equivalents": "cash_and_equivalents",
        "intangible_assets": "intangible_assets",
        "goodwill": "goodwill",
    }
    
    for fmp_col, ranker_col in balance_mapping.items():
        col_data = safe_get_column(df, fmp_col)
        if col_data.notna().any():
            output[ranker_col] = pd.to_numeric(col_data, errors="coerce")
        else:
            output[ranker_col] = np.nan
    
    # Map cash flow fields
    cashflow_mapping = {
        "operating_cash_flow": "operating_cash_flow",
        "capital_expenditure": "capital_expenditure",
        "free_cash_flow": "free_cash_flow",
        "dividends_paid": "dividends_paid",
        "stock_issued": "stock_issued",
        "stock_repurchased": "stock_repurchased",
    }
    
    for fmp_col, ranker_col in cashflow_mapping.items():
        col_data = safe_get_column(df, fmp_col)
        if col_data.notna().any():
            output[ranker_col] = pd.to_numeric(col_data, errors="coerce")
        else:
            output[ranker_col] = np.nan
    
    # Ensure market_cap is available
    market_cap_col = safe_get_column(df, "market_cap")
    if market_cap_col.notna().any():
        output["market_cap"] = pd.to_numeric(market_cap_col, errors="coerce")
    else:
        output["market_cap"] = np.nan
    
    # Compute derived metrics
    
    # gross_profitability = gross_profit / total_assets
    output["gross_profitability"] = _safe_divide(
        output["gross_profit"],
        output["total_assets"],
        default=np.nan
    )
    
    # reported_book_to_market = equity / market_cap
    # Use shareholders_equity if available, else total_stockholders_equity
    equity = output["shareholders_equity"].fillna(output["total_stockholders_equity"])
    output["reported_book_to_market"] = _safe_divide(
        equity,
        output["market_cap"],
        default=np.nan
    )
    
    # adjusted_book_to_market = (equity - intangibles) / market_cap
    # If intangible data is missing, intangibles should be NaN, not 0
    intangibles = output["intangible_assets"] + output["goodwill"]
    adjusted_equity = equity - intangibles
    output["adjusted_book_to_market"] = _safe_divide(
        adjusted_equity,
        output["market_cap"],
        default=np.nan
    )
    
    # dividend_yield = lastAnnualDividend / price  (from screener/profile data)
    # Try both column name variants that FMP returns
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
    output["safe_dividend_yield"] = computed_div_yield  # simplified — no payout safety gate yet

    # buyback_yield = |stock_repurchased| / market_cap
    # Note: stock_repurchased is typically negative in FMP data
    stock_repurchased_abs = output["stock_repurchased"].abs()
    output["buyback_yield"] = _safe_divide(
        stock_repurchased_abs,
        output["market_cap"],
        default=0.0
    )

    # shareholder_yield = dividend_yield + buyback_yield
    # Use fillna(0) so a missing component doesn't blank the whole yield
    output["shareholder_yield"] = (
        output["dividend_yield"].fillna(0.0) + output["buyback_yield"].fillna(0.0)
    ).where(output["dividend_yield"].notna() | output["buyback_yield"].notna(), other=np.nan)
    
    # payout_ratio = dividends_paid / net_income
    output["payout_ratio"] = _safe_divide(
        output["dividends_paid"].abs(),
        output["net_income"],
        default=np.nan
    )
    
    # Return on assets = net_income / total_assets
    output["reported_return_on_assets"] = _safe_divide(
        output["net_income"],
        output["total_assets"],
        default=np.nan
    )
    
    # Return on invested capital (simplified as ROIC = net_income / (debt + equity))
    # For now, use ROA as proxy
    output["reported_return_on_invested_capital"] = output["reported_return_on_assets"]
    
    # RD expense ratio = R&D / revenue
    output["rd_expense_ratio"] = _safe_divide(
        output["research_development"].fillna(output["rd_expenses"]),
        output["total_revenue"],
        default=np.nan
    )
    
    # Intangible adjustment eligibility
    output["intangible_adjustment_eligible"] = (
        (output["rd_expense_ratio"].fillna(0) >= 0.02) |
        (output["intangible_assets"].fillna(0) > 0)
    ).astype(float)
    
    # Intangible adjusted gross profitability
    # Add R&D and SGA to assets for intangible-adjusted profitability
    rd_capitalized = output["research_development"].fillna(output["rd_expenses"])
    sga_capitalized = output["sga_expenses"] * 0.30  # 30% of SGA capitalized
    intangible_adjusted_assets = output["total_assets"] + rd_capitalized + sga_capitalized
    
    output["intangible_adjusted_gross_profitability"] = _safe_divide(
        output["gross_profit"],
        intangible_adjusted_assets,
        default=np.nan
    )
    
    # Intangible adjusted equity
    intangible_adjusted_equity = equity + rd_capitalized + sga_capitalized
    
    output["intangible_adjusted_book_to_market"] = _safe_divide(
        intangible_adjusted_equity,
        output["market_cap"],
        default=np.nan
    )
    
    # Intangible adjusted return on assets
    output["intangible_adjusted_return_on_assets"] = _safe_divide(
        output["net_income"],
        intangible_adjusted_assets,
        default=np.nan
    )
    
    # Intangible adjusted return on invested capital
    output["intangible_adjusted_return_on_invested_capital"] = output["intangible_adjusted_return_on_assets"]
    
    # Determine which adjusted values to use
    output["intangible_adjustment_applied"] = (
        (output["intangible_adjustment_eligible"].fillna(0).astype(bool)) & 
        (output["intangible_adjusted_gross_profitability"].notna() | 
         output["intangible_adjusted_book_to_market"].notna())
    ).astype(float)
    
    # Use adjusted values where applicable
    output["gross_profitability"] = np.where(
        output["intangible_adjustment_applied"] == 1,
        output["intangible_adjusted_gross_profitability"],
        output["gross_profitability"]
    )
    
    output["adjusted_book_to_market"] = np.where(
        output["intangible_adjustment_applied"] == 1,
        output["intangible_adjusted_book_to_market"],
        output["adjusted_book_to_market"]
    )
    
    output["return_on_assets"] = np.where(
        output["intangible_adjustment_applied"] == 1,
        output["intangible_adjusted_return_on_assets"],
        output["reported_return_on_assets"]
    )
    
    output["return_on_invested_capital"] = np.where(
        output["intangible_adjustment_applied"] == 1,
        output["intangible_adjusted_return_on_invested_capital"],
        output["reported_return_on_invested_capital"]
    )
    
    return output


def map_beneish_components(
    income_df: pd.DataFrame,
    balance_df: pd.DataFrame,
    cashflow_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Map FMP data to Beneish M-Score components.
    
    The Beneish M-Score is used to detect earnings manipulation. It requires
    the following components:
    - Days Sales in Receivables Index (DSRI)
    - Gross Margin Index (GMI)
    - Asset Quality Index (AQI)
    - Sales Growth Index (SGI)
    - Depreciation Index (DEPI)
    - Sales, General and Administrative Expenses Index (SGAI)
    - Leverage Index (LVGI)
    - Total Accruals to Total Assets (TATA)
    
    Args:
        income_df: Income statement data (need at least 2 periods)
        balance_df: Balance sheet data (need at least 2 periods)
        cashflow_df: Cash flow data
        
    Returns:
        DataFrame with Beneish components and M-Score
    """
    if all(df.empty for df in [income_df, balance_df, cashflow_df]):
        return pd.DataFrame()
    
    # Get the last 2 periods for each symbol
    def get_last_n_periods(df: pd.DataFrame, n: int = 2) -> pd.DataFrame:
        if df.empty or "date" not in df.columns:
            return df
        return df.sort_values("date").groupby("symbol").tail(n).reset_index()
    
    income_2p = get_last_n_periods(income_df, 2)
    balance_2p = get_last_n_periods(balance_df, 2)
    cashflow_2p = get_last_n_periods(cashflow_df, 2)
    
    # Merge data
    df = income_2p.merge(balance_2p, on=["symbol", "date"], how="outer", suffixes=("", "_balance"))
    df = df.merge(cashflow_2p, on=["symbol", "date"], how="outer", suffixes=("", "_cashflow"))
    
    if df.empty:
        return pd.DataFrame()
    
    # Pivot to get current and prior periods
    df = df.sort_values(["symbol", "date"])
    
    # Get current (latest) and prior (previous) data
    current = df.groupby("symbol").last().reset_index()
    prior = df.groupby("symbol").nth(-2).reset_index() if len(df) >= 2 else pd.DataFrame()
    
    output = pd.DataFrame()
    output["symbol"] = current["symbol"]
    
    # Initialize components as NaN
    beneish_components = [
        "dsri", "gmi", "aqi", "sgi", "depi", "sgai", "lvgi", "tata"
    ]
    for comp in beneish_components:
        output[comp] = np.nan
    
    def _safe_col(df: pd.DataFrame, col: str, suffix: str = "") -> pd.Series:
        """Try multiple candidate column names (handles _cashflow/_balance rename suffixes).
        Returns a numeric Series of NaN if none found."""
        candidates = [
            f"{col}{suffix}",
            f"{col}_cashflow{suffix}",
            f"{col}_balance{suffix}",
        ]
        # Also try without the period suffix in case the join collapsed it
        if suffix:
            candidates += [f"{col}_cashflow", f"{col}_balance", col]
        for name in candidates:
            if name in df.columns:
                val = df[name]
                if isinstance(val, pd.DataFrame):
                    val = val.iloc[:, 0]
                return pd.to_numeric(val, errors="coerce")
        return pd.Series(np.nan, index=df.index, dtype=float)

    # Compute components if both periods are available
    if not prior.empty:
        # Merge current and prior with clear suffixes
        merged = current.merge(
            prior,
            on="symbol",
            how="left",
            suffixes=("_current", "_prior")
        )
        
        def mc(col): return _safe_col(merged, col, "_current")
        def mp(col): return _safe_col(merged, col, "_prior")

        # DSRI: Days Sales in Receivables Index
        receivables_current = mc("net_receivables")
        receivables_prior   = mp("net_receivables")
        sales_current       = mc("total_revenue")
        sales_prior         = mp("total_revenue")
        dsr_current = _safe_divide(receivables_current, sales_current, default=np.nan)
        dsr_prior   = _safe_divide(receivables_prior,   sales_prior,   default=np.nan)
        output["dsri"] = _safe_divide(dsr_current, dsr_prior, default=np.nan)
        
        # GMI: Gross Margin Index
        gross_profit_current = mc("gross_profit")
        gross_profit_prior   = mp("gross_profit")
        gm_current = _safe_divide(gross_profit_current, sales_current, default=np.nan)
        gm_prior   = _safe_divide(gross_profit_prior,   sales_prior,   default=np.nan)
        output["gmi"] = _safe_divide(gm_prior, gm_current, default=np.nan)
        
        # SGI: Sales Growth Index
        output["sgi"] = _safe_divide(sales_current, sales_prior, default=np.nan)
        
        # DEPI: Depreciation Index (proxy via capex / (assets + capex))
        capex_current = mc("capital_expenditure").abs()
        capex_prior   = mp("capital_expenditure").abs()
        ppe_current   = mc("total_assets")
        ppe_prior     = mp("total_assets")
        depi_current = _safe_divide(capex_current, ppe_current + capex_current, default=np.nan)
        depi_prior   = _safe_divide(capex_prior,   ppe_prior   + capex_prior,   default=np.nan)
        output["depi"] = _safe_divide(depi_prior, depi_current, default=np.nan)
        
        # SGAI: SG&A Index
        sga_current       = mc("sga_expenses")
        sga_prior         = mp("sga_expenses")
        sga_ratio_current = _safe_divide(sga_current, sales_current, default=np.nan)
        sga_ratio_prior   = _safe_divide(sga_prior,   sales_prior,   default=np.nan)
        output["sgai"] = _safe_divide(sga_ratio_current, sga_ratio_prior, default=np.nan)
        
        # LVGI: Leverage Index
        liabilities_current = mc("total_liabilities")
        liabilities_prior   = mp("total_liabilities")
        assets_current      = mc("total_assets")
        assets_prior        = mp("total_assets")
        leverage_current = _safe_divide(liabilities_current, assets_current, default=np.nan)
        leverage_prior   = _safe_divide(liabilities_prior,   assets_prior,   default=np.nan)
        output["lvgi"] = _safe_divide(leverage_current, leverage_prior, default=np.nan)
        
        # TATA: Total Accruals to Total Assets
        cfo_current    = mc("operating_cash_flow")
        income_current = mc("net_income")
        accruals = income_current - cfo_current
        output["tata"] = _safe_divide(accruals, assets_current, default=np.nan)
        
        # AQI: Asset Quality Index
        cash_current      = mc("cash_and_equivalents")
        cash_prior        = mp("cash_and_equivalents")
        inventory_current = mc("inventory")
        inventory_prior   = mp("inventory")
        current_assets_proxy       = cash_current + receivables_current + inventory_current
        current_assets_proxy_prior = cash_prior   + receivables_prior   + inventory_prior
        asset_quality_current = 1 - _safe_divide(current_assets_proxy,       assets_current, default=np.nan)
        asset_quality_prior   = 1 - _safe_divide(current_assets_proxy_prior, assets_prior,   default=np.nan)
        output["aqi"] = _safe_divide(asset_quality_current, asset_quality_prior, default=np.nan)
    
    # Beneish M-Score
    # M = -4.84 + 0.92*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI + 0.115*DEPI - 0.172*SGAI + 4.679*TATA - 0.327*LVGI
    output["beneish_m_score"] = (
        -4.84
        + 0.92  * output["dsri"]
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
    surprises_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Map FMP analyst estimates to ranker revision impulse metrics.
    
    This function computes:
    - revision_impulse_signal: 30-day change in EPS consensus
    - revision_jerk_recent_velocity: Acceleration of estimate changes
    - revision_impulse_analyst_count: Number of analysts
    - revision_impulse_disagreement: Dispersion/variance of estimates
    
    Args:
        estimates_df: Latest analyst estimates data from FMP
        historical_estimates_df: Historical time series of estimates (optional)
        surprises_df: Earnings surprises data (optional)
        
    Returns:
        DataFrame with revision impulse and jerk metrics
    """
    if estimates_df.empty:
        return pd.DataFrame()
    
    # --- CRITICAL: Deduplicate to latest record per symbol ---
    # We use the full historical data for signal computation later, 
    # but the base output frame must be 1-row-per-symbol.
    df_sorted = estimates_df.sort_values(["symbol", "date"] if "date" in estimates_df.columns else ["symbol"])
    df = df_sorted.groupby("symbol").last().reset_index()
    
    output = pd.DataFrame()
    output["symbol"] = df["symbol"]
    
    # Map basic fields
    def _col_or_nan(df, *candidates):
        for col in candidates:
            if col in df.columns:
                return pd.to_numeric(df[col], errors="coerce")
        return pd.Series(np.nan, index=df.index, dtype=float)

    output["revision_impulse_analyst_count"] = _col_or_nan(
        df,
        "numberAnalystEstimateRevenue",
        "numberAnalystEstimateEps",
        "number_of_analysts",
        "analystCount",
    ).fillna(0)
    
    # EPS estimates
    output["estimated_eps"] = _col_or_nan(df, "epsAvg", "estimated_eps", "eps")
    
    # Revenue estimates  
    output["estimated_revenue"] = _col_or_nan(df, "revenueAvg", "estimated_revenue", "revenue")

    # Initialize revision metrics
    output["revision_impulse_signal"] = np.nan
    output["revision_impulse_has_coverage"] = 0.0
    output["revision_jerk_signal"] = np.nan
    output["revision_jerk_has_coverage"] = 0.0
    output["revision_jerk_recent_velocity"] = np.nan
    output["revision_jerk_prior_velocity"] = np.nan
    output["revision_impulse_disagreement"] = np.nan
    output["revision_impulse_disagreement_penalty"] = np.nan
    
    # Initialize PEAD and SUE signals
    output["pead_signal"] = np.nan
    output["pead_signal_v2"] = np.nan
    output["sue_signal"] = np.nan
    output["sue_has_coverage"] = 0.0
    output["earnings_report_date"] = pd.NaT
    
    # Compute revision impulse from historical data if available
    if historical_estimates_df is not None and not historical_estimates_df.empty:
        logger.info("Computing revision impulse from historical estimates")
        
        # Group by symbol and compute estimate changes
        hist_df = historical_estimates_df.copy()
        
        if "date" in hist_df.columns and "estimated_eps" in hist_df.columns:
            hist_df = hist_df.sort_values(["symbol", "date"])
            
            # Compute 30-day change in EPS estimates
            hist_df["eps_30d_ago"] = hist_df.groupby("symbol")["estimated_eps"].shift(
                periods=1, fill_value=None
            )
            
            # Compute revision impulse as percent change
            hist_df["revision_impulse_raw"] = _safe_divide(
                (hist_df["estimated_eps"] - hist_df["eps_30d_ago"]).abs(),
                hist_df["eps_30d_ago"],
                default=np.nan
            )
            
            # Get latest revision impulse for each symbol
            latest_revision = hist_df.groupby("symbol").last().reset_index()
            
            # Merge with output
            output = output.merge(
                latest_revision[["symbol", "revision_impulse_raw"]],
                on="symbol",
                how="left"
            )
            
            # Normalize revision impulse to [-1, 1] range
            # Positive revision = upward estimate revision
            revision_col = "revision_impulse_raw"
            if revision_col in output.columns:
                # Use robust z-score normalization
                median_rev = output[revision_col].median()
                mad = (output[revision_col] - median_rev).abs().median()
                if mad > 0:
                    output["revision_impulse_signal"] = (
                        (output[revision_col] - median_rev) / (mad * 1.4826)
                    ).clip(-2.0, 2.0) / 2.0  # Normalize to [-1, 1]
                output["revision_impulse_has_coverage"] = (
                    output[revision_col].notna().astype(float)
                )
            
            # Compute revision jerk (acceleration of revision impulse)
            if "date" in hist_df.columns:
                hist_df["revision_jerk_raw"] = hist_df.groupby("symbol")["revision_impulse_raw"].diff()
                
                latest_jerk = hist_df.groupby("symbol").last().reset_index()
                output = output.merge(
                    latest_jerk[["symbol", "revision_jerk_raw"]],
                    on="symbol",
                    how="left"
                )
                
                if "revision_jerk_raw" in output.columns:
                    output["revision_jerk_recent_velocity"] = output["revision_jerk_raw"]
                    output["revision_jerk_has_coverage"] = (
                        output["revision_jerk_raw"].notna().astype(float)
                    )
    
    # Compute revision jerk and PEAD from surprises if available
    if surprises_df is not None and not surprises_df.empty:
        logger.info("Computing revision jerk and PEAD from earnings surprises")
        
        surprises_df = surprises_df.copy()
        
        # Derive surprise_percent if missing (common in bulk CSV)
        if "surprise_percent" not in surprises_df.columns and "actual_eps" in surprises_df.columns and "estimated_eps" in surprises_df.columns:
            surprises_df["actual_eps"] = pd.to_numeric(surprises_df["actual_eps"], errors="coerce")
            surprises_df["estimated_eps"] = pd.to_numeric(surprises_df["estimated_eps"], errors="coerce")
            surprises_df["surprise_percent"] = (
                (surprises_df["actual_eps"] - surprises_df["estimated_eps"]) / 
                surprises_df["estimated_eps"].abs().replace(0, np.nan)
            ) * 100.0
        
        if "surprise_percent" in surprises_df.columns:
            # Sort to get latest surprise
            sur_sorted = surprises_df.sort_values(["symbol", "date"])
            latest_sur = sur_sorted.groupby("symbol").last().reset_index()
            
            # Map latest surprise to signals
            sur_map = latest_sur.set_index("symbol")["surprise_percent"]
            output["pead_signal"] = output["symbol"].map(sur_map) / 100.0  # Normalize %
            output["sue_signal"] = output["pead_signal"] # Proxy SUE with surprise % for now
            output["sue_has_coverage"] = output["pead_signal"].notna().astype(float)
            
            if "date" in latest_sur.columns:
                report_dates = latest_sur.set_index("symbol")["date"]
                output["earnings_report_date"] = output["symbol"].map(report_dates)

            # Aggregate surprise by symbol for disagreement
            surprise_agg = surprises_df.groupby("symbol").agg({
                "surprise_percent": ["mean", "std", "count"]
            }).reset_index()
            
            surprise_agg.columns = ["symbol", "surprise_mean", "surprise_std", "surprise_count"]
            
            output = output.merge(surprise_agg, on="symbol", how="left")
            
            # Use surprise std as disagreement metric
            if "surprise_std" in output.columns:
                output["revision_impulse_disagreement"] = pd.to_numeric(
                    output["surprise_std"],
                    errors="coerce"
                ).fillna(0)
                
                # Disagreement penalty
                output["revision_impulse_disagreement_penalty"] = (
                    output["revision_impulse_disagreement"] / 10.0
                ).clip(0.0, 1.0)

    # Compute composite revision jerk signal
    if "revision_jerk_recent_velocity" in output.columns:
        output["revision_jerk_signal"] = output["revision_jerk_recent_velocity"]
    
    # Do not fill NaN with 0 - missing data should remain NaN for Alpha Aggregator
    
    return output


def map_institutional_ownership(institutional_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map FMP institutional ownership data to ranker metrics.
    
    Args:
        institutional_df: Institutional ownership data from FMP
        
    Returns:
        DataFrame with institutional_ownership_delta, top5_concentration_delta, etc.
    """
    if institutional_df.empty:
        return pd.DataFrame()
    
    df = institutional_df.copy()
    
    # Aggregate by symbol
    agg = df.groupby("symbol").agg({
        "ownership_percent": ["sum", "count", "max"],
        "value": "sum"
    }).reset_index()
    
    agg.columns = ["symbol", "total_ownership_pct", "holder_count", "max_ownership_pct", "total_value"]
    
    output = pd.DataFrame()
    output["symbol"] = agg["symbol"]
    output["percent_institutions"] = pd.to_numeric(agg["total_ownership_pct"], errors="coerce")
    output["institution_holder_count_latest"] = pd.to_numeric(agg["holder_count"], errors="coerce")
    output["institutional_ownership_from_holders"] = output["percent_institutions"]
    
    # Top 5 concentration (using max ownership as proxy)
    output["institutional_top5_concentration_delta"] = pd.to_numeric(
        agg["max_ownership_pct"],
        errors="coerce"
    )
    
    # Placeholder for delta (requires historical data)
    output["institutional_ownership_delta"] = np.nan
    output["institutional_breadth_delta"] = np.nan
    output["institution_holder_count_prev"] = np.nan
    
    return output


def map_insider_trading(insider_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map FMP insider trading data to ranker conviction metrics.
    
    Args:
        insider_df: Insider trading data from FMP
        
    Returns:
        DataFrame with insider_conviction_buy_cluster, buy_person_count, etc.
    """
    if insider_df.empty:
        return pd.DataFrame()
    
    df = insider_df.copy()
    
    # Normalize column names to handle both raw FMP and pre-renamed columns
    # FMP /stable/insider-trading/latest returns: reportingName, securitiesOwned,
    # transaction_type, acquisitionOrDisposition, etc.
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
    
    # Ensure numeric shares column
    if "shares" in df.columns:
        df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0)
    else:
        df["shares"] = 0
    
    # Ensure insider_name exists for aggregation
    if "insider_name" not in df.columns:
        df["insider_name"] = "unknown"
    
    # Filter for open-market purchases (exclude 10b5-1 plans)
    # FMP transaction_type uses: P-Purchase, S-Sale, A-Award, etc.
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
        # Aggregate buys by symbol
        buy_agg = open_market_buys.groupby("symbol").agg(
            total_shares_bought=("shares", "sum"),
            unique_buyers=("insider_name", "nunique"),
            trade_count=("shares", "count"),
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
    
    # Sell pressure
    if not open_market_sells.empty and "symbol" in open_market_sells.columns:
        sell_agg = open_market_sells.groupby("symbol").agg(
            total_shares_sold=("shares", "sum"),
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
    
    # Fill remaining fields
    output["insider_conviction_signal"] = np.nan
    output["insider_conviction_has_coverage"] = np.where(
        output["insider_conviction_buy_cluster"].notna() | output["insider_conviction_sell_pressure"].notna(),
        1.0, 0.0
    )
    output["insider_short_crowding_penalty"] = 0.0
    
    # Fill NaN for columns that might not have been set
    for col in ["insider_conviction_trade_count", "insider_conviction_buy_cluster",
                "insider_conviction_buy_person_count"]:
        if col not in output.columns:
            output[col] = np.nan
    
    return output


def merge_all_data(
    profiles_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    financials_df: pd.DataFrame,
    beneish_df: pd.DataFrame,
    estimates_df: pd.DataFrame,
    institutional_df: pd.DataFrame,
    insider_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge all mapped data into a single DataFrame for the ranker.
    """
    # --- Defensive Deduplication ---
    # Ensure all dataframes are 1-row-per-symbol before merging to avoid many-to-many explosions
    def _dedupe(df):
        if df.empty or "symbol" not in df.columns:
            return df
        return df.sort_values("symbol").groupby("symbol").last().reset_index()

    profiles_df = _dedupe(profiles_df)
    financials_df = _dedupe(financials_df)
    beneish_df = _dedupe(beneish_df)
    estimates_df = _dedupe(estimates_df)
    institutional_df = _dedupe(institutional_df)
    insider_df = _dedupe(insider_df)
    # ------------------------------

    # Start with profiles
    if not profiles_df.empty:
        merged = profiles_df.copy()
    else:
        merged = pd.DataFrame(columns=["symbol"])
    
    # Merge financials
    if not financials_df.empty:
        merged = merged.merge(financials_df, on="symbol", how="outer")
    
    # Merge Beneish
    if not beneish_df.empty:
        merged = merged.merge(beneish_df, on="symbol", how="outer")
    
    # Merge estimates
    if not estimates_df.empty:
        merged = merged.merge(estimates_df, on="symbol", how="outer")
    
    # Merge institutional
    if not institutional_df.empty:
        merged = merged.merge(institutional_df, on="symbol", how="outer")
    
    # Merge insider
    if not insider_df.empty:
        merged = merged.merge(insider_df, on="symbol", how="outer")
    
    # Merge prices (Batch Quote fields)
    if not prices_df.empty:
        # Get latest price/quote for each symbol
        if "date" in prices_df.columns:
            latest_prices = prices_df.sort_values("date").groupby("symbol").last().reset_index()
        else:
            latest_prices = prices_df
        
        # Technical Indicator Mapping from Batch Quote
        price_mapping = {
            "yearHigh": "52_week_high",
            "priceAvg200": "200_day_ma",
            "priceAvg50": "50_day_ma",
        }
        for fmp_col, internal_col in price_mapping.items():
            if fmp_col in latest_prices.columns and internal_col not in latest_prices.columns:
                latest_prices[internal_col] = latest_prices[fmp_col]
        
        merged = merged.merge(latest_prices, on="symbol", how="left")
    
    # Derived technicals if missing
    if "price" in merged.columns and "200_day_ma" in merged.columns:
        merged["price_to_200dma"] = merged["price"] / merged["200_day_ma"].replace(0, np.nan)
    if "price" in merged.columns and "52_week_high" in merged.columns:
        merged["distance_from_high"] = merged["price"] / merged["52_week_high"].replace(0, np.nan)

    # --- Deduplicate _x / _y column collisions from multiple merges ---
    # Prefer _x (earlier source = profile/screener data), then fill with _y
    x_cols = [c for c in merged.columns if c.endswith("_x")]
    for xcol in x_cols:
        base = xcol[:-2]
        ycol = f"{base}_y"
        if ycol in merged.columns:
            merged[base] = merged[xcol].combine_first(merged[ycol])
            merged.drop(columns=[xcol, ycol], inplace=True)
        else:
            merged.rename(columns={xcol: base}, inplace=True)
    
    # Ensure symbol is first column
    if "symbol" in merged.columns:
        cols = ["symbol"] + [col for col in merged.columns if col != "symbol"]
        merged = merged[cols]
    
    return merged


def create_raw_fmp_dataframe(
    bulk_data: Dict[str, pd.DataFrame],
    market: str = "us",
    financial_period: str = "annual",
    historical_estimates_df: Optional[pd.DataFrame] = None,
    surprises_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Create the raw DataFrame from FMP bulk data that the ranker expects.
    
    This is the main entry point for the mapper. It takes the raw bulk data
    from FMP and transforms it into the exact column structure that
    build_ranked_frame expects.
    
    Args:
        bulk_data: Dictionary of DataFrames from FMPClient.fetch_all_bulk_data
        market: Market identifier
        financial_period: "annual" or "quarterly"
        historical_estimates_df: Historical analyst estimates time series (optional)
        surprises_df: Earnings surprises data (optional)
        
    Returns:
        Raw DataFrame ready for build_ranked_frame
    """
    # Extract data from bulk_data
    profiles_df = bulk_data.get("profiles", pd.DataFrame())
    prices_df = bulk_data.get("prices", pd.DataFrame())
    income_df = bulk_data.get("income_statements", pd.DataFrame())
    balance_df = bulk_data.get("balance_sheets", pd.DataFrame())
    cashflow_df = bulk_data.get("cash_flows", pd.DataFrame())
    estimates_df = bulk_data.get("analyst_estimates", pd.DataFrame())
    institutional_df = bulk_data.get("institutional_ownership", pd.DataFrame())
    insider_df = bulk_data.get("insider_trading", pd.DataFrame())
    
    # Map each data type
    logger.info("Mapping FMP profile data")
    mapped_profiles = map_profile_data(profiles_df)
    
    logger.info("Mapping FMP financial statements")
    mapped_financials = map_financial_statements(income_df, balance_df, cashflow_df, profiles_df)
    
    logger.info("Mapping Beneish M-Score components")
    mapped_beneish = map_beneish_components(income_df, balance_df, cashflow_df)
    
    logger.info("Mapping analyst estimates")
    mapped_estimates = map_analyst_estimates(
        estimates_df,
        historical_estimates_df=historical_estimates_df,
        surprises_df=surprises_df
    )
    
    logger.info("Mapping institutional ownership")
    mapped_institutional = map_institutional_ownership(institutional_df)
    
    logger.info("Mapping insider trading")
    mapped_insider = map_insider_trading(insider_df)
    
    # Merge all mapped data
    logger.info("Merging all mapped data")
    raw_df = merge_all_data(
        mapped_profiles,
        prices_df,
        mapped_financials,
        mapped_beneish,
        mapped_estimates,
        mapped_institutional,
        mapped_insider
    )
    
    logger.info(f"Created raw DataFrame with {len(raw_df)} symbols and {len(raw_df.columns)} columns")
    
    return raw_df
