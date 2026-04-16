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
    
    # Merge profiles for market cap
    if not profiles_df.empty:
        df = df.merge(profiles_df[["symbol", "market_cap"]], on="symbol", how="left")
    
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
        if fmp_col in df.columns:
            output[ranker_col] = pd.to_numeric(df[fmp_col], errors="coerce")
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
        if fmp_col in df.columns:
            output[ranker_col] = pd.to_numeric(df[fmp_col], errors="coerce")
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
        if fmp_col in df.columns:
            output[ranker_col] = pd.to_numeric(df[fmp_col], errors="coerce")
        else:
            output[ranker_col] = np.nan
    
    # Ensure market_cap is available
    if "market_cap" in df.columns:
        output["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")
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
    
    # dividend_yield (will be computed from dividends data, placeholder for now)
    output["dividend_yield"] = np.nan
    
    # buyback_yield = stock_repurchased / market_cap
    # Note: stock_repurchased is typically negative in FMP data
    stock_repurchased_abs = output["stock_repurchased"].abs()
    output["buyback_yield"] = _safe_divide(
        stock_repurchased_abs,
        output["market_cap"],
        default=0.0
    )
    
    # shareholder_yield = dividend_yield + buyback_yield
    # If either component is missing, result should be NaN
    output["shareholder_yield"] = output["dividend_yield"] + output["buyback_yield"]
    
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
        output["intangible_adjustment_eligible"] & 
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
    
    # Compute components if both periods are available
    if not prior.empty:
        # Merge current and prior
        merged = current.merge(
            prior,
            on="symbol",
            how="left",
            suffixes=("_current", "_prior")
        )
        
        # DSRI: Days Sales in Receivables Index
        # (Receivables_current / Sales_current) / (Receivables_prior / Sales_prior)
        receivables_current = pd.to_numeric(merged["net_receivables_current"], errors="coerce")
        receivables_prior = pd.to_numeric(merged["net_receivables_prior"], errors="coerce")
        sales_current = pd.to_numeric(merged["total_revenue_current"], errors="coerce")
        sales_prior = pd.to_numeric(merged["total_revenue_prior"], errors="coerce")
        
        dsr_current = _safe_divide(receivables_current, sales_current, default=np.nan)
        dsr_prior = _safe_divide(receivables_prior, sales_prior, default=np.nan)
        output["dsri"] = _safe_divide(dsr_current, dsr_prior, default=np.nan)
        
        # GMI: Gross Margin Index
        # Gross Margin_prior / Gross Margin_current
        gross_profit_current = pd.to_numeric(merged["gross_profit_current"], errors="coerce")
        gross_profit_prior = pd.to_numeric(merged["gross_profit_prior"], errors="coerce")
        
        gm_current = _safe_divide(gross_profit_current, sales_current, default=np.nan)
        gm_prior = _safe_divide(gross_profit_prior, sales_prior, default=np.nan)
        output["gmi"] = _safe_divide(gm_prior, gm_current, default=np.nan)
        
        # SGI: Sales Growth Index
        # Sales_current / Sales_prior
        output["sgi"] = _safe_divide(sales_current, sales_prior, default=np.nan)
        
        # DEPI: Depreciation Index
        # (Depreciation_prior / (PP&E_prior + Depreciation_prior)) / 
        # (Depreciation_current / (PP&E_current + Depreciation_current))
        # Note: FMP doesn't provide depreciation directly, use capital expenditure as proxy
        capex_current = pd.to_numeric(merged["capital_expenditure_current"], errors="coerce").abs()
        capex_prior = pd.to_numeric(merged["capital_expenditure_prior"], errors="coerce").abs()
        ppe_current = pd.to_numeric(merged["total_assets_current"], errors="coerce")  # Proxy
        ppe_prior = pd.to_numeric(merged["total_assets_prior"], errors="coerce")
        
        depi_current = _safe_divide(capex_current, ppe_current + capex_current, default=np.nan)
        depi_prior = _safe_divide(capex_prior, ppe_prior + capex_prior, default=np.nan)
        output["depi"] = _safe_divide(depi_prior, depi_current, default=np.nan)
        
        # SGAI: SG&A Index
        # (SG&A_current / Sales_current) / (SG&A_prior / Sales_prior)
        sga_current = pd.to_numeric(merged["sga_expenses_current"], errors="coerce")
        sga_prior = pd.to_numeric(merged["sga_expenses_prior"], errors="coerce")
        
        sga_ratio_current = _safe_divide(sga_current, sales_current, default=np.nan)
        sga_ratio_prior = _safe_divide(sga_prior, sales_prior, default=np.nan)
        output["sgai"] = _safe_divide(sga_ratio_current, sga_ratio_prior, default=np.nan)
        
        # LVGI: Leverage Index
        # ((LTD_current + Current Liabilities_current) / Total Assets_current) /
        # ((LTD_prior + Current Liabilities_prior) / Total Assets_prior)
        # Simplified: Total Liabilities / Total Assets
        liabilities_current = pd.to_numeric(merged["total_liabilities_current"], errors="coerce")
        liabilities_prior = pd.to_numeric(merged["total_liabilities_prior"], errors="coerce")
        assets_current = pd.to_numeric(merged["total_assets_current"], errors="coerce")
        assets_prior = pd.to_numeric(merged["total_assets_prior"], errors="coerce")
        
        leverage_current = _safe_divide(liabilities_current, assets_current, default=np.nan)
        leverage_prior = _safe_divide(liabilities_prior, assets_prior, default=np.nan)
        output["lvgi"] = _safe_divide(leverage_current, leverage_prior, default=np.nan)
        
        # TATA: Total Accruals to Total Assets
        # (Income from Continuing Operations - Cash Flows from Operations) / Total Assets
        cfo_current = pd.to_numeric(merged["operating_cash_flow_current"], errors="coerce")
        income_current = pd.to_numeric(merged["net_income_current"], errors="coerce")
        
        accruals = income_current - cfo_current
        output["tata"] = _safe_divide(accruals, assets_current, default=np.nan)
        
        # AQI: Asset Quality Index
        # (1 - ((Current Assets + PP&E) / Total Assets))_current /
        # (1 - ((Current Assets + PP&E) / Total Assets))_prior
        # Simplified: (1 - (Cash + Receivables + Inventory) / Total Assets)
        cash_current = pd.to_numeric(merged["cash_and_equivalents_current"], errors="coerce")
        cash_prior = pd.to_numeric(merged["cash_and_equivalents_prior"], errors="coerce")
        inventory_current = pd.to_numeric(merged["inventory_current"], errors="coerce")
        inventory_prior = pd.to_numeric(merged["inventory_prior"], errors="coerce")
        
        current_assets_proxy = cash_current + receivables_current + inventory_current
        current_assets_proxy_prior = cash_prior + receivables_prior + inventory_prior
        
        asset_quality_current = 1 - _safe_divide(current_assets_proxy, assets_current, default=np.nan)
        asset_quality_prior = 1 - _safe_divide(current_assets_proxy_prior, assets_prior, default=np.nan)
        output["aqi"] = _safe_divide(asset_quality_current, asset_quality_prior, default=np.nan)
    
    # Compute Beneish M-Score
    # M = -4.84 + 0.92*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI + 0.115*DEPI - 0.172*SGAI + 4.679*TATA - 0.327*LVGI
    output["beneish_m_score"] = (
        -4.84 +
        0.92 * output["dsri"] +
        0.528 * output["gmi"] +
        0.404 * output["aqi"] +
        0.892 * output["sgi"] +
        0.115 * output["depi"] -
        0.172 * output["sgai"] +
        4.679 * output["tata"] -
        0.327 * output["lvgi"]
    )
    
    # Flag missing components
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
    
    df = estimates_df.copy()
    
    output = pd.DataFrame()
    output["symbol"] = df["symbol"]
    
    # Map basic fields
    # analyst_count is a count field, so 0 is appropriate if no analysts
    output["revision_impulse_analyst_count"] = pd.to_numeric(
        df.get("number_of_analysts", 0),
        errors="coerce"
    ).fillna(0)
    
    # EPS estimates
    if "estimated_eps" in df.columns:
        output["estimated_eps"] = pd.to_numeric(df["estimated_eps"], errors="coerce")
    else:
        output["estimated_eps"] = np.nan
    
    # Revenue estimates
    if "estimated_revenue" in df.columns:
        output["estimated_revenue"] = pd.to_numeric(df["estimated_revenue"], errors="coerce")
    else:
        output["estimated_revenue"] = np.nan
    
    # Initialize revision metrics
    output["revision_impulse_signal"] = np.nan
    output["revision_impulse_has_coverage"] = 0.0
    output["revision_jerk_signal"] = np.nan
    output["revision_jerk_has_coverage"] = 0.0
    output["revision_jerk_recent_velocity"] = np.nan
    output["revision_jerk_prior_velocity"] = np.nan
    output["revision_impulse_disagreement"] = np.nan
    output["revision_impulse_disagreement_penalty"] = np.nan
    
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
    
    # Compute revision jerk from surprises if available
    if surprises_df is not None and not surprises_df.empty:
        logger.info("Computing revision jerk from earnings surprises")
        
        surprises_df = surprises_df.copy()
        
        if "surprise_percent" in surprises_df.columns:
            # Aggregate surprise by symbol
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
                )
                
                # Disagreement penalty: higher std = higher penalty
                # If disagreement is NaN, penalty should be NaN
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
    
    # Filter for open-market purchases (exclude 10b5-1 plans)
    # Transaction types: P-Purchase, S-Sale, etc.
    if "transaction_type" in df.columns:
        open_market_buys = df[
            df["transaction_type"].str.contains("P", case=False, na=False) &
            ~df["transaction_type"].str.contains("10b5", case=False, na=False)
        ]
    else:
        open_market_buys = df
    
    if open_market_buys.empty:
        output = pd.DataFrame(columns=["symbol"])
        return output
    
    # Aggregate by symbol
    agg = open_market_buys.groupby("symbol").agg({
        "shares": "sum",
        "insider_name": "nunique",
        "value": "sum"
    }).reset_index()
    
    agg.columns = ["symbol", "total_shares_bought", "unique_buyers", "total_value"]
    
    output = pd.DataFrame()
    output["symbol"] = agg["symbol"]
    output["insider_conviction_buy_cluster"] = pd.to_numeric(
        agg["total_shares_bought"],
        errors="coerce"
    )
    output["insider_conviction_buy_person_count"] = pd.to_numeric(
        agg["unique_buyers"],
        errors="coerce"
    )
    
    # Placeholder for other metrics
    output["insider_conviction_signal"] = np.nan
    output["insider_conviction_has_coverage"] = 0.0
    output["insider_conviction_sell_pressure"] = np.nan
    output["insider_conviction_trade_count"] = np.nan
    output["insider_conviction_sell_person_count"] = np.nan
    output["insider_short_crowding_penalty"] = 0.0
    
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
    
    Args:
        profiles_df: Mapped profile data
        prices_df: Price data
        financials_df: Mapped financial statements
        beneish_df: Mapped Beneish components
        estimates_df: Mapped analyst estimates
        institutional_df: Mapped institutional ownership
        insider_df: Mapped insider trading
        
    Returns:
        Unified DataFrame with all columns expected by build_ranked_frame
    """
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
    
    # Merge prices
    if not prices_df.empty:
        # Get latest price for each symbol
        if "date" in prices_df.columns:
            latest_prices = prices_df.sort_values("date").groupby("symbol").last().reset_index()
        else:
            latest_prices = prices_df
        merged = merged.merge(latest_prices, on="symbol", how="left")
    
    # Ensure symbol is first column
    if "symbol" in merged.columns:
        cols = ["symbol"] + [col for col in merged.columns if col != "symbol"]
        merged = merged[cols]
    
    return merged


def create_raw_fmp_dataframe(
    bulk_data: Dict[str, pd.DataFrame],
    market: str = "us",
    financial_period: str = "annual",
    historical_estimates_df: Optional[pd.DataFrame] = None
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
        surprises_df=None
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
