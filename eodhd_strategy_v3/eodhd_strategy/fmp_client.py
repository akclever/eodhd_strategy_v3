"""
FMP (Financial Modeling Prep) Ultimate API Client

This module provides an asynchronous client for fetching bulk financial data.
Refactored to support Graceful Degradation: Missing endpoints (403/404) 
safely return empty arrays to feed NaNs into the dynamic Correlation Parity aggregator.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FMPConfig:
    """Configuration for FMP client."""
    api_key: str
    cache_dir: Path
    api_base_url: str = "https://financialmodelingprep.com/stable"
    max_retries: int = 5
    retry_delay: float = 1.0
    request_timeout: float = 30.0
    batch_size: int = 100


class FMPClient:
    """
    Asynchronous FMP API client with graceful degradation.
    """

    def __init__(self, config: FMPConfig):
        self.config = config
        self.cache_dir = config.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._api_base_url = config.api_base_url
        
    async def __aenter__(self):
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.request_timeout),
            headers={"User-Agent": "eodhd_strategy_v3/2.0"}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()
    
    async def _get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        retry_count: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Async GET request with exponential backoff and GRACEFUL DEGRADATION.
        Will never crash on 400, 401, 403, or 404. Returns [] instead.
        """
        if params is None:
            params = {}
        
        params["apikey"] = self.config.api_key
        url = f"{self._api_base_url}/{endpoint}"
        
        try:
            async with self._session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if isinstance(data, dict):
                        return data.get("data", [])
                    return data if isinstance(data, list) else [data]
                
                # Graceful Degradation: Do not crash if tier limits or missing endpoints are hit
                if response.status in [400, 401, 403, 404]:
                    # Silent skip - don't print anything to avoid garbled output
                    return []
                
                # Retry on rate limits (429) or server errors (5xx)
                if response.status in [429, 500, 502, 503, 504]:
                    if retry_count < self.config.max_retries:
                        delay = self.config.retry_delay * (2 ** retry_count)
                        await asyncio.sleep(delay)
                        return await self._get(endpoint, params, retry_count + 1)
                    else:
                        return []
                
                # Catch-all for unexpected statuses
                return []
                
        except aiohttp.ClientError as e:
            if retry_count < self.config.max_retries:
                delay = self.config.retry_delay * (2 ** retry_count)
                await asyncio.sleep(delay)
                return await self._get(endpoint, params, retry_count + 1)
            
            return []
    
    async def fetch_bulk_daily_prices(self, symbols: list[str] = None) -> pd.DataFrame:
        """Fetch daily prices using /stable/batch-quote for multiple symbols."""
        if not symbols:
            return pd.DataFrame()
            
        # Batch quote supports multiple symbols comma-separated
        symbol_str = ",".join(symbols[:100])  # Limit to 100 symbols per request
        endpoint = "batch-quote"
        params = {"symbols": symbol_str}
        
        data = await self._get(endpoint, params)
        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        available_mapping = {k: v for k, v in {
            "symbol": "symbol", "price": "close", 
            "change": "change", "changesPercentage": "change_percent",
            "dayLow": "low", "dayHigh": "high", 
            "volume": "volume", "marketCap": "market_cap"
        }.items() if k in df.columns}
        
        df = df.rename(columns=available_mapping)
        return df
    
    async def fetch_screener(
        self,
        market_cap_more_than: Optional[float] = None,
        market_cap_lower_than: Optional[float] = None,
        sector: Optional[str] = None,
        industry: Optional[str] = None,
        beta_more_than: Optional[float] = None,
        beta_lower_than: Optional[float] = None,
        price_more_than: Optional[float] = None,
        price_lower_than: Optional[float] = None,
        volume_more_than: Optional[float] = None,
        volume_lower_than: Optional[float] = None,
        exchange: Optional[str] = None,
        country: Optional[str] = None,
        is_etf: Optional[bool] = None,
        is_actively_trading: Optional[bool] = True,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Fetch stock screener results using /stable/company-screener."""
        endpoint = "company-screener"
        params = {"limit": limit}
        
        if market_cap_more_than is not None:
            params["marketCapMoreThan"] = market_cap_more_than
        if market_cap_lower_than is not None:
            params["marketCapLowerThan"] = market_cap_lower_than
        if sector:
            params["sector"] = sector
        if industry:
            params["industry"] = industry
        if beta_more_than is not None:
            params["betaMoreThan"] = beta_more_than
        if beta_lower_than is not None:
            params["betaLowerThan"] = beta_lower_than
        if price_more_than is not None:
            params["priceMoreThan"] = price_more_than
        if price_lower_than is not None:
            params["priceLowerThan"] = price_lower_than
        if volume_more_than is not None:
            params["volumeMoreThan"] = volume_more_than
        if volume_lower_than is not None:
            params["volumeLowerThan"] = volume_lower_than
        if exchange:
            params["exchange"] = exchange
        if country:
            params["country"] = country
        if is_etf is not None:
            params["isEtf"] = is_etf
        if is_actively_trading is not None:
            params["isActivelyTrading"] = is_actively_trading
        
        data = await self._get(endpoint, params)
        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        available_mapping = {k: v for k, v in {
            "symbol": "symbol", "companyName": "company_name",
            "marketCap": "market_cap", "sector": "sector", "industry": "industry",
            "beta": "beta", "price": "price", "volume": "volume",
            "exchange": "exchange", "country": "country",
            "isEtf": "is_etf", "isActivelyTrading": "is_actively_trading",
        }.items() if k in df.columns}
        
        df = df.rename(columns=available_mapping)
        return df
    
    async def fetch_profile_bulk(self, market: str = "us", part: int = 0) -> pd.DataFrame:
        """Fetch bulk profiles using /stable/profile-bulk with part parameter."""
        endpoint = "profile-bulk"
        params = {"part": part}
        
        data = await self._get(endpoint, params)
        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        available_mapping = {k: v for k, v in {
            "symbol": "symbol", "mktCap": "market_cap", "marketCap": "market_cap",
            "sector": "sector", "industry": "industry", 
            "isActivelyTrading": "is_actively_trading",
            "companyName": "company_name", "exchange": "exchange", "country": "country",
        }.items() if k in df.columns}
        
        df = df.rename(columns=available_mapping)
        if "is_actively_trading" in df.columns:
            df["is_actively_trading"] = df["is_actively_trading"].astype(bool)
        if "market_cap" in df.columns:
            df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")
        return df
    
    async def fetch_income_statement_bulk(
        self, symbols: list[str] = None, period: str = "annual", limit: int = 100
    ) -> pd.DataFrame:
        """Fetch income statements per-symbol using /stable/income-statement."""
        if not symbols:
            return pd.DataFrame()
            
        all_data = []
        for symbol in symbols[:limit]:
            endpoint = "income-statement"
            params = {"symbol": symbol, "period": period}
            data = await self._get(endpoint, params)
            if data:
                # Add symbol to each record if not present
                for record in data:
                    if "symbol" not in record:
                        record["symbol"] = symbol
                all_data.extend(data)
            
        if not all_data:
            return pd.DataFrame()
            
        df = pd.DataFrame(all_data)
        available_mapping = {k: v for k, v in {
            "symbol": "symbol", "date": "date", "period": "period",
            "revenue": "total_revenue", "totalRevenue": "total_revenue",
            "costOfRevenue": "cost_of_revenue",
            "grossProfit": "gross_profit", 
            "researchAndDevelopmentExpenses": "rd_expenses",
            "sellingGeneralAndAdministrativeExpenses": "sga_expenses",
            "operatingExpenses": "operating_expenses", "operatingIncome": "operating_income",
            "netIncome": "net_income", "ebitda": "ebitda", "eps": "eps"
        }.items() if k in df.columns}
        
        df = df.rename(columns=available_mapping)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df

    async def fetch_balance_sheet_bulk(
        self, symbols: list[str] = None, period: str = "annual", limit: int = 100
    ) -> pd.DataFrame:
        """Fetch balance sheets per-symbol using /stable/balance-sheet-statement."""
        if not symbols:
            return pd.DataFrame()
            
        all_data = []
        for symbol in symbols[:limit]:
            endpoint = "balance-sheet-statement"
            params = {"symbol": symbol, "period": period}
            data = await self._get(endpoint, params)
            if data:
                for record in data:
                    if "symbol" not in record:
                        record["symbol"] = symbol
                all_data.extend(data)
            
        if not all_data:
            return pd.DataFrame()
            
        df = pd.DataFrame(all_data)
        available_mapping = {k: v for k, v in {
            "symbol": "symbol", "date": "date", "period": "period",
            "totalAssets": "total_assets", "totalLiabilities": "total_liabilities",
            "totalStockholdersEquity": "shareholders_equity", 
            "totalStockholdersEquity": "total_stockholders_equity",
            "commonStockSharesOutstanding": "shares_outstanding",
            "netReceivables": "net_receivables", "inventory": "inventory",
            "accountPayables": "account_payables", "cashAndCashEquivalents": "cash_and_equivalents",
            "intangibleAssets": "intangible_assets", "goodWill": "goodwill",
        }.items() if k in df.columns}
        
        df = df.rename(columns=available_mapping)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df

    async def fetch_cash_flow_bulk(
        self, symbols: list[str] = None, period: str = "annual", limit: int = 100
    ) -> pd.DataFrame:
        """Fetch cash flow statements per-symbol using /stable/cash-flow-statement."""
        if not symbols:
            return pd.DataFrame()
            
        all_data = []
        for symbol in symbols[:limit]:
            endpoint = "cash-flow-statement"
            params = {"symbol": symbol, "period": period}
            data = await self._get(endpoint, params)
            if data:
                for record in data:
                    if "symbol" not in record:
                        record["symbol"] = symbol
                all_data.extend(data)
            
        if not all_data:
            return pd.DataFrame()
            
        df = pd.DataFrame(all_data)
        available_mapping = {k: v for k, v in {
            "symbol": "symbol", "date": "date", "period": "period",
            "netCashProvidedByOperatingActivities": "operating_cash_flow",
            "operatingCashFlow": "operating_cash_flow", "capitalExpenditure": "capital_expenditure",
            "freeCashFlow": "free_cash_flow", "dividendsPaid": "dividends_paid",
            "commonStockIssued": "stock_issued", "commonStockRepurchased": "stock_repurchased",
        }.items() if k in df.columns}
        
        df = df.rename(columns=available_mapping)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    
    async def fetch_analyst_estimates_bulk(
        self, symbols: list[str] = None, period: str = "annual", limit: int = 100
    ) -> pd.DataFrame:
        """Fetch analyst estimates per-symbol using /stable/analyst-estimates."""
        if not symbols:
            return pd.DataFrame()
            
        all_data = []
        for symbol in symbols[:limit]:
            endpoint = "analyst-estimates"
            params = {"symbol": symbol, "period": period, "page": 0, "limit": 10}
            data = await self._get(endpoint, params)
            if data:
                for record in data:
                    if "symbol" not in record:
                        record["symbol"] = symbol
                all_data.extend(data)
                
        return pd.DataFrame(all_data)

    async def fetch_institutional_ownership_bulk(
        self, symbols: list[str] = None, year: int = None, quarter: int = None
    ) -> pd.DataFrame:
        """Fetch institutional ownership per-symbol using /stable/institutional-ownership/symbol-positions-summary."""
        if not symbols:
            return pd.DataFrame()
        
        # Default to most recent quarter if not specified
        if year is None:
            year = datetime.now().year
        if quarter is None:
            quarter = 3  # Q3 as default
            
        all_data = []
        for symbol in symbols[:100]:
            endpoint = "institutional-ownership/symbol-positions-summary"
            params = {"symbol": symbol, "year": year, "quarter": quarter}
            data = await self._get(endpoint, params)
            if data:
                for record in data:
                    if "symbol" not in record:
                        record["symbol"] = symbol
                all_data.extend(data)
                
        return pd.DataFrame(all_data)
    
    async def fetch_insider_trading_bulk(self, page: int = 0, limit: int = 100) -> pd.DataFrame:
        """Fetch insider trading using /stable/insider-trading/latest."""
        endpoint = "insider-trading/latest"
        params = {"page": page, "limit": limit}
        
        data = await self._get(endpoint, params)
        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        available_mapping = {k: v for k, v in {
            "symbol": "symbol", "filingDate": "filing_date", "transactionDate": "transaction_date",
            "insiderName": "insider_name", "insiderTitle": "insider_title",
            "transactionType": "transaction_type", "shares": "shares", "price": "price",
            "value": "value", "acquistionOrDisposition": "acquisition_or_disposition",
        }.items() if k in df.columns}
        
        df = df.rename(columns=available_mapping)
        for date_col in ["filing_date", "transaction_date"]:
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col])
        return df
    
    async def fetch_all_bulk_data(
        self, market: str = "us", financial_period: str = "annual", symbols: list[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch all available data using /stable/ endpoints.
        
        Strategy:
        1. Get universe from screener or profile-bulk
        2. Fetch per-symbol data for financial statements
        3. Fetch bulk/latest data for insider trading
        """
        output = {}
        
        # Step 1: Get universe of symbols
        if symbols is None:
            # Try screener first for active stocks
            screener_df = await self.fetch_screener(
                is_actively_trading=True,
                market_cap_more_than=100_000_000,  # $100M+
                limit=5000
            )
            if not screener_df.empty and "symbol" in screener_df.columns:
                symbols = screener_df["symbol"].tolist()
                output["screener"] = screener_df
            else:
                # Fallback to profile-bulk
                profile_df = await self.fetch_profile_bulk(market=market, part=0)
                if not profile_df.empty and "symbol" in profile_df.columns:
                    symbols = profile_df["symbol"].tolist()[:500]  # Limit to 500
                    output["profiles"] = profile_df
                else:
                    symbols = []
        
        if not symbols:
            # Return empty data if we can't get symbols
            return {k: pd.DataFrame() for k in [
                "profiles", "prices", "income_statements", "balance_sheets", 
                "cash_flows", "insider_trading", "analyst_estimates", "institutional_ownership"
            ]}
        
        # Step 2: Fetch bulk data that works without symbols
        output["insider_trading"] = await self.fetch_insider_trading_bulk(page=0, limit=100)
        
        # Step 3: Fetch per-symbol data (limit to first 100 symbols to avoid rate limits)
        limited_symbols = symbols[:100]
        
        # Fetch prices for all symbols at once using batch-quote
        output["prices"] = await self.fetch_bulk_daily_prices(symbols=limited_symbols)
        
        # Fetch financial statements per-symbol
        output["income_statements"] = await self.fetch_income_statement_bulk(
            symbols=limited_symbols, period=financial_period
        )
        output["balance_sheets"] = await self.fetch_balance_sheet_bulk(
            symbols=limited_symbols, period=financial_period
        )
        output["cash_flows"] = await self.fetch_cash_flow_bulk(
            symbols=limited_symbols, period=financial_period
        )
        
        # Fetch estimates and institutional ownership per-symbol
        output["analyst_estimates"] = await self.fetch_analyst_estimates_bulk(
            symbols=limited_symbols, period=financial_period
        )
        output["institutional_ownership"] = await self.fetch_institutional_ownership_bulk(
            symbols=limited_symbols
        )
        
        # Ensure profiles exists (from screener or fetch separately)
        if "profiles" not in output and "screener" in output:
            output["profiles"] = output["screener"]
        elif "profiles" not in output:
            output["profiles"] = pd.DataFrame()
            
        return output
