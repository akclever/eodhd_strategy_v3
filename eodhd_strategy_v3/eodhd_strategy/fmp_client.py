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
                    text = await response.text()
                    text_lstrip = text.lstrip("\ufeff\n\r \t")
                    first_line = text_lstrip.splitlines()[0] if text_lstrip else ""
                    first_line_lower = first_line.lower()
                    looks_like_csv = (
                        text_lstrip.startswith("symbol,")
                        or text_lstrip.startswith('"symbol",')
                        or text_lstrip.startswith('symbol\r')
                        or text_lstrip.startswith('symbol\n')
                        or ("," in first_line and "symbol" in first_line_lower)
                    )
                    if looks_like_csv:
                        import io
                        import pandas as pd
                        try:
                            df = pd.read_csv(io.StringIO(text))
                            return df.to_dict(orient="records")
                        except Exception as e:
                            print(f"Error parsing CSV from {url}: {e}")
                            return []
                    else:
                        import json
                        try:
                            data = json.loads(text)
                            if isinstance(data, dict):
                                return data.get("data", [])
                            return data if isinstance(data, list) else [data]
                        except Exception as e:
                            print(f"Error parsing JSON from {url}: {e}")
                            return []
                
                # Client errors: raise RuntimeError to make endpoint failures visible during debugging
                if response.status in [400, 401, 403, 404]:
                    body = await response.text()
                    raise RuntimeError(
                        f"FMP API error: endpoint='{endpoint}', "
                        f"url='{url}', "
                        f"status={response.status}, "
                        f"body={body[:300]!r}"
                    )
                
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
            params["isEtf"] = str(is_etf).lower()
        if is_actively_trading is not None:
            params["isActivelyTrading"] = str(is_actively_trading).lower()
        
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
        self, symbols: list[str] = None, period: str = "annual", limit: int = 100, periods: int = 4
    ) -> pd.DataFrame:
        """Fetch income statements per-symbol using /stable/income-statement."""
        if not symbols:
            return pd.DataFrame()

        all_data = []
        for symbol in symbols[:limit]:
            endpoint = "income-statement"
            params = {"symbol": symbol, "period": period, "limit": periods}
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
            "revenue": "revenue", "costOfRevenue": "cost_of_revenue",
            "grossProfit": "gross_profit", "grossProfitRatio": "gross_margin",
            "researchAndDevelopmentExpenses": "rd_expense",
            "operatingExpenses": "operating_expenses", "operatingIncome": "operating_income",
            "netIncome": "net_income", "eps": "eps", "epsDiluted": "eps_diluted",
            "weightedAverageShsOut": "shares_outstanding",
        }.items() if k in df.columns}
        
        df = df.rename(columns=available_mapping)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df

    async def fetch_employee_count_bulk(
        self, symbols: list[str] = None, limit: int = 100
    ) -> pd.DataFrame:
        """Fetch employee counts per-symbol using /stable/employee-count."""
        if not symbols:
            return pd.DataFrame()

        all_data = []
        for symbol in symbols[:limit]:
            endpoint = "employee-count"
            params = {"symbol": symbol}
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
            "symbol": "symbol",
            "date": "date",
            "employeeCount": "full_time_employees",
            "fullTimeEmployees": "full_time_employees",
            "employees": "full_time_employees",
        }.items() if k in df.columns}

        df = df.rename(columns=available_mapping)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "full_time_employees" in df.columns:
            df["full_time_employees"] = pd.to_numeric(df["full_time_employees"], errors="coerce")
        return df

    async def fetch_financial_estimates_bulk(
        self,
        symbols: list[str] = None,
        period: str = "quarter",
        limit: int = 100,
        historical_limit: int = 20,
    ) -> pd.DataFrame:
        """Fetch financial estimates using /stable/analyst-estimates."""
        if not symbols:
            return pd.DataFrame()

        all_data = []
        period_value = "quarter" if period == "quarterly" else period
        for symbol in symbols[:limit]:
            endpoint = "analyst-estimates"
            params = {"symbol": symbol, "period": period_value, "page": 0, "limit": historical_limit}
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
            "symbol": "symbol",
            "date": "date",
            "period": "period",
            "estimatedEpsAvg": "estimated_eps",
            "epsAvg": "estimated_eps",
            "estimatedRevenueAvg": "estimated_revenue",
            "revenueAvg": "estimated_revenue",
            "numberAnalystEstimatedEps": "analyst_count",
            "numberAnalystEstimatedRevenue": "analyst_count_revenue",
        }.items() if k in df.columns}
        df = df.rename(columns=available_mapping)

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for col in ["estimated_eps", "estimated_revenue", "analyst_count", "analyst_count_revenue"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    async def fetch_earnings_company_bulk(
        self, symbols: list[str] = None, limit: int = 100
    ) -> pd.DataFrame:
        """Fetch per-symbol earnings history via /stable/earnings-company."""
        if not symbols:
            return pd.DataFrame()

        all_data = []
        for symbol in symbols[:limit]:
            endpoint = "earnings-company"
            params = {"symbol": symbol}
            logger.debug(f"FMP: calling endpoint={endpoint} for symbol={symbol}")
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
            "symbol": "symbol",
            "date": "date",
            "period": "period",
            "actualEps": "actual_eps",
            "epsActual": "actual_eps",
            "estimatedEps": "estimated_eps",
            "epsEstimated": "estimated_eps",
            "surprisePercent": "surprise_percent",
        }.items() if k in df.columns}
        df = df.rename(columns=available_mapping)

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for col in ["actual_eps", "estimated_eps", "surprise_percent"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    async def fetch_insider_trading_statistics_bulk(
        self, symbols: list[str] = None, limit: int = 100
    ) -> pd.DataFrame:
        """Fetch insider summary statistics via /stable/insider-trading/statistics."""
        if not symbols:
            return pd.DataFrame()

        all_data = []
        for symbol in symbols[:limit]:
            endpoint = "insider-trading/statistics"
            params = {"symbol": symbol}
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
            "symbol": "symbol",
            "totalBuy": "total_buy",
            "totalSell": "total_sell",
            "buyCount": "buy_count",
            "sellCount": "sell_count",
            "netActivity": "net_activity",
            "totalTransactions": "trade_count",
        }.items() if k in df.columns}
        df = df.rename(columns=available_mapping)

        for col in ["total_buy", "total_sell", "buy_count", "sell_count", "net_activity", "trade_count"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    async def fetch_income_statement_bulk_by_year(
        self, year: int, period: str = "annual", symbols: list[str] = None
    ) -> pd.DataFrame:
        """Fetch yearly income statement bulk via /stable/income-statement-bulk."""
        endpoint = "income-statement-bulk"
        params = {"year": year, "period": period}
        data = await self._get(endpoint, params)
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        if symbols and "symbol" in df.columns:
            df = df[df["symbol"].isin(set(symbols))]
        return df

    async def fetch_income_statement_growth_bulk_by_year(
        self, year: int, period: str = "annual", symbols: list[str] = None
    ) -> pd.DataFrame:
        """Fetch yearly income growth bulk via /stable/income-statement-growth-bulk."""
        endpoint = "income-statement-growth-bulk"
        params = {"year": year, "period": period}
        data = await self._get(endpoint, params)
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        if symbols and "symbol" in df.columns:
            df = df[df["symbol"].isin(set(symbols))]
        return df

    async def fetch_balance_sheet_bulk_by_year(
        self, year: int, period: str = "annual", symbols: list[str] = None
    ) -> pd.DataFrame:
        """Fetch yearly balance sheet bulk via /stable/balance-sheet-statement-bulk."""
        endpoint = "balance-sheet-statement-bulk"
        params = {"year": year, "period": period}
        data = await self._get(endpoint, params)
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        if symbols and "symbol" in df.columns:
            df = df[df["symbol"].isin(set(symbols))]
        return df

    async def fetch_cash_flow_bulk_by_year(
        self, year: int, period: str = "annual", symbols: list[str] = None
    ) -> pd.DataFrame:
        """Fetch yearly cash flow bulk via /stable/cash-flow-statement-bulk."""
        endpoint = "cash-flow-statement-bulk"
        params = {"year": year, "period": period}
        data = await self._get(endpoint, params)
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        if symbols and "symbol" in df.columns:
            df = df[df["symbol"].isin(set(symbols))]
        return df

    async def fetch_scores_bulk(self, symbols: list[str] = None) -> pd.DataFrame:
        """Fetch score set via /stable/scores-bulk (includes piotroski)."""
        endpoint = "scores-bulk"
        data = await self._get(endpoint, params={})
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        if symbols and "symbol" in df.columns:
            df = df[df["symbol"].isin(set(symbols))]
        return df

    async def fetch_historical_price_eod_bulk(
        self, symbols: list[str] = None, mode: str = "full", limit: int = 100
    ) -> pd.DataFrame:
        """Fetch historical EOD prices via /stable/historical-price-eod/{full|light}."""
        if not symbols:
            return pd.DataFrame()

        endpoint = "historical-price-eod/full" if mode == "full" else "historical-price-eod/light"
        all_data = []
        for symbol in symbols[:limit]:
            data = await self._get(endpoint, {"symbol": symbol})
            if not data:
                continue
            for record in data:
                if "symbol" not in record:
                    record["symbol"] = symbol
            all_data.extend(data)

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        mapping = {
            "symbol": "symbol",
            "date": "date",
            "close": "close",
            "adjClose": "adj_close",
            "adjustedClose": "adj_close",
            "volume": "volume",
            "open": "open",
            "high": "high",
            "low": "low",
        }
        available_mapping = {k: v for k, v in mapping.items() if k in df.columns}
        df = df.rename(columns=available_mapping)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for col in ["open", "high", "low", "close", "adj_close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    async def fetch_eod_bulk(self, date: str) -> pd.DataFrame:
        """Fetch one-day bulk EOD cross section via /stable/eod-bulk."""
        endpoint = "eod-bulk"
        data = await self._get(endpoint, {"date": date})
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    async def fetch_technical_indicator_bulk(
        self,
        indicator: str,
        symbols: list[str] = None,
        period_length: int = 14,
        timeframe: str = "1day",
        limit: int = 100,
    ) -> pd.DataFrame:
        """Fetch technical indicator series for symbols via /stable/technical-indicators/{indicator}."""
        if not symbols:
            return pd.DataFrame()

        endpoint = f"technical-indicators/{indicator}"
        all_data = []
        for symbol in symbols[:limit]:
            params = {"symbol": symbol, "periodLength": period_length, "timeframe": timeframe}
            data = await self._get(endpoint, params)
            if not data:
                continue
            for record in data:
                if "symbol" not in record:
                    record["symbol"] = symbol
                record["indicator_type"] = indicator
            all_data.extend(data)

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    async def fetch_peers_bulk(self) -> pd.DataFrame:
        """Fetch full peers list via /stable/peers-bulk."""
        data = await self._get("peers-bulk", params={})
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data)

    async def fetch_peers_for_symbols(self, symbols: list[str] = None, limit: int = 100) -> pd.DataFrame:
        """Fetch symbol peers via /stable/stock-peers?symbol=..."""
        if not symbols:
            return pd.DataFrame()
        all_data = []
        endpoint = "stock-peers"
        for symbol in symbols[:limit]:
            logger.debug(f"FMP: calling endpoint={endpoint} for symbol={symbol}")
            data = await self._get(endpoint, {"symbol": symbol})
            if not data:
                continue
            for record in data:
                if "symbol" not in record:
                    record["symbol"] = symbol
            all_data.extend(data)
        return pd.DataFrame(all_data) if all_data else pd.DataFrame()

    async def fetch_search_insider_trades_bulk(self, symbols: list[str] = None, limit: int = 100) -> pd.DataFrame:
        """Fetch insider trades per symbol via /stable/insider-trading/search."""
        if not symbols:
            return pd.DataFrame()
        all_data = []
        endpoint = "insider-trading/search"
        for symbol in symbols[:limit]:
            logger.debug(f"FMP: calling endpoint={endpoint} for symbol={symbol}")
            data = await self._get(endpoint, {"symbol": symbol})
            if not data:
                continue
            for record in data:
                if "symbol" not in record:
                    record["symbol"] = symbol
            all_data.extend(data)
        return pd.DataFrame(all_data) if all_data else pd.DataFrame()

    async def fetch_historical_employee_count_bulk(self, symbols: list[str] = None, limit: int = 100) -> pd.DataFrame:
        """Fetch historical employee counts via /stable/historical-employee-count."""
        if not symbols:
            return pd.DataFrame()
        all_data = []
        for symbol in symbols[:limit]:
            data = await self._get("historical-employee-count", {"symbol": symbol})
            if not data:
                continue
            for record in data:
                if "symbol" not in record:
                    record["symbol"] = symbol
            all_data.extend(data)
        if not all_data:
            return pd.DataFrame()
        df = pd.DataFrame(all_data)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    async def fetch_positions_summary_bulk(
        self, symbols: list[str] = None, limit: int = 100, year: int = None, quarter: int = None
    ) -> pd.DataFrame:
        """Fetch institutional summary via /stable/institutional-ownership/symbol-positions-summary."""
        if not symbols:
            return pd.DataFrame()
        all_data = []
        endpoint = "institutional-ownership/symbol-positions-summary"
        for symbol in symbols[:limit]:
            params = {"symbol": symbol}
            if year is not None:
                params["year"] = year
            if quarter is not None:
                params["quarter"] = quarter
            logger.debug(f"FMP: calling endpoint={endpoint} for symbol={symbol}")
            data = await self._get(endpoint, params)
            if not data:
                continue
            for record in data:
                if "symbol" not in record:
                    record["symbol"] = symbol
            all_data.extend(data)
        return pd.DataFrame(all_data) if all_data else pd.DataFrame()

    async def fetch_balance_sheet_bulk(
        self, symbols: list[str] = None, period: str = "annual", limit: int = 100, periods: int = 4
    ) -> pd.DataFrame:
        """Fetch balance sheets per-symbol using /stable/balance-sheet-statement."""
        if not symbols:
            return pd.DataFrame()

        all_data = []
        for symbol in symbols[:limit]:
            endpoint = "balance-sheet-statement"
            params = {"symbol": symbol, "period": period, "limit": periods}
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
            "totalStockholdersEquity": "total_stockholders_equity",
            "netReceivables": "net_receivables", "inventory": "inventory",
            "accountPayables": "account_payables", "cashAndCashEquivalents": "cash_and_equivalents",
            "intangibleAssets": "intangible_assets", "goodwill": "goodwill",
        }.items() if k in df.columns}
        
        df = df.rename(columns=available_mapping)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df

    async def fetch_cash_flow_bulk(
        self, symbols: list[str] = None, period: str = "annual", limit: int = 100, periods: int = 4
    ) -> pd.DataFrame:
        """Fetch cash flow statements per-symbol using /stable/cash-flow-statement."""
        if not symbols:
            return pd.DataFrame()

        all_data = []
        for symbol in symbols[:limit]:
            endpoint = "cash-flow-statement"
            params = {"symbol": symbol, "period": period, "limit": periods}
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
            "freeCashFlow": "free_cash_flow", 
            "commonDividendsPaid": "dividends_paid", "netDividendsPaid": "dividends_paid",
            "commonStockIssued": "stock_issued", "netCommonStockIssuance": "stock_issued",
            "commonStockRepurchased": "stock_repurchased",
        }.items() if k in df.columns}
        
        df = df.rename(columns=available_mapping)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    
    async def fetch_analyst_estimates_bulk(
        self, symbols: list[str] = None, period: str = "annual", limit: int = 100, historical_limit: int = 10
    ) -> pd.DataFrame:
        """Fetch analyst estimates per-symbol using /stable/analyst-estimates."""
        if not symbols:
            return pd.DataFrame()
            
        all_data = []
        for symbol in symbols[:limit]:
            endpoint = "analyst-estimates"
            params = {"symbol": symbol, "period": period, "page": 0, "limit": historical_limit}
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
            "symbol": "symbol", "date": "date",
            "period": "period",
            "estimatedEpsAvg": "estimated_eps", "estimatedEpsHigh": "estimated_eps_high",
            "estimatedEpsLow": "estimated_eps_low", "numberAnalystEstimatedEps": "analyst_count",
            "estimatedRevenueAvg": "estimated_revenue",
        }.items() if k in df.columns}
        
        df = df.rename(columns=available_mapping)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df

    async def fetch_earnings_surprises_bulk(
        self, year: int = 2026
    ) -> pd.DataFrame:
        """Fetch bulk earnings surprises using /stable/earnings-surprises-bulk."""
        endpoint = "earnings-surprises-bulk"
        params = {"year": year}
        data = await self._get(endpoint, params)
        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        available_mapping = {k: v for k, v in {
            "symbol": "symbol", "date": "date", 
            "actualEps": "actual_eps", "epsActual": "actual_eps",
            "estimatedEps": "estimated_eps", "epsEstimated": "estimated_eps",
            "surprisePercent": "surprise_percent",
        }.items() if k in df.columns}
        
        df = df.rename(columns=available_mapping)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df

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
