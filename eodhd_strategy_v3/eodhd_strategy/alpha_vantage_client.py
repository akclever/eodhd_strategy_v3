from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .client import TokenBucketRateLimiter

logger = logging.getLogger(__name__)

AV_BASE_URL = "https://www.alphavantage.co/query"
AV_USER_AGENT = "eodhd-smart-ranker/3.0"
_AV_CACHE_IO_LOCK = threading.RLock()


class AlphaVantageClient:
    """Alpha Vantage API client with caching, rate limiting, and retry logic.

    Mirrors the EODHDClient pattern so it can be used as a drop-in data source
    behind the unified DataProvider adapter.
    """

    def __init__(
        self,
        api_key: str,
        cache_dir: Path,
        refresh: bool = False,
        timeout: int = 30,
        rate_limit_per_minute: int = 75,
        burst: int = 5,
    ):
        self.api_key = api_key
        self.cache_dir = cache_dir / "alpha_vantage"
        self.refresh = refresh
        self.timeout = timeout
        self.rate_limiter = TokenBucketRateLimiter(rate_limit_per_minute, burst)
        self.session = requests.Session()
        retry = Retry(
            total=5,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
        self.session.mount("https://", adapter)
        self.session.headers.update({"User-Agent": AV_USER_AGENT})
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Core request helpers
    # ------------------------------------------------------------------

    def _cache_path(self, params: Dict[str, Any]) -> Path:
        filtered = {k: v for k, v in sorted(params.items()) if k != "apikey"}
        payload = json.dumps(filtered, sort_keys=True, default=str)
        key = hashlib.md5(payload.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.json"

    def get_json(
        self,
        params: Dict[str, Any],
        ttl_hours: Optional[float] = None,
        request_cost: int = 1,
    ) -> Any:
        """Make a cached, rate-limited GET to the Alpha Vantage query endpoint."""
        params = dict(params)
        params["apikey"] = self.api_key
        cache_path = self._cache_path(params)

        with _AV_CACHE_IO_LOCK:
            if not self.refresh and cache_path.exists():
                if ttl_hours is None:
                    return json.loads(cache_path.read_text(encoding="utf-8"))
                age_hours = (time.time() - cache_path.stat().st_mtime) / 3600.0
                if age_hours <= ttl_hours:
                    return json.loads(cache_path.read_text(encoding="utf-8"))

        self.rate_limiter.consume(max(1, request_cost))
        response = self.session.get(
            AV_BASE_URL, params=params, timeout=self.timeout
        )

        if response.status_code >= 400:
            body = response.text[:300].strip()
            raise RuntimeError(
                f"Alpha Vantage {params.get('function', '?')} failed HTTP "
                f"{response.status_code}: {body or 'No response body'}"
            )

        data = response.json()

        # Alpha Vantage returns errors inside JSON with specific keys
        if isinstance(data, dict):
            if "Error Message" in data:
                raise RuntimeError(
                    f"AV API error for {params.get('function', '?')}: "
                    f"{data['Error Message']}"
                )
            if "Information" in data and "rate limit" in data["Information"].lower():
                logger.warning("AV rate limit hit, sleeping 60s …")
                time.sleep(60)
                return self.get_json(params, ttl_hours, request_cost)
            if "Note" in data and "call frequency" in data.get("Note", "").lower():
                logger.warning("AV call-frequency note, sleeping 60s …")
                time.sleep(60)
                return self.get_json(params, ttl_hours, request_cost)

        payload_str = json.dumps(data)
        tmp_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
        with _AV_CACHE_IO_LOCK:
            tmp_path.write_text(payload_str, encoding="utf-8")
            tmp_path.replace(cache_path)
        return data

    # ------------------------------------------------------------------
    # Fundamentals
    # ------------------------------------------------------------------

    def get_income_statement(self, symbol: str) -> Dict[str, Any]:
        return self.get_json(
            {"function": "INCOME_STATEMENT", "symbol": symbol},
            ttl_hours=24,
        )

    def get_balance_sheet(self, symbol: str) -> Dict[str, Any]:
        return self.get_json(
            {"function": "BALANCE_SHEET", "symbol": symbol},
            ttl_hours=24,
        )

    def get_cash_flow(self, symbol: str) -> Dict[str, Any]:
        return self.get_json(
            {"function": "CASH_FLOW", "symbol": symbol},
            ttl_hours=24,
        )

    def get_company_overview(self, symbol: str) -> Dict[str, Any]:
        return self.get_json(
            {"function": "OVERVIEW", "symbol": symbol},
            ttl_hours=24,
        )

    # ------------------------------------------------------------------
    # Earnings
    # ------------------------------------------------------------------

    def get_earnings(self, symbol: str) -> Dict[str, Any]:
        return self.get_json(
            {"function": "EARNINGS", "symbol": symbol},
            ttl_hours=24,
        )

    # ------------------------------------------------------------------
    # Price data
    # ------------------------------------------------------------------

    def get_daily_prices(
        self,
        symbol: str,
        outputsize: str = "full",
    ) -> Dict[str, Any]:
        return self.get_json(
            {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": symbol,
                "outputsize": outputsize,
            },
            ttl_hours=12,
        )

    def get_weekly_prices(self, symbol: str) -> Dict[str, Any]:
        return self.get_json(
            {"function": "TIME_SERIES_WEEKLY_ADJUSTED", "symbol": symbol},
            ttl_hours=12,
        )

    # ------------------------------------------------------------------
    # Dividends
    # ------------------------------------------------------------------

    def get_dividends(self, symbol: str) -> Dict[str, Any]:
        """Uses the TIME_SERIES_DAILY_ADJUSTED which includes dividends,
        or the dedicated DIVIDENDS endpoint if available on premium."""
        return self.get_json(
            {"function": "DIVIDENDS", "symbol": symbol},
            ttl_hours=24,
        )

    # ------------------------------------------------------------------
    # Technical indicators
    # ------------------------------------------------------------------

    def get_technical_indicator(
        self,
        symbol: str,
        function: str,
        interval: str = "daily",
        time_period: int = 14,
        series_type: str = "close",
        **extra_params: Any,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "function": function,
            "symbol": symbol,
            "interval": interval,
            "time_period": time_period,
            "series_type": series_type,
        }
        params.update(extra_params)
        return self.get_json(params, ttl_hours=12)

    def get_rsi(self, symbol: str, time_period: int = 14) -> Dict[str, Any]:
        return self.get_technical_indicator(symbol, "RSI", time_period=time_period)

    def get_macd(self, symbol: str) -> Dict[str, Any]:
        return self.get_json(
            {
                "function": "MACD",
                "symbol": symbol,
                "interval": "daily",
                "series_type": "close",
            },
            ttl_hours=12,
        )

    def get_bbands(self, symbol: str, time_period: int = 20) -> Dict[str, Any]:
        return self.get_technical_indicator(symbol, "BBANDS", time_period=time_period)

    def get_adx(self, symbol: str, time_period: int = 14) -> Dict[str, Any]:
        return self.get_technical_indicator(symbol, "ADX", time_period=time_period)

    def get_stoch(self, symbol: str) -> Dict[str, Any]:
        return self.get_json(
            {"function": "STOCH", "symbol": symbol, "interval": "daily"},
            ttl_hours=12,
        )

    def get_vwap(self, symbol: str) -> Dict[str, Any]:
        return self.get_json(
            {"function": "VWAP", "symbol": symbol, "interval": "15min"},
            ttl_hours=12,
        )

    # ------------------------------------------------------------------
    # News & sentiment
    # ------------------------------------------------------------------

    def get_news_sentiment(
        self,
        tickers: str | None = None,
        topics: str | None = None,
        time_from: str | None = None,
        time_to: str | None = None,
        sort: str = "LATEST",
        limit: int = 50,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "function": "NEWS_SENTIMENT",
            "sort": sort,
            "limit": min(limit, 1000),
        }
        if tickers:
            params["tickers"] = tickers
        if topics:
            params["topics"] = topics
        if time_from:
            params["time_from"] = time_from
        if time_to:
            params["time_to"] = time_to
        return self.get_json(params, ttl_hours=6)

    # ------------------------------------------------------------------
    # Macro / economic indicators
    # ------------------------------------------------------------------

    def get_economic_indicator(self, function: str, interval: str = "monthly") -> Dict[str, Any]:
        """Fetch macro-economic data series from Alpha Vantage.

        Supported functions include: REAL_GDP, REAL_GDP_PER_CAPITA, TREASURY_YIELD,
        FEDERAL_FUNDS_RATE, CPI, INFLATION, RETAIL_SALES, DURABLES, UNEMPLOYMENT,
        NONFARM_PAYROLL.
        """
        params: Dict[str, Any] = {"function": function, "interval": interval}
        return self.get_json(params, ttl_hours=24)

    def get_treasury_yield(self, interval: str = "monthly", maturity: str = "10year") -> Dict[str, Any]:
        return self.get_json(
            {"function": "TREASURY_YIELD", "interval": interval, "maturity": maturity},
            ttl_hours=24,
        )

    def get_cpi(self, interval: str = "monthly") -> Dict[str, Any]:
        return self.get_economic_indicator("CPI", interval)

    def get_unemployment(self) -> Dict[str, Any]:
        return self.get_economic_indicator("UNEMPLOYMENT")

    def get_real_gdp(self, interval: str = "quarterly") -> Dict[str, Any]:
        return self.get_economic_indicator("REAL_GDP", interval)

    def get_federal_funds_rate(self, interval: str = "monthly") -> Dict[str, Any]:
        return self.get_economic_indicator("FEDERAL_FUNDS_RATE", interval)

    def get_inflation(self) -> Dict[str, Any]:
        return self.get_economic_indicator("INFLATION")

    # ------------------------------------------------------------------
    # Analyst / estimates (via Earnings endpoint)
    # ------------------------------------------------------------------

    def get_analyst_estimates(self, symbol: str) -> Dict[str, Any]:
        """Alpha Vantage embeds analyst estimates in the EARNINGS endpoint.
        This is an alias for convenience."""
        return self.get_earnings(symbol)
