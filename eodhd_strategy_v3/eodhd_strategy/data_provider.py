from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from .client import EODHDClient

logger = logging.getLogger(__name__)

ProviderMode = Literal["eodhd", "alpha_vantage", "hybrid", "fmp"]


class DataProvider:
    """Unified adapter that abstracts over EODHD, Alpha Vantage, and SEC EDGAR.

    In **eodhd** mode every call routes to the existing ``EODHDClient``.
    In **alpha_vantage** mode calls route to ``AlphaVantageClient`` with
    SEC EDGAR for institutional/insider data.
    In **hybrid** mode Alpha Vantage is preferred with EODHD as fallback;
    SEC EDGAR is always used for institutional/insider data when available.

    The public interface mirrors the methods already consumed by the codebase
    (``get_fundamentals``, ``get_price_history``, etc.) so that callers need
    only swap ``EODHDClient`` for ``DataProvider``.
    """

    def __init__(
        self,
        mode: ProviderMode = "eodhd",
        eodhd_client: Optional[EODHDClient] = None,
        av_client: Optional["AlphaVantageClient"] = None,  # noqa: F821 – lazy import
        edgar_client: Optional["SECEdgarClient"] = None,  # noqa: F821 – lazy import
        fmp_client: Optional["FMPClient"] = None,  # noqa: F821 – lazy import
    ):
        self.mode = mode
        self.eodhd = eodhd_client
        self.av = av_client
        self.edgar = edgar_client
        self.fmp = fmp_client

        if mode == "eodhd" and eodhd_client is None:
            raise ValueError("eodhd mode requires an EODHDClient")
        if mode == "alpha_vantage" and av_client is None:
            raise ValueError("alpha_vantage mode requires an AlphaVantageClient")
        if mode == "hybrid" and (eodhd_client is None or av_client is None):
            raise ValueError("hybrid mode requires both EODHDClient and AlphaVantageClient")
        if mode == "fmp" and fmp_client is None:
            raise ValueError("fmp mode requires an FMPClient")

    # ------------------------------------------------------------------
    # Pass-through helpers (keep API-surface compatible with EODHDClient)
    # ------------------------------------------------------------------

    @property
    def _eodhd(self) -> EODHDClient:
        assert self.eodhd is not None
        return self.eodhd

    # Expose the same attributes that callers may access on the old client
    @property
    def cache_dir(self) -> Path:
        if self.eodhd is not None:
            return self.eodhd.cache_dir
        if self.av is not None:
            return self.av.cache_dir.parent  # av stores in cache_dir/alpha_vantage
        if self.fmp is not None:
            return self.fmp.config.cache_dir
        raise RuntimeError("No client available for cache_dir")

    @property
    def refresh(self) -> bool:
        if self.eodhd is not None:
            return self.eodhd.refresh
        if self.av is not None:
            return self.av.refresh
        return False

    # ------------------------------------------------------------------
    # Universe / exchange helpers  (EODHD-only for now)
    # ------------------------------------------------------------------

    def get_exchange_symbols(self, exchange_code: str) -> list:
        if self.eodhd is None:
            raise RuntimeError("get_exchange_symbols requires EODHD; use --symbols or --symbols-file in alpha_vantage mode")
        return self._eodhd.get_exchange_symbols(exchange_code)

    def search_instruments(self, query: str, limit: int = 100, exchange: Optional[str] = None):
        if self.eodhd is None:
            raise RuntimeError("search_instruments requires EODHD; use --symbols or --symbols-file in alpha_vantage mode")
        return self._eodhd.search_instruments(query, limit, exchange)

    def get_supported_exchanges(self):
        if self.eodhd is None:
            return []
        return self._eodhd.get_supported_exchanges()

    def get_user(self) -> Dict[str, Any]:
        if self.eodhd is None:
            return {}
        return self._eodhd.get_user()

    # ------------------------------------------------------------------
    # Fundamentals
    # ------------------------------------------------------------------

    def get_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Return fundamentals in the EODHD-compatible dict structure.

        When Alpha Vantage is the source the response is normalised to the
        nested dict format that the rest of the codebase already expects.
        """
        if self.mode == "eodhd":
            data = self._eodhd.get_fundamentals(symbol)
            self._enrich_with_edgar_holders(data, symbol)
            return data

        if self.mode == "alpha_vantage":
            return self._av_fundamentals(symbol)

        # hybrid: try AV, fallback to EODHD
        try:
            data = self._av_fundamentals(symbol)
            if data and data.get("Financials"):
                return data
        except Exception as exc:
            logger.debug("AV fundamentals failed for %s, falling back: %s", symbol, exc)
        data = self._eodhd.get_fundamentals(symbol)
        self._enrich_with_edgar_holders(data, symbol)
        return data

    def _enrich_with_edgar_holders(self, fundamentals: Dict[str, Any], symbol: str) -> None:
        """Inject SEC EDGAR 13F holders into fundamentals if EDGAR is available
        and the existing Holders section is empty or missing."""
        if self.edgar is None:
            return
        existing = fundamentals.get("Holders") or {}
        has_institutions = bool(
            (existing.get("Institutions") if isinstance(existing, dict) else None)
        )
        if has_institutions:
            return
        try:
            holders = self._edgar_holders(symbol)
            if holders:
                fundamentals["Holders"] = holders
        except Exception as exc:
            logger.debug("EDGAR holders enrichment failed for %s: %s", symbol, exc)

    def _av_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Build an EODHD-shaped fundamentals dict from Alpha Vantage data."""
        assert self.av is not None
        ticker = symbol.split(".")[0]  # strip exchange suffix

        overview = self.av.get_company_overview(ticker)
        income = self.av.get_income_statement(ticker)
        balance = self.av.get_balance_sheet(ticker)
        cashflow = self.av.get_cash_flow(ticker)
        earnings = self.av.get_earnings(ticker)

        fundamentals: Dict[str, Any] = {
            "General": _av_overview_to_general(overview),
            "Highlights": _av_overview_to_highlights(overview),
            "Valuation": _av_overview_to_valuation(overview),
            "SharesStats": _av_overview_to_shares_stats(overview),
            "Financials": {
                "Income_Statement": {
                    "yearly": _av_reports_to_dict(income.get("annualReports", [])),
                    "quarterly": _av_reports_to_dict(income.get("quarterlyReports", [])),
                },
                "Balance_Sheet": {
                    "yearly": _av_reports_to_dict(balance.get("annualReports", [])),
                    "quarterly": _av_reports_to_dict(balance.get("quarterlyReports", [])),
                },
                "Cash_Flow": {
                    "yearly": _av_reports_to_dict(cashflow.get("annualReports", [])),
                    "quarterly": _av_reports_to_dict(cashflow.get("quarterlyReports", [])),
                },
            },
            "Earnings": _av_earnings_to_eodhd(earnings),
            "outstandingShares": _av_overview_to_outstanding_shares(overview),
        }

        # Enrich with SEC EDGAR institutional holders if available
        if self.edgar is not None:
            try:
                fundamentals["Holders"] = self._edgar_holders(symbol)
            except Exception as exc:
                logger.debug("EDGAR holders failed for %s: %s", symbol, exc)
                fundamentals["Holders"] = {}

        return fundamentals

    def _edgar_holders(self, symbol: str) -> Dict[str, Any]:
        """Fetch and normalise SEC EDGAR institutional holders into EODHD format."""
        assert self.edgar is not None
        ticker = symbol.split(".")[0]
        holdings = self.edgar.get_13f_holdings(ticker, lookback_quarters=4)

        if not holdings:
            return {}

        # Group holdings by report_date to aggregate per-quarter
        by_date: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for h in holdings:
            by_date[h.get("report_date", "")].append(h)

        # Deduplicate filers per date and build EODHD-compatible records.
        # Use value_x1000 as a proxy for relative ownership weight since
        # we don't have total shares outstanding to compute a percentage.
        institutions: List[Dict[str, Any]] = []
        for report_date, entries in by_date.items():
            seen_filers: set = set()
            for h in entries:
                filer = h.get("filer_name", "")
                filer_key = filer.lower().strip()
                if filer_key in seen_filers:
                    continue
                seen_filers.add(filer_key)
                # totalShares: use a small proxy value so _percentage_points_to_fraction
                # gives a non-zero result. Value_x1000 / 1e6 gives a rough % proxy.
                value_proxy = float(h.get("value_x1000", 0) or 0) / 1_000_000.0
                institutions.append({
                    "name": filer,
                    "date": report_date,
                    "totalShares": value_proxy,
                    "totalAssets": None,
                    "currentShares": h.get("shares", 0),
                    "change": None,
                    "change_p": None,
                    "_source": "edgar_13f",
                })

        return {"Institutions": institutions, "Funds": []}

    # ------------------------------------------------------------------
    # Price history
    # ------------------------------------------------------------------

    def get_price_history(
        self,
        symbol: str,
        from_date: str | None = None,
        to_date: str | None = None,
        period: str = "d",
    ) -> list:
        """Return daily OHLCV records as a list of dicts.

        EODHD returns ``[{date, open, high, low, close, adjusted_close, volume}, …]``.
        Alpha Vantage is normalised to the same shape.
        """
        if self.mode == "eodhd":
            return self._eodhd.get_price_history(symbol, from_date, to_date, period)

        if self.mode == "alpha_vantage":
            return self._av_price_history(symbol, from_date, to_date)

        # hybrid
        try:
            data = self._av_price_history(symbol, from_date, to_date)
            if data:
                return data
        except Exception as exc:
            logger.debug("AV price history failed for %s, falling back: %s", symbol, exc)
        return self._eodhd.get_price_history(symbol, from_date, to_date, period)

    def _av_price_history(
        self,
        symbol: str,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> list:
        assert self.av is not None
        ticker = symbol.split(".")[0]
        raw = self.av.get_daily_prices(ticker, outputsize="full")
        ts_key = "Time Series (Daily)"
        time_series = raw.get(ts_key, {})

        records = []
        for date_str, bar in sorted(time_series.items()):
            if from_date and date_str < from_date:
                continue
            if to_date and date_str > to_date:
                continue
            records.append({
                "date": date_str,
                "open": _safe_float(bar.get("1. open")),
                "high": _safe_float(bar.get("2. high")),
                "low": _safe_float(bar.get("3. low")),
                "close": _safe_float(bar.get("4. close")),
                "adjusted_close": _safe_float(bar.get("5. adjusted close")),
                "volume": _safe_float(bar.get("6. volume")),
            })
        return records

    # ------------------------------------------------------------------
    # Dividends
    # ------------------------------------------------------------------

    def get_dividends(self, symbol: str, start_date: str) -> list:
        if self.mode == "eodhd":
            return self._eodhd.get_dividends(symbol, start_date)

        if self.mode == "alpha_vantage":
            return self._av_dividends(symbol, start_date)

        try:
            data = self._av_dividends(symbol, start_date)
            if data:
                return data
        except Exception as exc:
            logger.debug("AV dividends failed for %s, falling back: %s", symbol, exc)
        return self._eodhd.get_dividends(symbol, start_date)

    def _av_dividends(self, symbol: str, start_date: str) -> list:
        assert self.av is not None
        ticker = symbol.split(".")[0]
        raw = self.av.get_dividends(ticker)
        data_list = raw.get("data", [])
        records = []
        for entry in data_list:
            ex_date = entry.get("ex_dividend_date", "")
            if ex_date < start_date:
                continue
            records.append({
                "date": ex_date,
                "value": _safe_float(entry.get("amount")),
                "unadjustedValue": _safe_float(entry.get("amount")),
                "currency": entry.get("currency", "USD"),
            })
        return records

    # ------------------------------------------------------------------
    # Earnings
    # ------------------------------------------------------------------

    def get_earnings(self, symbol: str) -> Any:
        if self.mode == "eodhd":
            return self._eodhd.get_earnings(symbol)

        if self.mode == "alpha_vantage":
            return self._av_earnings_calendar(symbol)

        try:
            data = self._av_earnings_calendar(symbol)
            if data:
                return data
        except Exception as exc:
            logger.debug("AV earnings failed for %s, falling back: %s", symbol, exc)
        return self._eodhd.get_earnings(symbol)

    def _av_earnings_calendar(self, symbol: str) -> list:
        """Normalise AV earnings to EODHD calendar format."""
        assert self.av is not None
        ticker = symbol.split(".")[0]
        raw = self.av.get_earnings(ticker)
        quarterly = raw.get("quarterlyEarnings", [])
        records = []
        for q in quarterly:
            records.append({
                "code": f"{ticker}.US",
                "report_date": q.get("reportedDate", ""),
                "date": q.get("fiscalDateEnding", ""),
                "actual": _safe_float(q.get("reportedEPS")),
                "estimate": _safe_float(q.get("estimatedEPS")),
                "difference": _safe_float(q.get("surprise")),
                "percent": _safe_float(q.get("surprisePercentage")),
            })
        return records

    # ------------------------------------------------------------------
    # Earnings calendar (dedicated endpoint)
    # ------------------------------------------------------------------

    def get_earnings_calendar(
        self,
        symbol: str,
        from_date: str | None = None,
        to_date: str | None = None,
    ):
        if self.eodhd is not None:
            return self._eodhd.get_earnings_calendar(symbol, from_date, to_date)
        return self._av_earnings_calendar(symbol)

    # ------------------------------------------------------------------
    # News / sentiment
    # ------------------------------------------------------------------

    def get_news(
        self,
        symbol: str | None = None,
        tag: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ):
        if self.mode == "eodhd":
            return self._eodhd.get_news(symbol, tag, start_date, end_date, limit, offset)

        if self.mode in ("alpha_vantage", "hybrid"):
            try:
                return self._av_news(symbol, start_date, end_date, limit)
            except Exception as exc:
                logger.debug("AV news failed for %s, falling back: %s", symbol, exc)
                if self.eodhd is not None:
                    return self._eodhd.get_news(symbol, tag, start_date, end_date, limit, offset)
                raise

        return []

    def _av_news(
        self,
        symbol: str | None,
        start_date: str | None,
        end_date: str | None,
        limit: int,
    ) -> list:
        """Fetch news from Alpha Vantage and normalise to EODHD shape."""
        assert self.av is not None
        ticker = symbol.split(".")[0] if symbol else None
        # AV expects dates as YYYYMMDDTHHMM
        time_from = f"{start_date.replace('-', '')}T0000" if start_date else None
        time_to = f"{end_date.replace('-', '')}T2359" if end_date else None

        raw = self.av.get_news_sentiment(
            tickers=ticker,
            time_from=time_from,
            time_to=time_to,
            limit=limit,
        )
        feed = raw.get("feed", [])
        records = []
        for article in feed:
            # Find ticker-specific sentiment score
            ticker_sentiment = 0.0
            relevance = 0.0
            for ts in article.get("ticker_sentiment", []):
                if ticker and ts.get("ticker", "").upper() == ticker.upper():
                    ticker_sentiment = _safe_float(ts.get("ticker_sentiment_score"))
                    relevance = _safe_float(ts.get("relevance_score"))
                    break

            records.append({
                "date": _av_datetime_to_date(article.get("time_published", "")),
                "datetime": article.get("time_published", ""),
                "title": article.get("title", ""),
                "content": article.get("summary", ""),
                "link": article.get("url", ""),
                "symbols": [ts.get("ticker", "") for ts in article.get("ticker_sentiment", [])],
                "tags": [topic.get("topic", "") for topic in article.get("topics", [])],
                "sentiment": {
                    "polarity": _safe_float(article.get("overall_sentiment_score")),
                    "neg": 0.0,
                    "neu": 0.0,
                    "pos": 0.0,
                },
                "av_ticker_sentiment": ticker_sentiment,
                "av_relevance": relevance,
                "av_overall_label": article.get("overall_sentiment_label", ""),
            })
        return records

    def get_sentiments(self, symbol: str, start_date: str, end_date: str):
        if self.mode == "eodhd":
            return self._eodhd.get_sentiments(symbol, start_date, end_date)

        if self.mode in ("alpha_vantage", "hybrid"):
            try:
                return self._av_sentiments(symbol, start_date, end_date)
            except Exception as exc:
                logger.debug("AV sentiments failed for %s, falling back: %s", symbol, exc)
                if self.eodhd is not None:
                    return self._eodhd.get_sentiments(symbol, start_date, end_date)
                raise

        return {}

    def _av_sentiments(self, symbol: str, start_date: str, end_date: str) -> dict:
        """Build EODHD-compatible sentiments dict from AV news sentiment.

        Uses per-ticker sentiment scores weighted by relevance when available,
        falling back to overall_sentiment_score for articles without ticker data.
        """
        assert self.av is not None
        ticker = symbol.split(".")[0]
        time_from = f"{start_date.replace('-', '')}T0000"
        time_to = f"{end_date.replace('-', '')}T2359"

        raw = self.av.get_news_sentiment(tickers=ticker, time_from=time_from, time_to=time_to, limit=200)
        feed = raw.get("feed", [])

        # Aggregate daily sentiments using ticker-specific scores when available
        daily: Dict[str, List[tuple]] = {}  # date -> [(score, weight), ...]
        for article in feed:
            date = _av_datetime_to_date(article.get("time_published", ""))
            if not date:
                continue

            # Try to extract ticker-specific sentiment and relevance
            ticker_score: float | None = None
            relevance: float = 0.0
            for ts in article.get("ticker_sentiment", []):
                if ts.get("ticker", "").upper() == ticker.upper():
                    ticker_score = _safe_float(ts.get("ticker_sentiment_score"))
                    relevance = _safe_float(ts.get("relevance_score"))
                    break

            if ticker_score is not None and relevance > 0.0:
                # Use relevance-weighted ticker-specific sentiment
                daily.setdefault(date, []).append((ticker_score, relevance))
            else:
                # Fallback to overall sentiment with unit weight
                score = _safe_float(article.get("overall_sentiment_score"))
                daily.setdefault(date, []).append((score, 1.0))

        result: Dict[str, Any] = {}
        for date, entries in sorted(daily.items()):
            total_weight = sum(w for _, w in entries)
            if total_weight > 0:
                weighted_avg = sum(s * w for s, w in entries) / total_weight
            else:
                weighted_avg = sum(s for s, _ in entries) / max(1, len(entries))
            result[date] = {
                "date": date,
                "count": len(entries),
                "normalized": weighted_avg,
            }
        return result

    # ------------------------------------------------------------------
    # Insider transactions
    # ------------------------------------------------------------------

    def get_insider_transactions(
        self,
        symbol: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 100,
    ):
        """Return insider transactions, preferring SEC EDGAR when available."""
        # Always prefer EDGAR for insider data
        if self.edgar is not None and symbol:
            try:
                lookback = 365
                txns = self.edgar.get_insider_transactions(symbol.split(".")[0], lookback)
                if txns:
                    return _edgar_insider_to_eodhd(txns, start_date, end_date)
            except Exception as exc:
                logger.debug("EDGAR insiders failed for %s: %s", symbol, exc)

        # Fallback to EODHD
        if self.eodhd is not None:
            return self._eodhd.get_insider_transactions(symbol, start_date, end_date, limit)
        return []

    # ------------------------------------------------------------------
    # Macro / economic indicators
    # ------------------------------------------------------------------

    def get_economic_events(self, start_date: str, end_date: str, country: Optional[str] = None):
        if self.eodhd is not None:
            return self._eodhd.get_economic_events(start_date, end_date, country)
        # AV doesn't have an exact equivalent; return empty
        return []

    def get_macro_indicator(self, country: str, indicator: str | None = None):
        if self.mode == "eodhd":
            return self._eodhd.get_macro_indicator(country, indicator)

        if self.mode in ("alpha_vantage", "hybrid") and self.av is not None:
            try:
                return self._av_macro_indicator(country, indicator)
            except Exception as exc:
                logger.debug("AV macro failed for %s/%s, falling back: %s", country, indicator, exc)
                if self.eodhd is not None:
                    return self._eodhd.get_macro_indicator(country, indicator)
                raise

        return self._eodhd.get_macro_indicator(country, indicator)

    def _av_macro_indicator(self, country: str, indicator: str | None) -> list:
        """Map EODHD macro-indicator names to AV economic-indicator calls."""
        assert self.av is not None
        # EODHD indicators → AV function mapping
        mapping: Dict[str, str] = {
            "gdp_current_usd": "REAL_GDP",
            "real_interest_rate": "FEDERAL_FUNDS_RATE",
            "unemployment_total": "UNEMPLOYMENT",
            "inflation_consumer_prices_annual": "CPI",
            "cpi": "CPI",
            "unemployment": "UNEMPLOYMENT",
            "gdp": "REAL_GDP",
            "treasury_yield": "TREASURY_YIELD",
        }

        av_function = mapping.get((indicator or "").lower(), None)
        if av_function is None:
            logger.warning("No AV mapping for macro indicator '%s', returning empty", indicator)
            return []

        raw = self.av.get_economic_indicator(av_function)
        data_list = raw.get("data", [])
        records = []
        for entry in data_list:
            records.append({
                "CountryCode": country,
                "Indicator": indicator or av_function,
                "Date": entry.get("date", ""),
                "Period": entry.get("date", ""),
                "Value": _safe_float(entry.get("value")),
            })
        return records

    # ------------------------------------------------------------------
    # Technical indicators (AV-only, new capability)
    # ------------------------------------------------------------------

    def get_rsi(self, symbol: str, time_period: int = 14) -> Dict[str, Any]:
        if self.av is None:
            return {}
        ticker = symbol.split(".")[0]
        return self.av.get_rsi(ticker, time_period)

    def get_macd(self, symbol: str) -> Dict[str, Any]:
        if self.av is None:
            return {}
        ticker = symbol.split(".")[0]
        return self.av.get_macd(ticker)

    def get_bbands(self, symbol: str, time_period: int = 20) -> Dict[str, Any]:
        if self.av is None:
            return {}
        ticker = symbol.split(".")[0]
        return self.av.get_bbands(ticker, time_period)

    def get_adx(self, symbol: str, time_period: int = 14) -> Dict[str, Any]:
        if self.av is None:
            return {}
        ticker = symbol.split(".")[0]
        return self.av.get_adx(ticker, time_period)

    def get_stoch(self, symbol: str) -> Dict[str, Any]:
        if self.av is None:
            return {}
        ticker = symbol.split(".")[0]
        return self.av.get_stoch(ticker)

    # ------------------------------------------------------------------
    # SEC EDGAR direct access
    # ------------------------------------------------------------------

    def get_13f_holdings(self, ticker: str, lookback_quarters: int = 4) -> List[Dict[str, Any]]:
        if self.edgar is None:
            return []
        return self.edgar.get_13f_holdings(ticker, lookback_quarters)

    def get_edgar_insider_summary(self, ticker: str, lookback_days: int = 90) -> Dict[str, Any]:
        if self.edgar is None:
            return {}
        return self.edgar.get_insider_summary(ticker, lookback_days)

    # ------------------------------------------------------------------
    # get_json pass-through (for any code that calls client.get_json directly)
    # ------------------------------------------------------------------

    def get_json(self, endpoint: str, params=None, ttl_hours=None, request_cost: int = 1):
        """Backward-compatible pass-through to EODHDClient.get_json."""
        if self.eodhd is None:
            raise RuntimeError("get_json requires EODHD client; not available in alpha_vantage-only mode")
        return self._eodhd.get_json(endpoint, params, ttl_hours, request_cost)


# ======================================================================
# Alpha Vantage → EODHD normalisation helpers
# ======================================================================

def _safe_float(val: Any) -> float:
    if val is None or val == "" or val == "None":
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _av_datetime_to_date(dt_str: str) -> str:
    """Convert AV datetime '20240115T1030' → '2024-01-15'."""
    if not dt_str or len(dt_str) < 8:
        return ""
    raw = dt_str.replace("-", "").replace("T", "")[:8]
    if len(raw) >= 8:
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    return dt_str[:10]


def _av_reports_to_dict(reports: list) -> dict:
    """Convert AV list of annual/quarterly reports to EODHD-style keyed dict.

    EODHD uses the date string as key:
    ``{"2024-01-31": {fields…}, "2023-01-31": {fields…}}``

    AV field names are normalised to match the camelCase keys the codebase
    already looks up via ``pick_first``.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for report in reports:
        date = report.get("fiscalDateEnding", "")
        if not date:
            continue
        mapped = _map_av_report_fields(report)
        mapped["dateFormatted"] = date
        mapped["date"] = date
        mapped["filing_date"] = report.get("reportedDate", date)
        out[date] = mapped
    return out


# Mapping of AV report field names → EODHD camelCase equivalents
_AV_FIELD_MAP: Dict[str, str] = {
    # Income statement
    "totalRevenue": "totalRevenue",
    "grossProfit": "grossProfit",
    "operatingIncome": "operatingIncome",
    "netIncome": "netIncome",
    "ebitda": "ebitda",
    "costOfRevenue": "costOfRevenue",
    "costofGoodsAndServicesSold": "costOfRevenue",
    "researchAndDevelopment": "researchAndDevelopmentExpenses",
    "sellingGeneralAndAdministrative": "sellingGeneralAndAdministrative",
    "operatingExpenses": "operatingExpenses",
    "interestExpense": "interestExpense",
    "interestIncome": "interestIncome",
    "incomeTaxExpense": "incomeTaxExpense",
    "depreciationAndAmortization": "depreciationAndAmortization",
    "incomeBeforeTax": "incomeBeforeTax",
    "netIncomeFromContinuingOps": "netIncomeFromContinuingOperations",
    # Balance sheet
    "totalAssets": "totalAssets",
    "totalCurrentAssets": "totalCurrentAssets",
    "totalNonCurrentAssets": "totalNonCurrentAssets",
    "totalLiabilities": "totalLiab",
    "totalCurrentLiabilities": "totalCurrentLiabilities",
    "totalNonCurrentLiabilities": "nonCurrentLiabilitiesTotal",
    "totalShareholderEquity": "totalStockholderEquity",
    "commonStock": "commonStock",
    "retainedEarnings": "retainedEarnings",
    "cashAndCashEquivalentsAtCarryingValue": "cash",
    "cashAndShortTermInvestments": "cashAndShortTermInvestments",
    "inventory": "inventory",
    "currentNetReceivables": "netReceivables",
    "shortTermDebt": "shortTermDebt",
    "longTermDebt": "longTermDebt",
    "shortLongTermDebtTotal": "shortLongTermDebtTotal",
    "propertyPlantEquipment": "propertyPlantEquipment",
    "goodwill": "goodwill",
    "intangibleAssets": "intangibleAssets",
    "intangibleAssetsExcludingGoodwill": "intangibleAssetsExcludingGoodwill",
    "accountsPayable": "accountsPayable",
    "otherCurrentAssets": "otherCurrentAssets",
    "otherNonCurrentAssets": "otherNonCurrentAssets",
    # Cash flow
    "operatingCashflow": "totalCashFromOperatingActivities",
    "capitalExpenditures": "capitalExpenditures",
    "cashflowFromInvestment": "totalCashFromInvestingActivities",
    "cashflowFromFinancing": "totalCashFromFinancingActivities",
    "dividendPayout": "dividendsPaid",
    "dividendPayoutCommonStock": "dividendsPaid",
    "paymentsForRepurchaseOfCommonStock": "salePurchaseOfStock",
    "changeInCashAndCashEquivalents": "changeInCash",
    "profitLoss": "netIncome",
    "depreciationDepletionAndAmortization": "depreciation",
    "changeInReceivables": "changeToNetincome",
    "changeInInventory": "changeToInventory",
    "changeInOperatingLiabilities": "changeToLiabilities",
}


def _map_av_report_fields(report: Dict[str, Any]) -> Dict[str, Any]:
    """Map Alpha Vantage report fields to EODHD-equivalent camelCase names."""
    mapped: Dict[str, Any] = {}
    for av_key, eodhd_key in _AV_FIELD_MAP.items():
        if av_key in report:
            val = report[av_key]
            if val is not None and val != "None":
                mapped[eodhd_key] = val
    # Also carry through any unmapped fields with original names
    for key, val in report.items():
        if key not in mapped and key not in ("fiscalDateEnding", "reportedCurrency"):
            mapped[key] = val
    return mapped


def _av_overview_to_general(overview: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "Code": overview.get("Symbol", ""),
        "Name": overview.get("Name", ""),
        "Exchange": overview.get("Exchange", ""),
        "CurrencyCode": overview.get("Currency", "USD"),
        "CurrencySymbol": "$",
        "CountryISO": overview.get("Country", ""),
        "Sector": overview.get("Sector", ""),
        "Industry": overview.get("Industry", ""),
        "Description": overview.get("Description", ""),
        "FullTimeEmployees": overview.get("FullTimeEmployees", ""),
        "IPODate": overview.get("IPODate", ""),
        "GicSector": overview.get("Sector", ""),
        "GicGroup": overview.get("Industry", ""),
    }


def _av_overview_to_highlights(overview: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "MarketCapitalization": _safe_float(overview.get("MarketCapitalization")),
        "EBITDA": _safe_float(overview.get("EBITDA")),
        "PERatio": _safe_float(overview.get("PERatio")),
        "PEGRatio": _safe_float(overview.get("PEGRatio")),
        "WallStreetTargetPrice": _safe_float(overview.get("AnalystTargetPrice")),
        "BookValue": _safe_float(overview.get("BookValue")),
        "DividendShare": _safe_float(overview.get("DividendPerShare")),
        "DividendYield": _safe_float(overview.get("DividendYield")),
        "EarningsShare": _safe_float(overview.get("EPS")),
        "EPSEstimateCurrentYear": _safe_float(overview.get("EPSEstimateCurrentYear")),
        "EPSEstimateNextYear": _safe_float(overview.get("EPSEstimateNextYear")),
        "EPSEstimateNextQuarter": _safe_float(overview.get("EPSEstimateNextQuarter")),
        "MostRecentQuarter": overview.get("LatestQuarter", ""),
        "ProfitMargin": _safe_float(overview.get("ProfitMargin")),
        "OperatingMarginTTM": _safe_float(overview.get("OperatingMarginTTM")),
        "ReturnOnAssetsTTM": _safe_float(overview.get("ReturnOnAssetsTTM")),
        "ReturnOnEquityTTM": _safe_float(overview.get("ReturnOnEquityTTM")),
        "RevenueTTM": _safe_float(overview.get("RevenueTTM")),
        "RevenuePerShareTTM": _safe_float(overview.get("RevenuePerShareTTM")),
        "GrossProfitTTM": _safe_float(overview.get("GrossProfitTTM")),
        "DilutedEpsTTM": _safe_float(overview.get("DilutedEPSTTM")),
        "QuarterlyEarningsGrowthYOY": _safe_float(overview.get("QuarterlyEarningsGrowthYOY")),
        "QuarterlyRevenueGrowthYOY": _safe_float(overview.get("QuarterlyRevenueGrowthYOY")),
    }


def _av_overview_to_valuation(overview: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "TrailingPE": _safe_float(overview.get("TrailingPE")),
        "ForwardPE": _safe_float(overview.get("ForwardPE")),
        "PriceSalesTTM": _safe_float(overview.get("PriceToSalesRatioTTM")),
        "PriceBookMRQ": _safe_float(overview.get("PriceToBookRatio")),
        "EnterpriseValue": _safe_float(overview.get("EnterpriseValue")),
        "EnterpriseValueRevenue": _safe_float(overview.get("EVToRevenue")),
        "EnterpriseValueEbitda": _safe_float(overview.get("EVToEBITDA")),
    }


def _av_overview_to_shares_stats(overview: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "SharesOutstanding": _safe_float(overview.get("SharesOutstanding")),
        "SharesFloat": _safe_float(overview.get("SharesFloat", 0)),
        "PercentInsiders": _safe_float(overview.get("PercentInsiders")),
        "PercentInstitutions": _safe_float(overview.get("PercentInstitutions")),
        "ShortRatio": _safe_float(overview.get("ShortRatio")),
        "ShortPercentOutstanding": _safe_float(overview.get("ShortPercentOutstanding")),
        "ShortPercentFloat": _safe_float(overview.get("ShortPercentFloat")),
    }


def _av_overview_to_outstanding_shares(overview: Dict[str, Any]) -> Dict[str, Any]:
    shares = _safe_float(overview.get("SharesOutstanding"))
    if shares <= 0:
        return {}
    return {
        "annual": {},
        "quarterly": {},
    }


def _av_earnings_to_eodhd(earnings_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert AV earnings response into the EODHD Earnings structure.

    The codebase accesses ``fundamentals["Earnings"]["History"]`` and
    ``fundamentals["Earnings"]["Trend"]``.
    """
    quarterly = earnings_data.get("quarterlyEarnings", [])
    annual = earnings_data.get("annualEarnings", [])

    # Build History section
    history: Dict[str, Dict[str, Any]] = {}
    for q in quarterly:
        date = q.get("reportedDate") or q.get("fiscalDateEnding", "")
        if not date:
            continue
        history[date] = {
            "reportDate": date,
            "date": q.get("fiscalDateEnding", ""),
            "epsActual": _safe_float(q.get("reportedEPS")),
            "epsEstimate": _safe_float(q.get("estimatedEPS")),
            "epsDifference": _safe_float(q.get("surprise")),
            "surprisePercent": _safe_float(q.get("surprisePercentage")),
        }

    # Build Trend section (from annual earnings for now)
    trend: Dict[str, Dict[str, Any]] = {}
    for a in annual:
        date = a.get("fiscalDateEnding", "")
        if not date:
            continue
        trend[date] = {
            "date": date,
            "period": "0y",
            "growth": None,
            "earningsEstimateAvg": _safe_float(a.get("reportedEPS")),
            "earningsEstimateLow": None,
            "earningsEstimateHigh": None,
            "earningsEstimateNumberOfAnalysts": None,
            "revenueEstimateAvg": None,
            "epsTrend7daysAgo": None,
            "epsTrend30daysAgo": None,
            "epsTrend60daysAgo": None,
            "epsTrend90daysAgo": None,
        }

    return {"History": history, "Trend": trend}


def _edgar_insider_to_eodhd(
    txns: List[Dict[str, Any]],
    start_date: str | None,
    end_date: str | None,
) -> list:
    """Convert EDGAR Form 4 transactions to EODHD insider-transactions format."""
    records = []
    for t in txns:
        txn_date = t.get("transaction_date", "")
        if start_date and txn_date < start_date:
            continue
        if end_date and txn_date > end_date:
            continue

        code = t.get("transaction_code", "")
        # Map Form 4 codes to EODHD transaction types
        if code == "P":
            txn_type = "Buy"
        elif code == "S":
            txn_type = "Sale"
        elif code == "A":
            txn_type = "Grant"
        elif code == "M":
            txn_type = "Exercise"
        else:
            txn_type = code

        records.append({
            "date": txn_date,
            "ownerName": t.get("owner_name", ""),
            "ownerCik": t.get("owner_cik", ""),
            "transactionType": txn_type,
            "transactionShares": t.get("shares", 0),
            "transactionPrice": t.get("price_per_share", 0),
            "sharesOwnedAfter": t.get("shares_owned_after", 0),
            "isDirector": t.get("is_director", False),
            "isOfficer": t.get("is_officer", False),
            "officerTitle": t.get("officer_title", ""),
        })
    return records
