from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = 'https://eodhd.com/api'
USER_AGENT = 'eodhd-smart-ranker/3.0'


class TokenBucketRateLimiter:
    def __init__(self, rate_per_minute: int = 900, burst: int = 200):
        self.rate_per_second = max(1.0, float(rate_per_minute) / 60.0)
        self.capacity = float(max(1, burst))
        self.tokens = float(max(1, burst))
        self.last_refill = time.monotonic()
        self.lock = threading.Lock()

    def consume(self, tokens: float = 1.0) -> None:
        tokens = float(max(0.0, tokens))
        while True:
            with self.lock:
                now = time.monotonic()
                elapsed = now - self.last_refill
                if elapsed > 0:
                    self.tokens = min(self.capacity, self.tokens + elapsed * self.rate_per_second)
                    self.last_refill = now
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return
                deficit = tokens - self.tokens
            time.sleep(max(0.02, deficit / self.rate_per_second))


class EODHDClient:
    def __init__(
        self,
        api_token: str,
        cache_dir: Path,
        refresh: bool = False,
        timeout: int = 30,
        rate_limit_per_minute: int = 900,
        burst: int = 200,
    ):
        self.api_token = api_token
        self.cache_dir = cache_dir
        self.refresh = refresh
        self.timeout = timeout
        self.rate_limiter = TokenBucketRateLimiter(rate_limit_per_minute, burst)
        self.session = requests.Session()
        retry = Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=['GET'],
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
        self.session.mount('https://', adapter)
        self.session.headers.update({'User-Agent': USER_AGENT})
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_earnings_calendar(
    self,
    symbol: str,
    from_date: str | None = None,
    to_date: str | None = None,
    ):
        params = {"symbols": symbol}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        return self.get_json("/calendar/earnings", params=params, ttl_hours=12)

    def _cache_path(self, endpoint: str, params: Dict[str, Any]) -> Path:
        payload = json.dumps({'endpoint': endpoint, 'params': params}, sort_keys=True, default=str)
        key = hashlib.md5(payload.encode('utf-8')).hexdigest()
        return self.cache_dir / f'{key}.json'

    def get_json(self, endpoint: str, params: Optional[Dict[str, Any]] = None, ttl_hours: Optional[float] = None, request_cost: int = 1) -> Any:
        params = dict(params or {})
        params['api_token'] = self.api_token
        params['fmt'] = 'json'
        cache_path = self._cache_path(endpoint, params)
        if not self.refresh and cache_path.exists():
            if ttl_hours is None:
                return json.loads(cache_path.read_text(encoding='utf-8'))
            age_hours = (time.time() - cache_path.stat().st_mtime) / 3600.0
            if age_hours <= ttl_hours:
                return json.loads(cache_path.read_text(encoding='utf-8'))
        self.rate_limiter.consume(max(1, request_cost))
        url = f'{BASE_URL}{endpoint}'
        response = self.session.get(url, params=params, timeout=self.timeout)
        if response.status_code >= 400:
            body = response.text[:300].strip()
            raise RuntimeError(f'{endpoint} failed with HTTP {response.status_code}: {body or "No response body"}')
        data = response.json()
        if isinstance(data, dict):
            lower = {str(k).lower(): v for k, v in data.items()}
            if 'error' in lower:
                raise RuntimeError(f'API error for {endpoint}: {lower["error"]}')
            if 'errors' in lower:
                raise RuntimeError(f'API errors for {endpoint}: {lower["errors"]}')
            if 'message' in lower and isinstance(lower['message'], str) and 'error' in lower['message'].lower():
                raise RuntimeError(f'API message for {endpoint}: {lower["message"]}')
        cache_path.write_text(json.dumps(data), encoding='utf-8')
        return data

    def get_user(self) -> Dict[str, Any]:
        return self.get_json('/user', ttl_hours=1, request_cost=1)

    def get_supported_exchanges(self):
        return self.get_json("/exchanges-list/", ttl_hours=24, request_cost=1)


    def _exchange_candidates(self, exchange_code: str) -> list[str]:
        code = str(exchange_code or "").strip().upper()
        candidates = [code]

        # Helpful hardcoded aliases first
        alias_map = {
            "US": ["US"],
            "SG": ["SG", "XSES", "SES"],
            "HK": ["HK", "XHKG", "HKEX", "HKSE"],
        }
        for alt in alias_map.get(code, []):
            if alt not in candidates:
                candidates.append(alt)

        # Then enrich from /exchanges-list
        try:
            exchanges = self.get_supported_exchanges()
        except Exception:
            return candidates

        if isinstance(exchanges, list):
            for item in exchanges:
                if not isinstance(item, dict):
                    continue

                item_code = str(item.get("Code") or "").strip().upper()
                item_name = str(item.get("Name") or "").strip().upper()
                item_country2 = str(item.get("CountryISO2") or "").strip().upper()
                item_country3 = str(item.get("CountryISO3") or "").strip().upper()
                item_country = str(item.get("Country") or "").strip().upper()
                item_mic = str(item.get("OperatingMIC") or "").strip().upper()

                is_match = False

                if code == item_code:
                    is_match = True
                elif code in {item_country2, item_country3}:
                    is_match = True
                elif code == "SG" and ("SINGAPORE" in item_country or "XSES" in item_mic or item_country2 == "SG"):
                    is_match = True
                elif code == "HK" and ("HONG KONG" in item_country or "XHKG" in item_mic or item_country2 == "HK"):
                    is_match = True

                if is_match:
                    for alt in [item_code] + [x.strip() for x in item_mic.split(",") if x.strip()]:
                        alt = alt.upper()
                        if alt and alt not in candidates:
                            candidates.append(alt)

        return candidates


    def get_exchange_symbols(self, exchange_code: str):
        errors = []

        for candidate in self._exchange_candidates(exchange_code):
            try:
                data = self.get_json(
                    f"/exchange-symbol-list/{candidate}",
                    ttl_hours=24,
                    request_cost=1,
                )
                if isinstance(data, list) and data:
                    return data
                if isinstance(data, list):
                    # empty but valid response
                    return data
            except Exception as exc:
                errors.append(f"{candidate}: {exc}")
                continue

        raise RuntimeError(
            f"Unable to fetch symbol list for exchange {exchange_code}. Tried: {', '.join(self._exchange_candidates(exchange_code))}. "
            f"Errors: {' | '.join(errors[:5])}"
        )

    def search_instruments(self, query: str, limit: int = 100, exchange: Optional[str] = None):
        params: Dict[str, Any] = {'limit': min(limit, 500)}
        if exchange:
            params['exchange'] = exchange
        encoded_query = quote(query.strip(), safe='')
        return self.get_json(f'/search/{encoded_query}', params=params, ttl_hours=24, request_cost=1)

    def get_fundamentals(self, symbol: str):
        return self.get_json(f'/fundamentals/{symbol}', ttl_hours=24, request_cost=10)

    def get_news(
        self,
        symbol: str | None = None,
        tag: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ):
        params: Dict[str, Any] = {
            'limit': max(1, min(1000, int(limit))),
            'offset': max(0, int(offset)),
        }
        if symbol:
            params['s'] = symbol
        if tag:
            params['t'] = tag
        if start_date:
            params['from'] = start_date
        if end_date:
            params['to'] = end_date

        request_cost = 10 if symbol else 5
        return self.get_json('/news', params=params, ttl_hours=6, request_cost=request_cost)

    def get_price_history(
        self,
        symbol: str,
        from_date: str | None = None,
        to_date: str | None = None,
        period: str = 'd',
    ):
        params: Dict[str, Any] = {'period': period}
        if from_date:
            params['from'] = from_date
        if to_date:
            params['to'] = to_date
        return self.get_json(f'/eod/{symbol}', params=params, ttl_hours=12, request_cost=1)

    def get_dividends(self, symbol: str, start_date: str):
        return self.get_json(f'/div/{symbol}', params={'from': start_date}, ttl_hours=24, request_cost=1)

    def get_earnings(self, symbol: str):
        return self.get_json('/calendar/earnings', params={'symbols': symbol}, ttl_hours=24, request_cost=1)

    def get_sentiments(self, symbol: str, start_date: str, end_date: str):
        return self.get_json('/sentiments', params={'s': symbol, 'from': start_date, 'to': end_date}, ttl_hours=12, request_cost=5)

    def get_insider_transactions(
        self,
        symbol: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 100,
    ):
        params: Dict[str, Any] = {'limit': max(1, min(1000, int(limit)))}
        if symbol:
            params['code'] = symbol
        if start_date:
            params['from'] = start_date
        if end_date:
            params['to'] = end_date
        return self.get_json('/insider-transactions', params=params, ttl_hours=12, request_cost=10)

    def get_economic_events(self, start_date: str, end_date: str, country: Optional[str] = None):
        params: Dict[str, Any] = {'from': start_date, 'to': end_date, 'limit': 1000}
        if country:
            params['country'] = country
        return self.get_json('/economic-events', params=params, ttl_hours=6, request_cost=1)
    
    def get_macro_indicator(self, country: str, indicator: str | None = None):
        params: Dict[str, Any] = {}
        if indicator:
            params["indicator"] = indicator
        return self.get_json(f"/macro-indicator/{country}", params=params, ttl_hours=24, request_cost=1)
