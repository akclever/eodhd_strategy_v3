from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .client import TokenBucketRateLimiter

logger = logging.getLogger(__name__)

EDGAR_BASE = "https://efts.sec.gov/LATEST"
EDGAR_FULL_TEXT = "https://efts.sec.gov/LATEST/search-index"
EDGAR_FILINGS = "https://www.sec.gov/cgi-bin/browse-edgar"
EDGAR_DATA = "https://data.sec.gov"
EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions"
_EDGAR_CACHE_IO_LOCK = threading.RLock()


class SECEdgarClient:
    """SEC EDGAR client for 13F institutional holdings and Form 4 insider transactions.

    Implements caching, rate limiting (10 req/s per SEC fair-access policy),
    and retry logic. Requires an email address for the User-Agent header
    as mandated by SEC.
    """

    def __init__(
        self,
        email: str,
        cache_dir: Path,
        refresh: bool = False,
        timeout: int = 30,
        rate_limit_per_second: int = 10,
    ):
        if not email or "@" not in email:
            raise ValueError(
                "SEC EDGAR requires a valid email in the User-Agent header. "
                "Set sec_edgar_email in your config."
            )
        self.email = email
        self.cache_dir = cache_dir / "sec_edgar"
        self.refresh = refresh
        self.timeout = timeout
        self.rate_limiter = TokenBucketRateLimiter(
            rate_per_minute=rate_limit_per_second * 60,
            burst=rate_limit_per_second,
        )
        self.session = requests.Session()
        retry = Retry(
            total=4,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=5, pool_maxsize=5)
        self.session.mount("https://", adapter)
        self.session.headers.update({
            "User-Agent": f"eodhd-smart-ranker/3.0 ({email})",
            "Accept-Encoding": "gzip, deflate",
        })
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # CIK lookup cache (ticker → CIK string, zero-padded to 10 digits)
        self._cik_map: Dict[str, str] = {}
        self._cik_map_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_path(self, namespace: str, key_data: Dict[str, Any]) -> Path:
        payload = json.dumps(
            {"ns": namespace, **key_data}, sort_keys=True, default=str
        )
        digest = hashlib.md5(payload.encode("utf-8")).hexdigest()
        subdir = self.cache_dir / namespace
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / f"{digest}.json"

    def _get_cached(self, cache_path: Path, ttl_hours: Optional[float]) -> Any | None:
        with _EDGAR_CACHE_IO_LOCK:
            if not self.refresh and cache_path.exists():
                if ttl_hours is None:
                    return json.loads(cache_path.read_text(encoding="utf-8"))
                age_hours = (time.time() - cache_path.stat().st_mtime) / 3600.0
                if age_hours <= ttl_hours:
                    return json.loads(cache_path.read_text(encoding="utf-8"))
        return None

    def _write_cache(self, cache_path: Path, data: Any) -> None:
        payload = json.dumps(data, default=str)
        tmp = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
        with _EDGAR_CACHE_IO_LOCK:
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(cache_path)

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _get(self, url: str, params: Optional[Dict[str, Any]] = None) -> requests.Response:
        self.rate_limiter.consume(1)
        resp = self.session.get(url, params=params, timeout=self.timeout)
        if resp.status_code == 429:
            logger.warning("SEC EDGAR 429 – sleeping 12 s …")
            time.sleep(12)
            return self._get(url, params)
        resp.raise_for_status()
        return resp

    def _get_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        resp = self._get(url, params)
        return resp.json()

    # ------------------------------------------------------------------
    # CIK resolution
    # ------------------------------------------------------------------

    def ticker_to_cik(self, ticker: str) -> str:
        """Resolve a stock ticker to a zero-padded 10-digit CIK string."""
        ticker_upper = ticker.upper().split(".")[0]  # strip exchange suffix

        with self._cik_map_lock:
            if ticker_upper in self._cik_map:
                return self._cik_map[ticker_upper]

        cache_key = self._cache_path("cik", {"ticker": ticker_upper})
        cached = self._get_cached(cache_key, ttl_hours=168)  # 7 days
        if cached is not None:
            with self._cik_map_lock:
                self._cik_map[ticker_upper] = cached
            return cached

        # Try the company tickers JSON (bulk lookup)
        try:
            data = self._get_json(f"{EDGAR_DATA}/files/company_tickers.json")
            for entry in data.values():
                t = str(entry.get("ticker", "")).upper()
                c = str(entry.get("cik_str", "")).zfill(10)
                with self._cik_map_lock:
                    self._cik_map[t] = c
            with self._cik_map_lock:
                if ticker_upper in self._cik_map:
                    self._write_cache(cache_key, self._cik_map[ticker_upper])
                    return self._cik_map[ticker_upper]
        except Exception as exc:
            logger.warning("Failed to load company_tickers.json: %s", exc)

        # Fallback: EDGAR full-text search
        try:
            data = self._get_json(
                f"{EDGAR_BASE}/search-index",
                params={"q": ticker_upper, "dateRange": "custom", "forms": "10-K"},
            )
            hits = data.get("hits", {}).get("hits", [])
            if hits:
                cik = str(hits[0]["_source"].get("entity_id", "")).zfill(10)
                with self._cik_map_lock:
                    self._cik_map[ticker_upper] = cik
                self._write_cache(cache_key, cik)
                return cik
        except Exception as exc:
            logger.warning("CIK fallback search failed for %s: %s", ticker_upper, exc)

        raise ValueError(f"Could not resolve CIK for ticker '{ticker}'")

    # ------------------------------------------------------------------
    # Company submissions (filings index)
    # ------------------------------------------------------------------

    def get_submissions(self, cik: str) -> Dict[str, Any]:
        """Get all recent filings for a CIK from the submissions API."""
        cik = cik.zfill(10)
        cache_path = self._cache_path("submissions", {"cik": cik})
        cached = self._get_cached(cache_path, ttl_hours=24)
        if cached is not None:
            return cached
        data = self._get_json(f"{EDGAR_SUBMISSIONS}/CIK{cik}.json")
        self._write_cache(cache_path, data)
        return data

    # ------------------------------------------------------------------
    # 13F Institutional Holdings
    # ------------------------------------------------------------------

    def get_13f_holdings(
        self,
        ticker: str,
        lookback_quarters: int = 4,
    ) -> List[Dict[str, Any]]:
        """Retrieve recent 13F-HR filings that hold the given ticker.

        Returns a list of dicts with keys:
        - filer_name, filer_cik
        - report_date
        - shares, value (in $1000s as reported)
        - put_call (if options)
        """
        cache_path = self._cache_path(
            "13f", {"ticker": ticker, "lookback_q": lookback_quarters}
        )
        cached = self._get_cached(cache_path, ttl_hours=72)  # 3 days
        if cached is not None:
            return cached

        # Use EFTS full-text search for 13F filings mentioning the ticker
        cutoff = datetime.utcnow() - timedelta(days=lookback_quarters * 93)
        cutoff_str = cutoff.strftime("%Y-%m-%d")

        try:
            search_data = self._get_json(
                f"{EDGAR_BASE}/search-index",
                params={
                    "q": f'"{ticker.upper().split(".")[0]}"',
                    "forms": "13F-HR",
                    "dateRange": "custom",
                    "startdt": cutoff_str,
                    "enddt": datetime.utcnow().strftime("%Y-%m-%d"),
                },
            )
        except Exception:
            # EFTS may not be available; try the EDGAR full-text search API
            try:
                search_data = self._get_json(
                    f"{EDGAR_BASE}/efts/search-index",
                    params={
                        "q": f'"{ticker.upper().split(".")[0]}"',
                        "forms": "13F-HR",
                        "dateRange": "custom",
                        "startdt": cutoff_str,
                        "enddt": datetime.utcnow().strftime("%Y-%m-%d"),
                    },
                )
            except Exception as exc:
                logger.warning("13F search failed for %s: %s", ticker, exc)
                self._write_cache(cache_path, [])
                return []

        hits = search_data.get("hits", {}).get("hits", [])
        holdings: List[Dict[str, Any]] = []

        for hit in hits[:20]:  # limit to 20 filings to control API calls
            source = hit.get("_source", {})
            filing_url = source.get("file_url", "")
            filer_name = source.get("display_names", ["Unknown"])[0] if source.get("display_names") else "Unknown"
            filer_cik = source.get("entity_id", "")
            report_date = source.get("period_of_report", source.get("file_date", ""))

            if not filing_url:
                continue

            # Try to parse the information table XML from the filing
            try:
                info_table = self._fetch_13f_info_table(filing_url, ticker)
                for entry in info_table:
                    entry["filer_name"] = filer_name
                    entry["filer_cik"] = filer_cik
                    entry["report_date"] = report_date
                    holdings.append(entry)
            except Exception as exc:
                logger.debug("Could not parse 13F info table from %s: %s", filing_url, exc)

        self._write_cache(cache_path, holdings)
        return holdings

    def _fetch_13f_info_table(self, filing_index_url: str, ticker: str) -> List[Dict[str, Any]]:
        """Parse 13F information table XML for holdings matching a ticker."""
        # Get the filing index page to find the info table document
        resp = self._get(filing_index_url)
        text = resp.text

        # Look for the information table XML link
        xml_pattern = re.compile(r'href="([^"]*infotable[^"]*\.xml)"', re.IGNORECASE)
        match = xml_pattern.search(text)
        if not match:
            # Try primary_doc pattern
            xml_pattern2 = re.compile(r'href="([^"]*\.xml)"', re.IGNORECASE)
            match = xml_pattern2.search(text)
        if not match:
            return []

        xml_url = match.group(1)
        if not xml_url.startswith("http"):
            xml_url = f"https://www.sec.gov{xml_url}"

        xml_resp = self._get(xml_url)
        return self._parse_13f_xml(xml_resp.text, ticker)

    def _parse_13f_xml(self, xml_text: str, ticker: str) -> List[Dict[str, Any]]:
        """Extract holdings for a specific ticker from 13F XML."""
        results: List[Dict[str, Any]] = []
        ticker_clean = ticker.upper().split(".")[0]

        try:
            root = ElementTree.fromstring(xml_text)
        except ElementTree.ParseError:
            return results

        # Handle various namespaces used in 13F filings
        ns = {"": ""}
        for elem in root.iter():
            if elem.tag.startswith("{"):
                ns_uri = elem.tag.split("}")[0] + "}"
                ns["ns"] = ns_uri[1:-1]
                break

        # Look for infoTable entries
        for entry in root.iter():
            tag = entry.tag.split("}")[-1] if "}" in entry.tag else entry.tag
            if tag.lower() != "infotable":
                continue

            name_of_issuer = ""
            cusip = ""
            value = 0
            shares = 0
            put_call = ""

            for child in entry:
                child_tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                child_tag_lower = child_tag.lower()
                text = (child.text or "").strip()

                if child_tag_lower == "nameofissuer":
                    name_of_issuer = text
                elif child_tag_lower == "cusip":
                    cusip = text
                elif child_tag_lower == "value":
                    value = int(text) if text.isdigit() else 0
                elif child_tag_lower in ("sshprnamt", "shrsorprnamt"):
                    # Shares or principal amount
                    for sub in child:
                        sub_tag = sub.tag.split("}")[-1] if "}" in sub.tag else sub.tag
                        if sub_tag.lower() in ("sshprnamt", "shrsorprnamt"):
                            shares = int(sub.text or "0")
                elif child_tag_lower == "putcall":
                    put_call = text

            # Rough match: check if ticker appears in issuer name
            if ticker_clean.lower() in name_of_issuer.lower() or (
                cusip and ticker_clean.lower() in cusip.lower()
            ):
                results.append({
                    "name_of_issuer": name_of_issuer,
                    "cusip": cusip,
                    "value_x1000": value,
                    "shares": shares,
                    "put_call": put_call,
                })

        return results

    # ------------------------------------------------------------------
    # Form 4 Insider Transactions
    # ------------------------------------------------------------------

    def get_insider_transactions(
        self,
        ticker: str,
        lookback_days: int = 365,
    ) -> List[Dict[str, Any]]:
        """Get Form 4 insider transaction filings for a given ticker.

        Returns a list of dicts with keys:
        - owner_name, owner_cik, is_director, is_officer, officer_title
        - transaction_date, transaction_code (P=purchase, S=sale, etc.)
        - shares, price_per_share, acquired_disposed (A or D)
        - shares_owned_after
        """
        cache_path = self._cache_path(
            "form4", {"ticker": ticker, "lookback_days": lookback_days}
        )
        cached = self._get_cached(cache_path, ttl_hours=24)
        if cached is not None:
            return cached

        try:
            cik = self.ticker_to_cik(ticker)
        except ValueError as exc:
            logger.warning("Cannot resolve CIK for %s: %s", ticker, exc)
            self._write_cache(cache_path, [])
            return []

        # Get recent filings from submissions
        try:
            submissions = self.get_submissions(cik)
        except Exception as exc:
            logger.warning("Failed to get submissions for %s: %s", ticker, exc)
            self._write_cache(cache_path, [])
            return []

        recent = submissions.get("filings", {}).get("recent", {})
        if not recent:
            recent = submissions.get("recent", {})

        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])

        cutoff = (datetime.utcnow() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        transactions: List[Dict[str, Any]] = []

        for i, form in enumerate(forms):
            if form not in ("4", "4/A"):
                continue
            filing_date = dates[i] if i < len(dates) else ""
            if filing_date < cutoff:
                continue

            accession = accessions[i] if i < len(accessions) else ""
            primary_doc = primary_docs[i] if i < len(primary_docs) else ""

            if not accession or not primary_doc:
                continue

            # Fetch and parse the Form 4 XML
            accession_clean = accession.replace("-", "")
            doc_url = (
                f"{EDGAR_DATA}/Archives/edgar/data/{cik.lstrip('0')}"
                f"/{accession_clean}/{primary_doc}"
            )

            try:
                parsed = self._parse_form4(doc_url, filing_date)
                transactions.extend(parsed)
            except Exception as exc:
                logger.debug("Could not parse Form 4 at %s: %s", doc_url, exc)

        self._write_cache(cache_path, transactions)
        return transactions

    def _parse_form4(self, url: str, filing_date: str) -> List[Dict[str, Any]]:
        """Parse a Form 4 XML document and extract transaction details."""
        resp = self._get(url)
        results: List[Dict[str, Any]] = []

        try:
            root = ElementTree.fromstring(resp.text)
        except ElementTree.ParseError:
            return results

        # Extract owner info
        owner_name = ""
        owner_cik = ""
        is_director = False
        is_officer = False
        officer_title = ""

        for owner_elem in root.iter():
            tag = owner_elem.tag.split("}")[-1] if "}" in owner_elem.tag else owner_elem.tag
            if tag == "reportingOwner":
                for child in owner_elem.iter():
                    ctag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                    text = (child.text or "").strip()
                    if ctag == "rptOwnerName":
                        owner_name = text
                    elif ctag == "rptOwnerCik":
                        owner_cik = text
                    elif ctag == "isDirector":
                        is_director = text == "1" or text.lower() == "true"
                    elif ctag == "isOfficer":
                        is_officer = text == "1" or text.lower() == "true"
                    elif ctag == "officerTitle":
                        officer_title = text

        # Extract non-derivative transactions
        for txn_elem in root.iter():
            tag = txn_elem.tag.split("}")[-1] if "}" in txn_elem.tag else txn_elem.tag
            if tag != "nonDerivativeTransaction":
                continue

            txn_date = ""
            txn_code = ""
            shares = 0.0
            price = 0.0
            acquired_disposed = ""
            shares_after = 0.0

            for child in txn_elem.iter():
                ctag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                text = (child.text or "").strip()

                if ctag == "transactionDate":
                    # The value is in a nested <value> element
                    for sub in child.iter():
                        stag = sub.tag.split("}")[-1] if "}" in sub.tag else sub.tag
                        if stag == "value" and sub.text:
                            txn_date = sub.text.strip()
                elif ctag == "transactionCode":
                    txn_code = text
                elif ctag == "transactionShares":
                    for sub in child.iter():
                        stag = sub.tag.split("}")[-1] if "}" in sub.tag else sub.tag
                        if stag == "value" and sub.text:
                            try:
                                shares = float(sub.text.strip())
                            except ValueError:
                                pass
                elif ctag == "transactionPricePerShare":
                    for sub in child.iter():
                        stag = sub.tag.split("}")[-1] if "}" in sub.tag else sub.tag
                        if stag == "value" and sub.text:
                            try:
                                price = float(sub.text.strip())
                            except ValueError:
                                pass
                elif ctag == "transactionAcquiredDisposedCode":
                    for sub in child.iter():
                        stag = sub.tag.split("}")[-1] if "}" in sub.tag else sub.tag
                        if stag == "value" and sub.text:
                            acquired_disposed = sub.text.strip()
                elif ctag == "sharesOwnedFollowingTransaction":
                    for sub in child.iter():
                        stag = sub.tag.split("}")[-1] if "}" in sub.tag else sub.tag
                        if stag == "value" and sub.text:
                            try:
                                shares_after = float(sub.text.strip())
                            except ValueError:
                                pass

            results.append({
                "owner_name": owner_name,
                "owner_cik": owner_cik,
                "is_director": is_director,
                "is_officer": is_officer,
                "officer_title": officer_title,
                "filing_date": filing_date,
                "transaction_date": txn_date or filing_date,
                "transaction_code": txn_code,
                "shares": shares,
                "price_per_share": price,
                "acquired_disposed": acquired_disposed,
                "shares_owned_after": shares_after,
            })

        return results

    # ------------------------------------------------------------------
    # Convenience: aggregated insider summary
    # ------------------------------------------------------------------

    def get_insider_summary(
        self,
        ticker: str,
        lookback_days: int = 90,
    ) -> Dict[str, Any]:
        """Return an aggregated summary of insider transactions.

        Keys: buy_count, sell_count, net_shares, buy_value, sell_value,
              unique_buyers, unique_sellers, cluster_buy (bool)
        """
        txns = self.get_insider_transactions(ticker, lookback_days)

        buy_count = 0
        sell_count = 0
        buy_shares = 0.0
        sell_shares = 0.0
        buy_value = 0.0
        sell_value = 0.0
        buyers: set[str] = set()
        sellers: set[str] = set()

        for t in txns:
            code = t.get("transaction_code", "").upper()
            ad = t.get("acquired_disposed", "").upper()
            sh = float(t.get("shares", 0) or 0)
            pr = float(t.get("price_per_share", 0) or 0)
            name = t.get("owner_name", "")

            if code == "P" or (code in ("A", "M") and ad == "A"):
                buy_count += 1
                buy_shares += sh
                buy_value += sh * pr
                if name:
                    buyers.add(name)
            elif code == "S" or (ad == "D"):
                sell_count += 1
                sell_shares += sh
                sell_value += sh * pr
                if name:
                    sellers.add(name)

        net_shares = buy_shares - sell_shares
        cluster_buy = len(buyers) >= 3 and buy_count >= 3

        return {
            "buy_count": buy_count,
            "sell_count": sell_count,
            "net_shares": net_shares,
            "buy_shares": buy_shares,
            "sell_shares": sell_shares,
            "buy_value": buy_value,
            "sell_value": sell_value,
            "unique_buyers": len(buyers),
            "unique_sellers": len(sellers),
            "cluster_buy": cluster_buy,
        }
