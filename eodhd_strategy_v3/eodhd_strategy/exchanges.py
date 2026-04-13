from __future__ import annotations

from typing import Any, Iterable


GLOBAL_EXCHANGE_ALIAS_MAP: dict[str, str] = {
    "US": "US",
    "NYSE": "US",
    "NASDAQ": "US",
    "AMEX": "US",
    "ARCA": "US",
    "XNYS": "US",
    "XNAS": "US",
    "HK": "HK",
    "XHKG": "HK",
    "HKEX": "HK",
    "HKSE": "HK",
    "HONG KONG": "HK",
    "AU": "AU",
    "XASX": "AU",
    "ASX": "AU",
    "AUSTRALIA": "AU",
    "XETRA": "XETRA",
    "XETR": "XETRA",
    "DE": "XETRA",
    "GERMANY": "XETRA",
    "PA": "PA",
    "XPAR": "PA",
    "FR": "PA",
    "FRANCE": "PA",
    "AS": "AS",
    "XAMS": "AS",
    "NL": "AS",
    "NETHERLANDS": "AS",
    "SW": "SW",
    "XSWX": "SW",
    "SWX": "SW",
    "CH": "SW",
    "SWITZERLAND": "SW",
}

CANONICAL_EXCHANGE_REGION_MAP: dict[str, str] = {
    "US": "US",
    "HK": "HK",
    "AU": "AU",
    "XETRA": "DE",
    "PA": "FR",
    "AS": "NL",
    "SW": "CH",
}


def normalize_exchange_code(exchange: str | None) -> str:
    code = str(exchange or "").strip().upper()
    if not code:
        return ""
    return GLOBAL_EXCHANGE_ALIAS_MAP.get(code, code)


def requested_exchange_aliases(raw_exchanges: str | Iterable[str] | None) -> set[str]:
    if raw_exchanges is None:
        return set()

    if isinstance(raw_exchanges, str):
        tokens = [chunk.strip() for chunk in raw_exchanges.split(",") if chunk.strip()]
    else:
        tokens = [str(chunk).strip() for chunk in raw_exchanges if str(chunk).strip()]

    aliases: set[str] = set()
    for token in tokens:
        upper = token.upper()
        normalized = normalize_exchange_code(upper)
        if upper:
            aliases.add(upper)
        if normalized:
            aliases.add(normalized)

    return aliases


def symbol_suffix_aliases(symbols: Iterable[str]) -> set[str]:
    aliases: set[str] = set()
    for symbol in symbols:
        text = str(symbol or "").strip().upper()
        if "." not in text:
            continue
        _, suffix = text.rsplit(".", 1)
        aliases.update(requested_exchange_aliases([suffix]))
    return aliases


def infer_listing_region(symbol: str | None = None, exchange: str | None = None) -> str:
    exchange_code = normalize_exchange_code(exchange)
    if not exchange_code and symbol:
        text = str(symbol).strip().upper()
        if "." in text:
            _, suffix = text.rsplit(".", 1)
            exchange_code = normalize_exchange_code(suffix)
    return CANONICAL_EXCHANGE_REGION_MAP.get(exchange_code, "")


def _iter_listing_records(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            yield from _iter_listing_records(item)
        return

    if not isinstance(payload, dict):
        return

    if any(key in payload for key in ("Exchange", "exchange", "Code", "code")):
        yield payload

    for value in payload.values():
        if isinstance(value, (dict, list)):
            yield from _iter_listing_records(value)


def extract_listing_exchange_codes(general: dict[str, Any] | None) -> set[str]:
    if not isinstance(general, dict):
        return set()

    exchanges: set[str] = set()

    current_exchange = normalize_exchange_code(general.get("Exchange"))
    if current_exchange:
        exchanges.add(current_exchange)

    for record in _iter_listing_records(general.get("Listings")):
        exchange = normalize_exchange_code(record.get("Exchange") or record.get("exchange"))
        if exchange:
            exchanges.add(exchange)

    return exchanges


def extract_listing_symbols(general: dict[str, Any] | None) -> list[tuple[str, str]]:
    if not isinstance(general, dict):
        return []

    symbols: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for record in _iter_listing_records(general.get("Listings")):
        code = str(record.get("Code") or record.get("code") or "").strip().upper()
        exchange = normalize_exchange_code(record.get("Exchange") or record.get("exchange"))
        if not code or not exchange:
            continue
        pair = (code, exchange)
        if pair in seen:
            continue
        seen.add(pair)
        symbols.append(pair)

    return symbols
