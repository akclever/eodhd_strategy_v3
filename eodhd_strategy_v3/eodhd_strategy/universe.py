from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from .client import EODHDClient
from .exchanges import normalize_exchange_code, requested_exchange_aliases


def is_common_stock(item: Dict[str, object]) -> bool:
    type_text = " ".join(
        str(item.get(k, "")) for k in ["Type", "type", "asset_type", "AssetType", "GeneralType"]
    ).lower()

    name_text = str(item.get("Name", "") or item.get("name", "")).lower()
    code = str(item.get("Code", "") or item.get("code", "") or "").upper()
    exchange = str(item.get("Exchange", "") or item.get("exchange", "") or "").upper()

    if type_text and not ("common stock" in type_text or type_text == "commonstock"):
        return False

    bad_name_terms = [
        "acquisition corp",
        "shell",
        "spac",
        "unit",
        "rights",
        "warrant",
        "depositary",
        "adr",
        "ads",
        "etf",
        "etn",
        "fund",
        "trust",
        "preferred",
    ]
    if any(term in name_text for term in bad_name_terms):
        return False

    # More conservative suffix filter than before
    if len(code) > 1 and (code.endswith("W") or code.endswith("U")):
        return False

    if exchange in {"PINK", "OTC", "OTCM", "GREY"}:
        return False

    return True


def _normalize_exchange_for_symbol(exchange: str, region: str) -> str:
    return normalize_exchange_code(exchange)


def normalize_symbol(code: str, exchange: str, region: str) -> str:
    code = str(code or "").strip().upper()
    exchange = _normalize_exchange_for_symbol(exchange, region)

    if not code:
        return ""

    if "." in code:
        left, right = code.rsplit(".", 1)
        right = _normalize_exchange_for_symbol(right, region)
        return f"{left.upper()}.{right}" if right else code.upper()

    return f"{code}.{exchange}" if exchange else code


def _stable_shuffle(symbols: List[str], seed: str = "eodhd_strategy_v3") -> List[str]:
    def key_fn(sym: str) -> str:
        return hashlib.md5(f"{seed}|{sym}".encode("utf-8")).hexdigest()
    return sorted(symbols, key=key_fn)


def clean_symbol_list(
    items: Iterable[Dict[str, object]],
    region: str,
    allowed_exchanges: set[str] | None = None,
) -> List[str]:
    region = str(region or "US").upper()
    symbols: List[str] = []
    seen = set()

    region_exchange_aliases = {
        "US": {"US", "NYSE", "NASDAQ", "AMEX", "ARCA", "XNYS", "XNAS"},
        "HK": {"HK", "XHKG", "HKEX", "HKSE", "HONG KONG"},
        "AU": {"AU", "XASX", "ASX", "AUSTRALIA"},
        "DE": {"XETRA", "XETR", "DE", "GERMANY"},
        "FR": {"PA", "XPAR", "FR", "FRANCE"},
        "NL": {"AS", "XAMS", "NL", "NETHERLANDS"},
        "CH": {"SW", "XSWX", "SWX", "CH", "SWITZERLAND"},
    }
    allowed_exchange_aliases = {
        str(value).strip().upper()
        for value in (allowed_exchanges or region_exchange_aliases.get(region, set()))
        if str(value).strip()
    }
    allowed_exchange_aliases.update(
        normalize_exchange_code(value) for value in list(allowed_exchange_aliases) if value
    )

    for item in items:
        if not isinstance(item, dict):
            continue

        symbol = str(
            item.get("Code")
            or item.get("code")
            or item.get("symbol")
            or item.get("Symbol")
            or ""
        ).strip()

        exchange = str(item.get("Exchange") or item.get("exchange") or "").upper()

        normalized = normalize_symbol(symbol, exchange, region)
        if not normalized or normalized in seen:
            continue

        if not is_common_stock(item):
            continue

        ex_norm = _normalize_exchange_for_symbol(exchange, region)
        if exchange and ex_norm not in allowed_exchange_aliases and exchange not in allowed_exchange_aliases:
            continue

        seen.add(normalized)
        symbols.append(normalized)

    return symbols


def load_symbols_from_file(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        for col in ["symbol", "Symbol", "code", "Code"]:
            if col in df.columns:
                vals = [str(x).strip().upper() for x in df[col].dropna().astype(str).tolist()]
                return [x for x in vals if x]
        raise ValueError("CSV file must contain one of: symbol, Symbol, code, Code")

    return [line.strip().upper() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def collect_universe(client: EODHDClient, args) -> List[str]:
    region = str(getattr(args, "region", "US")).upper()
    symbols: List[str] = []
    requested_exchanges = requested_exchange_aliases(getattr(args, "exchanges", ""))
    requested_search_exchange = requested_exchange_aliases(getattr(args, "search_exchange", ""))
    allowed_exchanges = requested_search_exchange or requested_exchanges or None

    if args.symbols_file:
        symbols.extend(load_symbols_from_file(args.symbols_file))

    if args.symbols:
        raw = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
        symbols.extend(raw)

    if args.search_query:
        results = client.search_instruments(
            args.search_query,
            limit=max(args.limit or 500, 500),
            exchange=args.search_exchange or None,
        )
        symbols.extend(clean_symbol_list(results, region, allowed_exchanges=allowed_exchanges))

    if not symbols:
        for ex in [x.strip() for x in args.exchanges.split(",") if x.strip()]:
            items = client.get_exchange_symbols(ex)
            symbols.extend(clean_symbol_list(items, region, allowed_exchanges=requested_exchanges or None))

    deduped: List[str] = []
    seen = set()
    for s in symbols:
        if s not in seen:
            deduped.append(s)
            seen.add(s)

    deduped = _stable_shuffle(deduped)

    if args.limit and args.limit > 0:
        deduped = deduped[: args.limit]

    return deduped
