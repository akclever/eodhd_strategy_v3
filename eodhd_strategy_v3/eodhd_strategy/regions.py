from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Set

from .exchanges import normalize_exchange_code


@dataclass(frozen=True)
class RegionPreset:
    region: str
    exchange: str
    macro_country: str
    rebalance_country: str
    default_top_n_positions: int
    default_max_position_weight: float
    default_sector_cap: float
    exchange_aliases: Set[str]
    country_aliases: Set[str]


REGION_PRESETS = {
    "US": RegionPreset(
        region="US",
        exchange="US",
        macro_country="USA",
        rebalance_country="US",
        default_top_n_positions=20,
        default_max_position_weight=0.08,
        default_sector_cap=0.25,
        exchange_aliases={"US", "NYSE", "NASDAQ", "AMEX", "ARCA", "XNYS", "XNAS"},
        country_aliases={"USA", "UNITED STATES", "US"},
    ),
    "HK": RegionPreset(
        region="HK",
        exchange="HK",
        macro_country="HKG",
        rebalance_country="HK",
        default_top_n_positions=20,
        default_max_position_weight=0.08,
        default_sector_cap=0.25,
        exchange_aliases={"HK", "XHKG", "HKEX", "HKSE", "HONG KONG"},
        country_aliases={"HONG KONG", "HK", "HKG"},
    ),
    "AU": RegionPreset(
        region="AU",
        exchange="AU",
        macro_country="AUS",
        rebalance_country="AU",
        default_top_n_positions=20,
        default_max_position_weight=0.08,
        default_sector_cap=0.25,
        exchange_aliases={"AU", "XASX", "ASX", "AUSTRALIA"},
        country_aliases={"AUSTRALIA", "AU", "AUS"},
    ),
    "DE": RegionPreset(
        region="DE",
        exchange="XETRA",
        macro_country="DEU",
        rebalance_country="DE",
        default_top_n_positions=20,
        default_max_position_weight=0.08,
        default_sector_cap=0.25,
        exchange_aliases={"XETRA", "XETR", "DE", "GERMANY"},
        country_aliases={"GERMANY", "DE", "DEU"},
    ),
    "FR": RegionPreset(
        region="FR",
        exchange="PA",
        macro_country="FRA",
        rebalance_country="FR",
        default_top_n_positions=20,
        default_max_position_weight=0.08,
        default_sector_cap=0.25,
        exchange_aliases={"PA", "XPAR", "FR", "FRANCE"},
        country_aliases={"FRANCE", "FR", "FRA"},
    ),
    "NL": RegionPreset(
        region="NL",
        exchange="AS",
        macro_country="NLD",
        rebalance_country="NL",
        default_top_n_positions=20,
        default_max_position_weight=0.08,
        default_sector_cap=0.25,
        exchange_aliases={"AS", "XAMS", "NL", "NETHERLANDS"},
        country_aliases={"NETHERLANDS", "NL", "NLD"},
    ),
    "CH": RegionPreset(
        region="CH",
        exchange="SW",
        macro_country="CHE",
        rebalance_country="CH",
        default_top_n_positions=20,
        default_max_position_weight=0.08,
        default_sector_cap=0.25,
        exchange_aliases={"SW", "XSWX", "SWX", "CH", "SWITZERLAND"},
        country_aliases={"SWITZERLAND", "CH", "CHE"},
    ),
}


def get_region_preset(region: Optional[str]) -> RegionPreset:
    key = str(region or "US").upper()
    return REGION_PRESETS.get(key, REGION_PRESETS["US"])


def apply_rank_defaults(args):
    preset = get_region_preset(getattr(args, "region", "US"))

    if getattr(args, "exchanges", "US") == "US" and preset.exchange != "US":
        args.exchanges = preset.exchange

    if getattr(args, "use_macro", False) and getattr(args, "macro_country", "USA") == "USA":
        args.macro_country = preset.macro_country

    return args


def apply_rebalance_defaults(args):
    preset = get_region_preset(getattr(args, "region", "US"))

    if getattr(args, "rebalance_country", "US") == "US" and preset.rebalance_country != "US":
        args.rebalance_country = preset.rebalance_country

    if int(getattr(args, "top_n_positions", 20)) == 20 and preset.default_top_n_positions != 20:
        args.top_n_positions = preset.default_top_n_positions

    if float(getattr(args, "max_position_weight", 0.08)) == 0.08 and preset.default_max_position_weight != 0.08:
        args.max_position_weight = preset.default_max_position_weight

    if float(getattr(args, "sector_cap", 0.25)) == 0.25 and preset.default_sector_cap != 0.25:
        args.sector_cap = preset.default_sector_cap

    return args


def region_allows_listing(
    symbol: str,
    metrics: dict,
    region: str,
    strict_issuer_country: bool = False,
    allowed_exchange_aliases: Set[str] | None = None,
) -> bool:
    preset = get_region_preset(region)

    symbol_text = str(symbol or "").upper().strip()
    exchange_text = str(metrics.get("exchange") or "").upper().strip()
    country_text = str(metrics.get("country") or "").upper().strip()

    exchange_aliases = {
        str(alias).strip().upper()
        for alias in (allowed_exchange_aliases or preset.exchange_aliases)
        if str(alias).strip()
    }
    exchange_aliases.update(normalize_exchange_code(alias) for alias in list(exchange_aliases) if alias)
    issuer_exchange_aliases = {
        str(alias).strip().upper()
        for alias in preset.exchange_aliases
        if str(alias).strip()
    }
    issuer_exchange_aliases.update(
        normalize_exchange_code(alias) for alias in list(issuer_exchange_aliases) if alias
    )

    suffix = symbol_text.split(".")[-1] if "." in symbol_text else ""
    suffix_norm = normalize_exchange_code(suffix)
    exchange_norm = normalize_exchange_code(exchange_text)
    suffix_ok = suffix in exchange_aliases or suffix_norm in exchange_aliases
    exchange_ok = (
        exchange_text in exchange_aliases or exchange_norm in exchange_aliases
        if exchange_text
        else suffix_ok
    )

    listing_ok = suffix_ok or exchange_ok
    if not listing_ok:
        return False

    if strict_issuer_country:
        country_iso_text = str(metrics.get("country_iso") or "").upper().strip()
        isin_text = str(metrics.get("isin") or "").upper().strip()
        primary_ticker_text = str(metrics.get("primary_ticker") or "").upper().strip()

        if country_text in preset.country_aliases:
            return True

        if country_iso_text in preset.country_aliases:
            return True

        if primary_ticker_text and "." in primary_ticker_text:
            _, primary_suffix = primary_ticker_text.rsplit(".", 1)
            primary_suffix_norm = normalize_exchange_code(primary_suffix)
            if primary_suffix in issuer_exchange_aliases or primary_suffix_norm in issuer_exchange_aliases:
                return True

        if isin_text[:2] in preset.country_aliases:
            return True

        return False

    return True
