from __future__ import annotations

from types import SimpleNamespace

from eodhd_strategy.exchanges import extract_listing_exchange_codes, extract_listing_symbols, infer_listing_region
from eodhd_strategy.regions import region_allows_listing
from eodhd_strategy.universe import clean_symbol_list, collect_universe


def test_clean_symbol_list_allows_mixed_requested_exchanges() -> None:
    items = [
        {"Code": "MSFT", "Exchange": "NASDAQ", "Type": "Common Stock", "Name": "Microsoft Corporation"},
        {"Code": "SAP", "Exchange": "XETRA", "Type": "Common Stock", "Name": "SAP SE"},
        {"Code": "MC", "Exchange": "PA", "Type": "Common Stock", "Name": "LVMH Moet Hennessy Louis Vuitton"},
        {"Code": "AD", "Exchange": "AS", "Type": "Common Stock", "Name": "Ahold Delhaize"},
    ]

    symbols = clean_symbol_list(items, region="US", allowed_exchanges={"US", "XETRA", "PA", "AS"})

    assert set(symbols) == {"MSFT.US", "SAP.XETRA", "MC.PA", "AD.AS"}


def test_collect_universe_can_mix_us_and_eur_exchanges() -> None:
    class FakeClient:
        def get_exchange_symbols(self, exchange_code: str):
            payloads = {
                "US": [
                    {"Code": "MSFT", "Exchange": "NASDAQ", "Type": "Common Stock", "Name": "Microsoft Corporation"},
                ],
                "XETRA": [
                    {"Code": "SAP", "Exchange": "XETRA", "Type": "Common Stock", "Name": "SAP SE"},
                ],
                "PA": [
                    {"Code": "MC", "Exchange": "PA", "Type": "Common Stock", "Name": "LVMH Moet Hennessy Louis Vuitton"},
                ],
                "AS": [
                    {"Code": "AD", "Exchange": "AS", "Type": "Common Stock", "Name": "Ahold Delhaize"},
                ],
            }
            return payloads[exchange_code]

    args = SimpleNamespace(
        region="US",
        symbols_file="",
        symbols="",
        search_query="",
        search_exchange="",
        exchanges="US,XETRA,PA,AS",
        limit=0,
    )

    symbols = collect_universe(FakeClient(), args)

    assert set(symbols) == {"MSFT.US", "SAP.XETRA", "MC.PA", "AD.AS"}


def test_region_allows_listing_can_use_requested_exchange_aliases() -> None:
    assert region_allows_listing(
        "SAP.XETRA",
        {"exchange": "XETRA", "country": "GERMANY"},
        region="US",
        allowed_exchange_aliases={"US", "XETRA", "PA", "AS"},
    )

    assert not region_allows_listing(
        "NESN.SW",
        {"exchange": "SWX", "country": "SWITZERLAND"},
        region="US",
        allowed_exchange_aliases={"US", "XETRA", "PA", "AS"},
    )


def test_region_allows_listing_strict_issuer_country_accepts_us_primary_ticker() -> None:
    assert region_allows_listing(
        "APC.XETRA",
        {
            "exchange": "XETRA",
            "country": "GERMANY",
            "country_iso": "DE",
            "primary_ticker": "AAPL.US",
        },
        region="US",
        strict_issuer_country=True,
        allowed_exchange_aliases={"XETRA", "PA", "AS"},
    )


def test_region_allows_listing_strict_issuer_country_accepts_us_isin() -> None:
    assert region_allows_listing(
        "QCI.XETRA",
        {
            "exchange": "XETRA",
            "country": "GERMANY",
            "country_iso": "DE",
            "isin": "US7475251036",
        },
        region="US",
        strict_issuer_country=True,
        allowed_exchange_aliases={"XETRA", "PA", "AS"},
    )


def test_infer_listing_region_uses_exchange_or_symbol_suffix() -> None:
    assert infer_listing_region(symbol="SAP.XETRA") == "DE"
    assert infer_listing_region(exchange="PA") == "FR"


def test_extract_listing_exchange_codes_normalizes_nested_general_listings() -> None:
    general = {
        "Exchange": "NYSE",
        "Listings": {
            "0": {"Code": "APC", "Exchange": "XETRA"},
            "1": {"Code": "AAPL", "Exchange": "US"},
            "2": {"Code": "APC", "Exchange": "XETR"},
        },
    }

    assert extract_listing_exchange_codes(general) == {"US", "XETRA"}


def test_extract_listing_symbols_returns_normalized_pairs() -> None:
    general = {
        "Listings": {
            "0": {"Code": "AAPL", "Exchange": "US"},
            "1": {"Code": "APC", "Exchange": "XETR"},
            "2": {"Code": "AAPL", "Exchange": "US"},
        },
    }

    assert extract_listing_symbols(general) == [("AAPL", "US"), ("APC", "XETRA")]
