"""Phase 4 tests — Alpha Vantage, SEC EDGAR, DataProvider integration.

Covers:
- DataProvider instantiation and mode routing
- EDGAR → insider conviction field compatibility
- EDGAR → institutional breadth enrichment
- Technical indicator metrics computation
- Pipeline wiring (config, CLI args)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from eodhd_strategy.config import RankerConfig
from eodhd_strategy.data_provider import DataProvider
from eodhd_strategy.features import (
    _insider_role_weight,
    _extract_av_article_sentiment,
    _av_topic_categories,
    add_overlay_metrics,
    compute_insider_conviction_metrics,
    compute_news_event_metrics,
    compute_news_theme_drift_metrics,
    compute_sentiment_metrics,
    compute_technical_indicator_metrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config(**overrides) -> RankerConfig:
    base = {
        "api_token": "test",
        "cache_dir": Path("."),
        "refresh": False,
        "workers": 1,
        "min_market_cap": 100.0,
        "dividend_source": "hybrid",
        "regime": "neutral",
        "use_pead": False,
        "pead_lookback_days": 120,
        "pead_half_life_days": 45,
        "min_pead_analysts": 3,
        "use_revision_impulse": False,
        "min_revision_analysts": 4,
        "revision_impulse_weight": 0.06,
        "use_estimate_term_structure": False,
        "estimate_term_structure_weight": 0.04,
        "use_growth_acceleration": False,
        "growth_weight": 0.10,
        "alpha_factor_spec": "legacy",
        "use_residual_valuation": False,
        "use_compounder_persistence": False,
        "use_intangible_adjustments": False,
        "use_price_momentum": False,
        "require_real_momentum_coverage": False,
        "momentum_weight": 0.10,
        "use_life_cycle": False,
        "life_cycle_tilt_strength": 0.35,
        "use_sentiment": False,
        "sentiment_lookback_days": 14,
        "min_sentiment_accel": -0.02,
        "min_sentiment_articles_recent": 3,
        "use_news_events": False,
        "news_lookback_days": 10,
        "min_news_articles": 3,
        "news_event_weight": 0.06,
        "use_news_peer_spillover": False,
        "news_peer_spillover_weight": 0.25,
        "use_news_novelty_saturation": False,
        "use_news_confirmation": False,
        "news_confirmation_weight": 0.20,
        "use_news_macro_weighting": False,
        "use_beneish": False,
        "use_accrual_volatility": False,
        "use_working_capital_stress": False,
        "forensic_weight": 0.10,
        "missing_beneish_penalty": 0.25,
        "use_capital_allocation_quality": False,
        "capital_allocation_weight": 0.04,
        "use_recovery_transition": False,
        "recovery_transition_weight": 0.03,
        "use_insider_conviction": False,
        "insider_conviction_weight": 0.03,
        "use_news_theme_drift": False,
        "news_theme_drift_weight": 0.03,
        "use_peer_relative_anomalies": False,
        "peer_relative_anomaly_weight": 0.04,
        "exclude_binary_biotech": False,
        "binary_biotech_min_revenue": 1_000_000_000.0,
        "dividend_payout_cap": 0.85,
        "max_distance_from_high": 0.15,
        "require_above_200dma": False,
        "neutralize_by": "sector",
        "min_group_size": 1,
        "overlay_top_n": 50,
        "output": Path("ranked.csv"),
        "min_sentiment_days": 3,
        "min_piotroski_score": 5,
        "pead_max_abs_surprise_pct": 100.0,
        "pead_max_age_days": 45,
        "macro_state": "neutral",
        "universe_size": 200,
        "use_employee_efficiency": False,
        "employee_efficiency_weight": 0.05,
        "analysis_from_primary_ticker": False,
    }
    base.update(overrides)
    return RankerConfig(**base)


def _mock_eodhd_client() -> MagicMock:
    client = MagicMock()
    client.get_fundamentals.return_value = {"General": {}, "Financials": {}}
    client.get_price_history.return_value = []
    client.get_user.return_value = {"apiRequests": 100}
    client.get_insider_transactions.return_value = []
    client.get_exchange_symbols.return_value = []
    client.get_supported_exchanges.return_value = []
    client.get_json.return_value = {}
    return client


# ===========================================================================
# DataProvider instantiation and mode routing
# ===========================================================================

class TestDataProviderModes:

    def test_eodhd_mode_creates_with_eodhd_client(self):
        eodhd = _mock_eodhd_client()
        dp = DataProvider(mode="eodhd", eodhd_client=eodhd)
        assert dp.eodhd is not None
        assert dp.av is None
        assert dp.edgar is None

    def test_alpha_vantage_mode_without_eodhd(self):
        av = MagicMock()
        dp = DataProvider(mode="alpha_vantage", av_client=av)
        assert dp.eodhd is None
        assert dp.av is not None

    def test_hybrid_mode_has_both_clients(self):
        eodhd = _mock_eodhd_client()
        av = MagicMock()
        dp = DataProvider(mode="hybrid", eodhd_client=eodhd, av_client=av)
        assert dp.eodhd is not None
        assert dp.av is not None

    def test_get_user_returns_empty_without_eodhd(self):
        dp = DataProvider(mode="alpha_vantage", av_client=MagicMock())
        assert dp.get_user() == {}

    def test_get_supported_exchanges_returns_empty_without_eodhd(self):
        dp = DataProvider(mode="alpha_vantage", av_client=MagicMock())
        assert dp.get_supported_exchanges() == []

    def test_get_exchange_symbols_raises_without_eodhd(self):
        dp = DataProvider(mode="alpha_vantage", av_client=MagicMock())
        with pytest.raises(RuntimeError, match="requires EODHD"):
            dp.get_exchange_symbols("US")

    def test_get_json_raises_without_eodhd(self):
        dp = DataProvider(mode="alpha_vantage", av_client=MagicMock())
        with pytest.raises(RuntimeError, match="requires EODHD client"):
            dp.get_json("/some/endpoint")


# ===========================================================================
# DataProvider fundamentals enrichment with EDGAR
# ===========================================================================

class TestEdgarHoldersEnrichment:

    def test_eodhd_mode_enriches_with_edgar_when_holders_missing(self):
        eodhd = _mock_eodhd_client()
        eodhd.get_fundamentals.return_value = {
            "General": {"Name": "Test"},
            "Financials": {},
            "Holders": {},
        }
        edgar = MagicMock()
        edgar.get_13f_holdings.return_value = [
            {
                "filer_name": "Big Fund",
                "filer_cik": "123",
                "report_date": "2025-12-31",
                "shares": 50000,
                "value_x1000": 5000,
            },
            {
                "filer_name": "Small Fund",
                "filer_cik": "456",
                "report_date": "2025-12-31",
                "shares": 10000,
                "value_x1000": 1000,
            },
        ]

        dp = DataProvider(mode="eodhd", eodhd_client=eodhd, edgar_client=edgar)
        result = dp.get_fundamentals("TEST.US")

        holders = result.get("Holders", {})
        institutions = holders.get("Institutions", [])
        assert len(institutions) == 2
        assert institutions[0]["name"] == "Big Fund"
        assert institutions[0]["date"] == "2025-12-31"
        assert institutions[0]["_source"] == "edgar_13f"

    def test_eodhd_mode_skips_edgar_when_holders_present(self):
        eodhd = _mock_eodhd_client()
        eodhd.get_fundamentals.return_value = {
            "General": {},
            "Financials": {},
            "Holders": {
                "Institutions": {
                    "0": {"name": "Existing Fund", "date": "2025-12-31", "totalShares": 5.0}
                }
            },
        }
        edgar = MagicMock()
        dp = DataProvider(mode="eodhd", eodhd_client=eodhd, edgar_client=edgar)
        dp.get_fundamentals("TEST.US")

        edgar.get_13f_holdings.assert_not_called()

    def test_edgar_holders_deduplicates_filers_per_date(self):
        eodhd = _mock_eodhd_client()
        eodhd.get_fundamentals.return_value = {
            "General": {},
            "Financials": {},
        }
        edgar = MagicMock()
        edgar.get_13f_holdings.return_value = [
            {"filer_name": "Big Fund", "filer_cik": "123", "report_date": "2025-12-31", "shares": 50000, "value_x1000": 5000},
            {"filer_name": "Big Fund", "filer_cik": "123", "report_date": "2025-12-31", "shares": 60000, "value_x1000": 6000},
        ]

        dp = DataProvider(mode="eodhd", eodhd_client=eodhd, edgar_client=edgar)
        result = dp.get_fundamentals("TEST.US")

        institutions = result.get("Holders", {}).get("Institutions", [])
        assert len(institutions) == 1


# ===========================================================================
# DataProvider insider transactions — EDGAR preference
# ===========================================================================

class TestEdgarInsiderTransactions:

    def test_prefers_edgar_over_eodhd(self):
        eodhd = _mock_eodhd_client()
        av = MagicMock()
        edgar = MagicMock()
        edgar.get_insider_transactions.return_value = [
            {
                "owner_name": "John CEO",
                "owner_cik": "789",
                "transaction_date": "2026-03-01",
                "transaction_code": "P",
                "shares": 1000,
                "price_per_share": 50.0,
                "shares_owned_after": 5000,
                "is_director": False,
                "is_officer": True,
                "officer_title": "CEO",
            }
        ]

        dp = DataProvider(mode="hybrid", eodhd_client=eodhd, av_client=av, edgar_client=edgar)
        result = dp.get_insider_transactions("AAPL.US", start_date="2026-01-01", end_date="2026-04-01")

        assert len(result) >= 1
        assert result[0]["ownerName"] == "John CEO"
        assert result[0]["transactionType"] == "Buy"
        assert result[0]["transactionShares"] == 1000
        assert result[0]["transactionPrice"] == 50.0
        assert result[0]["isOfficer"] is True
        eodhd.get_insider_transactions.assert_not_called()

    def test_falls_back_to_eodhd_when_edgar_empty(self):
        eodhd = _mock_eodhd_client()
        eodhd.get_insider_transactions.return_value = [{"date": "2026-03-01", "transactionCode": "P"}]
        av = MagicMock()
        edgar = MagicMock()
        edgar.get_insider_transactions.return_value = []

        dp = DataProvider(mode="hybrid", eodhd_client=eodhd, av_client=av, edgar_client=edgar)
        result = dp.get_insider_transactions("AAPL.US")

        eodhd.get_insider_transactions.assert_called_once()
        assert len(result) == 1

    def test_returns_empty_when_no_clients(self):
        dp = DataProvider(mode="alpha_vantage", av_client=MagicMock())
        result = dp.get_insider_transactions("AAPL.US")
        assert result == []


# ===========================================================================
# Insider conviction — field compatibility with EDGAR-normalized data
# ===========================================================================

class TestInsiderConvictionEdgarFields:

    def _mock_client_with_edgar_insiders(self):
        import datetime as _dt
        today = _dt.date.today()
        d1 = (today - _dt.timedelta(days=3)).isoformat()
        d2 = (today - _dt.timedelta(days=7)).isoformat()
        d3 = (today - _dt.timedelta(days=12)).isoformat()
        client = MagicMock()
        client.get_insider_transactions.return_value = [
            {
                "date": d1,
                "ownerName": "Jane CFO",
                "transactionType": "Buy",
                "transactionShares": 5000,
                "transactionPrice": 100.0,
                "isOfficer": True,
                "isDirector": False,
                "officerTitle": "Chief Financial Officer",
            },
            {
                "date": d2,
                "ownerName": "Bob Director",
                "transactionType": "Buy",
                "transactionShares": 2000,
                "transactionPrice": 98.0,
                "isOfficer": False,
                "isDirector": True,
                "officerTitle": "",
            },
            {
                "date": d3,
                "ownerName": "Alice VP",
                "transactionType": "Sale",
                "transactionShares": 500,
                "transactionPrice": 102.0,
                "isOfficer": True,
                "isDirector": False,
                "officerTitle": "VP Engineering",
            },
        ]
        return client

    def test_recognizes_edgar_buy_sale_codes(self):
        client = self._mock_client_with_edgar_insiders()
        result = compute_insider_conviction_metrics(client, "AAPL.US", lookback_days=30)

        assert result["insider_fetch_status"] == "ok"
        assert result["insider_conviction_trade_count"] == 3.0
        assert result["insider_conviction_buy_person_count"] == 2.0
        assert result["insider_conviction_sell_person_count"] == 1.0
        assert result["insider_conviction_signal"] is not None

    def test_uses_transactionShares_field(self):
        import datetime as _dt
        recent = ((_dt.date.today()) - _dt.timedelta(days=2)).isoformat()
        client = MagicMock()
        client.get_insider_transactions.return_value = [
            {
                "date": recent,
                "ownerName": "Test",
                "transactionType": "Buy",
                "transactionShares": 10000,
                "transactionPrice": 50.0,
                "isOfficer": True,
            },
        ]
        result = compute_insider_conviction_metrics(client, "TEST.US", lookback_days=30)
        assert result["insider_conviction_signal"] is not None
        assert result["insider_conviction_signal"] > 0

    def test_handles_empty_transactions(self):
        client = MagicMock()
        client.get_insider_transactions.return_value = []
        result = compute_insider_conviction_metrics(client, "TEST.US")
        assert result["insider_fetch_status"] == "empty"
        assert result["insider_conviction_signal"] is None

    def test_handles_fetch_error(self):
        client = MagicMock()
        client.get_insider_transactions.side_effect = ConnectionError("timeout")
        result = compute_insider_conviction_metrics(client, "TEST.US")
        assert result["insider_fetch_status"] == "error"
        assert result["insider_fetch_error"] == 1.0
        assert result["insider_fetch_error_type"] == "ConnectionError"


# ===========================================================================
# _insider_role_weight — EDGAR boolean flags
# ===========================================================================

class TestInsiderRoleWeight:

    def test_ceo_from_title_string(self):
        w = _insider_role_weight({"officerTitle": "Chief Executive Officer"})
        assert w == 2.0

    def test_cfo_from_title_string(self):
        w = _insider_role_weight({"officerTitle": "Chief Financial Officer"})
        assert w == 2.0

    def test_director_from_string(self):
        w = _insider_role_weight({"ownerRelationship": "Director"})
        assert w == 1.0

    def test_is_officer_boolean_flag(self):
        w = _insider_role_weight({"isOfficer": True})
        assert w == 1.0

    def test_is_director_boolean_flag(self):
        w = _insider_role_weight({"isDirector": True})
        assert w == 1.0

    def test_unknown_role_default(self):
        w = _insider_role_weight({})
        assert w == 0.75

    def test_unknown_role_strict(self):
        w = _insider_role_weight({}, strict=True)
        assert w == 0.5


# ===========================================================================
# Technical indicator metrics
# ===========================================================================

class TestTechnicalIndicatorMetrics:

    def _mock_av_client(
        self,
        rsi: float = 45.0,
        macd_hist: float = 0.5,
        bb_upper: float = 110.0,
        bb_lower: float = 90.0,
        bb_middle: float = 100.0,
        adx: float = 30.0,
        stoch_k: float = 35.0,
        stoch_d: float = 40.0,
    ):
        client = MagicMock()
        client.get_rsi.return_value = {
            "Technical Analysis: RSI": {"2026-04-14": {"RSI": str(rsi)}}
        }
        client.get_macd.return_value = {
            "Technical Analysis: MACD": {"2026-04-14": {"MACD_Hist": str(macd_hist)}}
        }
        client.get_bbands.return_value = {
            "Technical Analysis: BBANDS": {
                "2026-04-14": {
                    "Real Upper Band": str(bb_upper),
                    "Real Lower Band": str(bb_lower),
                    "Real Middle Band": str(bb_middle),
                }
            }
        }
        client.get_adx.return_value = {
            "Technical Analysis: ADX": {"2026-04-14": {"ADX": str(adx)}}
        }
        client.get_stoch.return_value = {
            "Technical Analysis: STOCH": {
                "2026-04-14": {"SlowK": str(stoch_k), "SlowD": str(stoch_d)}
            }
        }
        return client

    def test_all_indicators_produce_signal(self):
        client = self._mock_av_client()
        result = compute_technical_indicator_metrics(client, "AAPL")

        assert result["technical_fetch_status"] == "ok"
        assert result["technical_momentum_signal"] is not None
        assert -1.0 <= result["technical_momentum_signal"] <= 1.0
        assert result["technical_momentum_has_coverage"] == 1.0
        assert result["technical_rsi_14"] == 45.0
        assert result["technical_adx_14"] == 30.0

    def test_oversold_produces_positive_signal(self):
        client = self._mock_av_client(rsi=20.0, stoch_k=15.0, stoch_d=18.0, macd_hist=0.3)
        result = compute_technical_indicator_metrics(client, "AAPL")
        assert result["technical_momentum_signal"] > 0

    def test_overbought_produces_negative_signal(self):
        client = self._mock_av_client(rsi=80.0, stoch_k=85.0, stoch_d=82.0, macd_hist=-0.5)
        result = compute_technical_indicator_metrics(client, "AAPL")
        assert result["technical_momentum_signal"] < 0

    def test_returns_empty_when_no_get_rsi(self):
        client = MagicMock(spec=[])  # no attributes
        result = compute_technical_indicator_metrics(client, "AAPL")
        assert result["technical_fetch_status"] == "empty"
        assert result["technical_momentum_signal"] is None

    def test_handles_partial_indicator_failure(self):
        client = self._mock_av_client()
        client.get_macd.side_effect = Exception("API error")
        client.get_bbands.side_effect = Exception("API error")

        result = compute_technical_indicator_metrics(client, "AAPL")

        assert result["technical_fetch_status"] == "ok"
        assert result["technical_momentum_signal"] is not None
        assert result["technical_momentum_has_coverage"] < 1.0
        assert result["technical_rsi_14"] is not None
        assert result["technical_macd_histogram"] is None

    def test_adx_modulates_confidence(self):
        # Low ADX = weak trend = lower signal magnitude
        low_adx_client = self._mock_av_client(adx=10.0, rsi=25.0)
        high_adx_client = self._mock_av_client(adx=50.0, rsi=25.0)

        low_result = compute_technical_indicator_metrics(low_adx_client, "AAPL")
        high_result = compute_technical_indicator_metrics(high_adx_client, "AAPL")

        # Both should be positive (oversold RSI) but high ADX should have stronger signal
        assert low_result["technical_momentum_signal"] > 0
        assert high_result["technical_momentum_signal"] > 0
        assert abs(high_result["technical_momentum_signal"]) >= abs(low_result["technical_momentum_signal"])


# ===========================================================================
# Technical momentum overlay wiring in add_overlay_metrics
# ===========================================================================

class TestTechnicalMomentumOverlayWiring:

    def test_enabled_calls_compute(self, monkeypatch):
        fake_result = {
            "technical_rsi_14": 45.0,
            "technical_macd_histogram": 0.5,
            "technical_bbands_pct_b": 0.5,
            "technical_adx_14": 30.0,
            "technical_stoch_k": 50.0,
            "technical_stoch_d": 50.0,
            "technical_momentum_signal": 0.3,
            "technical_momentum_has_coverage": 1.0,
            "technical_fetch_status": "ok",
            "technical_fetch_error": 0.0,
        }
        monkeypatch.setattr(
            "eodhd_strategy.features.compute_technical_indicator_metrics",
            lambda client, symbol: fake_result,
        )

        row = {"symbol": "AAPL.US"}
        cfg = _config(use_technical_momentum=True)
        out = add_overlay_metrics(object(), row, cfg)

        assert out["technical_momentum_signal"] == 0.3
        assert out["technical_fetch_status"] == "ok"

    def test_disabled_sets_defaults(self):
        row = {"symbol": "AAPL.US"}
        cfg = _config(use_technical_momentum=False)
        out = add_overlay_metrics(object(), row, cfg)

        assert out["technical_momentum_signal"] is None
        assert out["technical_momentum_has_coverage"] == 0.0
        assert out["technical_fetch_status"] == "not_requested"


# ===========================================================================
# DataProvider technical indicator passthroughs
# ===========================================================================

class TestDataProviderTechnicalPassthroughs:

    def test_get_rsi_delegates_to_av(self):
        av = MagicMock()
        av.get_rsi.return_value = {"Technical Analysis: RSI": {"2026-04-14": {"RSI": "50"}}}
        dp = DataProvider(mode="alpha_vantage", av_client=av)

        result = dp.get_rsi("AAPL.US")
        av.get_rsi.assert_called_once_with("AAPL", 14)
        assert "Technical Analysis: RSI" in result

    def test_get_rsi_returns_empty_without_av(self):
        dp = DataProvider(mode="eodhd", eodhd_client=_mock_eodhd_client())
        assert dp.get_rsi("AAPL.US") == {}

    def test_get_macd_delegates_to_av(self):
        av = MagicMock()
        av.get_macd.return_value = {"Technical Analysis: MACD": {}}
        dp = DataProvider(mode="alpha_vantage", av_client=av)
        dp.get_macd("MSFT.US")
        av.get_macd.assert_called_once_with("MSFT")

    def test_get_bbands_delegates_to_av(self):
        av = MagicMock()
        av.get_bbands.return_value = {"Technical Analysis: BBANDS": {}}
        dp = DataProvider(mode="alpha_vantage", av_client=av)
        dp.get_bbands("AAPL.US")
        av.get_bbands.assert_called_once_with("AAPL", 20)

    def test_get_adx_delegates_to_av(self):
        av = MagicMock()
        av.get_adx.return_value = {"Technical Analysis: ADX": {}}
        dp = DataProvider(mode="alpha_vantage", av_client=av)
        dp.get_adx("AAPL.US")
        av.get_adx.assert_called_once_with("AAPL", 14)

    def test_get_stoch_delegates_to_av(self):
        av = MagicMock()
        av.get_stoch.return_value = {"Technical Analysis: STOCH": {}}
        dp = DataProvider(mode="alpha_vantage", av_client=av)
        dp.get_stoch("AAPL.US")
        av.get_stoch.assert_called_once_with("AAPL")


# ===========================================================================
# DataProvider EDGAR direct access
# ===========================================================================

class TestDataProviderEdgarDirect:

    def test_get_13f_holdings_delegates(self):
        edgar = MagicMock()
        edgar.get_13f_holdings.return_value = [{"filer_name": "Fund", "shares": 100}]
        dp = DataProvider(mode="eodhd", eodhd_client=_mock_eodhd_client(), edgar_client=edgar)

        result = dp.get_13f_holdings("AAPL")
        edgar.get_13f_holdings.assert_called_once_with("AAPL", 4)
        assert len(result) == 1

    def test_get_13f_holdings_empty_without_edgar(self):
        dp = DataProvider(mode="eodhd", eodhd_client=_mock_eodhd_client())
        assert dp.get_13f_holdings("AAPL") == []

    def test_get_edgar_insider_summary_delegates(self):
        edgar = MagicMock()
        edgar.get_insider_summary.return_value = {
            "buy_count": 3,
            "sell_count": 1,
            "net_value": 50000,
            "cluster_detected": True,
        }
        dp = DataProvider(mode="eodhd", eodhd_client=_mock_eodhd_client(), edgar_client=edgar)

        result = dp.get_edgar_insider_summary("AAPL")
        assert result["cluster_detected"] is True

    def test_get_edgar_insider_summary_empty_without_edgar(self):
        dp = DataProvider(mode="eodhd", eodhd_client=_mock_eodhd_client())
        assert dp.get_edgar_insider_summary("AAPL") == {}


# ===========================================================================
# Config fields
# ===========================================================================

class TestConfigIntegration:

    def test_data_provider_field_defaults(self):
        cfg = _config()
        assert cfg.data_provider == "eodhd"
        assert cfg.alpha_vantage_api_key == ""
        assert cfg.sec_edgar_email == ""

    def test_technical_momentum_field_defaults(self):
        cfg = _config()
        assert cfg.use_technical_momentum is False
        assert cfg.technical_momentum_weight == 0.05

    def test_config_accepts_all_modes(self):
        for mode in ("eodhd", "alpha_vantage", "hybrid"):
            cfg = _config(data_provider=mode)
            assert cfg.data_provider == mode


# ===========================================================================
# run_rank CLI arg parsing
# ===========================================================================

class TestRunRankCLIArgs:

    def test_data_provider_args_parsed(self):
        from run_rank import parse_args
        args = parse_args([
            "--api-token", "test",
            "--data-provider", "hybrid",
            "--alpha-vantage-api-key", "av_key_123",
            "--sec-edgar-email", "test@example.com",
            "--symbols", "AAPL.US",
        ])
        assert args.data_provider == "hybrid"
        assert args.alpha_vantage_api_key == "av_key_123"
        assert args.sec_edgar_email == "test@example.com"

    def test_technical_momentum_args_parsed(self):
        from run_rank import parse_args
        args = parse_args([
            "--api-token", "test",
            "--use-technical-momentum",
            "--technical-momentum-weight", "0.07",
            "--symbols", "AAPL.US",
        ])
        assert args.use_technical_momentum is True
        assert args.technical_momentum_weight == 0.07

    def test_defaults_when_not_specified(self):
        from run_rank import parse_args
        args = parse_args(["--api-token", "test", "--symbols", "AAPL.US"])
        assert args.data_provider == "eodhd"
        assert args.use_technical_momentum is False


# ===========================================================================
# _make_client integration
# ===========================================================================

class TestMakeClient:

    def test_eodhd_mode_creates_data_provider(self):
        from run_rank import _make_client
        cfg = _config(api_token="fake_token", cache_dir=Path(".eodhd_cache"))
        client = _make_client(cfg)
        assert isinstance(client, DataProvider)
        assert client.eodhd is not None

    def test_alpha_vantage_only_mode(self):
        from run_rank import _make_client
        cfg = _config(
            api_token="",
            cache_dir=Path(".eodhd_cache"),
            data_provider="alpha_vantage",
            alpha_vantage_api_key="test_av_key",
        )
        client = _make_client(cfg)
        assert isinstance(client, DataProvider)
        assert client.eodhd is None
        assert client.av is not None


# ===========================================================================
# AV News Sentiment Enhancement Tests
# ===========================================================================

import datetime
import pandas as pd


def _make_av_article(
    title: str = "Test Article",
    date: str | None = None,
    polarity: float = 0.0,
    av_ticker_sentiment: float | None = None,
    av_relevance: float | None = None,
    av_overall_label: str | None = None,
    tags: list | None = None,
    symbols: list | None = None,
):
    """Build a mock article dict with optional AV enrichment."""
    if date is None:
        date = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    article = {
        "date": date,
        "title": title,
        "content": "",
        "link": "",
        "symbols": symbols or ["ACME"],
        "tags": tags or [],
        "sentiment": {"polarity": polarity, "neg": 0, "neu": 0, "pos": 0},
    }
    if av_ticker_sentiment is not None:
        article["av_ticker_sentiment"] = av_ticker_sentiment
    if av_relevance is not None:
        article["av_relevance"] = av_relevance
    if av_overall_label is not None:
        article["av_overall_label"] = av_overall_label
    return article


class TestExtractAvArticleSentiment:

    def test_returns_has_av_false_for_eodhd_article(self):
        article = {"title": "Test", "sentiment": {"polarity": 0.3}}
        _, _, _, has_av = _extract_av_article_sentiment(article)
        assert has_av is False

    def test_returns_has_av_true_with_ticker_sentiment(self):
        article = {"av_ticker_sentiment": 0.6, "av_relevance": 0.8, "av_overall_label": "Bullish"}
        ticker_sent, relevance, label_bias, has_av = _extract_av_article_sentiment(article)
        assert has_av is True
        assert ticker_sent == 0.6
        assert relevance == 0.8
        assert label_bias == 1.0

    def test_label_bias_mapping(self):
        for label, expected in [("Bullish", 1.0), ("Somewhat-Bullish", 0.5),
                                ("Neutral", 0.0), ("Somewhat-Bearish", -0.5),
                                ("Bearish", -1.0)]:
            article = {"av_ticker_sentiment": 0.1, "av_overall_label": label}
            _, _, bias, _ = _extract_av_article_sentiment(article)
            assert bias == expected, f"Expected {expected} for label '{label}', got {bias}"

    def test_zero_ticker_sentiment_detects_relevance(self):
        article = {"av_ticker_sentiment": 0.0, "av_relevance": 0.5}
        _, _, _, has_av = _extract_av_article_sentiment(article)
        assert has_av is True  # relevance alone triggers has_av


class TestAvTopicCategories:

    def test_maps_known_topics(self):
        article = {"tags": ["earnings", "mergers_and_acquisitions"]}
        cats = _av_topic_categories(article)
        assert "earnings_positive" in cats
        assert "corporate_positive" in cats

    def test_ignores_unmapped_topics(self):
        article = {"tags": ["technology", "real_estate"]}
        cats = _av_topic_categories(article)
        assert cats == []

    def test_handles_missing_tags(self):
        assert _av_topic_categories({}) == []
        assert _av_topic_categories({"tags": None}) == []

    def test_maps_ipo_to_corporate_positive(self):
        article = {"tags": ["ipo"]}
        cats = _av_topic_categories(article)
        assert cats == ["corporate_positive"]


class TestNewsEventMetricsWithAV:

    def _fake_client(self, articles):
        client = MagicMock()
        client.get_news.return_value = articles
        return client

    def test_av_ticker_sentiment_overrides_polarity(self):
        today = datetime.date.today().isoformat()
        articles = [
            _make_av_article(
                title="ACME beats estimates",
                date=today,
                polarity=-0.2,  # overall polarity is negative
                av_ticker_sentiment=0.7,  # but ticker-specific is very positive
                av_relevance=0.9,
                av_overall_label="Bullish",
            ),
        ]
        result = compute_news_event_metrics(self._fake_client(articles), "ACME.US", lookback_days=7)
        # Signal should be positive because AV ticker sentiment dominates
        assert result["news_event_signal"] is not None
        assert result["news_event_signal"] > 0

    def test_av_relevance_used_as_weight(self):
        today = datetime.date.today().isoformat()
        # Two articles: one highly relevant, one not
        articles = [
            _make_av_article(date=today, polarity=0.5, av_ticker_sentiment=0.5, av_relevance=0.95),
            _make_av_article(date=today, title="Low relevance", polarity=-0.5,
                             av_ticker_sentiment=-0.5, av_relevance=0.1),
        ]
        result = compute_news_event_metrics(self._fake_client(articles), "ACME.US", lookback_days=7)
        assert result["news_event_signal"] is not None
        # High-relevance positive article should dominate over low-relevance negative
        assert result["news_event_signal"] > 0

    def test_av_label_bias_boosts_signal(self):
        today = datetime.date.today().isoformat()
        # Article with no keyword matches, weak polarity, but strong AV label
        articles = [
            _make_av_article(
                title="Some unrelated headline about ACME",
                date=today,
                polarity=0.1,
                av_ticker_sentiment=0.1,
                av_relevance=0.8,
                av_overall_label="Bullish",
            ),
        ]
        result = compute_news_event_metrics(self._fake_client(articles), "ACME.US", lookback_days=7)
        assert result["news_event_signal"] is not None
        # Label bias should push signal slightly more positive
        assert result["news_event_signal"] > 0.0

    def test_av_topic_tags_supplement_categories(self):
        today = datetime.date.today().isoformat()
        articles = [
            _make_av_article(
                title="Some headline",  # no keyword matches
                date=today,
                polarity=0.3,
                av_ticker_sentiment=0.3,
                av_relevance=0.8,
                tags=["earnings"],
            ),
        ]
        result = compute_news_event_metrics(self._fake_client(articles), "ACME.US", lookback_days=7)
        assert result["news_event_signal"] is not None
        assert result["news_event_breadth"] >= 1.0  # at least one category matched

    def test_fallback_to_eodhd_when_no_av_fields(self):
        today = datetime.date.today().isoformat()
        articles = [
            {
                "date": today,
                "title": "ACME beats estimates with strong revenue growth",
                "content": "",
                "symbols": ["ACME"],
                "tags": [],
                "sentiment": {"polarity": 0.4, "neg": 0, "neu": 0, "pos": 0},
            },
        ]
        result = compute_news_event_metrics(self._fake_client(articles), "ACME.US", lookback_days=7)
        assert result["news_event_signal"] is not None
        assert result["news_event_signal"] > 0  # keyword match + positive polarity


class TestNewsThemeDriftMetricsWithAV:

    def _fake_client(self, articles):
        client = MagicMock()
        client.get_news.return_value = articles
        return client

    def test_av_fields_affect_theme_drift(self):
        today = datetime.date.today()
        recent = (today - datetime.timedelta(days=5)).isoformat()
        baseline = (today - datetime.timedelta(days=45)).isoformat()
        articles = [
            _make_av_article(date=recent, polarity=0.1, av_ticker_sentiment=0.8,
                             av_relevance=0.9, av_overall_label="Bullish"),
            _make_av_article(date=baseline, polarity=-0.1, av_ticker_sentiment=-0.5,
                             av_relevance=0.7, av_overall_label="Bearish"),
        ]
        result = compute_news_theme_drift_metrics(
            self._fake_client(articles), "ACME.US",
            recent_window_days=30, baseline_window_days=90,
        )
        assert result["news_theme_drift_signal"] is not None
        # Recent bullish + baseline bearish → positive drift
        assert result["news_theme_drift_signal"] > 0

    def test_eodhd_articles_still_work(self):
        today = datetime.date.today()
        recent = (today - datetime.timedelta(days=5)).isoformat()
        baseline = (today - datetime.timedelta(days=45)).isoformat()
        articles = [
            {"date": recent, "title": "ACME revenue growth strong", "symbols": ["ACME"],
             "tags": [], "sentiment": {"polarity": 0.4}},
            {"date": baseline, "title": "ACME weak outlook", "symbols": ["ACME"],
             "tags": [], "sentiment": {"polarity": -0.3}},
        ]
        result = compute_news_theme_drift_metrics(
            self._fake_client(articles), "ACME.US",
            recent_window_days=30, baseline_window_days=90,
        )
        assert result["news_theme_drift_signal"] is not None


class TestSentimentMetricsAVFormat:

    def _fake_client(self, sentiments_payload):
        client = MagicMock()
        client.get_sentiments.return_value = sentiments_payload
        return client

    def test_handles_av_date_keyed_dict(self):
        """AV _av_sentiments returns {date: {date, count, normalized}}."""
        today = datetime.date.today()
        payload = {}
        for i in range(5):
            d = (today - datetime.timedelta(days=i)).isoformat()
            payload[d] = {"date": d, "count": 3, "normalized": 0.2 + 0.05 * i}

        result = compute_sentiment_metrics(self._fake_client(payload), "ACME.US", lookback_days=14)
        assert result["sentiment_fetch_status"] == "ok"
        assert result["sentiment_latest"] is not None
        assert result["sentiment_count_days"] == 5

    def test_handles_eodhd_symbol_keyed_list(self):
        """EODHD returns {symbol: [{date, normalized, count}]}."""
        today = datetime.date.today()
        series = []
        for i in range(5):
            d = (today - datetime.timedelta(days=i)).isoformat()
            series.append({"date": d, "count": 2, "normalized": 0.1 * i})

        payload = {"ACME.US": series}
        result = compute_sentiment_metrics(self._fake_client(payload), "ACME.US", lookback_days=14)
        assert result["sentiment_fetch_status"] == "ok"
        assert result["sentiment_count_days"] == 5

    def test_empty_payload_returns_empty(self):
        result = compute_sentiment_metrics(self._fake_client({}), "ACME.US", lookback_days=14)
        assert result["sentiment_fetch_status"] == "empty"


class TestAvSentimentsRelevanceWeighting:

    def test_ticker_sentiment_weighted_by_relevance(self):
        """Verify _av_sentiments uses ticker-specific scores weighted by relevance."""
        from eodhd_strategy.data_provider import DataProvider

        av_mock = MagicMock()
        av_mock.get_news_sentiment.return_value = {
            "feed": [
                {
                    "time_published": "20250415T1200",
                    "overall_sentiment_score": 0.1,
                    "ticker_sentiment": [
                        {"ticker": "ACME", "ticker_sentiment_score": 0.8, "relevance_score": 0.9},
                    ],
                },
                {
                    "time_published": "20250415T1400",
                    "overall_sentiment_score": -0.3,
                    "ticker_sentiment": [
                        {"ticker": "ACME", "ticker_sentiment_score": -0.2, "relevance_score": 0.1},
                    ],
                },
            ]
        }

        dp = DataProvider(mode="alpha_vantage", av_client=av_mock)
        result = dp._av_sentiments("ACME.US", "2025-04-10", "2025-04-16")

        assert "2025-04-15" in result
        entry = result["2025-04-15"]
        assert entry["count"] == 2
        # Weighted avg: (0.8*0.9 + -0.2*0.1) / (0.9 + 0.1) = (0.72 - 0.02) / 1.0 = 0.70
        assert abs(entry["normalized"] - 0.70) < 0.01

    def test_falls_back_to_overall_when_no_ticker_match(self):
        from eodhd_strategy.data_provider import DataProvider

        av_mock = MagicMock()
        av_mock.get_news_sentiment.return_value = {
            "feed": [
                {
                    "time_published": "20250415T1200",
                    "overall_sentiment_score": 0.5,
                    "ticker_sentiment": [
                        {"ticker": "OTHER", "ticker_sentiment_score": 0.9, "relevance_score": 0.8},
                    ],
                },
            ]
        }

        dp = DataProvider(mode="alpha_vantage", av_client=av_mock)
        result = dp._av_sentiments("ACME.US", "2025-04-10", "2025-04-16")

        assert "2025-04-15" in result
        entry = result["2025-04-15"]
        # No ACME ticker match → falls back to overall_sentiment_score = 0.5
        assert abs(entry["normalized"] - 0.5) < 0.01
