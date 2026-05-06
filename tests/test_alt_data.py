"""Tests for alt data providers: FundingRateProvider and FearGreedProvider."""
from __future__ import annotations

import json
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest


# ── FundingRateProvider ───────────────────────────────────────────────────────

class TestFundingRateProvider:
    def test_import(self):
        from privateye.data.providers.funding_rates import FundingRateProvider
        p = FundingRateProvider("BTC/USDT")
        assert p.symbol == "BTC/USDT"

    def test_empty_df_columns(self):
        from privateye.data.providers.funding_rates import _empty_funding_df
        df = _empty_funding_df()
        assert list(df.columns) == ["timestamp", "funding_rate", "open_interest"]
        assert len(df) == 0

    def test_fetch_latest_mocked(self):
        from privateye.data.providers.funding_rates import FundingRateProvider

        mock_fr = {"fundingRate": 0.0001}
        mock_oi = {"openInterestAmount": 12345.0}

        provider = FundingRateProvider("BTC/USDT")
        mock_exchange = MagicMock()
        mock_exchange.fetch_funding_rate.return_value = mock_fr
        mock_exchange.fetch_open_interest.return_value = mock_oi
        provider._exchange = mock_exchange

        result = asyncio.get_event_loop().run_until_complete(provider.fetch_latest())

        assert "timestamp" in result
        assert abs(result["funding_rate"] - 0.0001) < 1e-9
        assert abs(result["open_interest"] - 12345.0) < 1e-9

    def test_fetch_history_mocked(self):
        from privateye.data.providers.funding_rates import FundingRateProvider

        mock_rows = [
            {"timestamp": 1_700_000_000_000, "fundingRate": 0.0001},
            {"timestamp": 1_700_008_000_000, "fundingRate": -0.0002},
            {"timestamp": 1_700_016_000_000, "fundingRate": 0.0003},
        ]

        provider = FundingRateProvider("BTC/USDT")
        mock_exchange = MagicMock()
        mock_exchange.fetch_funding_rate_history.return_value = mock_rows
        provider._exchange = mock_exchange

        df = asyncio.get_event_loop().run_until_complete(
            provider.fetch_history(since_ms=1_699_000_000_000, limit=100)
        )

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["timestamp", "funding_rate", "open_interest"]
        assert len(df) == 3
        assert df["funding_rate"].iloc[0] == pytest.approx(0.0001)
        assert df["funding_rate"].iloc[1] == pytest.approx(-0.0002)

    def test_fetch_history_error_returns_empty(self):
        from privateye.data.providers.funding_rates import FundingRateProvider

        provider = FundingRateProvider("BTC/USDT")
        mock_exchange = MagicMock()
        mock_exchange.fetch_funding_rate_history.side_effect = Exception("network error")
        provider._exchange = mock_exchange

        df = asyncio.get_event_loop().run_until_complete(provider.fetch_history())
        assert df.empty

    def test_fetch_history_sorted(self):
        from privateye.data.providers.funding_rates import FundingRateProvider

        mock_rows = [
            {"timestamp": 1_700_016_000_000, "fundingRate": 0.0003},
            {"timestamp": 1_700_000_000_000, "fundingRate": 0.0001},
            {"timestamp": 1_700_008_000_000, "fundingRate": -0.0002},
        ]

        provider = FundingRateProvider("BTC/USDT")
        mock_exchange = MagicMock()
        mock_exchange.fetch_funding_rate_history.return_value = mock_rows
        provider._exchange = mock_exchange

        df = asyncio.get_event_loop().run_until_complete(provider.fetch_history())
        assert df["timestamp"].is_monotonic_increasing


# ── FearGreedProvider ─────────────────────────────────────────────────────────

_MOCK_FG_RESPONSE = json.dumps({
    "data": [
        {"timestamp": "1700000000", "value": "25", "value_classification": "Fear"},
        {"timestamp": "1700086400", "value": "50", "value_classification": "Neutral"},
        {"timestamp": "1700172800", "value": "75", "value_classification": "Greed"},
    ]
})


class TestFearGreedProvider:
    def test_import(self):
        from privateye.data.providers.sentiment import FearGreedProvider
        p = FearGreedProvider()
        assert p.timeout == 10.0

    def test_empty_df_columns(self):
        from privateye.data.providers.sentiment import _empty_fg_df
        df = _empty_fg_df()
        assert list(df.columns) == ["timestamp", "fear_greed", "classification"]
        assert len(df) == 0

    def test_fetch_history_mocked(self):
        from privateye.data.providers.sentiment import FearGreedProvider, _http_get

        provider = FearGreedProvider()

        with patch("privateye.data.providers.sentiment._http_get", return_value=_MOCK_FG_RESPONSE):
            df = asyncio.get_event_loop().run_until_complete(provider.fetch_history(limit=3))

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["timestamp", "fear_greed", "classification"]
        assert len(df) == 3
        assert df["fear_greed"].iloc[0] == pytest.approx(25.0)
        assert df["classification"].iloc[2] == "Greed"

    def test_fetch_latest_mocked(self):
        from privateye.data.providers.sentiment import FearGreedProvider

        single = json.dumps({
            "data": [{"timestamp": "1700000000", "value": "30", "value_classification": "Fear"}]
        })
        provider = FearGreedProvider()

        with patch("privateye.data.providers.sentiment._http_get", return_value=single):
            result = asyncio.get_event_loop().run_until_complete(provider.fetch_latest())

        assert result["fear_greed"] == pytest.approx(30.0)
        assert result["classification"] == "Fear"
        assert "timestamp" in result

    def test_fetch_history_sorted(self):
        from privateye.data.providers.sentiment import FearGreedProvider

        # Reversed order in response
        reversed_resp = json.dumps({
            "data": [
                {"timestamp": "1700172800", "value": "75", "value_classification": "Greed"},
                {"timestamp": "1700086400", "value": "50", "value_classification": "Neutral"},
                {"timestamp": "1700000000", "value": "25", "value_classification": "Fear"},
            ]
        })
        provider = FearGreedProvider()

        with patch("privateye.data.providers.sentiment._http_get", return_value=reversed_resp):
            df = asyncio.get_event_loop().run_until_complete(provider.fetch_history(limit=3))

        assert df["timestamp"].is_monotonic_increasing

    def test_fetch_history_error_returns_empty(self):
        from privateye.data.providers.sentiment import FearGreedProvider

        provider = FearGreedProvider()
        with patch("privateye.data.providers.sentiment._http_get", side_effect=Exception("timeout")):
            df = asyncio.get_event_loop().run_until_complete(provider.fetch_history())

        assert df.empty

    def test_fear_greed_range(self):
        from privateye.data.providers.sentiment import FearGreedProvider

        provider = FearGreedProvider()
        with patch("privateye.data.providers.sentiment._http_get", return_value=_MOCK_FG_RESPONSE):
            df = asyncio.get_event_loop().run_until_complete(provider.fetch_history(limit=3))

        assert (df["fear_greed"] >= 0).all()
        assert (df["fear_greed"] <= 100).all()
