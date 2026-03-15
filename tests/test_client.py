"""Tests for the Binance client wrapper."""
import pytest
from unittest.mock import MagicMock, patch

from binance.exceptions import BinanceAPIException

from bot.client import get_balance, get_symbol_price, get_klines


def _api_error():
    return BinanceAPIException(
        MagicMock(status_code=400), 400, '{"msg":"test error","code":-1000}'
    )


class TestGetBalance:
    def test_returns_float_balance(self):
        client = MagicMock()
        client.get_asset_balance.return_value = {"free": "123.456"}
        assert get_balance(client, "USDT") == pytest.approx(123.456)

    def test_returns_zero_on_none_response(self):
        client = MagicMock()
        client.get_asset_balance.return_value = None
        assert get_balance(client, "USDT") == 0.0

    def test_returns_zero_on_api_error(self):
        client = MagicMock()
        client.get_asset_balance.side_effect = _api_error()
        assert get_balance(client, "USDT") == 0.0

    def test_returns_free_balance_only(self):
        client = MagicMock()
        client.get_asset_balance.return_value = {"free": "50.0", "locked": "25.0"}
        assert get_balance(client, "BTC") == pytest.approx(50.0)


class TestGetSymbolPrice:
    def test_returns_float_price(self):
        client = MagicMock()
        client.get_symbol_ticker.return_value = {"price": "45000.50"}
        assert get_symbol_price(client, "BTCUSDT") == pytest.approx(45000.50)

    def test_returns_zero_on_api_error(self):
        client = MagicMock()
        client.get_symbol_ticker.side_effect = _api_error()
        assert get_symbol_price(client, "BTCUSDT") == 0.0


class TestGetKlines:
    def test_returns_list(self):
        client = MagicMock()
        client.get_klines.return_value = [[1, 2, 3]] * 5
        result = get_klines(client, "BTCUSDT", "15m")
        assert isinstance(result, list)
        assert len(result) == 5

    def test_returns_empty_list_on_api_error(self):
        client = MagicMock()
        client.get_klines.side_effect = _api_error()
        result = get_klines(client, "BTCUSDT", "15m")
        assert result == []

    def test_passes_limit_to_client(self):
        client = MagicMock()
        client.get_klines.return_value = []
        get_klines(client, "BTCUSDT", "1h", limit=50)
        client.get_klines.assert_called_once_with(
            symbol="BTCUSDT", interval="1h", limit=50
        )
