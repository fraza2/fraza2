"""Tests for the trader module."""
import pytest
from unittest.mock import MagicMock, patch

from bot.trader import (
    Portfolio,
    Trade,
    _step_size_decimals,
    calculate_quantity,
    close_trade,
    manage_open_trades,
    open_trade,
)
from bot.strategy import Signal, StrategyResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(price: float = 50000.0, signal: Signal = Signal.BUY) -> StrategyResult:
    return StrategyResult(
        signal=signal,
        rsi=25.0,
        macd=10.0,
        macd_signal=5.0,
        ema_short=50100.0,
        ema_long=49900.0,
        price=price,
        reason="test",
    )


def _make_trade(entry_price: float = 50000.0) -> Trade:
    return Trade(
        symbol="BTCUSDT",
        side="BUY",
        quantity=0.001,
        entry_price=entry_price,
        stop_loss=entry_price * 0.98,
        take_profit=entry_price * 1.04,
    )


def _mock_client(balance: float = 1000.0, symbol_info=None):
    client = MagicMock()
    client.get_asset_balance.return_value = {"free": str(balance)}
    client.get_symbol_info.return_value = symbol_info or {
        "filters": [
            {"filterType": "LOT_SIZE", "minQty": "0.001", "maxQty": "9000.0", "stepSize": "0.001"},
        ]
    }
    client.order_market.return_value = {"orderId": 12345, "status": "FILLED"}
    return client


# ---------------------------------------------------------------------------
# _step_size_decimals
# ---------------------------------------------------------------------------

class TestStepSizeDecimals:
    def test_three_decimals(self):
        assert _step_size_decimals("0.001000") == 3

    def test_one_decimal(self):
        assert _step_size_decimals("0.1") == 1

    def test_integer_step(self):
        assert _step_size_decimals("1.00000") == 0

    def test_eight_decimals(self):
        assert _step_size_decimals("0.00000001") == 8


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

class TestPortfolio:
    def test_open_trades_empty_initially(self):
        p = Portfolio()
        assert p.open_trades == []

    def test_open_trades_count(self):
        p = Portfolio()
        p.trades.append(_make_trade())
        p.trades.append(_make_trade())
        assert len(p.open_trades) == 2

    def test_closed_trades_count(self):
        p = Portfolio()
        t = _make_trade()
        t.closed = True
        t.pnl = 50.0
        p.trades.append(t)
        assert len(p.closed_trades) == 1

    def test_total_pnl(self):
        p = Portfolio()
        for pnl in (100.0, -30.0, 50.0):
            t = _make_trade()
            t.closed = True
            t.pnl = pnl
            p.trades.append(t)
        assert p.total_pnl == pytest.approx(120.0)

    def test_summary_is_string(self):
        p = Portfolio()
        assert isinstance(p.summary(), str)


# ---------------------------------------------------------------------------
# calculate_quantity
# ---------------------------------------------------------------------------

class TestCalculateQuantity:
    def test_quantity_positive(self):
        client = _mock_client(balance=1000.0)
        qty = calculate_quantity(client, "BTCUSDT", 50000.0)
        assert qty > 0

    def test_quantity_zero_when_no_balance(self):
        client = _mock_client(balance=0.0)
        qty = calculate_quantity(client, "BTCUSDT", 50000.0)
        assert qty == 0.0

    def test_quantity_respects_step_size(self):
        client = _mock_client(balance=1000.0)
        qty = calculate_quantity(client, "BTCUSDT", 50000.0)
        # qty should be a multiple of 0.001
        assert round(qty / 0.001) == pytest.approx(qty / 0.001, abs=1e-6)

    def test_quantity_within_trade_percent(self):
        """Quantity * price should be ~5% of balance."""
        balance = 1000.0
        price = 50000.0
        client = _mock_client(balance=balance)
        qty = calculate_quantity(client, "BTCUSDT", price)
        trade_value = qty * price
        expected = balance * 0.05
        assert trade_value == pytest.approx(expected, rel=0.1)


# ---------------------------------------------------------------------------
# open_trade
# ---------------------------------------------------------------------------

class TestOpenTrade:
    def test_opens_trade_on_buy_signal(self):
        client = _mock_client()
        portfolio = Portfolio()
        result = _make_result(signal=Signal.BUY)
        trade = open_trade(client, portfolio, result)
        assert trade is not None
        assert len(portfolio.open_trades) == 1

    def test_skips_when_max_trades_reached(self):
        client = _mock_client()
        portfolio = Portfolio()
        # Fill up to max
        for _ in range(3):
            portfolio.trades.append(_make_trade())

        result = _make_result(signal=Signal.BUY)
        trade = open_trade(client, portfolio, result)
        assert trade is None

    def test_trade_has_stop_loss_and_take_profit(self):
        client = _mock_client()
        portfolio = Portfolio()
        price = 50000.0
        result = _make_result(price=price, signal=Signal.BUY)
        trade = open_trade(client, portfolio, result)
        assert trade is not None
        assert trade.stop_loss < price
        assert trade.take_profit > price

    def test_returns_none_when_order_fails(self):
        client = _mock_client()
        client.order_market.return_value = None
        # Make place_order return None by raising exception
        from binance.exceptions import BinanceAPIException
        client.order_market.side_effect = BinanceAPIException(
            MagicMock(status_code=400), 400, '{"msg":"error","code":-1000}'
        )
        portfolio = Portfolio()
        result = _make_result(signal=Signal.BUY)
        trade = open_trade(client, portfolio, result)
        assert trade is None


# ---------------------------------------------------------------------------
# close_trade
# ---------------------------------------------------------------------------

class TestCloseTrade:
    def test_closes_trade(self):
        client = _mock_client()
        trade = _make_trade(entry_price=50000.0)
        result = close_trade(client, trade, 52000.0, "TAKE_PROFIT")
        assert result is True
        assert trade.closed is True

    def test_calculates_pnl(self):
        client = _mock_client()
        trade = _make_trade(entry_price=50000.0)
        trade.quantity = 0.001
        close_trade(client, trade, 52000.0, "TAKE_PROFIT")
        expected_pnl = (52000.0 - 50000.0) * 0.001
        assert trade.pnl == pytest.approx(expected_pnl)

    def test_negative_pnl_on_stop_loss(self):
        client = _mock_client()
        trade = _make_trade(entry_price=50000.0)
        trade.quantity = 0.001
        close_trade(client, trade, 49000.0, "STOP_LOSS")
        assert trade.pnl < 0

    def test_returns_false_when_order_fails(self):
        from binance.exceptions import BinanceAPIException
        client = _mock_client()
        client.order_market.side_effect = BinanceAPIException(
            MagicMock(status_code=400), 400, '{"msg":"error","code":-1000}'
        )
        trade = _make_trade()
        result = close_trade(client, trade, 49000.0, "STOP_LOSS")
        assert result is False
        assert trade.closed is False


# ---------------------------------------------------------------------------
# manage_open_trades
# ---------------------------------------------------------------------------

class TestManageOpenTrades:
    def test_closes_on_stop_loss(self):
        client = _mock_client()
        portfolio = Portfolio()
        trade = _make_trade(entry_price=50000.0)  # SL at 49000
        portfolio.trades.append(trade)
        manage_open_trades(client, portfolio, current_price=48000.0)
        assert trade.closed is True

    def test_closes_on_take_profit(self):
        client = _mock_client()
        portfolio = Portfolio()
        trade = _make_trade(entry_price=50000.0)  # TP at 52000
        portfolio.trades.append(trade)
        manage_open_trades(client, portfolio, current_price=53000.0)
        assert trade.closed is True

    def test_holds_within_range(self):
        client = _mock_client()
        portfolio = Portfolio()
        trade = _make_trade(entry_price=50000.0)
        portfolio.trades.append(trade)
        manage_open_trades(client, portfolio, current_price=50500.0)
        assert trade.closed is False

    def test_handles_empty_portfolio(self):
        client = _mock_client()
        portfolio = Portfolio()
        # Should not raise
        manage_open_trades(client, portfolio, current_price=50000.0)
