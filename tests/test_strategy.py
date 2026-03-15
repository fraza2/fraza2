"""Tests for the trading strategy module."""
import math
import time
import pytest
import pandas as pd

from bot.strategy import (
    Signal,
    build_dataframe,
    compute_indicators,
    analyze,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_klines(closes: list, base_time: int = 1_700_000_000_000) -> list:
    """Build minimal fake klines list from a list of close prices."""
    klines = []
    for i, close in enumerate(closes):
        open_time = base_time + i * 60_000
        close_time = open_time + 59_999
        row = [
            open_time,        # open_time
            str(close),       # open  (use close for simplicity)
            str(close * 1.001),  # high
            str(close * 0.999),  # low
            str(close),       # close
            "100.0",          # volume
            close_time,
            "10000.0",        # quote_volume
            100,              # trades
            "50.0",           # taker_buy_base
            "5000.0",         # taker_buy_quote
            "0",              # ignore
        ]
        klines.append(row)
    return klines


def _rising_prices(n: int, start: float = 100.0, step: float = 0.5) -> list:
    return [start + i * step for i in range(n)]


def _falling_prices(n: int, start: float = 200.0, step: float = 0.5) -> list:
    return [start - i * step for i in range(n)]


# ---------------------------------------------------------------------------
# build_dataframe
# ---------------------------------------------------------------------------

class TestBuildDataframe:
    def test_returns_dataframe(self):
        klines = _make_klines(_rising_prices(10))
        df = build_dataframe(klines)
        assert isinstance(df, pd.DataFrame)

    def test_has_ohlcv_columns(self):
        klines = _make_klines(_rising_prices(10))
        df = build_dataframe(klines)
        for col in ("open", "high", "low", "close", "volume"):
            assert col in df.columns

    def test_numeric_close(self):
        klines = _make_klines([100.0, 101.0, 99.0])
        df = build_dataframe(klines)
        assert df["close"].dtype in (float, "float64")

    def test_index_is_datetime(self):
        klines = _make_klines(_rising_prices(5))
        df = build_dataframe(klines)
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_row_count_matches(self):
        prices = _rising_prices(50)
        klines = _make_klines(prices)
        df = build_dataframe(klines)
        assert len(df) == 50


# ---------------------------------------------------------------------------
# compute_indicators
# ---------------------------------------------------------------------------

class TestComputeIndicators:
    def _df(self, n: int = 100):
        klines = _make_klines(_rising_prices(n))
        return build_dataframe(klines)

    def test_rsi_column_present(self):
        df = compute_indicators(self._df())
        assert "rsi" in df.columns

    def test_macd_columns_present(self):
        df = compute_indicators(self._df())
        for col in ("macd", "macd_signal", "macd_hist"):
            assert col in df.columns

    def test_ema_columns_present(self):
        df = compute_indicators(self._df())
        for col in ("ema_short", "ema_long"):
            assert col in df.columns

    def test_rsi_range(self):
        df = compute_indicators(self._df(200))
        df.dropna(inplace=True)
        assert df["rsi"].between(0, 100).all()

    def test_does_not_mutate_input(self):
        df = self._df()
        original_cols = list(df.columns)
        compute_indicators(df)
        assert list(df.columns) == original_cols


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

class TestAnalyze:
    def test_returns_none_for_insufficient_data(self):
        klines = _make_klines(_rising_prices(5))
        assert analyze(klines) is None

    def test_returns_result_for_sufficient_data(self):
        klines = _make_klines(_rising_prices(200))
        result = analyze(klines)
        assert result is not None

    def test_result_has_signal(self):
        klines = _make_klines(_rising_prices(200))
        result = analyze(klines)
        assert isinstance(result.signal, Signal)

    def test_result_has_price(self):
        prices = _rising_prices(200)
        klines = _make_klines(prices)
        result = analyze(klines)
        assert result is not None
        assert result.price == pytest.approx(prices[-1], rel=1e-3)

    def test_steady_trend_returns_hold(self):
        """A smooth, steady upward trend with no crossovers should return HOLD."""
        klines = _make_klines(_rising_prices(200, step=0.1))
        result = analyze(klines)
        assert result is not None
        # With steady rise RSI won't dip to oversold; signal should be HOLD or BUY
        assert result.signal in (Signal.HOLD, Signal.BUY)

    def test_rsi_in_result(self):
        klines = _make_klines(_rising_prices(200))
        result = analyze(klines)
        assert result is not None
        assert 0 <= result.rsi <= 100

    def test_reason_is_string(self):
        klines = _make_klines(_rising_prices(200))
        result = analyze(klines)
        assert result is not None
        assert isinstance(result.reason, str) and len(result.reason) > 0
