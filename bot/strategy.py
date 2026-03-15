from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd
import pandas_ta as ta

from bot.config import Config
from bot.logger import get_logger

logger = get_logger(__name__)


class Signal(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class StrategyResult:
    signal: Signal
    rsi: float
    macd: float
    macd_signal: float
    ema_short: float
    ema_long: float
    price: float
    reason: str


def build_dataframe(klines: list) -> pd.DataFrame:
    """Convert raw Binance klines to a DataFrame with OHLCV columns."""
    df = pd.DataFrame(klines, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("open_time", inplace=True)
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI, MACD, and EMA indicators to the DataFrame."""
    df = df.copy()

    # RSI
    df["rsi"] = ta.rsi(df["close"], length=Config.RSI_PERIOD)

    # MACD
    macd = ta.macd(
        df["close"],
        fast=Config.MACD_FAST,
        slow=Config.MACD_SLOW,
        signal=Config.MACD_SIGNAL,
    )
    df["macd"] = macd[f"MACD_{Config.MACD_FAST}_{Config.MACD_SLOW}_{Config.MACD_SIGNAL}"]
    df["macd_signal"] = macd[f"MACDs_{Config.MACD_FAST}_{Config.MACD_SLOW}_{Config.MACD_SIGNAL}"]
    df["macd_hist"] = macd[f"MACDh_{Config.MACD_FAST}_{Config.MACD_SLOW}_{Config.MACD_SIGNAL}"]

    # EMA
    df["ema_short"] = ta.ema(df["close"], length=Config.EMA_SHORT)
    df["ema_long"] = ta.ema(df["close"], length=Config.EMA_LONG)

    return df


def analyze(klines: list) -> Optional[StrategyResult]:
    """
    Analyze klines and return a trading signal.

    Strategy logic:
    - BUY when:
        * RSI < oversold threshold (30) — asset is undervalued
        * MACD line crosses above signal line (bullish momentum)
        * EMA short > EMA long (uptrend confirmation)
    - SELL when:
        * RSI > overbought threshold (70) — asset is overvalued
        * MACD line crosses below signal line (bearish momentum)
        * EMA short < EMA long (downtrend confirmation)
    - HOLD otherwise
    """
    if len(klines) < Config.MACD_SLOW + Config.MACD_SIGNAL + 5:
        logger.warning("Not enough klines data to compute indicators")
        return None

    df = build_dataframe(klines)
    df = compute_indicators(df)
    df.dropna(inplace=True)

    if len(df) < 2:
        logger.warning("Not enough rows after dropping NaN")
        return None

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    rsi = latest["rsi"]
    macd = latest["macd"]
    macd_sig = latest["macd_signal"]
    prev_macd = prev["macd"]
    prev_macd_sig = prev["macd_signal"]
    ema_short = latest["ema_short"]
    ema_long = latest["ema_long"]
    price = latest["close"]

    macd_crossed_up = prev_macd < prev_macd_sig and macd > macd_sig
    macd_crossed_down = prev_macd > prev_macd_sig and macd < macd_sig

    # BUY conditions
    if rsi < Config.RSI_OVERSOLD and macd_crossed_up and ema_short > ema_long:
        signal = Signal.BUY
        reason = f"RSI={rsi:.1f} (oversold), MACD bullish crossover, EMA uptrend"

    # SELL conditions
    elif rsi > Config.RSI_OVERBOUGHT and macd_crossed_down and ema_short < ema_long:
        signal = Signal.SELL
        reason = f"RSI={rsi:.1f} (overbought), MACD bearish crossover, EMA downtrend"

    else:
        signal = Signal.HOLD
        reason = f"RSI={rsi:.1f}, MACD crossover={'up' if macd_crossed_up else 'down' if macd_crossed_down else 'none'}"

    result = StrategyResult(
        signal=signal,
        rsi=rsi,
        macd=macd,
        macd_signal=macd_sig,
        ema_short=ema_short,
        ema_long=ema_long,
        price=price,
        reason=reason,
    )
    logger.info(f"Signal: {signal.value} | {reason} | Price: {price:.4f}")
    return result
