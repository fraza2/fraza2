"""
Backtester pentru strategia RSI+MACD+EMA pe 30 zile de date reale BNBUSDT.
Nu necesita API key — foloseste endpoint-ul public Binance pentru date istorice.
Indicatorii sunt implementati nativ cu pandas/numpy (fara librarii externe).

Rulare:
    python backtest.py
"""

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np
import requests

# ── Config backtesting ──────────────────────────────────────────────────────
SYMBOL = "BNBUSDT"
INTERVAL = "15m"
DAYS = 30
STARTING_BALANCE_USDT = 800.0
TRADE_QUANTITY_PERCENT = 5.0
MAX_OPEN_TRADES = 3
STOP_LOSS_PCT = 2.0
TAKE_PROFIT_PCT = 4.0

RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
EMA_SHORT, EMA_LONG = 9, 21


# ── Indicatori (implementare nativa) ───────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rsi"] = rsi(df["close"], RSI_PERIOD)
    df["macd"], df["macd_sig"], df["macd_hist"] = macd(df["close"], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    df["ema_s"] = ema(df["close"], EMA_SHORT)
    df["ema_l"] = ema(df["close"], EMA_LONG)
    return df


# ── Date istorice (public API sau sintetice) ────────────────────────────────

def _fetch_from_api(symbol: str, interval: str, days: int) -> pd.DataFrame:
    """Descarca de pe Binance public API (fara cheie)."""
    base_url = "https://api.binance.com/api/v3/klines"
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - (days + 5) * 24 * 3600 * 1000

    all_rows = []
    current_start = start_ms
    while current_start < end_ms:
        params = {
            "symbol": symbol, "interval": interval,
            "startTime": current_start, "endTime": end_ms, "limit": 1000,
        }
        resp = requests.get(base_url, params=params, timeout=10)
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            break
        all_rows.extend(rows)
        current_start = rows[-1][0] + 1
        if len(rows) < 1000:
            break

    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.set_index("open_time", inplace=True)
    return df


def _generate_synthetic(symbol: str, interval_min: int, days: int, seed: int = 42) -> pd.DataFrame:
    """
    Genereaza date OHLCV sintetice realiste cu Geometric Brownian Motion.
    Parametri calibrati dupa volatilitatea istorica BNB (~3% pe zi, 15m candle).
    """
    np.random.seed(seed)
    n = days * 24 * (60 // interval_min)

    # BNB ~$650 la inceputul simularii (pret realist feb-mar 2026)
    start_price = 650.0
    dt = interval_min / (365 * 24 * 60)   # fractie de an per candle
    mu = 0.0                               # drift neutru
    sigma = 0.55                           # volatilitate anualizata (~55%, realista BNB)

    # GBM
    z = np.random.standard_normal(n)
    log_returns = (mu - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z
    close_prices = start_price * np.exp(np.cumsum(log_returns))

    # Construim OHLCV din close
    intracandle_vol = sigma * math.sqrt(dt)
    opens = np.roll(close_prices, 1)
    opens[0] = start_price

    high_mult = np.exp(np.abs(np.random.normal(0, intracandle_vol, n)))
    low_mult  = np.exp(-np.abs(np.random.normal(0, intracandle_vol, n)))

    highs  = np.maximum(opens, close_prices) * high_mult
    lows   = np.minimum(opens, close_prices) * low_mult
    volume = np.random.uniform(500, 3000, n)  # BNB volum per candle

    # Index timestamp
    start_ts = pd.Timestamp("2026-02-13", tz="UTC")
    idx = pd.date_range(start=start_ts, periods=n, freq=f"{interval_min}min")

    df = pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": close_prices, "volume": volume,
    }, index=idx)
    df.index.name = "open_time"
    return df


def fetch_klines(symbol: str, interval: str, days: int) -> pd.DataFrame:
    interval_min_map = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240}
    interval_min = interval_min_map.get(interval, 15)

    try:
        print(f"  Conectare la Binance API ({symbol} {interval})...")
        df = _fetch_from_api(symbol, interval, days)
        print(f"  {len(df)} candle-uri descarcate din API.")
        return df
    except Exception as e:
        print(f"  API indisponibil ({e.__class__.__name__}). Folosesc date sintetice realiste.")
        df = _generate_synthetic(symbol, interval_min, days + 5)
        print(f"  {len(df)} candle-uri sintetice generate (GBM, start $650, sigma=55%/an).")
        return df


# ── Modele ──────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    entry_time: object
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    exit_time: Optional[object] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None

    @property
    def closed(self) -> bool:
        return self.exit_time is not None


@dataclass
class BacktestState:
    balance: float = STARTING_BALANCE_USDT
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)

    @property
    def open_trades(self):
        return [t for t in self.trades if not t.closed]

    @property
    def closed_trades(self):
        return [t for t in self.trades if t.closed]


# ── Logica tranzactionare ───────────────────────────────────────────────────

def get_signal(row, prev_row) -> str:
    """
    Strategie MACD crossover cu confirmare EMA trend si filtru RSI.

    BUY  : MACD line traverseaza in sus signal line
           + EMA9 > EMA21 (uptrend confirmat)
           + RSI < 70 (nu supracumparat — evitam intrare la varf)

    SELL : MACD line traverseaza in jos signal line
           + EMA9 < EMA21 (downtrend confirmat)
           + RSI > 30 (nu supravandut — evitam vanzare la fund)
    """
    for val in (row["rsi"], row["macd"], row["macd_sig"], row["ema_s"], row["ema_l"]):
        if pd.isna(val):
            return "HOLD"

    macd_crossed_up   = prev_row["macd"] < prev_row["macd_sig"] and row["macd"] > row["macd_sig"]
    macd_crossed_down = prev_row["macd"] > prev_row["macd_sig"] and row["macd"] < row["macd_sig"]

    if macd_crossed_up and row["ema_s"] > row["ema_l"] and row["rsi"] < RSI_OVERBOUGHT:
        return "BUY"
    if macd_crossed_down and row["ema_s"] < row["ema_l"] and row["rsi"] > RSI_OVERSOLD:
        return "SELL"
    return "HOLD"


def manage_trades(state: BacktestState, candle, ts):
    for trade in list(state.open_trades):
        if candle["low"] <= trade.stop_loss:
            trade.exit_time = ts
            trade.exit_price = trade.stop_loss
            trade.exit_reason = "STOP_LOSS"
            trade.pnl = (trade.stop_loss - trade.entry_price) * trade.quantity
            state.balance += trade.quantity * trade.stop_loss
        elif candle["high"] >= trade.take_profit:
            trade.exit_time = ts
            trade.exit_price = trade.take_profit
            trade.exit_reason = "TAKE_PROFIT"
            trade.pnl = (trade.take_profit - trade.entry_price) * trade.quantity
            state.balance += trade.quantity * trade.take_profit


def try_open_trade(state: BacktestState, candle, ts):
    if len(state.open_trades) >= MAX_OPEN_TRADES:
        return
    price = candle["close"]
    trade_value = state.balance * (TRADE_QUANTITY_PERCENT / 100)
    if trade_value < 1.0:
        return
    quantity = math.floor((trade_value / price) * 1000) / 1000
    if quantity <= 0:
        return
    state.balance -= quantity * price
    state.trades.append(Trade(
        entry_time=ts,
        entry_price=price,
        quantity=quantity,
        stop_loss=round(price * (1 - STOP_LOSS_PCT / 100), 6),
        take_profit=round(price * (1 + TAKE_PROFIT_PCT / 100), 6),
    ))


def close_all_open(state: BacktestState, last_price: float, last_ts):
    for trade in state.open_trades:
        trade.exit_time = last_ts
        trade.exit_price = last_price
        trade.exit_reason = "END_OF_SIM"
        trade.pnl = (last_price - trade.entry_price) * trade.quantity
        state.balance += trade.quantity * last_price


# ── Backtest principal ──────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame) -> BacktestState:
    state = BacktestState(balance=STARTING_BALANCE_USDT)
    rows = df.reset_index()

    for i in range(1, len(rows)):
        row = rows.iloc[i]
        prev = rows.iloc[i - 1]
        ts = row["open_time"]

        manage_trades(state, row, ts)

        signal = get_signal(row, prev)
        if signal == "BUY":
            try_open_trade(state, row, ts)

        open_value = sum(t.quantity * row["close"] for t in state.open_trades)
        state.equity_curve.append({"time": ts, "equity": state.balance + open_value})

    close_all_open(state, rows.iloc[-1]["close"], rows.iloc[-1]["open_time"])
    return state


# ── Raport ──────────────────────────────────────────────────────────────────

def print_report(state: BacktestState, df: pd.DataFrame):
    closed = state.closed_trades
    wins = [t for t in closed if t.pnl and t.pnl > 0]
    losses = [t for t in closed if t.pnl and t.pnl <= 0]
    tp_hits = [t for t in closed if t.exit_reason == "TAKE_PROFIT"]
    sl_hits = [t for t in closed if t.exit_reason == "STOP_LOSS"]
    eod = [t for t in closed if t.exit_reason == "END_OF_SIM"]

    total_pnl = sum(t.pnl for t in closed if t.pnl)
    win_rate = len(wins) / len(closed) * 100 if closed else 0
    avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0
    gross_win = abs(sum(t.pnl for t in wins))
    gross_loss = abs(sum(t.pnl for t in losses))
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")

    eq_series = pd.Series([e["equity"] for e in state.equity_curve])
    rolling_max = eq_series.cummax()
    drawdown_pct = (eq_series - rolling_max) / rolling_max * 100
    max_drawdown = drawdown_pct.min()

    final_balance = state.balance
    roi = (final_balance - STARTING_BALANCE_USDT) / STARTING_BALANCE_USDT * 100

    first_price = df["close"].iloc[0]
    last_price = df["close"].iloc[-1]
    bh_roi = (last_price - first_price) / first_price * 100
    bh_value = STARTING_BALANCE_USDT * (1 + bh_roi / 100)

    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  RAPORT SIMULARE  {SYMBOL} | {INTERVAL} | {DAYS} zile")
    print(f"  {str(df.index[0])[:10]}  →  {str(df.index[-1])[:10]}")
    print(sep)
    print(f"  Capital initial       : ${STARTING_BALANCE_USDT:>10,.2f}")
    print(f"  Capital final (bot)   : ${final_balance:>10,.2f}   ({roi:+.2f}%)")
    print(f"  Capital final (B&H)   : ${bh_value:>10,.2f}   ({bh_roi:+.2f}%)")
    print(f"  Profit net            : ${total_pnl:>+10,.4f}")
    print(sep)
    print(f"  Tranzactii totale     : {len(closed)}")
    print(f"  Win rate              : {win_rate:.1f}%  ({len(wins)} castig / {len(losses)} pierdere)")
    print(f"  Take Profit hits      : {len(tp_hits)}")
    print(f"  Stop Loss hits        : {len(sl_hits)}")
    print(f"  Inchise la final sim  : {len(eod)}")
    print(f"  Profit mediu / trade  : ${avg_win:>+.4f} win  |  ${avg_loss:>+.4f} loss")
    print(f"  Profit Factor         : {profit_factor:.2f}")
    print(f"  Max Drawdown          : {max_drawdown:.2f}%")
    print(sep)

    if closed:
        header = f"  {'Data intrare':<20} {'Intrare':>9} {'Iesire':>9} {'Qty':>7} {'PnL':>10}  Motiv"
        print(f"\n{header}")
        print(f"  {'-'*20} {'-'*9} {'-'*9} {'-'*7} {'-'*10}  {'-'*12}")
        show = closed[-25:] if len(closed) > 25 else closed
        for t in show:
            pnl_str = f"{t.pnl:>+.4f}" if t.pnl is not None else "N/A"
            print(
                f"  {str(t.entry_time)[:19]:<20}"
                f" {t.entry_price:>9.3f}"
                f" {t.exit_price:>9.3f}"
                f" {t.quantity:>7.4f}"
                f" {pnl_str:>10}"
                f"  {t.exit_reason}"
            )
        if len(closed) > 25:
            print(f"  ... (primele {len(closed) - 25} tranzactii omise)")

    print(f"\n{'=' * 62}\n")


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 62)
    print("  Binance Backtester — RSI + MACD + EMA")
    print("=" * 62)
    print(f"  Simbol     : {SYMBOL}")
    print(f"  Interval   : {INTERVAL}")
    print(f"  Perioada   : {DAYS} zile")
    print(f"  Budget     : ${STARTING_BALANCE_USDT:.2f} USDT")
    print(f"  SL / TP    : {STOP_LOSS_PCT}% / {TAKE_PROFIT_PCT}%")
    print(f"  Marime     : {TRADE_QUANTITY_PERCENT}% din balance / trade")
    print(f"  Max trades : {MAX_OPEN_TRADES} simultan\n")

    df_raw = fetch_klines(SYMBOL, INTERVAL, DAYS)
    df_ind = add_indicators(df_raw)
    df_ind.dropna(inplace=True)

    # Pastreaza doar ultimele DAYS zile
    cutoff = df_ind.index[-1] - pd.Timedelta(days=DAYS)
    df_sim = df_ind[df_ind.index >= cutoff].copy()
    print(f"  Candle-uri simulate: {len(df_sim)}\n")

    state = run_backtest(df_sim)
    print_report(state, df_sim)
