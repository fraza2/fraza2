from dataclasses import dataclass, field
from typing import Optional
import math

from binance.client import Client
from binance.exceptions import BinanceAPIException

from bot.client import get_balance, get_symbol_price
from bot.config import Config
from bot.logger import get_logger
from bot.strategy import Signal, StrategyResult

logger = get_logger(__name__)


@dataclass
class Trade:
    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: float
    entry_price: float
    stop_loss: float
    take_profit: float
    order_id: Optional[int] = None
    closed: bool = False
    exit_price: Optional[float] = None
    pnl: Optional[float] = None


@dataclass
class Portfolio:
    trades: list = field(default_factory=list)

    @property
    def open_trades(self) -> list:
        return [t for t in self.trades if not t.closed]

    @property
    def closed_trades(self) -> list:
        return [t for t in self.trades if t.closed]

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.closed_trades if t.pnl is not None)

    def summary(self) -> str:
        return (
            f"Open trades: {len(self.open_trades)} | "
            f"Closed trades: {len(self.closed_trades)} | "
            f"Total PnL: {self.total_pnl:.4f} USDT"
        )


def _step_size_decimals(step_size: str) -> int:
    """Return the number of decimal places from a step size string like '0.001000'."""
    step_size = step_size.rstrip("0")
    if "." in step_size:
        return len(step_size.split(".")[1])
    return 0


def get_lot_size(client: Client, symbol: str):
    """Fetch LOT_SIZE filter for the symbol to get min/max/step constraints."""
    try:
        info = client.get_symbol_info(symbol)
        for f in info["filters"]:
            if f["filterType"] == "LOT_SIZE":
                return float(f["minQty"]), float(f["maxQty"]), f["stepSize"]
    except (BinanceAPIException, KeyError, TypeError):
        pass
    return 0.001, 9999999.0, "0.001"


def calculate_quantity(client: Client, symbol: str, price: float) -> float:
    """Calculate trade quantity based on % of USDT balance."""
    base_asset = "USDT"
    balance = get_balance(client, base_asset)
    trade_value = balance * (Config.TRADE_QUANTITY_PERCENT / 100)

    if trade_value <= 0:
        logger.warning("Insufficient USDT balance")
        return 0.0

    quantity = trade_value / price
    min_qty, max_qty, step_size = get_lot_size(client, symbol)
    decimals = _step_size_decimals(step_size)
    quantity = math.floor(quantity / float(step_size)) * float(step_size)
    quantity = round(quantity, decimals)
    quantity = max(min_qty, min(quantity, max_qty))

    logger.debug(
        f"Calculated quantity: {quantity} {symbol} "
        f"(balance={balance:.2f} USDT, price={price:.4f})"
    )
    return quantity


def place_order(client: Client, symbol: str, side: str, quantity: float) -> Optional[dict]:
    """Place a market order on Binance (testnet or live)."""
    try:
        order = client.order_market(
            symbol=symbol,
            side=side,
            quantity=quantity,
        )
        logger.info(f"Order placed: {side} {quantity} {symbol} | id={order['orderId']}")
        return order
    except BinanceAPIException as e:
        logger.error(f"Order failed ({side} {quantity} {symbol}): {e}")
        return None


def open_trade(client: Client, portfolio: Portfolio, result: StrategyResult) -> Optional[Trade]:
    """Open a new BUY trade if conditions are met."""
    if len(portfolio.open_trades) >= Config.MAX_OPEN_TRADES:
        logger.info("Max open trades reached, skipping")
        return None

    price = result.price
    quantity = calculate_quantity(client, Config.SYMBOL, price)
    if quantity <= 0:
        return None

    order = place_order(client, Config.SYMBOL, "BUY", quantity)
    if not order:
        return None

    stop_loss = price * (1 - Config.STOP_LOSS_PERCENT / 100)
    take_profit = price * (1 + Config.TAKE_PROFIT_PERCENT / 100)

    trade = Trade(
        symbol=Config.SYMBOL,
        side="BUY",
        quantity=quantity,
        entry_price=price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        order_id=order["orderId"],
    )
    portfolio.trades.append(trade)
    logger.info(
        f"Trade opened | entry={price:.4f} SL={stop_loss:.4f} TP={take_profit:.4f}"
    )
    return trade


def close_trade(client: Client, trade: Trade, current_price: float, reason: str) -> bool:
    """Close an existing trade with a SELL market order."""
    order = place_order(client, trade.symbol, "SELL", trade.quantity)
    if not order:
        return False

    trade.closed = True
    trade.exit_price = current_price
    trade.pnl = (current_price - trade.entry_price) * trade.quantity
    logger.info(
        f"Trade closed ({reason}) | entry={trade.entry_price:.4f} "
        f"exit={current_price:.4f} PnL={trade.pnl:.4f} USDT"
    )
    return True


def manage_open_trades(client: Client, portfolio: Portfolio, current_price: float):
    """Check stop loss / take profit for all open trades."""
    for trade in portfolio.open_trades:
        if current_price <= trade.stop_loss:
            close_trade(client, trade, current_price, "STOP_LOSS")
        elif current_price >= trade.take_profit:
            close_trade(client, trade, current_price, "TAKE_PROFIT")
