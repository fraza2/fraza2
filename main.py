"""Entry point for the Binance trading bot."""
import time
import signal
import sys

from bot.client import create_client, get_klines, get_symbol_price
from bot.config import Config
from bot.logger import get_logger
from bot.strategy import Signal, analyze
from bot.trader import Portfolio, manage_open_trades, open_trade

logger = get_logger("main")

_running = True


def _handle_signal(signum, frame):
    global _running
    logger.info("Shutdown signal received, stopping bot...")
    _running = False


def run():
    logger.info("=" * 60)
    logger.info(f"Starting Binance Trading Bot")
    logger.info(f"  Symbol   : {Config.SYMBOL}")
    logger.info(f"  Interval : {Config.INTERVAL}")
    logger.info(f"  Testnet  : {Config.TESTNET}")
    logger.info(f"  Strategy : RSI({Config.RSI_PERIOD}) + MACD({Config.MACD_FAST},{Config.MACD_SLOW},{Config.MACD_SIGNAL}) + EMA({Config.EMA_SHORT},{Config.EMA_LONG})")
    logger.info("=" * 60)

    client = create_client()
    portfolio = Portfolio()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    while _running:
        try:
            klines = get_klines(client, Config.SYMBOL, Config.INTERVAL, limit=200)
            if not klines:
                logger.warning("No klines received, skipping iteration")
                time.sleep(Config.LOOP_INTERVAL)
                continue

            result = analyze(klines)
            if result is None:
                time.sleep(Config.LOOP_INTERVAL)
                continue

            current_price = get_symbol_price(client, Config.SYMBOL)
            manage_open_trades(client, portfolio, current_price)

            if result.signal == Signal.BUY:
                open_trade(client, portfolio, result)
            elif result.signal == Signal.SELL:
                logger.info("SELL signal — no short selling in this strategy, holding")

            logger.info(portfolio.summary())

        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}", exc_info=True)

        time.sleep(Config.LOOP_INTERVAL)

    logger.info("Bot stopped. Final state:")
    logger.info(portfolio.summary())


if __name__ == "__main__":
    run()
