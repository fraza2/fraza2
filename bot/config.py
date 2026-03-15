import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    API_KEY: str = os.getenv("BINANCE_API_KEY", "")
    API_SECRET: str = os.getenv("BINANCE_API_SECRET", "")
    TESTNET: bool = os.getenv("TESTNET", "true").lower() == "true"

    # Trading pair and timeframe
    SYMBOL: str = os.getenv("SYMBOL", "BTCUSDT")
    INTERVAL: str = os.getenv("INTERVAL", "15m")  # 1m, 5m, 15m, 1h, 4h

    # Risk management
    TRADE_QUANTITY_PERCENT: float = float(os.getenv("TRADE_QUANTITY_PERCENT", "5"))  # % of balance per trade
    MAX_OPEN_TRADES: int = int(os.getenv("MAX_OPEN_TRADES", "3"))
    STOP_LOSS_PERCENT: float = float(os.getenv("STOP_LOSS_PERCENT", "2.0"))   # 2% stop loss
    TAKE_PROFIT_PERCENT: float = float(os.getenv("TAKE_PROFIT_PERCENT", "4.0"))  # 4% take profit

    # RSI settings
    RSI_PERIOD: int = int(os.getenv("RSI_PERIOD", "14"))
    RSI_OVERSOLD: float = float(os.getenv("RSI_OVERSOLD", "30"))
    RSI_OVERBOUGHT: float = float(os.getenv("RSI_OVERBOUGHT", "70"))

    # MACD settings
    MACD_FAST: int = int(os.getenv("MACD_FAST", "12"))
    MACD_SLOW: int = int(os.getenv("MACD_SLOW", "26"))
    MACD_SIGNAL: int = int(os.getenv("MACD_SIGNAL", "9"))

    # EMA settings
    EMA_SHORT: int = int(os.getenv("EMA_SHORT", "9"))
    EMA_LONG: int = int(os.getenv("EMA_LONG", "21"))

    # Bot loop interval in seconds
    LOOP_INTERVAL: int = int(os.getenv("LOOP_INTERVAL", "60"))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/bot.log")
