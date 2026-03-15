from binance.client import Client
from binance.exceptions import BinanceAPIException
from bot.config import Config
from bot.logger import get_logger

logger = get_logger(__name__)

TESTNET_BASE_URL = "https://testnet.binance.vision/api"


def create_client() -> Client:
    client = Client(Config.API_KEY, Config.API_SECRET, testnet=Config.TESTNET)
    if Config.TESTNET:
        client.API_URL = TESTNET_BASE_URL
        logger.info("Connected to Binance TESTNET")
    else:
        logger.info("Connected to Binance LIVE")
    return client


def get_balance(client: Client, asset: str) -> float:
    try:
        balance = client.get_asset_balance(asset=asset)
        return float(balance["free"]) if balance else 0.0
    except BinanceAPIException as e:
        logger.error(f"Failed to get balance for {asset}: {e}")
        return 0.0


def get_symbol_price(client: Client, symbol: str) -> float:
    try:
        ticker = client.get_symbol_ticker(symbol=symbol)
        return float(ticker["price"])
    except BinanceAPIException as e:
        logger.error(f"Failed to get price for {symbol}: {e}")
        return 0.0


def get_klines(client: Client, symbol: str, interval: str, limit: int = 200):
    try:
        return client.get_klines(symbol=symbol, interval=interval, limit=limit)
    except BinanceAPIException as e:
        logger.error(f"Failed to get klines for {symbol}: {e}")
        return []
