"""
NSE Data Collector - Fetches historical and live NSE market data.
Uses yfinance for 30-year historical data and OpenAlgo for live data.
"""
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class NSEDataCollector:
    """Collects historical and live NSE data for all stocks."""

    # Major NSE stocks (NIFTY 500 representative subset + full list loaded dynamically)
    NIFTY50_SYMBOLS = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR",
        "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "LT", "AXISBANK",
        "ASIANPAINT", "MARUTI", "TITAN", "SUNPHARMA", "BAJFINANCE",
        "WIPRO", "HCLTECH", "NTPC", "ONGC", "POWERGRID", "TATAMOTORS",
        "JSWSTEEL", "TATASTEEL", "ADANIENT", "ADANIPORTS", "ULTRACEMCO",
        "NESTLEIND", "TECHM", "M&M", "INDUSINDBK", "BAJAJFINSV",
        "COALINDIA", "GRASIM", "CIPLA", "DRREDDY", "BRITANNIA",
        "APOLLOHOSP", "DIVISLAB", "EICHERMOT", "HEROMOTOCO", "TATACONSUM",
        "SBILIFE", "HDFCLIFE", "BPCL", "HINDALCO", "LTIM", "BAJAJ-AUTO",
    ]

    def __init__(self, config):
        self.config = config
        self.cache_dir = config.data.cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self._symbols_cache: Optional[List[str]] = None
        self._data_cache: Dict[str, pd.DataFrame] = {}

    def get_nse_symbols(self) -> List[str]:
        """Get list of all NSE traded symbols."""
        if self._symbols_cache:
            return self._symbols_cache

        try:
            # Try fetching from NSE website
            url = self.config.data.nse_symbols_url
            df = pd.read_csv(url)
            symbols = df["SYMBOL"].tolist()
            self._symbols_cache = symbols
            logger.info(f"Loaded {len(symbols)} NSE symbols from exchange")
            return symbols
        except Exception as e:
            logger.warning(f"Failed to fetch NSE symbols list: {e}. Using NIFTY50 fallback.")
            self._symbols_cache = self.NIFTY50_SYMBOLS.copy()
            return self._symbols_cache

    def fetch_historical_data(
        self,
        symbol: str,
        years: int = 30,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch historical data for a symbol using yfinance."""
        cache_file = os.path.join(self.cache_dir, f"{symbol}_{years}y_{interval}.parquet")

        # Check cache (refresh if older than 1 day)
        if os.path.exists(cache_file):
            mtime = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - mtime < timedelta(days=1):
                df = pd.read_parquet(cache_file)
                self._data_cache[symbol] = df
                return df

        try:
            # yfinance uses .NS suffix for NSE stocks
            ticker = f"{symbol}.NS"
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years * 365)

            df = yf.download(
                ticker,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval=interval,
                progress=False,
                auto_adjust=True,
            )

            if df.empty:
                logger.warning(f"No data for {symbol}")
                return pd.DataFrame()

            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.index.name = "Date"
            df = df.reset_index()

            # Save to cache
            df.to_parquet(cache_file, index=False)
            self._data_cache[symbol] = df
            logger.info(f"Fetched {len(df)} rows for {symbol} ({years}yr history)")
            return df

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_all_historical(self, symbols: Optional[List[str]] = None, years: int = 30) -> Dict[str, pd.DataFrame]:
        """Fetch historical data for all symbols."""
        if symbols is None:
            symbols = self.get_nse_symbols()

        all_data = {}
        total = len(symbols)
        for i, symbol in enumerate(symbols):
            logger.info(f"[{i + 1}/{total}] Fetching {symbol}...")
            df = self.fetch_historical_data(symbol, years=years)
            if not df.empty:
                all_data[symbol] = df

        logger.info(f"Fetched historical data for {len(all_data)}/{total} symbols")
        return all_data

    def fetch_live_data_openalgo(self, client, symbol: str, interval: str = "5m", days: int = 5) -> pd.DataFrame:
        """Fetch recent data via OpenAlgo API."""
        try:
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

            result = client.history(
                symbol=symbol,
                exchange="NSE",
                interval=interval,
                start_date=start_date,
                end_date=end_date,
            )

            if isinstance(result, pd.DataFrame) and not result.empty:
                return result
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"OpenAlgo live data error for {symbol}: {e}")
            return pd.DataFrame()

    def get_live_quote_openalgo(self, client, symbol: str) -> dict:
        """Get real-time quote via OpenAlgo."""
        try:
            return client.quotes(symbol=symbol, exchange="NSE")
        except Exception as e:
            logger.error(f"Quote error for {symbol}: {e}")
            return {}

    def prepare_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare raw OHLCV data for model training."""
        if df.empty:
            return df

        df = df.copy()

        # Calculate returns
        df["returns"] = df["Close"].pct_change()
        df["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))

        # Volatility (rolling 20-day)
        df["volatility_20d"] = df["returns"].rolling(window=20).std()

        # Volume change
        if "Volume" in df.columns:
            df["volume_change"] = df["Volume"].pct_change()
            df["volume_sma_20"] = df["Volume"].rolling(window=20).mean()
            df["volume_ratio"] = df["Volume"] / df["volume_sma_20"]

        # Price features
        df["high_low_range"] = (df["High"] - df["Low"]) / df["Close"]
        df["open_close_range"] = (df["Close"] - df["Open"]) / df["Open"]

        # Drop NaN rows
        df.dropna(inplace=True)
        return df
