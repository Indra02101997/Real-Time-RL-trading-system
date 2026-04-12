"""
Technical Indicators Module.
Computes RSI, MACD, Bollinger Bands, Stochastic, SMA/EMA crossovers,
ATR, VWAP, OBV, and generates trading signals.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Computes technical indicators and generates trading signals."""

    def __init__(self, config):
        self.config = config.strategy

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all technical indicators on OHLCV DataFrame."""
        df = df.copy()
        df = self.compute_rsi(df)
        df = self.compute_macd(df)
        df = self.compute_bollinger_bands(df)
        df = self.compute_stochastic(df)
        df = self.compute_sma(df)
        df = self.compute_ema(df)
        df = self.compute_atr(df)
        if "Volume" in df.columns:
            df = self.compute_vwap(df)
            df = self.compute_obv(df)
        df = self.compute_momentum(df)
        df = self.compute_cci(df)
        return df

    def compute_rsi(self, df: pd.DataFrame, period: Optional[int] = None) -> pd.DataFrame:
        """Relative Strength Index."""
        period = period or self.config.rsi_period
        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        df["rsi"] = 100 - (100 / (1 + rs))
        df["rsi_signal"] = 0
        df.loc[df["rsi"] < self.config.rsi_oversold, "rsi_signal"] = 1  # Buy
        df.loc[df["rsi"] > self.config.rsi_overbought, "rsi_signal"] = -1  # Sell
        return df

    def compute_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """MACD (Moving Average Convergence Divergence)."""
        ema_fast = df["Close"].ewm(span=self.config.macd_fast, adjust=False).mean()
        ema_slow = df["Close"].ewm(span=self.config.macd_slow, adjust=False).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_signal_line"] = df["macd"].ewm(span=self.config.macd_signal, adjust=False).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal_line"]

        df["macd_signal"] = 0
        # Bullish crossover
        df.loc[
            (df["macd"] > df["macd_signal_line"]) &
            (df["macd"].shift(1) <= df["macd_signal_line"].shift(1)),
            "macd_signal"
        ] = 1
        # Bearish crossover
        df.loc[
            (df["macd"] < df["macd_signal_line"]) &
            (df["macd"].shift(1) >= df["macd_signal_line"].shift(1)),
            "macd_signal"
        ] = -1
        return df

    def compute_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bollinger Bands."""
        period = self.config.bollinger_period
        std_dev = self.config.bollinger_std
        df["bb_middle"] = df["Close"].rolling(window=period).mean()
        rolling_std = df["Close"].rolling(window=period).std()
        df["bb_upper"] = df["bb_middle"] + (std_dev * rolling_std)
        df["bb_lower"] = df["bb_middle"] - (std_dev * rolling_std)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        df["bb_pct"] = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, 1e-10)

        df["bb_signal"] = 0
        df.loc[df["Close"] < df["bb_lower"], "bb_signal"] = 1  # Buy (oversold)
        df.loc[df["Close"] > df["bb_upper"], "bb_signal"] = -1  # Sell (overbought)
        return df

    def compute_stochastic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stochastic Oscillator."""
        k_period = self.config.stochastic_k
        d_period = self.config.stochastic_d
        low_min = df["Low"].rolling(window=k_period).min()
        high_max = df["High"].rolling(window=k_period).max()
        df["stoch_k"] = 100 * (df["Close"] - low_min) / (high_max - low_min).replace(0, 1e-10)
        df["stoch_d"] = df["stoch_k"].rolling(window=d_period).mean()

        df["stoch_signal"] = 0
        df.loc[
            (df["stoch_k"] < 20) & (df["stoch_k"] > df["stoch_d"]),
            "stoch_signal"
        ] = 1  # Buy
        df.loc[
            (df["stoch_k"] > 80) & (df["stoch_k"] < df["stoch_d"]),
            "stoch_signal"
        ] = -1  # Sell
        return df

    def compute_sma(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simple Moving Average crossover."""
        df["sma_short"] = df["Close"].rolling(window=self.config.sma_short).mean()
        df["sma_long"] = df["Close"].rolling(window=self.config.sma_long).mean()

        df["sma_signal"] = 0
        df.loc[
            (df["sma_short"] > df["sma_long"]) &
            (df["sma_short"].shift(1) <= df["sma_long"].shift(1)),
            "sma_signal"
        ] = 1  # Golden cross
        df.loc[
            (df["sma_short"] < df["sma_long"]) &
            (df["sma_short"].shift(1) >= df["sma_long"].shift(1)),
            "sma_signal"
        ] = -1  # Death cross
        return df

    def compute_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Exponential Moving Average crossover."""
        df["ema_short"] = df["Close"].ewm(span=self.config.ema_short, adjust=False).mean()
        df["ema_long"] = df["Close"].ewm(span=self.config.ema_long, adjust=False).mean()

        df["ema_signal"] = 0
        df.loc[
            (df["ema_short"] > df["ema_long"]) &
            (df["ema_short"].shift(1) <= df["ema_long"].shift(1)),
            "ema_signal"
        ] = 1
        df.loc[
            (df["ema_short"] < df["ema_long"]) &
            (df["ema_short"].shift(1) >= df["ema_long"].shift(1)),
            "ema_signal"
        ] = -1
        return df

    def compute_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Average True Range."""
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift(1)).abs()
        low_close = (df["Low"] - df["Close"].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = true_range.rolling(window=self.config.atr_period).mean()
        df["atr_pct"] = df["atr"] / df["Close"]

        df["atr_signal"] = 0
        # High volatility = potential breakout
        atr_sma = df["atr"].rolling(window=self.config.atr_period * 2).mean()
        df.loc[df["atr"] > 1.5 * atr_sma, "atr_signal"] = 1
        return df

    def compute_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume Weighted Average Price."""
        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
        df["vwap"] = (typical_price * df["Volume"]).rolling(
            window=self.config.vwap_period
        ).sum() / df["Volume"].rolling(window=self.config.vwap_period).sum().replace(0, 1e-10)

        df["vwap_signal"] = 0
        df.loc[df["Close"] > df["vwap"], "vwap_signal"] = 1  # Bullish
        df.loc[df["Close"] < df["vwap"], "vwap_signal"] = -1  # Bearish
        return df

    def compute_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """On-Balance Volume."""
        obv = [0]
        for i in range(1, len(df)):
            if df["Close"].iloc[i] > df["Close"].iloc[i - 1]:
                obv.append(obv[-1] + df["Volume"].iloc[i])
            elif df["Close"].iloc[i] < df["Close"].iloc[i - 1]:
                obv.append(obv[-1] - df["Volume"].iloc[i])
            else:
                obv.append(obv[-1])

        df["obv"] = obv
        df["obv_sma"] = df["obv"].rolling(window=self.config.obv_period).mean()

        df["obv_signal"] = 0
        df.loc[df["obv"] > df["obv_sma"], "obv_signal"] = 1
        df.loc[df["obv"] < df["obv_sma"], "obv_signal"] = -1
        return df

    def compute_cci(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Commodity Channel Index — detects mean-reversion extremes."""
        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mean_dev = typical_price.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        )
        df["cci"] = (typical_price - sma_tp) / (0.015 * mean_dev.replace(0, 1e-10))

        df["cci_signal"] = 0
        df.loc[df["cci"] < -100, "cci_signal"] = 1   # Oversold → buy
        df.loc[df["cci"] > 100, "cci_signal"] = -1   # Overbought → sell
        return df

    def compute_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Price momentum."""
        df["momentum_10"] = df["Close"].pct_change(periods=10)
        df["momentum_20"] = df["Close"].pct_change(periods=20)

        df["momentum_signal"] = 0
        df.loc[
            (df["momentum_10"] > 0) & (df["momentum_20"] > 0),
            "momentum_signal"
        ] = 1
        df.loc[
            (df["momentum_10"] < 0) & (df["momentum_20"] < 0),
            "momentum_signal"
        ] = -1
        return df

    def get_signal_columns(self) -> List[str]:
        """Get list of all signal column names."""
        return [
            "rsi_signal", "macd_signal", "bb_signal", "stoch_signal",
            "sma_signal", "ema_signal", "atr_signal", "vwap_signal",
            "obv_signal", "momentum_signal", "cci_signal",
        ]

    def get_feature_columns(self) -> List[str]:
        """Get list of all feature column names for ML input."""
        return [
            "rsi", "macd", "macd_histogram", "bb_pct", "bb_width",
            "stoch_k", "stoch_d", "sma_short", "sma_long",
            "ema_short", "ema_long", "atr", "atr_pct",
            "momentum_10", "momentum_20", "cci",
        ]

    def get_combined_signal(self, row: pd.Series, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Get weighted combined signal from all indicators.
        Returns value in [-1, 1]. Positive = buy, negative = sell.
        """
        signal_cols = self.get_signal_columns()
        if weights is None:
            weights = {col: 1.0 / len(signal_cols) for col in signal_cols}

        total_signal = 0.0
        total_weight = 0.0
        for col in signal_cols:
            if col in row.index and not np.isnan(row[col]):
                w = weights.get(col, 1.0 / len(signal_cols))
                total_signal += row[col] * w
                total_weight += abs(w)

        if total_weight == 0:
            return 0.0
        return total_signal / total_weight
