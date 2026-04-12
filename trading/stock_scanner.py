"""
Stock Profitability Scanner.
Scans all NSE stocks every 5 minutes during trading hours,
ranks them by profitability signals, and decides BUY / SHORT / HOLD.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import AppConfig
from data.nse_data_collector import NSEDataCollector
from indicators.technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


class ScanAction(Enum):
    BUY = "BUY"
    SHORT = "SHORT"
    HOLD = "HOLD"


@dataclass
class ScanResult:
    symbol: str
    action: ScanAction
    score: float  # Composite profitability score in [-1, 1]
    price: float
    change_pct: float  # Intraday change %
    volume_ratio: float  # Volume vs 20-day avg
    rsi: float
    macd_hist: float
    bb_pct: float
    sentiment: float
    sharpe_5m: float  # 5-minute rolling Sharpe
    momentum: float
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def confidence(self) -> str:
        s = abs(self.score)
        if s > 0.7:
            return "HIGH"
        if s > 0.4:
            return "MEDIUM"
        return "LOW"


class StockScanner:
    """
    Scans all NSE stocks on a 5-minute cadence:
      1. Fetches latest 5m candle data
      2. Computes technical indicators + rolling Sharpe ratio
      3. Ranks stocks by composite profitability score
      4. Decides BUY / SHORT / HOLD for each
    """

    # Minimum thresholds for action
    BUY_THRESHOLD = 0.25
    SHORT_THRESHOLD = -0.25

    def __init__(self, config: AppConfig):
        self.config = config
        self.data_collector = NSEDataCollector(config)
        self.indicators = TechnicalIndicators(config)
        self._prev_scan: Dict[str, ScanResult] = {}
        self._scan_history: List[Dict[str, ScanResult]] = []

    def scan_all(
        self,
        symbols: List[str],
        trader=None,
        sentiment_cache: Optional[Dict[str, float]] = None,
        strategy_weights: Optional[Dict[str, float]] = None,
    ) -> List[ScanResult]:
        """
        Run a full scan across all symbols.
        Returns results sorted by absolute score (most actionable first).
        """
        results: List[ScanResult] = []
        sentiment_cache = sentiment_cache or {}

        for symbol in symbols:
            try:
                result = self._scan_symbol(
                    symbol, trader, sentiment_cache, strategy_weights
                )
                if result is not None:
                    results.append(result)
            except Exception as e:
                logger.debug(f"Scan skip {symbol}: {e}")

        # Sort: strongest signals first
        results.sort(key=lambda r: abs(r.score), reverse=True)

        # Cache for next cycle
        self._prev_scan = {r.symbol: r for r in results}
        self._scan_history.append(self._prev_scan.copy())

        self._log_scan_summary(results)
        return results

    def _scan_symbol(
        self,
        symbol: str,
        trader,
        sentiment_cache: Dict[str, float],
        strategy_weights: Optional[Dict[str, float]],
    ) -> Optional[ScanResult]:
        """Scan a single symbol and produce a ScanResult."""
        # --- Get 5-minute intraday data ---
        df = None
        if trader is not None:
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
            df_result = trader.get_historical(
                symbol, interval="5m", start_date=start_date, end_date=end_date
            )
            if isinstance(df_result, pd.DataFrame) and not df_result.empty:
                df = df_result

        if df is None or df.empty:
            df = self.data_collector.fetch_historical_data(symbol, years=1, interval="1d")

        if df is None or df.empty or len(df) < 50:
            return None

        # --- Compute indicators ---
        df = self.indicators.compute_all(df)
        df.dropna(inplace=True)
        if df.empty:
            return None

        row = df.iloc[-1]
        price = float(row.get("Close", 0))
        if price <= 0:
            return None

        # --- Intraday change ---
        open_price = float(row.get("Open", price))
        change_pct = (price - open_price) / open_price if open_price > 0 else 0.0

        # --- Volume ratio ---
        volume = float(row.get("Volume", 0))
        vol_sma = float(df["Volume"].rolling(20).mean().iloc[-1]) if "Volume" in df.columns else 1.0
        volume_ratio = volume / max(vol_sma, 1.0)

        # --- 5-minute rolling Sharpe ---
        returns = df["Close"].pct_change().dropna()
        recent_returns = returns.tail(12)  # Last ~1 hour of 5m bars
        sharpe_5m = 0.0
        if len(recent_returns) > 1 and recent_returns.std() > 0:
            sharpe_5m = float(
                np.sqrt(252 * 78) * recent_returns.mean() / recent_returns.std()
            )  # 78 five-min bars per trading day, annualised

        # --- Indicator values ---
        rsi = float(row.get("rsi", 50))
        macd_hist = float(row.get("macd_histogram", 0))
        bb_pct = float(row.get("bb_pct", 0.5))
        momentum = float(row.get("momentum_10", 0))
        sentiment = sentiment_cache.get(symbol, 0.0)

        # --- Composite score ---
        score = self._compute_composite_score(
            row, sentiment, volume_ratio, sharpe_5m, strategy_weights
        )

        # --- Decide action ---
        action = ScanAction.HOLD
        if score >= self.BUY_THRESHOLD:
            action = ScanAction.BUY
        elif score <= self.SHORT_THRESHOLD:
            action = ScanAction.SHORT

        return ScanResult(
            symbol=symbol,
            action=action,
            score=score,
            price=price,
            change_pct=change_pct,
            volume_ratio=volume_ratio,
            rsi=rsi,
            macd_hist=macd_hist,
            bb_pct=bb_pct,
            sentiment=sentiment,
            sharpe_5m=sharpe_5m,
            momentum=momentum,
        )

    def _compute_composite_score(
        self,
        row: pd.Series,
        sentiment: float,
        volume_ratio: float,
        sharpe_5m: float,
        strategy_weights: Optional[Dict[str, float]],
    ) -> float:
        """
        Compute a composite profitability score in [-1, 1].
        Blends:
          - Technical indicator signals (weighted by Thompson Sampling)
          - Sentiment score
          - Volume confirmation
          - Short-term Sharpe ratio
        """
        # 1. Weighted technical signal
        tech_signal = self.indicators.get_combined_signal(row, strategy_weights)

        # 2. Sentiment component (clamped)
        sent_score = np.clip(sentiment, -1, 1)

        # 3. Volume confirmation: amplifies signal when volume spikes
        vol_mult = min(volume_ratio, 3.0) / 3.0  # normalise to [0,1]

        # 4. Sharpe component (normalised)
        sharpe_norm = np.clip(sharpe_5m / 5.0, -1, 1)

        # 5. RSI extremes bonus
        rsi = float(row.get("rsi", 50))
        rsi_bonus = 0.0
        if rsi < 30:
            rsi_bonus = 0.15  # oversold → buy bias
        elif rsi > 70:
            rsi_bonus = -0.15  # overbought → short bias

        # Blend
        raw = (
            0.35 * tech_signal
            + 0.15 * sent_score
            + 0.15 * sharpe_norm
            + 0.10 * rsi_bonus
            + 0.10 * vol_mult * np.sign(tech_signal)  # volume confirms direction
            + 0.15 * np.clip(float(row.get("momentum_10", 0)) * 10, -1, 1)
        )
        return float(np.clip(raw, -1, 1))

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    def get_top_buys(self, results: List[ScanResult], top_k: int = 10) -> List[ScanResult]:
        return [r for r in results if r.action == ScanAction.BUY][:top_k]

    def get_top_shorts(self, results: List[ScanResult], top_k: int = 10) -> List[ScanResult]:
        return [r for r in results if r.action == ScanAction.SHORT][:top_k]

    def _log_scan_summary(self, results: List[ScanResult]):
        buys = [r for r in results if r.action == ScanAction.BUY]
        shorts = [r for r in results if r.action == ScanAction.SHORT]
        holds = [r for r in results if r.action == ScanAction.HOLD]
        logger.info(
            f"Scan complete: {len(results)} stocks | "
            f"BUY={len(buys)} SHORT={len(shorts)} HOLD={len(holds)}"
        )
        for r in buys[:5]:
            logger.info(
                f"  ▲ BUY  {r.symbol:>12s}  score={r.score:+.3f}  "
                f"price={r.price:.2f}  chg={r.change_pct:+.2%}  "
                f"vol_ratio={r.volume_ratio:.1f}x  conf={r.confidence}"
            )
        for r in shorts[:5]:
            logger.info(
                f"  ▼ SHORT {r.symbol:>12s}  score={r.score:+.3f}  "
                f"price={r.price:.2f}  chg={r.change_pct:+.2%}  "
                f"vol_ratio={r.volume_ratio:.1f}x  conf={r.confidence}"
            )

    def format_scan_table(self, results: List[ScanResult]) -> str:
        """Format scan results as a readable table."""
        header = (
            f"{'Symbol':>12s} {'Action':>6s} {'Score':>7s} {'Price':>10s} "
            f"{'Chg%':>7s} {'Vol':>5s} {'RSI':>5s} {'Sharpe':>7s} {'Conf':>6s}"
        )
        lines = [header, "-" * len(header)]
        for r in results:
            lines.append(
                f"{r.symbol:>12s} {r.action.value:>6s} {r.score:+.3f} "
                f"{r.price:>10.2f} {r.change_pct:>+6.2%} {r.volume_ratio:>5.1f} "
                f"{r.rsi:>5.1f} {r.sharpe_5m:>+7.2f} {r.confidence:>6s}"
            )
        return "\n".join(lines)
