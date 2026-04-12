"""
Portfolio Manager - Manages positions, P&L tracking, risk management,
and capital allocation across all NSE stocks.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Position:
    symbol: str
    quantity: int
    avg_price: float
    current_price: float
    entry_time: datetime
    unrealized_pnl: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0

    @property
    def value(self) -> float:
        return self.quantity * self.current_price

    @property
    def pnl_pct(self) -> float:
        if self.avg_price == 0:
            return 0.0
        return (self.current_price - self.avg_price) / self.avg_price


@dataclass
class TradeRecord:
    symbol: str
    action: str  # "BUY" or "SELL"
    quantity: int
    price: float
    timestamp: datetime
    pnl: float = 0.0
    strategy_signals: Dict[str, int] = field(default_factory=dict)
    sentiment_score: float = 0.0
    q_values: Optional[List[float]] = None


class PortfolioManager:
    """
    Manages the trading portfolio across all NSE stocks.
    Handles capital allocation, risk management, and P&L tracking.
    """

    def __init__(self, config):
        self.config = config
        self.initial_capital = config.trading.initial_capital
        self.cash = config.trading.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[TradeRecord] = []
        self.realized_pnl = 0.0
        self.peak_value = self.initial_capital
        self.total_brokerage_paid = 0.0
        self.brokerage_per_trade = config.trading.brokerage_per_trade

    @property
    def total_position_value(self) -> float:
        return sum(p.value for p in self.positions.values())

    @property
    def portfolio_value(self) -> float:
        return self.cash + self.total_position_value

    @property
    def total_return(self) -> float:
        return (self.portfolio_value - self.initial_capital) / self.initial_capital

    @property
    def max_drawdown(self) -> float:
        if self.peak_value == 0:
            return 0.0
        return (self.peak_value - self.portfolio_value) / self.peak_value

    def update_peak(self):
        """Track peak portfolio value for drawdown calculation."""
        current = self.portfolio_value
        if current > self.peak_value:
            self.peak_value = current

    def can_buy(self, symbol: str, price: float) -> Tuple[bool, int]:
        """Check if we can buy a stock and how many shares."""
        if symbol in self.positions:
            return False, 0  # Already holding

        if len(self.positions) >= self.config.trading.max_positions:
            return False, 0

        # Max allocation per stock
        max_allocation = self.portfolio_value * self.config.trading.max_position_pct
        available = min(self.cash, max_allocation)

        if available < price:
            return False, 0

        shares = int(available / price)
        return shares > 0, shares

    def execute_buy(self, symbol: str, quantity: int, price: float,
                    strategy_signals: Dict[str, int] = None,
                    sentiment_score: float = 0.0,
                    q_values: list = None) -> bool:
        """Record a buy execution (includes brokerage cost)."""
        brokerage = self.brokerage_per_trade
        cost = quantity * price + brokerage
        if cost > self.cash:
            logger.warning(f"Insufficient cash for {symbol}: need {cost:.2f} (incl. ₹{brokerage} brokerage), have {self.cash:.2f}")
            return False

        self.cash -= cost
        self.total_brokerage_paid += brokerage
        sl = price * (1 - self.config.trading.stop_loss_pct)
        tp = price * (1 + self.config.trading.take_profit_pct)

        self.positions[symbol] = Position(
            symbol=symbol, quantity=quantity, avg_price=price,
            current_price=price, entry_time=datetime.now(),
            stop_loss=sl, take_profit=tp,
        )

        self.trade_history.append(TradeRecord(
            symbol=symbol, action="BUY", quantity=quantity,
            price=price, timestamp=datetime.now(),
            strategy_signals=strategy_signals or {},
            sentiment_score=sentiment_score,
            q_values=q_values,
        ))

        logger.info(f"BUY {quantity} {symbol} @ {price:.2f} | SL={sl:.2f} TP={tp:.2f} | Brokerage=₹{brokerage:.0f}")
        return True

    def execute_sell(self, symbol: str, price: float,
                     strategy_signals: Dict[str, int] = None,
                     sentiment_score: float = 0.0,
                     q_values: list = None) -> Tuple[bool, float]:
        """Record a sell execution (includes brokerage cost). Returns (success, net_pnl)."""
        if symbol not in self.positions:
            logger.warning(f"No position to sell: {symbol}")
            return False, 0.0

        pos = self.positions[symbol]
        brokerage = self.brokerage_per_trade
        revenue = pos.quantity * price - brokerage
        pnl = revenue - (pos.quantity * pos.avg_price)
        # pnl already accounts for sell-side brokerage;
        # buy-side brokerage was deducted from cash at entry.

        self.cash += revenue
        self.realized_pnl += pnl
        self.total_brokerage_paid += brokerage

        self.trade_history.append(TradeRecord(
            symbol=symbol, action="SELL", quantity=pos.quantity,
            price=price, timestamp=datetime.now(), pnl=pnl,
            strategy_signals=strategy_signals or {},
            sentiment_score=sentiment_score,
            q_values=q_values,
        ))

        del self.positions[symbol]
        self.update_peak()

        logger.info(f"SELL {pos.quantity} {symbol} @ {price:.2f} | PnL=₹{pnl:.2f} (net of ₹{brokerage:.0f} brokerage)")
        return True, pnl

    def update_prices(self, prices: Dict[str, float]):
        """Update current prices for all positions."""
        for symbol, price in prices.items():
            if symbol in self.positions:
                pos = self.positions[symbol]
                pos.current_price = price
                pos.unrealized_pnl = (price - pos.avg_price) * pos.quantity

        self.update_peak()

    def check_stop_loss_take_profit(self, prices: Dict[str, float]) -> List[Tuple[str, str]]:
        """Check if any positions hit stop loss or take profit. Returns [(symbol, reason)]."""
        triggers = []
        for symbol, pos in list(self.positions.items()):
            price = prices.get(symbol, pos.current_price)
            if price <= pos.stop_loss:
                triggers.append((symbol, "stop_loss"))
            elif price >= pos.take_profit:
                triggers.append((symbol, "take_profit"))
        return triggers

    def get_portfolio_summary(self) -> Dict:
        """Get complete portfolio summary."""
        position_details = []
        for symbol, pos in self.positions.items():
            position_details.append({
                "symbol": symbol,
                "quantity": pos.quantity,
                "avg_price": pos.avg_price,
                "current_price": pos.current_price,
                "unrealized_pnl": pos.unrealized_pnl,
                "pnl_pct": pos.pnl_pct,
            })

        winning = sum(1 for t in self.trade_history if t.action == "SELL" and t.pnl > 0)
        total_sells = sum(1 for t in self.trade_history if t.action == "SELL")

        return {
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "total_position_value": self.total_position_value,
            "portfolio_value": self.portfolio_value,
            "total_return": self.total_return,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": sum(p.unrealized_pnl for p in self.positions.values()),
            "max_drawdown": self.max_drawdown,
            "num_positions": len(self.positions),
            "total_trades": len(self.trade_history),
            "win_rate": winning / max(total_sells, 1),
            "total_brokerage_paid": self.total_brokerage_paid,
            "positions": position_details,
        }

    def get_reward_for_rl(self, symbol: str) -> float:
        """
        Calculate RL reward based on latest trade outcome.
        Uses Sharpe-ratio-aware scaling (ML4T inspired).
        """
        sells = [t for t in self.trade_history if t.symbol == symbol and t.action == "SELL"]
        if not sells:
            return 0.0

        latest = sells[-1]
        # Normalize PnL by initial capital
        reward = latest.pnl / self.initial_capital

        # Penalise volatility: reduce reward by rolling drawdown proportion
        if self.max_drawdown > 0.05:
            reward -= 0.1 * self.max_drawdown

        # Bonus for good risk-adjusted returns
        if latest.pnl > 0:
            reward *= 1.5  # Reward winning trades more

        return reward
