"""
Main Orchestrator - NSE RL Trader
Ties together all components: FinBERT, Q-Learning, Technical Indicators,
OpenAlgo trading, and real-time model updates.

Usage:
    python main.py --mode pretrain    # Pre-train on 30yr historical data
    python main.py --mode trade       # Run live trading with real-time RL updates
    python main.py --mode backtest    # Run backtest on recent data
"""
import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import AppConfig
from data.nse_data_collector import NSEDataCollector
from data.news_scraper import NewsScraper
from indicators.technical_indicators import TechnicalIndicators
from models.finbert_sentiment import FinBERTSentiment
from models.q_learning_agent import ACTION_BUY, ACTION_HOLD, ACTION_SELL, ACTION_NAMES
from models.strategy_selector import StrategySelector
from trading.openalgo_trader import OpenAlgoTrader
from trading.portfolio_manager import PortfolioManager
from trading.stock_scanner import StockScanner, ScanAction
from trading.telegram_notifier import TelegramNotifier
from training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("nse_rl_trader.log"),
    ],
)
logger = logging.getLogger(__name__)


class NSERLTrader:
    """
    Main trading system that:
    1. Collects news sentiment via FinBERT
    2. Computes technical indicators (RSI, MACD, BB, Stochastic, etc.)
    3. Uses Q-Learning to make trading decisions
    4. Executes trades via OpenAlgo across all NSE stocks
    5. Continuously updates models in real-time
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig()

        # Core components
        self.data_collector = NSEDataCollector(self.config)
        self.news_scraper = NewsScraper(self.config)
        self.indicators = TechnicalIndicators(self.config)
        self.trainer = Trainer(self.config)
        self.portfolio = PortfolioManager(self.config)
        self.strategy_selector = self.trainer.strategy_selector

        # OpenAlgo trader (initialized on trade mode)
        self.trader: Optional[OpenAlgoTrader] = None

        # Stock scanner (5-min profitability ranking)
        self.scanner = StockScanner(self.config)

        # Telegram notifications
        self.telegram = TelegramNotifier(self.config)

        # FinBERT (lazy loaded)
        self.finbert: Optional[FinBERTSentiment] = None

        # Runtime state
        self._running = False
        self._last_news_fetch = datetime.min
        self._last_model_update = datetime.min
        self._symbol_states: Dict[str, np.ndarray] = {}
        self._symbol_data: Dict[str, pd.DataFrame] = {}
        self._sentiment_cache: Dict[str, float] = {}

    def initialize(self, load_models: bool = True):
        """Initialize all components."""
        logger.info("Initializing NSE RL Trader...")

        if load_models:
            self.trainer.load_all()
            self.finbert = self.trainer.finbert

        if self.finbert is None:
            self.finbert = self.trainer.initialize_finbert()

        logger.info("Initialization complete.")

    def run_pretrain(self, symbols: Optional[List[str]] = None, episodes: int = 50):
        """Run pre-training on historical data."""
        logger.info("=" * 60)
        logger.info("STARTING PRE-TRAINING ON HISTORICAL DATA")
        logger.info("=" * 60)

        self.initialize(load_models=False)
        report = self.trainer.pretrain_on_historical(symbols=symbols, num_episodes=episodes)

        logger.info("=" * 60)
        logger.info("PRE-TRAINING COMPLETE")
        logger.info(f"Symbols trained: {report['symbols_trained']}")
        logger.info(f"Total episodes: {report['total_episodes']}")
        logger.info(f"Best strategies: {report['best_strategies']}")
        logger.info("=" * 60)
        return report

    def run_live_trading(self):
        """
        Main live trading loop.
        Runs during NSE trading hours (9:15 AM - 3:15 PM IST).
        """
        logger.info("=" * 60)
        logger.info("STARTING LIVE TRADING")
        logger.info(f"Initial Capital: ₹{self.config.trading.initial_capital:,.2f}")
        logger.info("=" * 60)

        self.initialize(load_models=True)
        self.trader = OpenAlgoTrader(self.config)

        # Get trading symbols
        symbols = self.data_collector.get_nse_symbols()
        logger.info(f"Trading across {len(symbols)} NSE stocks")

        # Telegram: market-open alert
        self.telegram.notify_market_open(self.config.trading.initial_capital, len(symbols))

        self._running = True

        try:
            while self._running:
                now = datetime.now()

                # Check if within trading hours
                if not self._is_trading_hours(now):
                    if self._is_post_market(now):
                        self._run_end_of_day()
                        self._running = False
                        break
                    logger.info("Waiting for market open...")
                    time.sleep(60)
                    continue

                # 1. Fetch news and update sentiment
                self._update_news_sentiment()

                # 2. Run 5-minute profitability scan
                strategy_weights = self.strategy_selector.get_signal_weights()
                scan_results = self.scanner.scan_all(
                    symbols,
                    trader=self.trader,
                    sentiment_cache=self._sentiment_cache,
                    strategy_weights=strategy_weights,
                )

                # 3. Process scan results: execute BUY / SHORT / HOLD
                for result in scan_results:
                    try:
                        self._process_scan_result(result)
                    except Exception as e:
                        logger.error(f"Error processing scan {result.symbol}: {e}")

                # 4. Also run Q-learning pass on existing positions
                for symbol in list(self.portfolio.positions.keys()):
                    try:
                        self._process_symbol(symbol)
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")

                # 5. Check stop-loss / take-profit
                self._check_risk_management()

                # 6. Periodic model update
                if (now - self._last_model_update).seconds > self.config.qlearning.update_interval_minutes * 60:
                    self._periodic_model_update()
                    self._last_model_update = now

                # 7. Log portfolio status
                self._log_portfolio_status()

                # Wait before next cycle (5 minutes for 5m candle resolution)
                logger.info("Sleeping for 5 minutes...")
                time.sleep(300)

        except KeyboardInterrupt:
            logger.info("Trading stopped by user.")
        finally:
            self._cleanup()

    def _process_symbol(self, symbol: str):
        """Process a single symbol: analyze, decide, and optionally trade."""
        # Get latest data
        if self.trader:
            quote = self.trader.get_quote(symbol)
            if not quote or "data" not in quote:
                return

        # Build state vector
        state = self._build_state(symbol)
        if state is None:
            return

        # Get Q-learning action
        action = self.trainer.agent.select_action(state, training=True)
        q_values = self.trainer.agent.get_q_values(state)

        # Get strategy weights for signal interpretation
        strategy_weights = self.strategy_selector.get_signal_weights()

        # Get current price
        current_price = self._get_current_price(symbol)
        if current_price <= 0:
            return

        # Sentiment score for this symbol
        sentiment = self._sentiment_cache.get(symbol, 0.0)

        # Execute action
        if action == ACTION_BUY:
            self._handle_buy(symbol, current_price, q_values, sentiment)
        elif action == ACTION_SELL:
            self._handle_sell(symbol, current_price, q_values, sentiment)
        # HOLD: do nothing

        # Store state for next iteration
        self._symbol_states[symbol] = state

    def _process_scan_result(self, result):
        """
        Act on a StockScanner result.
        BUY  → go long if not already in position
        SHORT → sell existing position or place short (MIS intraday)
        HOLD → do nothing
        """
        symbol = result.symbol
        price = result.price
        sentiment = result.sentiment

        if result.action == ScanAction.BUY:
            if symbol not in self.portfolio.positions:
                state = self._build_state(symbol)
                q_values = self.trainer.agent.get_q_values(state) if state is not None else np.zeros(3)
                self._handle_buy(symbol, price, q_values, sentiment)

        elif result.action == ScanAction.SHORT:
            if symbol in self.portfolio.positions:
                # Close the long before shorting
                state = self._build_state(symbol)
                q_values = self.trainer.agent.get_q_values(state) if state is not None else np.zeros(3)
                self._handle_sell(symbol, price, q_values, sentiment)
            else:
                self._handle_short(symbol, price, sentiment)

    def _handle_short(self, symbol: str, price: float, sentiment: float):
        """Place a short (sell-first) intraday MIS order via OpenAlgo."""
        if not self.trader:
            return
        if len(self.portfolio.positions) >= self.config.trading.max_positions:
            return

        # Compute quantity
        max_allocation = self.portfolio.portfolio_value * self.config.trading.max_position_pct
        available = min(self.portfolio.cash, max_allocation)
        quantity = max(1, int(available / price))

        result = self.trader.place_sell_order(symbol, quantity, strategy="scanner_short")
        if result.get("status") == "success":
            logger.info(f"SHORT {quantity} {symbol} @ {price:.2f} (scanner)")
            self.telegram.notify_short(symbol, quantity, price)

    def _build_state(self, symbol: str) -> Optional[np.ndarray]:
        """Build state vector for Q-learning from market data + sentiment."""
        try:
            # Get recent data (try OpenAlgo first, fallback to yfinance)
            df = None
            if self.trader:
                end_date = datetime.now().strftime("%Y-%m-%d")
                start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
                df_result = self.trader.get_historical(
                    symbol, interval="D", start_date=start_date, end_date=end_date
                )
                if isinstance(df_result, pd.DataFrame) and not df_result.empty:
                    df = df_result

            if df is None or df.empty:
                df = self.data_collector.fetch_historical_data(symbol, years=1, interval="1d")

            if df is None or df.empty or len(df) < 50:
                return None

            # Add indicators
            df = self.indicators.compute_all(df)
            df.dropna(inplace=True)

            if df.empty:
                return None

            # Build feature vector from latest row
            row = df.iloc[-1]
            features = []

            # Technical indicator features
            for col in self.indicators.get_feature_columns():
                if col in row.index:
                    val = row[col]
                    features.append(0.0 if (np.isnan(val) or np.isinf(val)) else float(val))

            # Signal features
            for col in self.indicators.get_signal_columns():
                if col in row.index:
                    features.append(float(row[col]))

            # Price features
            if "Close" in row.index:
                close = row["Close"]
                features.extend([
                    row.get("returns", 0.0),
                    row.get("volatility_20d", 0.0),
                    row.get("high_low_range", 0.0),
                ])

            # Sentiment feature
            sentiment = self._sentiment_cache.get(symbol, 0.0)
            features.append(sentiment)

            # Portfolio features
            portfolio_value = self.portfolio.portfolio_value
            in_position = 1.0 if symbol in self.portfolio.positions else 0.0
            unrealized_pnl = 0.0
            if symbol in self.portfolio.positions:
                pos = self.portfolio.positions[symbol]
                unrealized_pnl = pos.pnl_pct

            features.extend([
                self.portfolio.cash / max(portfolio_value, 1),
                in_position,
                unrealized_pnl,
                self.portfolio.total_return,
            ])

            # Pad/truncate to state_size
            state = np.array(features[:self.config.qlearning.state_size], dtype=np.float32)
            if len(state) < self.config.qlearning.state_size:
                state = np.pad(state, (0, self.config.qlearning.state_size - len(state)))

            return np.clip(state, -10, 10)

        except Exception as e:
            logger.error(f"State build error for {symbol}: {e}")
            return None

    def _handle_buy(self, symbol: str, price: float, q_values, sentiment: float):
        """Handle buy decision."""
        can_buy, max_shares = self.portfolio.can_buy(symbol, price)
        if not can_buy:
            return

        # Adjust quantity based on Q-value confidence
        confidence = (q_values[ACTION_BUY] - q_values[ACTION_HOLD]) / max(abs(q_values[ACTION_HOLD]), 0.01)
        quantity = max(1, int(max_shares * min(abs(confidence), 1.0)))

        # Execute via OpenAlgo
        if self.trader:
            result = self.trader.place_buy_order(symbol, quantity)
            if result.get("status") == "success":
                self.portfolio.execute_buy(
                    symbol, quantity, price,
                    sentiment_score=sentiment, q_values=q_values.tolist(),
                )
                self.telegram.notify_buy(symbol, quantity, price, confidence=confidence, sentiment=sentiment)
        else:
            # Paper trading without OpenAlgo
            self.portfolio.execute_buy(
                symbol, quantity, price,
                sentiment_score=sentiment, q_values=q_values.tolist(),
            )
            self.telegram.notify_buy(symbol, quantity, price, confidence=confidence, sentiment=sentiment)

    def _handle_sell(self, symbol: str, price: float, q_values, sentiment: float):
        """Handle sell decision."""
        if symbol not in self.portfolio.positions:
            return

        pos = self.portfolio.positions[symbol]

        # Execute via OpenAlgo
        if self.trader:
            result = self.trader.place_sell_order(symbol, pos.quantity)
            if result.get("status") == "success":
                success, pnl = self.portfolio.execute_sell(
                    symbol, price,
                    sentiment_score=sentiment, q_values=q_values.tolist(),
                )
                if success:
                    pnl_pct = pnl / (pos.quantity * pos.avg_price) if pos.avg_price else 0
                    self.telegram.notify_sell(symbol, pos.quantity, price, pnl=pnl, pnl_pct=pnl_pct)
                    self._provide_rl_feedback(symbol, pnl)
        else:
            success, pnl = self.portfolio.execute_sell(
                symbol, price,
                sentiment_score=sentiment, q_values=q_values.tolist(),
            )
            if success:
                pnl_pct = pnl / (pos.quantity * pos.avg_price) if pos.avg_price else 0
                self.telegram.notify_sell(symbol, pos.quantity, price, pnl=pnl, pnl_pct=pnl_pct)
                self._provide_rl_feedback(symbol, pnl)

    def _provide_rl_feedback(self, symbol: str, pnl: float):
        """Provide feedback to all RL components after a completed trade."""
        reward = pnl / self.config.trading.initial_capital

        # Update strategy selector
        latest_trade = self.portfolio.trade_history[-1] if self.portfolio.trade_history else None
        if latest_trade and latest_trade.strategy_signals:
            self.strategy_selector.update_batch(latest_trade.strategy_signals, reward)

        # Update Q-agent if we have previous state
        if symbol in self._symbol_states:
            prev_state = self._symbol_states[symbol]
            current_state = self._build_state(symbol)
            if current_state is not None:
                self.trainer.online_update(
                    prev_state, ACTION_SELL, reward, current_state, done=False,
                )

    def _update_news_sentiment(self):
        """Fetch and analyze news sentiment."""
        now = datetime.now()
        if (now - self._last_news_fetch).seconds < self.config.news.fetch_interval_minutes * 60:
            return

        logger.info("Fetching latest news for sentiment analysis...")
        articles = self.news_scraper.fetch_all_news()

        if not articles:
            return

        # Analyze sentiment
        texts = [f"{a.title}. {a.content}" for a in articles]
        results = self.finbert.analyze_batch(texts)

        # Update articles with sentiment
        for article, result in zip(articles, results):
            article.sentiment_score = result["score"]
            article.sentiment_label = result["label"]

        # Build per-symbol sentiment cache
        summary = self.news_scraper.get_symbol_sentiment_summary()
        for symbol, data in summary.items():
            self._sentiment_cache[symbol] = data.get("avg_sentiment", 0.0)

        self._last_news_fetch = now
        logger.info(f"Processed {len(articles)} articles. Sentiment for {len(summary)} symbols.")

    def _check_risk_management(self):
        """Check stop-loss and take-profit triggers."""
        prices = {}
        for symbol in list(self.portfolio.positions.keys()):
            price = self._get_current_price(symbol)
            if price > 0:
                prices[symbol] = price

        self.portfolio.update_prices(prices)
        triggers = self.portfolio.check_stop_loss_take_profit(prices)

        for symbol, reason in triggers:
            logger.warning(f"Risk trigger: {symbol} hit {reason}")
            price = prices.get(symbol, 0)
            if price > 0:
                pos = self.portfolio.positions.get(symbol)
                if pos and reason == "stop_loss":
                    self.telegram.notify_stop_loss(symbol, price, abs(pos.pnl_pct))
                elif pos and reason == "take_profit":
                    self.telegram.notify_take_profit(symbol, price, pos.pnl_pct)
                self._handle_sell(symbol, price, np.zeros(3), 0.0)

    def _get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        try:
            if self.trader:
                quote = self.trader.get_quote(symbol)
                if quote and "data" in quote:
                    data = quote["data"]
                    return float(data.get("ltp", data.get("close", 0)))

            # Fallback: use latest from cached data
            df = self.data_collector.fetch_historical_data(symbol, years=1)
            if not df.empty:
                return float(df.iloc[-1]["Close"])
        except Exception as e:
            logger.error(f"Price fetch error for {symbol}: {e}")
        return 0.0

    def _periodic_model_update(self):
        """Periodic update of strategy weights and Q-agent during trading."""
        logger.info("Running periodic model update...")
        self.strategy_selector.sample_weights()

        # Log current strategy performance
        best = self.strategy_selector.get_best_strategies(top_k=5)
        logger.info(f"Top strategies: {best}")

    def _run_end_of_day(self):
        """End of day: square off, update models, save."""
        logger.info("Market closed. Running end-of-day procedures...")

        # Square off all positions
        prices = {}
        for symbol in list(self.portfolio.positions.keys()):
            price = self._get_current_price(symbol)
            if price > 0:
                prices[symbol] = price
                self._handle_sell(symbol, price, np.zeros(3), 0.0)

        if self.trader:
            self.trader.close_all_positions()

        # Run end-of-day model updates
        self.trainer.end_of_day_update(self.portfolio.trade_history)

        # Clean old news
        self.news_scraper.clear_old_articles()

        # Final portfolio summary
        summary = self.portfolio.get_portfolio_summary()
        logger.info("=" * 60)
        logger.info("END OF DAY SUMMARY")
        logger.info(f"Portfolio Value: ₹{summary['portfolio_value']:,.2f}")
        logger.info(f"Total Return: {summary['total_return']:.2%}")
        logger.info(f"Realized P&L: ₹{summary['realized_pnl']:,.2f}")
        logger.info(f"Total Trades: {summary['total_trades']}")
        logger.info(f"Win Rate: {summary['win_rate']:.2%}")
        logger.info(f"Max Drawdown: {summary['max_drawdown']:.2%}")
        logger.info("=" * 60)

    def _cleanup(self):
        """Cleanup on shutdown."""
        logger.info("Cleaning up...")
        self.trainer.save_all()
        if self.trader:
            self.trader.cancel_all_orders()

    def _log_portfolio_status(self):
        """Log current portfolio status."""
        summary = self.portfolio.get_portfolio_summary()
        logger.info(
            f"Portfolio: ₹{summary['portfolio_value']:,.2f} | "
            f"Return: {summary['total_return']:.2%} | "
            f"Positions: {summary['num_positions']} | "
            f"Cash: ₹{summary['cash']:,.2f}"
        )

    def _is_trading_hours(self, now: datetime) -> bool:
        """Check if current time is within NSE trading hours."""
        tc = self.config.trading
        market_open = now.replace(hour=tc.trading_start_hour, minute=tc.trading_start_minute, second=0)
        market_close = now.replace(hour=tc.trading_end_hour, minute=tc.trading_end_minute, second=0)
        return market_open <= now <= market_close

    def _is_post_market(self, now: datetime) -> bool:
        """Check if market has closed for the day."""
        tc = self.config.trading
        market_close = now.replace(hour=tc.trading_end_hour, minute=tc.trading_end_minute, second=0)
        return now > market_close

    def run_backtest(self, symbols: Optional[List[str]] = None, years: int = 2):
        """
        Run backtest on recent historical data.
        Simulates trading without OpenAlgo connection.
        """
        logger.info("=" * 60)
        logger.info(f"STARTING BACKTEST ({years} years)")
        logger.info("=" * 60)

        self.initialize(load_models=True)
        self.trainer.agent.epsilon = 0.05  # Low exploration for backtest

        if symbols is None:
            symbols = self.data_collector.NIFTY50_SYMBOLS[:10]

        for symbol in symbols:
            df = self.prepare_symbol_data(symbol, years)
            if df is None:
                continue

            logger.info(f"Backtesting {symbol} ({len(df)} bars)...")

            for i in range(50, len(df)):
                row = df.iloc[i]
                price = row["Close"]

                # Build state from row
                features = []
                for col in self.indicators.get_feature_columns():
                    if col in row.index:
                        val = row[col]
                        features.append(0.0 if np.isnan(val) else float(val))

                for col in self.indicators.get_signal_columns():
                    if col in row.index:
                        features.append(float(row[col]))

                features.extend([0.0, 0.0, 0.0, self.portfolio.total_return])

                state = np.array(features[:self.config.qlearning.state_size], dtype=np.float32)
                if len(state) < self.config.qlearning.state_size:
                    state = np.pad(state, (0, self.config.qlearning.state_size - len(state)))
                state = np.clip(state, -10, 10)

                action = self.trainer.agent.select_action(state, training=False)

                if action == ACTION_BUY:
                    can_buy, qty = self.portfolio.can_buy(symbol, price)
                    if can_buy:
                        self.portfolio.execute_buy(symbol, qty, price)
                elif action == ACTION_SELL:
                    if symbol in self.portfolio.positions:
                        self.portfolio.execute_sell(symbol, price)

        # Final results
        summary = self.portfolio.get_portfolio_summary()
        logger.info("=" * 60)
        logger.info("BACKTEST RESULTS")
        logger.info(f"Initial Capital: ₹{self.config.trading.initial_capital:,.2f}")
        logger.info(f"Final Value: ₹{summary['portfolio_value']:,.2f}")
        logger.info(f"Total Return: {summary['total_return']:.2%}")
        logger.info(f"Total Trades: {summary['total_trades']}")
        logger.info(f"Win Rate: {summary['win_rate']:.2%}")
        logger.info(f"Max Drawdown: {summary['max_drawdown']:.2%}")
        logger.info("=" * 60)
        return summary

    def prepare_symbol_data(self, symbol: str, years: int) -> Optional[pd.DataFrame]:
        """Prepare data for backtesting."""
        return self.trainer.prepare_symbol_data(symbol, years)

    def run_scan(self, symbols: Optional[List[str]] = None):
        """
        Run the 5-minute profitability scanner once and print results.
        Useful for a quick look at which stocks are actionable right now.
        """
        logger.info("=" * 60)
        logger.info("RUNNING 5-MINUTE PROFITABILITY SCAN")
        logger.info("=" * 60)

        self.initialize(load_models=True)

        if symbols is None:
            symbols = self.data_collector.get_nse_symbols()

        # Optionally connect to OpenAlgo for live quotes
        trader = None
        if self.config.openalgo.api_key:
            trader = OpenAlgoTrader(self.config)

        # Run sentiment pass
        self._update_news_sentiment()

        strategy_weights = self.strategy_selector.get_signal_weights()
        results = self.scanner.scan_all(
            symbols,
            trader=trader,
            sentiment_cache=self._sentiment_cache,
            strategy_weights=strategy_weights,
        )

        # Print formatted table
        print("\n" + self.scanner.format_scan_table(results) + "\n")

        buys = self.scanner.get_top_buys(results)
        shorts = self.scanner.get_top_shorts(results)
        logger.info(f"Top BUY candidates: {[r.symbol for r in buys]}")
        logger.info(f"Top SHORT candidates: {[r.symbol for r in shorts]}")
        return results


def main():
    parser = argparse.ArgumentParser(description="NSE RL Trader - FinBERT + Q-Learning")
    parser.add_argument(
        "--mode", choices=["pretrain", "trade", "backtest", "scan"],
        default="pretrain",
        help="Operating mode: pretrain, trade, backtest, or scan (5-min profitability scanner)",
    )
    parser.add_argument("--episodes", type=int, default=50, help="Training episodes per symbol")
    parser.add_argument("--symbols", nargs="+", default=None, help="Specific symbols to trade/train")
    parser.add_argument("--capital", type=float, default=10000.0, help="Initial capital in INR")
    parser.add_argument("--api-key", type=str, default="", help="OpenAlgo API key")
    parser.add_argument("--host", type=str, default="http://127.0.0.1:5000", help="OpenAlgo host URL")
    parser.add_argument("--paper", action="store_true", default=True, help="Use paper trading (analyzer) mode")
    parser.add_argument("--years", type=int, default=2, help="Years for backtest")

    args = parser.parse_args()

    # Configure
    config = AppConfig()
    config.trading.initial_capital = args.capital

    if args.api_key:
        config.openalgo.api_key = args.api_key
    if args.host:
        config.openalgo.host = args.host
    config.openalgo.use_analyzer = args.paper

    # Run
    system = NSERLTrader(config)

    if args.mode == "pretrain":
        system.run_pretrain(symbols=args.symbols, episodes=args.episodes)
    elif args.mode == "trade":
        system.run_live_trading()
    elif args.mode == "backtest":
        system.run_backtest(symbols=args.symbols, years=args.years)
    elif args.mode == "scan":
        system.run_scan(symbols=args.symbols)


if __name__ == "__main__":
    main()
