"""
Trainer Module - Handles pre-training on historical data and
continuous learning during live trading.
"""
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config.settings import AppConfig
from data.nse_data_collector import NSEDataCollector
from indicators.technical_indicators import TechnicalIndicators
from models.finbert_sentiment import FinBERTSentiment
from models.q_learning_agent import QLearningAgent, TradingEnvironment
from models.strategy_selector import StrategySelector

logger = logging.getLogger(__name__)


class Trainer:
    """
    Orchestrates training of the Q-Learning agent:
    1. Pre-training on 30 years of historical NSE data
    2. Continuous online learning during live trading
    3. Strategy weight optimization
    4. FinBERT feedback integration
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.data_collector = NSEDataCollector(config)
        self.indicators = TechnicalIndicators(config)
        self.agent = QLearningAgent(config)
        self.strategy_selector = StrategySelector(config)
        self.finbert: Optional[FinBERTSentiment] = None

        os.makedirs(config.model_save_dir, exist_ok=True)

    def initialize_finbert(self):
        """Lazy-load FinBERT (heavy model)."""
        if self.finbert is None:
            self.finbert = FinBERTSentiment(self.config)
        return self.finbert

    def prepare_symbol_data(self, symbol: str, years: int = 30) -> Optional[pd.DataFrame]:
        """Fetch and prepare data for a single symbol."""
        df = self.data_collector.fetch_historical_data(symbol, years=years)
        if df.empty:
            return None

        # Add technical indicators
        df = self.indicators.compute_all(df)

        # Add basic features
        df = self.data_collector.prepare_training_data(df)

        # Drop rows with NaN
        df.dropna(inplace=True)

        if len(df) < 100:
            logger.warning(f"Insufficient data for {symbol}: {len(df)} rows")
            return None

        return df

    def pretrain_on_historical(self, symbols: Optional[List[str]] = None,
                                num_episodes: int = 50) -> Dict:
        """
        Pre-train the Q-Learning agent on historical data.
        Iterates over multiple symbols to build a general trading policy.
        """
        if symbols is None:
            symbols = self.data_collector.NIFTY50_SYMBOLS[:20]  # Start with top 20

        logger.info(f"Pre-training on {len(symbols)} symbols for {num_episodes} episodes each")

        all_rewards = {}
        total_symbols_trained = 0

        for sym_idx, symbol in enumerate(symbols):
            logger.info(f"[{sym_idx + 1}/{len(symbols)}] Preparing {symbol}...")
            df = self.prepare_symbol_data(symbol, years=self.config.data.history_years)
            if df is None:
                continue

            # Create trading environment
            env = TradingEnvironment(
                df=df,
                initial_capital=self.config.trading.initial_capital,
                state_size=self.config.qlearning.state_size,
                max_position_pct=self.config.trading.max_position_pct,
            )

            # Train agent
            episode_rewards = self.agent.train_on_historical(env, num_episodes=num_episodes)
            all_rewards[symbol] = episode_rewards
            total_symbols_trained += 1

            # Update strategy weights based on performance
            self._update_strategy_weights_from_training(df, episode_rewards)

            # Save checkpoint every 5 symbols
            if (sym_idx + 1) % 5 == 0:
                self.save_all()

        # Final save
        self.save_all()

        # Generate training report
        report = {
            "symbols_trained": total_symbols_trained,
            "total_episodes": total_symbols_trained * num_episodes,
            "final_epsilon": self.agent.epsilon,
            "avg_rewards": {
                s: float(np.mean(r[-10:])) for s, r in all_rewards.items()
            },
            "best_strategies": self.strategy_selector.get_best_strategies(),
        }

        logger.info(f"Pre-training complete. Trained on {total_symbols_trained} symbols.")
        logger.info(f"Best strategies: {report['best_strategies']}")
        return report

    def _update_strategy_weights_from_training(self, df: pd.DataFrame, rewards: List[float]):
        """Update strategy selector based on which indicator signals correlated with rewards."""
        signal_cols = self.indicators.get_signal_columns()

        # For each signal column, check correlation with successful episodes
        avg_reward = np.mean(rewards) if rewards else 0

        for col in signal_cols:
            if col not in df.columns:
                continue

            # Check if this indicator had active signals during profitable periods
            recent = df.tail(len(rewards)) if len(rewards) <= len(df) else df
            signal_active = recent[col].abs().sum()

            strategy_name = None
            for name, sig_col in self.strategy_selector.SIGNAL_COL_MAP.items():
                if sig_col == col:
                    strategy_name = name
                    break

            if strategy_name and signal_active > 0:
                self.strategy_selector.update(strategy_name, avg_reward)

        self.strategy_selector.sample_weights()

    def online_update(self, state: np.ndarray, action: int, reward: float,
                      next_state: np.ndarray, done: bool,
                      strategy_signals: Dict[str, int] = None,
                      sentiment_text: str = None):
        """
        Real-time update during live trading.
        Updates Q-agent, strategy selector, and FinBERT.
        """
        # Update Q-Learning agent
        loss = self.agent.update_live(state, action, reward, next_state, done)

        # Update strategy weights
        if strategy_signals:
            self.strategy_selector.update_batch(strategy_signals, reward)
            self.strategy_selector.sample_weights()

        # Feed reward back to FinBERT
        if sentiment_text and self.finbert:
            result = self.finbert.analyze(sentiment_text)
            label_idx = list(self.finbert.LABEL_MAP.keys())[
                list(self.finbert.LABEL_MAP.values()).index(result["label"])
            ]
            self.finbert.add_rl_feedback(sentiment_text, label_idx, reward)

        return loss

    def end_of_day_update(self, trade_records: list):
        """
        End-of-day batch update:
        1. Fine-tune FinBERT with accumulated RL feedback
        2. Evolve strategies
        3. Run additional training episodes
        """
        logger.info("Running end-of-day model updates...")

        # 1. Fine-tune FinBERT
        if self.finbert:
            self.finbert.fine_tune_with_rl_feedback()

        # 2. Evolve strategies
        self.strategy_selector.evolve_strategies()
        logger.info(f"Strategy report: {json.dumps(self.strategy_selector.get_performance_report(), indent=2, default=str)}")

        # 3. Save everything
        self.save_all()

        logger.info("End-of-day update complete.")

    def save_all(self):
        """Save all model states."""
        save_dir = self.config.model_save_dir

        # Save Q-learning agent
        self.agent.save(os.path.join(save_dir, "q_agent.pt"))

        # Save strategy selector
        strategy_state = self.strategy_selector.save_state()
        with open(os.path.join(save_dir, "strategy_selector.json"), "w") as f:
            json.dump(strategy_state, f, indent=2, default=str)

        # Save FinBERT
        if self.finbert:
            self.finbert.save_model(os.path.join(save_dir, "finbert"))

        logger.info(f"All models saved to {save_dir}")

    def load_all(self):
        """Load all model states."""
        save_dir = self.config.model_save_dir

        # Load Q-learning agent
        agent_path = os.path.join(save_dir, "q_agent.pt")
        if os.path.exists(agent_path):
            self.agent.load(agent_path)

        # Load strategy selector
        strategy_path = os.path.join(save_dir, "strategy_selector.json")
        if os.path.exists(strategy_path):
            with open(strategy_path) as f:
                self.strategy_selector.load_state(json.load(f))

        # Load FinBERT
        finbert_path = os.path.join(save_dir, "finbert")
        if os.path.exists(finbert_path):
            self.initialize_finbert()
            self.finbert.load_model(finbert_path)

        logger.info("All models loaded")
