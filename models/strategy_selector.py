"""
Strategy Selector - Uses RL to learn optimal combination of technical indicators.
Evolves strategy weights over time based on trading performance.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class StrategySelector:
    """
    Learns optimal weights for combining technical strategies.
    Uses a multi-armed bandit approach with Thompson Sampling
    to select and weight the best strategies.
    """

    STRATEGY_NAMES = [
        "rsi", "macd", "bollinger_bands", "stochastic",
        "sma_crossover", "ema_crossover", "vwap", "atr",
        "obv", "momentum", "mean_reversion", "cci",
    ]

    SIGNAL_COL_MAP = {
        "rsi": "rsi_signal",
        "macd": "macd_signal",
        "bollinger_bands": "bb_signal",
        "stochastic": "stoch_signal",
        "sma_crossover": "sma_signal",
        "ema_crossover": "ema_signal",
        "vwap": "vwap_signal",
        "atr": "atr_signal",
        "obv": "obv_signal",
        "momentum": "momentum_signal",
        "mean_reversion": "bb_signal",  # Mean reversion uses Bollinger
        "cci": "cci_signal",
    }

    def __init__(self, config):
        self.config = config
        self.enabled = config.strategy.enabled_strategies

        # Thompson Sampling: Beta distribution parameters for each strategy
        # alpha = successes + 1, beta = failures + 1
        self.alpha = {s: 1.0 for s in self.STRATEGY_NAMES}
        self.beta_param = {s: 1.0 for s in self.STRATEGY_NAMES}

        # Running weights (softmax of Thompson samples)
        self.current_weights: Dict[str, float] = {
            s: 1.0 / len(self.enabled) for s in self.enabled
        }

        # Performance tracking
        self.strategy_rewards: Dict[str, List[float]] = {s: [] for s in self.STRATEGY_NAMES}
        self.strategy_trades: Dict[str, int] = {s: 0 for s in self.STRATEGY_NAMES}

    def sample_weights(self) -> Dict[str, float]:
        """
        Sample new strategy weights using Thompson Sampling.
        Returns weights that define how much each strategy contributes.
        """
        samples = {}
        for strategy in self.enabled:
            # Sample from Beta distribution
            sample = np.random.beta(self.alpha[strategy], self.beta_param[strategy])
            samples[strategy] = sample

        # Normalize to sum to 1
        total = sum(samples.values())
        if total > 0:
            self.current_weights = {s: v / total for s, v in samples.items()}
        else:
            self.current_weights = {s: 1.0 / len(self.enabled) for s in self.enabled}

        return self.current_weights

    def get_signal_weights(self) -> Dict[str, float]:
        """Get weights mapped to signal column names for TechnicalIndicators."""
        return {
            self.SIGNAL_COL_MAP[s]: w
            for s, w in self.current_weights.items()
            if s in self.SIGNAL_COL_MAP
        }

    def update(self, strategy: str, reward: float):
        """
        Update strategy performance based on trading reward.
        reward > 0: strategy signal was profitable
        reward < 0: strategy signal led to loss
        """
        if strategy not in self.alpha:
            return

        self.strategy_rewards[strategy].append(reward)
        self.strategy_trades[strategy] += 1

        if reward > 0:
            self.alpha[strategy] += reward
        else:
            self.beta_param[strategy] += abs(reward)

    def update_batch(self, strategy_signals: Dict[str, int], trade_reward: float):
        """
        Update all strategies that contributed to a trade decision.
        strategy_signals: {strategy_name: signal_value} for the trade
        """
        for strategy, signal in strategy_signals.items():
            if signal != 0:
                self.update(strategy, trade_reward * self.current_weights.get(strategy, 0))

    def get_best_strategies(self, top_k: int = 5) -> List[Tuple[str, float]]:
        """Get top performing strategies by average reward."""
        avg_rewards = {}
        for s in self.enabled:
            rewards = self.strategy_rewards[s]
            if rewards:
                avg_rewards[s] = np.mean(rewards)
            else:
                avg_rewards[s] = 0.0

        sorted_strategies = sorted(avg_rewards.items(), key=lambda x: x[1], reverse=True)
        return sorted_strategies[:top_k]

    def get_performance_report(self) -> Dict:
        """Generate a performance report for all strategies."""
        report = {}
        for s in self.STRATEGY_NAMES:
            rewards = self.strategy_rewards[s]
            report[s] = {
                "trades": self.strategy_trades[s],
                "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
                "total_reward": float(sum(rewards)) if rewards else 0.0,
                "win_rate": float(sum(1 for r in rewards if r > 0) / max(len(rewards), 1)),
                "current_weight": self.current_weights.get(s, 0.0),
                "alpha": self.alpha[s],
                "beta": self.beta_param[s],
            }
        return report

    def evolve_strategies(self):
        """
        Evolutionary step: disable consistently poor strategies,
        increase weight on consistently good ones.
        """
        report = self.get_performance_report()
        for strategy, stats in report.items():
            if stats["trades"] > 50 and stats["win_rate"] < 0.3:
                if strategy in self.enabled:
                    logger.info(f"Disabling poor strategy: {strategy} (win_rate={stats['win_rate']:.2%})")
                    self.enabled.remove(strategy)

        # Re-normalize weights
        if self.enabled:
            self.sample_weights()
        else:
            # Reset if all disabled
            self.enabled = self.STRATEGY_NAMES.copy()
            self.alpha = {s: 1.0 for s in self.STRATEGY_NAMES}
            self.beta_param = {s: 1.0 for s in self.STRATEGY_NAMES}
            self.sample_weights()

    def save_state(self) -> Dict:
        """Serialize state for saving."""
        return {
            "alpha": self.alpha,
            "beta_param": self.beta_param,
            "enabled": self.enabled,
            "current_weights": self.current_weights,
            "strategy_trades": self.strategy_trades,
        }

    def load_state(self, state: Dict):
        """Load serialized state."""
        self.alpha = state.get("alpha", self.alpha)
        self.beta_param = state.get("beta_param", self.beta_param)
        self.enabled = state.get("enabled", self.enabled)
        self.current_weights = state.get("current_weights", self.current_weights)
        self.strategy_trades = state.get("strategy_trades", self.strategy_trades)
