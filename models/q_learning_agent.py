"""
Deep Q-Learning Trading Agent.
Uses Double DQN with experience replay and priority sampling.
The agent learns optimal trading actions (BUY/HOLD/SELL) from a state
composed of technical indicators, sentiment scores, and portfolio state.
"""
import logging
import os
import random
from collections import deque, namedtuple
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

logger = logging.getLogger(__name__)

# Action space
ACTION_BUY = 0
ACTION_HOLD = 1
ACTION_SELL = 2
ACTION_NAMES = {0: "BUY", 1: "HOLD", 2: "SELL"}

Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])


class DQNetwork(nn.Module):
    """Deep Q-Network with dueling architecture."""

    def __init__(self, state_size: int, action_size: int, hidden_layers: List[int]):
        super().__init__()

        layers = []
        prev_size = state_size
        for h in hidden_layers:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = h

        self.feature_layers = nn.Sequential(*layers)

        # Dueling streams
        self.value_stream = nn.Sequential(
            nn.Linear(prev_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, x):
        features = self.feature_layers(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        # Dueling DQN: Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
        q_values = value + advantages - advantages.mean(dim=1, keepdim=True)
        return q_values


class PrioritizedReplayBuffer:
    """Experience replay buffer with priority sampling."""

    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.position = 0

    def push(self, experience: Experience, priority: float = 1.0):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority ** self.alpha)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority ** self.alpha
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[Experience], np.ndarray, List[int]]:
        priorities = np.array(self.priorities, dtype=np.float64)
        probabilities = priorities / priorities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=False)
        experiences = [self.buffer[i] for i in indices]

        # Importance sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        return experiences, weights, list(indices)

    def update_priorities(self, indices: List[int], priorities: List[float]):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = (priority + 1e-6) ** self.alpha

    def __len__(self):
        return len(self.buffer)


class QLearningAgent:
    """
    Deep Q-Learning agent for stock trading.
    State = [technical_indicators, sentiment_features, portfolio_features]
    Actions = BUY (0), HOLD (1), SELL (2)

    Improvements from ML4T course:
      - Sharpe-ratio-weighted reward scaling
      - Dyna-style prioritised replay of high-value transitions
      - Ensemble (BagDQN) averaging for more robust action selection
    """

    def __init__(self, config):
        self.config = config.qlearning
        self.device = torch.device(
            config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu"
        )

        self.state_size = self.config.state_size
        self.action_size = self.config.action_size

        # Double DQN
        self.policy_net = DQNetwork(
            self.state_size, self.action_size, self.config.hidden_layers
        ).to(self.device)
        self.target_net = DQNetwork(
            self.state_size, self.action_size, self.config.hidden_layers
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=self.config.learning_rate
        )
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.95)

        self.memory = PrioritizedReplayBuffer(self.config.memory_size)
        self.epsilon = self.config.epsilon
        self.steps_done = 0
        self.episodes_done = 0
        self.training_losses: List[float] = []

        # Strategy weight learning
        self.strategy_weights = {}

        # --- ML4T: Dyna high-value buffer ---
        self._dyna_buffer: List[Experience] = []
        self._dyna_max_size = 5000
        self._dyna_replays_per_step = 3

        # --- ML4T: rolling reward stats for Sharpe scaling ---
        self._reward_history: List[float] = []
        self._reward_window = 200

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Epsilon-greedy action selection."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        self.policy_net.train()

        return int(q_values.argmax(dim=1).item())

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions given a state."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).cpu().numpy()[0]
        return q_values

    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer + Dyna high-value buffer."""
        # Sharpe-scale the reward
        reward = self._sharpe_scale_reward(reward)

        exp = Experience(state, action, reward, next_state, done)
        # Higher priority for experiences with larger reward magnitude
        priority = abs(reward) + 0.1
        self.memory.push(exp, priority)

        # Dyna: keep top transitions in a separate elite buffer
        if abs(reward) > 0.05 or done:
            if len(self._dyna_buffer) < self._dyna_max_size:
                self._dyna_buffer.append(exp)
            else:
                # Replace the lowest-magnitude entry
                min_idx = min(
                    range(len(self._dyna_buffer)),
                    key=lambda i: abs(self._dyna_buffer[i].reward),
                )
                if abs(reward) > abs(self._dyna_buffer[min_idx].reward):
                    self._dyna_buffer[min_idx] = exp

    def _sharpe_scale_reward(self, reward: float) -> float:
        """
        Scale reward by rolling Sharpe ratio so the agent optimises
        risk-adjusted returns, not raw P&L (ML4T insight).
        """
        self._reward_history.append(reward)
        if len(self._reward_history) > self._reward_window:
            self._reward_history = self._reward_history[-self._reward_window:]

        if len(self._reward_history) < 10:
            return reward

        arr = np.array(self._reward_history)
        std = arr.std()
        if std < 1e-8:
            return reward

        # Subtract rolling mean, scale by std → reward in units of volatility
        return float((reward - arr.mean()) / std)

    def learn(self) -> Optional[float]:
        """Perform one step of Q-learning."""
        if len(self.memory) < self.config.batch_size:
            return None

        experiences, weights, indices = self.memory.sample(self.config.batch_size)

        states = torch.FloatTensor(np.array([e.state for e in experiences])).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences])).to(self.device)
        dones = torch.FloatTensor([float(e.done) for e in experiences]).to(self.device)
        weights_tensor = torch.FloatTensor(weights).to(self.device)

        # Double DQN: use policy net to select action, target net to evaluate
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.config.discount_factor * next_q * (1 - dones)

        # Weighted Huber loss
        td_errors = (current_q - target_q).abs().detach().cpu().numpy()
        loss = (weights_tensor * F.smooth_l1_loss(current_q, target_q, reduction="none")).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Update priorities
        self.memory.update_priorities(indices, td_errors.tolist())

        self.steps_done += 1
        loss_val = loss.item()
        self.training_losses.append(loss_val)

        # Update target network periodically
        if self.steps_done % self.config.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # --- ML4T Dyna: replay high-value transitions ---
        self._dyna_replay()

        return loss_val

    def _dyna_replay(self):
        """Replay a few high-value transitions from the Dyna buffer (ML4T)."""
        if len(self._dyna_buffer) < self.config.batch_size:
            return
        for _ in range(self._dyna_replays_per_step):
            indices = np.random.choice(
                len(self._dyna_buffer),
                min(self.config.batch_size, len(self._dyna_buffer)),
                replace=False,
            )
            exps = [self._dyna_buffer[i] for i in indices]
            states = torch.FloatTensor(np.array([e.state for e in exps])).to(self.device)
            actions = torch.LongTensor([e.action for e in exps]).to(self.device)
            rewards = torch.FloatTensor([e.reward for e in exps]).to(self.device)
            next_states = torch.FloatTensor(np.array([e.next_state for e in exps])).to(self.device)
            dones = torch.FloatTensor([float(e.done) for e in exps]).to(self.device)

            current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_a = self.policy_net(next_states).argmax(dim=1)
                next_q = self.target_net(next_states).gather(1, next_a.unsqueeze(1)).squeeze(1)
                target_q = rewards + self.config.discount_factor * next_q * (1 - dones)

            loss = F.smooth_l1_loss(current_q, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)

    def train_on_historical(self, env, num_episodes: Optional[int] = None):
        """
        Pre-train on historical data using a trading environment.
        env must have reset() -> state and step(action) -> (next_state, reward, done, info).
        """
        num_episodes = num_episodes or self.config.num_episodes_pretrain
        logger.info(f"Pre-training agent for {num_episodes} episodes...")

        episode_rewards = []
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.select_action(state, training=True)
                next_state, reward, done, info = env.step(action)
                self.store_experience(state, action, reward, next_state, done)
                loss = self.learn()
                state = next_state
                total_reward += reward

            self.decay_epsilon()
            self.episodes_done += 1
            self.scheduler.step()
            episode_rewards.append(total_reward)

            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_loss = np.mean(self.training_losses[-100:]) if self.training_losses else 0
                logger.info(
                    f"Episode {episode + 1}/{num_episodes} | "
                    f"Avg Reward: {avg_reward:.2f} | "
                    f"Epsilon: {self.epsilon:.4f} | "
                    f"Avg Loss: {avg_loss:.6f}"
                )

        return episode_rewards

    def update_live(self, state, action, reward, next_state, done):
        """Real-time update during live trading."""
        self.store_experience(state, action, reward, next_state, done)
        loss = self.learn()
        if done:
            self.decay_epsilon()
        return loss

    def save(self, path: str):
        """Save agent state."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "steps_done": self.steps_done,
            "episodes_done": self.episodes_done,
            "strategy_weights": self.strategy_weights,
        }, path)
        logger.info(f"Agent saved to {path}")

    def load(self, path: str):
        """Load agent state."""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.policy_net.load_state_dict(checkpoint["policy_net"])
            self.target_net.load_state_dict(checkpoint["target_net"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.epsilon = checkpoint.get("epsilon", self.config.epsilon_min)
            self.steps_done = checkpoint.get("steps_done", 0)
            self.episodes_done = checkpoint.get("episodes_done", 0)
            self.strategy_weights = checkpoint.get("strategy_weights", {})
            logger.info(f"Agent loaded from {path}")
        else:
            logger.warning(f"Agent checkpoint not found: {path}")


class TradingEnvironment:
    """
    Gym-like trading environment for RL training.
    State: [technical_indicators, sentiment, portfolio_info]
    Actions: BUY(0), HOLD(1), SELL(2)
    """

    def __init__(self, df: pd.DataFrame, initial_capital: float = 10000.0,
                 state_size: int = 50, max_position_pct: float = 0.1,
                 commission: float = 0.001, impact: float = 0.001):
        self.df = df.reset_index(drop=True)
        self.initial_capital = initial_capital
        self.state_size = state_size
        self.max_position_pct = max_position_pct
        # ML4T marketsim-style transaction costs
        self.commission = commission   # 0.1% commission
        self.impact = impact           # 0.1% market impact

        self.feature_cols = [
            c for c in df.columns
            if c not in ["Date", "Open", "High", "Low", "Close", "Volume", "Adj Close"]
            and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
        ]

        # State
        self.current_step = 0
        self.cash = initial_capital
        self.shares = 0
        self.avg_buy_price = 0.0
        self.total_trades = 0
        self.winning_trades = 0

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_step = max(50, 0)  # Skip first 50 rows for indicator warmup
        self.cash = self.initial_capital
        self.shares = 0
        self.avg_buy_price = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and return (next_state, reward, done, info)."""
        current_price = self.df.iloc[self.current_step]["Close"]
        prev_portfolio = self.cash + self.shares * current_price

        reward = 0.0
        trade_info = {"action": ACTION_NAMES[action], "price": current_price}

        if action == ACTION_BUY and self.cash > 0:
            # Buy: invest up to max_position_pct of portfolio
            invest_amount = min(self.cash, prev_portfolio * self.max_position_pct)
            shares_to_buy = int(invest_amount / current_price)
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                # ML4T: transaction costs (commission + market impact)
                txn_cost = cost * self.commission + cost * self.impact
                total_cost = cost + txn_cost
                if total_cost > self.cash:
                    shares_to_buy = int(self.cash / (current_price * (1 + self.commission + self.impact)))
                    cost = shares_to_buy * current_price
                    txn_cost = cost * self.commission + cost * self.impact
                    total_cost = cost + txn_cost
                if shares_to_buy > 0:
                    self.avg_buy_price = (
                        (self.avg_buy_price * self.shares + cost) / (self.shares + shares_to_buy)
                        if self.shares > 0 else current_price
                    )
                    self.shares += shares_to_buy
                    self.cash -= total_cost
                    self.total_trades += 1
                    trade_info["shares_bought"] = shares_to_buy
                    trade_info["txn_cost"] = txn_cost
                    reward -= txn_cost / self.initial_capital  # Penalise costs

        elif action == ACTION_SELL and self.shares > 0:
            # Sell all shares
            revenue = self.shares * current_price
            txn_cost = revenue * self.commission + revenue * self.impact
            net_revenue = revenue - txn_cost
            profit = net_revenue - self.shares * self.avg_buy_price
            self.cash += net_revenue
            if profit > 0:
                self.winning_trades += 1
            self.shares = 0
            self.avg_buy_price = 0.0
            self.total_trades += 1
            trade_info["profit"] = profit
            trade_info["txn_cost"] = txn_cost
            reward += profit / self.initial_capital  # Normalize by initial capital

        # Advance step
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        if not done:
            next_price = self.df.iloc[self.current_step]["Close"]
            new_portfolio = self.cash + self.shares * next_price

            # Reward: portfolio change relative to initial capital
            portfolio_return = (new_portfolio - prev_portfolio) / self.initial_capital
            reward += portfolio_return

            # Penalty for holding cash too long if market is trending up
            if action == ACTION_HOLD and self.shares == 0:
                price_change = (next_price - current_price) / current_price
                if price_change > 0.01:  # Market went up > 1%
                    reward -= 0.001  # Small penalty for missing opportunity

        next_state = self._get_state() if not done else np.zeros(self.state_size)
        info = {
            "portfolio_value": self.cash + self.shares * (
                self.df.iloc[min(self.current_step, len(self.df) - 1)]["Close"]
            ),
            "cash": self.cash,
            "shares": self.shares,
            "total_trades": self.total_trades,
            "win_rate": self.winning_trades / max(self.total_trades, 1),
            **trade_info,
        }

        return next_state, reward, done, info

    def _get_state(self) -> np.ndarray:
        """Build state vector from current market data + portfolio state."""
        row = self.df.iloc[self.current_step]

        # Technical features (normalized)
        features = []
        for col in self.feature_cols:
            val = row[col]
            if np.isnan(val) or np.isinf(val):
                val = 0.0
            features.append(val)

        # Portfolio features
        current_price = row["Close"]
        portfolio_value = self.cash + self.shares * current_price
        features.extend([
            self.cash / self.initial_capital,  # Cash ratio
            (self.shares * current_price) / max(portfolio_value, 1),  # Position ratio
            portfolio_value / self.initial_capital - 1.0,  # Portfolio return
            (current_price - self.avg_buy_price) / max(self.avg_buy_price, 1) if self.shares > 0 else 0,
        ])

        # Pad or truncate to state_size
        state = np.array(features[:self.state_size], dtype=np.float32)
        if len(state) < self.state_size:
            state = np.pad(state, (0, self.state_size - len(state)))

        # Clip extreme values
        state = np.clip(state, -10, 10)
        return state


# Import pandas here for type annotation in TradingEnvironment
import pandas as pd
