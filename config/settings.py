"""
Configuration settings for NSE RL Trader.
"""
import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class OpenAlgoConfig:
    api_key: str = os.getenv("OPENALGO_API_KEY", "")
    host: str = os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")
    ws_url: str = os.getenv("OPENALGO_WS_URL", "ws://127.0.0.1:8765")
    exchange: str = "NSE"
    product: str = "MIS"  # MIS for intraday, CNC for delivery
    use_analyzer: bool = True  # Sandbox/paper trading mode


@dataclass
class TradingConfig:
    initial_capital: float = 10000.0
    max_position_pct: float = 0.10  # Max 10% of capital per stock
    max_positions: int = 20
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.05  # 5% take profit
    brokerage_per_trade: float = 20.0  # Broker commission per order (₹20 for Zerodha)
    trading_start_hour: int = 9
    trading_start_minute: int = 15
    trading_end_hour: int = 15
    trading_end_minute: int = 15
    square_off_hour: int = 15
    square_off_minute: int = 10


@dataclass
class QLearningConfig:
    learning_rate: float = 0.001
    discount_factor: float = 0.95
    epsilon: float = 1.0  # Exploration rate
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    batch_size: int = 64
    memory_size: int = 100000
    state_size: int = 50  # Feature vector size
    action_size: int = 3  # BUY, HOLD, SELL
    hidden_layers: List[int] = field(default_factory=lambda: [256, 128, 64])
    target_update_freq: int = 10
    num_episodes_pretrain: int = 100
    update_interval_minutes: int = 15


@dataclass
class FinBERTConfig:
    model_name: str = "ProsusAI/finbert"
    max_seq_length: int = 512
    batch_size: int = 16
    fine_tune_lr: float = 2e-5
    fine_tune_epochs: int = 3
    sentiment_weight: float = 0.3  # Weight of sentiment in state
    cache_dir: str = os.path.join(os.path.dirname(__file__), "..", "cache", "models")


@dataclass
class DataConfig:
    history_years: int = 30
    nse_symbols_url: str = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    cache_dir: str = os.path.join(os.path.dirname(__file__), "..", "cache", "data")
    historical_interval: str = "D"  # Daily for historical
    live_interval: str = "5m"  # 5-minute for live trading


@dataclass
class NewsConfig:
    sources: List[str] = field(default_factory=lambda: [
        "https://economictimes.indiatimes.com/markets/rss",
        "https://www.moneycontrol.com/rss/marketreports.xml",
        "https://www.livemint.com/rss/markets",
    ])
    google_news_query: str = "NSE stock market India"
    fetch_interval_minutes: int = 10
    max_articles_per_fetch: int = 50


@dataclass
class StrategyConfig:
    # Technical indicator parameters
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    stochastic_k: int = 14
    stochastic_d: int = 3
    sma_short: int = 20
    sma_long: int = 50
    ema_short: int = 12
    ema_long: int = 26
    atr_period: int = 14
    vwap_period: int = 20
    obv_period: int = 20
    # Strategy weights (learned via RL)
    enabled_strategies: List[str] = field(default_factory=lambda: [
        "rsi", "macd", "bollinger_bands", "stochastic",
        "sma_crossover", "ema_crossover", "vwap", "atr",
        "obv", "momentum", "mean_reversion", "cci", "pe_ratio"
    ])


@dataclass
class TelegramConfig:
    bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    enabled: bool = True  # Set False to disable even if token is present


@dataclass
class AppConfig:
    openalgo: OpenAlgoConfig = field(default_factory=OpenAlgoConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    qlearning: QLearningConfig = field(default_factory=QLearningConfig)
    finbert: FinBERTConfig = field(default_factory=FinBERTConfig)
    data: DataConfig = field(default_factory=DataConfig)
    news: NewsConfig = field(default_factory=NewsConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    log_level: str = "INFO"
    model_save_dir: str = os.path.join(os.path.dirname(__file__), "..", "saved_models")
    state_dir: str = os.path.join(os.path.dirname(__file__), "..", "state")
    device: str = "cuda"  # "cuda" or "cpu"
