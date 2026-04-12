# NSE RL Trader — FinBERT + Deep Q-Learning Trading System

A reinforcement learning–based algorithmic trading system for the **National Stock Exchange (NSE)** that combines **FinBERT** sentiment analysis with **Deep Q-Learning** and integrates with **OpenAlgo** for trade execution.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     NSE RL TRADER                           │
├─────────────┬─────────────┬─────────────┬──────────────────┤
│  News       │  FinBERT    │  Technical  │  Q-Learning      │
│  Scraper    │  Sentiment  │  Indicators │  Agent (DQN)     │
│  (RSS,      │  Analysis   │  RSI, MACD  │  Double DQN +    │
│  Google)    │  + RL       │  BB, Stoch  │  Prioritized     │
│             │  Fine-tune  │  SMA, EMA   │  Replay          │
├─────────────┴─────────────┴──────┬──────┴──────────────────┤
│              Strategy Selector   │  Portfolio Manager       │
│         (Thompson Sampling)      │  (Risk Management)       │
├──────────────────────────────────┴─────────────────────────┤
│                    OpenAlgo Integration                     │
│              (NSE Order Execution via REST API)             │
└────────────────────────────────────────────────────────────┘
```

## Components

| Module | File | Description |
|--------|------|-------------|
| **Config** | `config/settings.py` | All configuration dataclasses |
| **NSE Data** | `data/nse_data_collector.py` | 30-year historical + live data via yfinance & OpenAlgo |
| **News Scraper** | `data/news_scraper.py` | RSS feeds, Google News, entity extraction |
| **FinBERT** | `models/finbert_sentiment.py` | Financial sentiment with RL feedback fine-tuning |
| **Q-Learning** | `models/q_learning_agent.py` | Double DQN with dueling architecture + trading environment |
| **Strategy Selector** | `models/strategy_selector.py` | Thompson Sampling for optimal indicator weighting |
| **Technical Indicators** | `indicators/technical_indicators.py` | RSI, MACD, Bollinger, Stochastic, SMA/EMA, ATR, VWAP, OBV |
| **OpenAlgo Trader** | `trading/openalgo_trader.py` | OpenAlgo API integration for order execution |
| **Stock Scanner** | `trading/stock_scanner.py` | 5-min profitability scanner with BUY/SHORT/HOLD signals |
| **Portfolio Manager** | `trading/portfolio_manager.py` | Position tracking, risk management, P&L |
| **Trainer** | `training/trainer.py` | Pre-training + continuous online learning |
| **Main** | `main.py` | Orchestrator with CLI |

## Setup

### 1. Install Python Dependencies

```bash
cd nse_rl_trader

# Install dependencies
pip install -r requirements.txt

# Set environment variables
set OPENALGO_API_KEY=your_api_key_here
set OPENALGO_HOST=http://127.0.0.1:5000
```

### 2. Set Up OpenAlgo (Broker Bridge)

OpenAlgo is a locally-hosted bridge that connects this system to your NSE broker. It exposes a unified REST API so the RL trader can place orders, fetch quotes, and manage positions.

#### Step 1 — Install OpenAlgo

```bash
# Clone the OpenAlgo repo
git clone https://github.com/marketcalls/openalgo.git
cd openalgo

# Install dependencies (Python 3.10+)
pip install -r requirements.txt
```

#### Step 2 — Configure your broker

1. Copy the sample config and edit it:
   ```bash
   copy .env.sample .env        # Windows
   # cp .env.sample .env        # Linux/macOS
   ```
2. Open `.env` and set your broker credentials:
   ```env
   BROKER_API_KEY=your_broker_api_key
   BROKER_API_SECRET=your_broker_secret
   BROKER_NAME=zerodha           # or fyers, angel, dhan, etc.
   ```
3. Supported brokers: **Zerodha, Fyers, Angel One, Dhan, Finvasia, IIFL, Kotak Neo, Upstox, 5Paisa, AliceBlue** and more. See [OpenAlgo docs](https://docs.openalgo.in) for the full list.

#### Step 3 — Start the OpenAlgo server

```bash
python app.py
# Server starts at http://127.0.0.1:5000
```

#### Step 4 — Generate an API key

1. Open `http://127.0.0.1:5000` in your browser.
2. Log in with your broker credentials (redirects to broker login).
3. After broker auth, copy the **API Key** shown on the dashboard.

#### Step 5 — Connect the RL Trader

```bash
# Option A: environment variables
set OPENALGO_API_KEY=<your_api_key>
set OPENALGO_HOST=http://127.0.0.1:5000

# Option B: CLI flags
python main.py --mode trade --api-key <your_api_key> --host http://127.0.0.1:5000 --paper
```

#### Paper Trading (Analyzer Mode)

By default the system runs in **Analyzer (paper) mode** with ₹1 Cr virtual capital. To switch to live trading, pass `--no-paper`:
```bash
python main.py --mode trade --api-key <key> --no-paper
```

> **Warning**: Live trading involves real money. Always validate with paper mode first.

## Usage

### 1. Pre-train on 30 Years of Historical Data

```bash
python main.py --mode pretrain --episodes 100 --capital 10000
```

This will:
- Download 30 years of OHLCV data for NIFTY 50 stocks via yfinance
- Compute all technical indicators (RSI, MACD, BB, Stochastic, etc.)
- Train the Deep Q-Learning agent across multiple stocks
- Optimize strategy weights using Thompson Sampling
- Save trained models to `saved_models/`

### 2. Run Live Trading via OpenAlgo

```bash
python main.py --mode trade --api-key YOUR_KEY --capital 10000 --paper
```

This will:
- Load pre-trained models
- Connect to OpenAlgo (paper/analyzer mode by default)
- During NSE hours (9:15 AM – 3:15 PM):
  - Fetch news → Analyze sentiment with FinBERT
  - Compute technical indicators for all NSE stocks
  - Q-Learning agent decides BUY/HOLD/SELL for each stock
  - Execute trades via OpenAlgo API
  - Update models in real-time after each trade
  - Check stop-loss/take-profit continuously
- At market close:
  - Square off all positions
  - Run end-of-day model fine-tuning
  - FinBERT fine-tuned with RL feedback from today's trades
  - Strategy weights evolved (poor strategies disabled)

### 3. Backtest

```bash
python main.py --mode backtest --years 2 --capital 10000
```

### 4. Run 5-Minute Profitability Scanner

```bash
python main.py --mode scan
```

This gives a one-shot scan of all NSE stocks and prints a ranked table:

```
      Symbol Action   Score      Price    Chg%   Vol   RSI  Sharpe   Conf
----------------------------------------------------------------------
    RELIANCE    BUY  +0.612    2854.30  +1.23%   2.1  34.2   +1.87   HIGH
         TCS  SHORT  -0.530    3432.10  -0.87%   1.8  74.5   -1.23 MEDIUM
    HDFCBANK   HOLD  +0.120    1678.50  +0.04%   0.9  52.1   +0.32    LOW
```

**During live trading (`--mode trade`)**, the scanner runs automatically every 5 minutes:
- **BUY** signals trigger long entries
- **SHORT** signals trigger sell-first MIS intraday positions
- **HOLD** signals are skipped
- Positions are ranked by a composite score that blends technical signals, FinBERT sentiment, volume confirmation, and rolling Sharpe ratio

## How It Works

### Reinforcement Learning (Q-Learning)

- **State**: 50-dimensional vector containing:
  - Technical indicator values (RSI, MACD histogram, BB%, Stochastic K/D, etc.)
  - Technical signals (-1/0/+1 from each strategy)
  - FinBERT sentiment score for the stock
  - Portfolio state (cash ratio, position exposure, P&L, returns)

- **Actions**: BUY (0), HOLD (1), SELL (2)

- **Reward**: Sharpe-ratio-scaled P&L from completed trades + portfolio value change (penalised by drawdown)

- **Architecture**: Double DQN with Dueling heads, Prioritized Experience Replay, and Dyna-style high-value replay buffer

### FinBERT Sentiment Loop

1. News scraped from RSS feeds + Google News every 10 minutes
2. FinBERT classifies each article as positive/negative/neutral
3. Per-stock sentiment aggregated from entity-matched articles
4. Sentiment score fed into Q-Learning state vector
5. After trades complete, reward signal fed back to FinBERT:
   - Profitable trade + positive sentiment → reinforce
   - Loss + positive sentiment → adjust labels

### Strategy Selection (Thompson Sampling)

The system learns which combination of technical indicators works best:

| Strategy | Signal Source |
|----------|-------------|
| RSI | Overbought/Oversold levels |
| MACD | Signal line crossovers |
| Bollinger Bands | Price touching bands |
| Stochastic | K/D crossover at extremes |
| SMA Crossover | Golden/Death cross |
| EMA Crossover | Fast/slow EMA cross |
| VWAP | Price vs VWAP |
| ATR | Volatility breakout |
| OBV | Volume trend confirmation |
| Momentum | 10/20-day price momentum |
| CCI | Commodity Channel Index mean-reversion |

Weights are sampled from Beta distributions and updated based on trade outcomes. Consistently poor strategies are automatically disabled.

### OpenAlgo Integration

- Uses OpenAlgo Python SDK for unified NSE broker access
- Supports **Analyzer Mode** (paper trading with ₹1Cr virtual capital)
- Places MARKET orders via `placeorder()` API
- Monitors positions via `positionbook()` and `quotes()`
- Square-off via `closeposition()` at end of day

## Configuration

Edit `config/settings.py` or pass CLI arguments:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_capital` | ₹10,000 | Starting capital |
| `max_position_pct` | 10% | Max allocation per stock |
| `max_positions` | 20 | Max concurrent positions |
| `stop_loss_pct` | 2% | Stop loss trigger |
| `take_profit_pct` | 5% | Take profit trigger |
| `epsilon` | 1.0 → 0.01 | Exploration rate (decays) |
| `learning_rate` | 0.001 | Q-network learning rate |
| `discount_factor` | 0.95 | Future reward discount |
| `sentiment_weight` | 0.3 | FinBERT weight in state |

## Directory Structure

```
nse_rl_trader/
├── config/
│   ├── __init__.py
│   └── settings.py              # All configuration
├── data/
│   ├── __init__.py
│   ├── nse_data_collector.py    # Historical + live NSE data
│   └── news_scraper.py          # News from RSS/Google
├── models/
│   ├── __init__.py
│   ├── finbert_sentiment.py     # FinBERT + RL fine-tuning
│   ├── q_learning_agent.py      # Double DQN agent + environment
│   └── strategy_selector.py     # Thompson Sampling optimizer
├── indicators/
│   ├── __init__.py
│   └── technical_indicators.py  # All technical indicators
├── trading/
│   ├── __init__.py
│   ├── openalgo_trader.py       # OpenAlgo API wrapper
│   ├── stock_scanner.py         # 5-min profitability scanner
│   └── portfolio_manager.py     # Portfolio + risk management
├── training/
│   ├── __init__.py
│   └── trainer.py               # Training orchestration
├── main.py                      # CLI entry point
├── requirements.txt
└── README.md
```

## Disclaimer

This software is for **educational and research purposes only**. Do not risk money you cannot afford to lose. Always test in Analyzer/paper mode before any live deployment. Past performance does not guarantee future results. Trading involves substantial risk of loss.
