# NSE RL Trader — Operations Guide

> Complete step-by-step guide to set up, train, trade, and automate the system.

---

## Table of Contents

1. [Quick-Start Checklist](#1-quick-start-checklist)
2. [Hardware Requirements](#2-hardware-requirements)
3. [Environment Setup](#3-environment-setup)
4. [Broker Setup & Paper Trading Account](#4-broker-setup--paper-trading-account)
5. [Phase 1 — Pre-training](#5-phase-1--pre-training)
6. [Phase 2 — Backtesting](#6-phase-2--backtesting)
7. [Phase 3 — Paper Trading](#7-phase-3--paper-trading)
8. [Phase 4 — Live Trading](#8-phase-4--live-trading)
9. [Daily Workflow](#9-daily-workflow)
10. [Automation](#10-automation)
11. [Time Estimates](#11-time-estimates)
12. [FAQ](#12-faq)

---

## 1. Quick-Start Checklist

```
[ ] Python 3.10+ installed
[ ] CUDA GPU available (recommended) OR CPU (slower)
[ ] pip install -r requirements.txt
[ ] OpenAlgo installed and running
[ ] Broker account created (Zerodha recommended)
[ ] API key generated from OpenAlgo dashboard
[ ] Pre-training completed (saved_models/ populated)
[ ] Backtest results validated
[ ] Paper trading run for 2+ weeks
[ ] Confident in results → switch to live
```

---

## 2. Hardware Requirements

### For Pre-training (Model Creation)

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | Not strictly required | NVIDIA GPU with 4+ GB VRAM (RTX 3060+) |
| **CPU** | 4-core, 2.5 GHz | 8-core, 3.5 GHz+ |
| **RAM** | 8 GB | 16 GB+ |
| **Disk** | 5 GB free | 20 GB SSD |

**Do you NEED a GPU?**
- **Q-Learning DQN**: The neural network is small (50 → 256 → 128 → 64 → 3). It trains fine on CPU. A GPU speeds it up ~3-5x but is not required.
- **FinBERT**: This is a BERT-class model (~110M parameters). Fine-tuning benefits greatly from a GPU. On CPU, initial FinBERT loading takes ~2 min and inference is ~0.5 sec/article (vs ~0.05 sec on GPU).
- **Bottom line**: You CAN pretrain entirely on CPU. It will take ~2-4 hours instead of ~30-60 minutes with a GPU.

The system auto-detects your hardware. The config has `device: "cuda"` by default — if no GPU is found, it falls back to CPU automatically (PyTorch handles this).

### For Live Trading

| Component | Requirement |
|-----------|-------------|
| **Machine** | Any modern PC/laptop (CPU is fine) |
| **Internet** | Stable broadband (for OpenAlgo + news fetching) |
| **Uptime** | Must run 9:00 AM – 3:30 PM IST continuously |

---

## 3. Environment Setup

### Step 1 — Install Python & Dependencies

```powershell
# Verify Python version
python --version   # Needs 3.10+

# Create virtual environment (recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Verify PyTorch + GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 2 — Set Environment Variables

```powershell
# OpenAlgo credentials
$env:OPENALGO_API_KEY = "your_api_key_here"
$env:OPENALGO_HOST = "http://127.0.0.1:5000"

# Telegram notifications (optional but recommended)
$env:TELEGRAM_BOT_TOKEN = "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
$env:TELEGRAM_CHAT_ID = "987654321"

# Or set permanently (System → Advanced → Environment Variables)
[System.Environment]::SetEnvironmentVariable("OPENALGO_API_KEY", "your_api_key", "User")
[System.Environment]::SetEnvironmentVariable("OPENALGO_HOST", "http://127.0.0.1:5000", "User")
[System.Environment]::SetEnvironmentVariable("TELEGRAM_BOT_TOKEN", "your_bot_token", "User")
[System.Environment]::SetEnvironmentVariable("TELEGRAM_CHAT_ID", "your_chat_id", "User")
```

### Step 3 — Set Up Telegram Notifications

The system sends real-time alerts to your Telegram for every trade:

1. **Create a bot**: Open Telegram → search `@BotFather` → `/newbot` → follow prompts → copy the **bot token**
2. **Get your chat ID**: Message `@userinfobot` or `@RawDataBot` on Telegram → it replies with your chat ID
3. Set the env vars as shown above

**Notifications you’ll receive:**
| Event | Example |
|-------|---------|
| 🟢 BUY executed | "🟢 BUY RELIANCE — Qty: 5 @ ₹2,854.30" |
| 🔴 SELL executed | "🔴 SELL TCS — P&L: ₹+1,230" |
| 🚻 SHORT placed | "🚻 SHORT HDFCBANK — Qty: 10 @ ₹1,678" |
| 🛑 Stop-loss hit | "🛑 STOP LOSS INFY — Loss: 2.0%" |
| 🎯 Take-profit hit | "🎯 TAKE PROFIT RELIANCE — Gain: 5.1%" |
| 🏁 End-of-day | Full portfolio summary with P&L, win rate, drawdown |

---

## 4. Broker Setup & Paper Trading Account

### Recommended Broker: **Zerodha** (Best Overall) or **Dhan** (Zero Cost)

**Why Zerodha?**
- Best-supported by OpenAlgo (most tested integration)
- ₹20/trade for intraday (the system accounts for this cost automatically)
- Robust API (Kite Connect)
- Largest broker in India — highest liquidity
- Very reliable login / token refresh

> **Brokerage Cost Note**: The system deducts ₹20 brokerage per trade (buy + sell = ₹40 round-trip) from your P&L automatically. This is factored into the RL reward signal so the agent learns to avoid low-margin trades that don’t cover costs.

### Zero/Low-Cost Broker Alternatives

| Broker | Brokerage | OpenAlgo Support | Best For | Notes |
|--------|-----------|------------------|----------|-------|
| **Dhan** | **₹0** (free intraday & delivery) | ✅ Supported | Cost-sensitive traders | Modern API, zero brokerage on all segments. Set `brokerage_per_trade: 0` in config. |
| **Finvasia / Shoonya** | **₹0** (truly zero) | ✅ Supported | Scalping, high-frequency | No brokerage, no hidden charges. Excellent for this RL system. |
| **MStock (Mirae Asset)** | **₹0** (delivery), ₹20 (intraday) | ✅ Supported | Delivery trades | Free delivery, intraday same as Zerodha |
| **Zerodha** | ₹20/trade | ✅ Best supported | Reliability | Most battle-tested OpenAlgo integration |
| **Angel One** | ₹20/trade | ✅ Supported | Beginners | Free 1st year on some plans |
| **Fyers** | ₹20/trade | ✅ Supported | Data feeds | Free real-time data included |

**Recommendation:**
- **For lowest cost**: Use **Dhan** or **Finvasia/Shoonya** (₹0 brokerage). Set `brokerage_per_trade: 0.0` in `config/settings.py`.
- **For reliability**: Use **Zerodha**. The ₹20/trade cost is already factored into the system.
- **Best of both**: Start with **Dhan** (free), switch to Zerodha if you face API stability issues.

**Other good options:**
| Broker | Pros | Cons |
|--------|------|------|
| **Dhan** | **Zero brokerage**, modern API | Newer, less community documentation |
| **Finvasia/Shoonya** | **Zero brokerage**, no hidden fees | UI is basic, but API works well |
| **Zerodha** | Best API, most stable | ₹20/trade |
| **Fyers** | Good API, free data feed | ₹20/trade |
| **Angel One** | Free for first year | API quirks with token expiry |
| **IIFL** | Good for F&O | Higher brokerage |

### Account Creation Steps (Zerodha)

1. **Open account** at [zerodha.com](https://zerodha.com) — takes 15 min online, activated in 2-3 days
2. **Fund your account** with minimum ₹500 (for paper trading you don't need real money, but the account must exist for API access)
3. **Enable API access**: Log into [kite.zerodha.com](https://kite.zerodha.com) → profile → get Kite Connect API key (₹2000/month for full API, but **OpenAlgo bypasses this** — you just need a regular trading account)

### OpenAlgo Setup (Paper Trading Bridge)

OpenAlgo acts as a **local broker bridge** — it logs into your broker and exposes a REST API. The **Analyzer mode** is OpenAlgo's built-in paper trading:

```powershell
# Clone and install OpenAlgo
git clone https://github.com/marketcalls/openalgo.git
cd openalgo
pip install -r requirements.txt

# Configure broker
copy .env.sample .env
# Edit .env → set BROKER_NAME=zerodha, BROKER_API_KEY, BROKER_API_SECRET

# Start OpenAlgo server
python app.py
# Opens at http://127.0.0.1:5000
```

4. Open http://127.0.0.1:5000 → Login with broker credentials
5. Copy the **API Key** from the dashboard
6. **Analyzer mode** (paper trading) is ON by default — trades are simulated with ₹1 Cr virtual capital

> **You do NOT need real money for paper trading.** OpenAlgo's Analyzer mode simulates everything locally.

---

## 5. Phase 1 — Pre-training

### What Happens

Pre-training downloads **30 years of daily OHLCV data** for NIFTY 50 stocks and trains the DQN agent through simulated episodes. This builds the initial trading policy.

### Command

```powershell
# Full pre-training (recommended first run)
python main.py --mode pretrain --episodes 100 --capital 10000

# Shorter test run (to verify setup)
python main.py --mode pretrain --episodes 10 --capital 10000 --symbols RELIANCE TCS INFY

# Custom capital matching your real allocation
python main.py --mode pretrain --episodes 100 --capital 500000
```

### What Gets Created

```
saved_models/
├── q_agent.pt                  # DQN weights (the main "brain")
├── strategy_selector.json      # Thompson Sampling weights (α/β per strategy)
└── finbert/                    # Fine-tuned FinBERT checkpoint
cache/
├── data/                       # Downloaded OHLCV data (parquet)
└── models/                     # HuggingFace FinBERT cache
```

### When to Re-train from Scratch

- After major code changes (new indicators, architecture changes)
- If the agent's performance degrades significantly
- After a major market regime shift (e.g., COVID-level disruption)
- Otherwise: **you do NOT need to re-pretrain**. The live trading mode handles continuous learning.

### Time Estimate

| Scenario | GPU | CPU |
|----------|-----|-----|
| 10 symbols × 10 episodes (test) | ~5 min | ~15 min |
| 20 symbols × 50 episodes | ~20 min | ~1 hr |
| 20 symbols × 100 episodes (full) | ~40 min | ~2-4 hrs |
| 50 symbols × 100 episodes (extended) | ~2 hrs | ~6-8 hrs |

> Data download time depends on network speed. First run downloads ~500 MB of historical data (cached for subsequent runs).

---

## 6. Phase 2 — Backtesting

### When to Backtest

- **After pre-training** — to validate the model before paper trading
- **Weekly** — to check if the model is still performing well on recent data
- **NOT daily** — backtesting on the same data every day doesn't add value

### Command

```powershell
# Backtest on last 2 years of data (default)
python main.py --mode backtest --years 2 --capital 10000

# Backtest on specific stocks
python main.py --mode backtest --symbols RELIANCE TCS HDFCBANK --years 3 --capital 500000

# Quick 6-month backtest
python main.py --mode backtest --years 1 --capital 10000
```

### Interpreting Results

```
BACKTEST RESULTS
Initial Capital: ₹10,000.00
Final Value:     ₹12,450.00
Total Return:    24.50%
Total Trades:    342
Win Rate:        58.2%            ← Aim for >55%
Max Drawdown:    6.3%             ← Keep below 10%
```

**Good results to look for:**
- Win rate > 55%
- Max drawdown < 10%
- Positive total return that beats NIFTY 50 benchmark
- If win rate < 50% or drawdown > 15% → re-pretrain with more episodes

### Time Estimate

| Scope | Time |
|-------|------|
| 10 stocks × 2 years | ~2-5 min |
| 50 stocks × 2 years | ~10-20 min |

---

## 7. Phase 3 — Paper Trading

### What Happens

Paper trading runs the **full live pipeline** (news → sentiment → indicators → Q-agent → orders) but all trades go to OpenAlgo's Analyzer mode (no real money). This validates:
- OpenAlgo connectivity
- News scraping reliability
- 5-minute scan cycle stability
- Model performance on live data
- Stop-loss/take-profit triggers

### Command

```powershell
# Start paper trading (--paper is default)
python main.py --mode trade --api-key YOUR_KEY --capital 500000 --paper
```

### How it Works During Market Hours

```
9:00 AM   System starts, waits for market open
9:15 AM   Market opens → loads models → begins trading loop
9:15      Fetch news → FinBERT sentiment → per-stock scores
9:20      5-min scan: rank all stocks → BUY/SHORT/HOLD
9:20      Q-agent decides on each signal → place paper orders
9:25      Next 5-min cycle begins...
...       (repeats every 5 minutes)
3:10 PM   Square off all positions
3:15 PM   Market close → end-of-day training
3:15      FinBERT fine-tuned with RL feedback
3:15      Strategy weights evolved (poor ones disabled)
3:15      Models saved to saved_models/
3:30 PM   System exits
```

### How Long to Paper Trade

**Minimum 2 weeks.** Ideally 1 month. You need to observe:
- Consistency across different market conditions (up/down/flat days)
- Net positive P&L over the observation period
- Win rate sustained above 55%
- No unexpected crashes or errors

### Do You Need to Re-train After Trading Hours?

**No. The `trade` mode handles everything automatically:**
1. During market hours: online RL updates after each trade (Q-network + replay buffer)
2. At market close (3:15 PM): end-of-day batch training
   - FinBERT fine-tuned with RL feedback from today's trades
   - Thompson Sampling strategy weights evolved
   - All models saved
3. Next morning: system loads the updated models and continues

**You do NOT need to manually run pretrain again.** The model improves itself every day.

---

## 8. Phase 4 — Live Trading

### Prerequisites

```
[✓] Pre-training completed
[✓] Backtest shows win rate > 55%, drawdown < 10%
[✓] Paper trading profitable for 2+ weeks
[✓] Broker account funded
[✓] Risk management parameters reviewed
```

### Command

```powershell
# Switch to live trading (real money!)
python main.py --mode trade --api-key YOUR_KEY --capital 500000 --no-paper
```

> **Warning**: `--no-paper` uses REAL money. Start small (₹10,000–50,000) and scale up only after proving profitability.

---

## 9. Daily Workflow

### Automated Daily Schedule

```
BEFORE MARKET (8:30 AM - 9:15 AM)
├── Start OpenAlgo server:  python app.py       (in openalgo directory)
├── Login to broker via OpenAlgo dashboard
├── Start trading system:   python main.py --mode trade --api-key KEY --paper
└── System waits for 9:15 AM automatically

DURING MARKET (9:15 AM - 3:15 PM)
├── System runs automatically (no manual intervention)
├── Monitor logs:  tail -f nse_rl_trader.log    (optional)
└── Run quick scan anytime:  python main.py --mode scan

AFTER MARKET (3:15 PM - 3:30 PM)
├── System auto-squares off all positions at 3:10 PM
├── End-of-day training runs automatically
├── Models saved automatically
└── System exits

WEEKLY (Saturday/Sunday)
├── Review performance: check nse_rl_trader.log
├── Run backtest on latest data:
│   python main.py --mode backtest --years 1 --capital 500000
└── If performance degraded badly → consider re-pretraining
```

### Do You Need to Re-train Daily?

**No.** The `--mode trade` loop handles continuous learning:
- **After each trade**: Q-network gets an online RL update (state, action, reward → replay buffer → mini-batch train)
- **At market close**: FinBERT fine-tuned, strategies evolved, models saved
- **Next day**: Updated models loaded automatically

**When you DO re-pretrain:**
- Model performance degrades over several weeks
- You add new indicators (like the PE ratio addition)
- Major code or architecture changes
- Market regime shift (pandemic, recession, etc.)

---

## 10. Automation

### Windows Task Scheduler

The system can be fully automated so you don't need to manually start it each morning. See `scripts/` folder for ready-made automation scripts:

| Script | Purpose |
|--------|---------|
| `scripts/start_trading.ps1` | Start OpenAlgo + trading system |
| `scripts/daily_trading.bat` | Batch file for Task Scheduler |
| `scripts/weekly_backtest.ps1` | Weekly backtest validation |
| `scripts/setup_scheduler.ps1` | Auto-configure Windows Task Scheduler |

### Setup Steps

1. Run `scripts/setup_scheduler.ps1` as Administrator
2. This creates two scheduled tasks:
   - **NSE_RL_DailyTrading** — runs at 8:45 AM every weekday (Mon-Fri)
   - **NSE_RL_WeeklyBacktest** — runs at 10:00 AM every Saturday
3. Edit the scripts to set your API key and paths

---

## 11. Time Estimates

| Operation | Duration | Hardware |
|-----------|----------|----------|
| **Initial setup** (install deps + download FinBERT) | 10-15 min | Any |
| **Data download** (30yr for 50 stocks, first time) | 5-15 min | Internet dependent |
| **Pre-training** (20 stocks × 100 episodes) | 30-60 min (GPU) / 2-4 hrs (CPU) | GPU recommended |
| **Backtest** (10 stocks × 2 years) | 2-5 min | Any |
| **Daily trading session** (9:15 AM – 3:15 PM) | 6 hours | Runs in background |
| **End-of-day training** (automatic) | 5-15 min | Any |
| **Weekly backtest** | 10-20 min | Any |
| **5-min scan** (one-shot) | 1-2 min | Any |
| **FinBERT sentiment** (per article) | 50ms (GPU) / 500ms (CPU) | GPU preferred |

---

## 12. FAQ

### Q: Do I need a GPU?
**A**: Not strictly. The DQN is small enough for CPU. FinBERT is slower on CPU but works. For faster pre-training, a GPU helps. For daily trading, CPU is fine — you're only running inference.

### Q: Can I run this on a cloud VM?
**A**: Yes. A ₹2000/month Azure/AWS VM with 4 vCPUs is sufficient for live trading. For pre-training, use a GPU instance temporarily.

### Q: What if my internet disconnects during trading?
**A**: The system will lose the current 5-min cycle. When it reconnects, it resumes. Open positions have stop-loss/take-profit set, so you're protected. OpenAlgo also has its own connection recovery.

### Q: How much money should I start with?
**A**: Paper trade with ₹10 lakh virtual capital first. For live, start with ₹10,000–50,000. Scale up only after 1 month of consistent profitability.

### Q: Which stocks does it trade?
**A**: By default, NIFTY 50. You can expand to NIFTY 500 or the full NSE universe, but more stocks = slower scans.

### Q: What does PE ratio add?
**A**: The PE ratio indicator adds a **fundamental valuation signal** alongside the existing technical signals. It flags stocks trading significantly below their rolling PE mean as undervalued (buy) and those above as overvalued (sell). This adds a longer-term valuation dimension to the system's primarily momentum/mean-reversion technical signals.

---

*End of Operations Guide*
