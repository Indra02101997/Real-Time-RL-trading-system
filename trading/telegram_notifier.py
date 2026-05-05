"""
Telegram Notifier — Sends real-time trade alerts and daily summaries
to a Telegram chat via the Bot API.

Setup:
    1. Message @BotFather on Telegram → /newbot → copy the token
    2. Get your chat ID: message @userinfobot or @RawDataBot
    3. Set environment variables:
        TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
        TELEGRAM_CHAT_ID=987654321
"""
import logging
import os
from datetime import datetime
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Sends trade notifications and portfolio updates to Telegram."""

    BASE_URL = "https://api.telegram.org/bot{token}"

    def __init__(self, config):
        self.bot_token = config.telegram.bot_token
        self.chat_id = config.telegram.chat_id
        self.enabled = config.telegram.enabled and bool(self.bot_token) and bool(self.chat_id)

        if self.enabled:
            logger.info("Telegram notifications enabled.")
        else:
            logger.info("Telegram notifications disabled (no token/chat_id or disabled in config).")

    def _send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send a message via the Telegram Bot API."""
        if not self.enabled:
            return False

        url = f"{self.BASE_URL.format(token=self.bot_token)}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
        }

        try:
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                return True
            logger.warning(f"Telegram API returned {resp.status_code}: {resp.text}")
            return False
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False

    # ── Trade Alerts ─────────────────────────────────────────

    def notify_buy(self, symbol: str, quantity: int, price: float,
                   confidence: float = 0.0, sentiment: float = 0.0):
        """Notify on a BUY order execution."""
        text = (
            f"🟢 <b>BUY</b> {symbol}\n"
            f"Qty: {quantity} @ ₹{price:,.2f}\n"
            f"Cost: ₹{quantity * price:,.2f}\n"
            f"Confidence: {confidence:.2f} | Sentiment: {sentiment:+.2f}\n"
            f"⏰ {datetime.now().strftime('%I:%M %p')}"
        )
        self._send_message(text)

    def notify_sell(self, symbol: str, quantity: int, price: float,
                    pnl: float = 0.0, pnl_pct: float = 0.0,
                    reason: str = "signal"):
        """Notify on a SELL order execution."""
        emoji = "🔴" if pnl < 0 else "✅"
        pnl_emoji = "📉" if pnl < 0 else "📈"
        text = (
            f"{emoji} <b>SELL</b> {symbol}\n"
            f"Qty: {quantity} @ ₹{price:,.2f}\n"
            f"{pnl_emoji} P&L: ₹{pnl:,.2f} ({pnl_pct:+.2%})\n"
            f"Reason: {reason}\n"
            f"⏰ {datetime.now().strftime('%I:%M %p')}"
        )
        self._send_message(text)

    def notify_short(self, symbol: str, quantity: int, price: float):
        """Notify on a SHORT order execution."""
        text = (
            f"🔻 <b>SHORT</b> {symbol}\n"
            f"Qty: {quantity} @ ₹{price:,.2f}\n"
            f"⏰ {datetime.now().strftime('%I:%M %p')}"
        )
        self._send_message(text)

    def notify_stop_loss(self, symbol: str, price: float, loss_pct: float):
        """Notify on stop-loss trigger."""
        text = (
            f"🛑 <b>STOP LOSS</b> triggered for {symbol}\n"
            f"Price: ₹{price:,.2f} | Loss: {loss_pct:.2%}\n"
            f"⏰ {datetime.now().strftime('%I:%M %p')}"
        )
        self._send_message(text)

    def notify_take_profit(self, symbol: str, price: float, gain_pct: float):
        """Notify on take-profit trigger."""
        text = (
            f"🎯 <b>TAKE PROFIT</b> hit for {symbol}\n"
            f"Price: ₹{price:,.2f} | Gain: {gain_pct:.2%}\n"
            f"⏰ {datetime.now().strftime('%I:%M %p')}"
        )
        self._send_message(text)

    # ── Summaries ────────────────────────────────────────────

    def notify_market_open(self, capital: float, num_models_loaded: int):
        """Send market-open summary."""
        text = (
            f"📊 <b>Market Open — Trading Started</b>\n"
            f"Capital: ₹{capital:,.2f}\n"
            f"Models loaded: {num_models_loaded}\n"
            f"⏰ {datetime.now().strftime('%I:%M %p')}"
        )
        self._send_message(text)

    def notify_square_off(self, cash: float, portfolio_value: float, realized_pnl: float):
        """Notify that the system has squared off all intraday positions."""
        emoji = "📈" if realized_pnl >= 0 else "📉"
        text = (
            f"🏳️ <b>Intraday Square-Off Complete</b>\n"
            f"All positions closed. No carry-over.\n"
            f"Cash: ₹{cash:,.2f}\n"
            f"Portfolio Value: ₹{portfolio_value:,.2f}\n"
            f"{emoji} Today's Realized P&L: ₹{realized_pnl:,.2f}\n"
            f"⏰ {datetime.now().strftime('%I:%M %p')}"
        )
        self._send_message(text)

    def notify_end_of_day(self, summary: dict):
        """Send end-of-day portfolio summary."""
        pv = summary.get("portfolio_value", 0)
        ret = summary.get("total_return", 0)
        rpnl = summary.get("realized_pnl", 0)
        trades = summary.get("total_trades", 0)
        wr = summary.get("win_rate", 0)
        dd = summary.get("max_drawdown", 0)

        emoji = "📈" if ret >= 0 else "📉"
        text = (
            f"🏁 <b>End of Day Summary</b>\n"
            f"{'━' * 28}\n"
            f"Portfolio: ₹{pv:,.2f}\n"
            f"{emoji} Return: {ret:+.2%}\n"
            f"Realized P&L: ₹{rpnl:,.2f}\n"
            f"Trades: {trades} | Win Rate: {wr:.1%}\n"
            f"Max Drawdown: {dd:.2%}\n"
            f"⏰ {datetime.now().strftime('%I:%M %p')}"
        )
        self._send_message(text)

    def notify_error(self, error_msg: str):
        """Send an error alert."""
        text = (
            f"⚠️ <b>ERROR</b>\n"
            f"{error_msg[:500]}\n"
            f"⏰ {datetime.now().strftime('%I:%M %p')}"
        )
        self._send_message(text)
