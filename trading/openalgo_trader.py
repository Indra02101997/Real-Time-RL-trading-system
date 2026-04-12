"""
OpenAlgo Trading Integration.
Connects to OpenAlgo for order execution, portfolio management,
and real-time market data across all NSE stocks.
"""
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional

from openalgo import api as OpenAlgoAPI

logger = logging.getLogger(__name__)


class OpenAlgoTrader:
    """
    Interfaces with OpenAlgo for live/paper trading on NSE.
    Supports placing orders, checking positions, and managing portfolio.
    """

    def __init__(self, config):
        self.config = config
        self.client = OpenAlgoAPI(
            api_key=config.openalgo.api_key,
            host=config.openalgo.host,
        )
        self.exchange = config.openalgo.exchange
        self.product = config.openalgo.product

        # Enable analyzer (paper trading) mode if configured
        if config.openalgo.use_analyzer:
            try:
                self.client.analyzertoggle(mode=True)
                logger.info("OpenAlgo Analyzer (paper trading) mode ENABLED")
            except Exception as e:
                logger.warning(f"Could not enable analyzer mode: {e}")

        self._order_history: List[Dict] = []

    def place_buy_order(self, symbol: str, quantity: int, price_type: str = "MARKET",
                        price: float = 0, strategy: str = "rl_trader") -> Dict:
        """Place a BUY order."""
        try:
            params = {
                "symbol": symbol,
                "exchange": self.exchange,
                "action": "BUY",
                "quantity": quantity,
                "price_type": price_type,
                "product": self.product,
            }
            if price_type == "LIMIT" and price > 0:
                params["price"] = str(price)

            result = self.client.placeorder(**params)
            order_record = {
                "time": datetime.now().isoformat(),
                "symbol": symbol, "action": "BUY",
                "quantity": quantity, "price_type": price_type,
                "result": result, "strategy": strategy,
            }
            self._order_history.append(order_record)
            logger.info(f"BUY {quantity} {symbol} @ {price_type}: {result}")
            return result

        except Exception as e:
            logger.error(f"Buy order failed for {symbol}: {e}")
            return {"status": "error", "message": str(e)}

    def place_sell_order(self, symbol: str, quantity: int, price_type: str = "MARKET",
                         price: float = 0, strategy: str = "rl_trader") -> Dict:
        """Place a SELL order."""
        try:
            params = {
                "symbol": symbol,
                "exchange": self.exchange,
                "action": "SELL",
                "quantity": quantity,
                "price_type": price_type,
                "product": self.product,
            }
            if price_type == "LIMIT" and price > 0:
                params["price"] = str(price)

            result = self.client.placeorder(**params)
            order_record = {
                "time": datetime.now().isoformat(),
                "symbol": symbol, "action": "SELL",
                "quantity": quantity, "price_type": price_type,
                "result": result, "strategy": strategy,
            }
            self._order_history.append(order_record)
            logger.info(f"SELL {quantity} {symbol} @ {price_type}: {result}")
            return result

        except Exception as e:
            logger.error(f"Sell order failed for {symbol}: {e}")
            return {"status": "error", "message": str(e)}

    def place_smart_order(self, symbol: str, action: str, quantity: int,
                          position_size: int, price_type: str = "MARKET") -> Dict:
        """Place a smart order with position sizing via OpenAlgo."""
        try:
            result = self.client.placesmartorder(
                symbol=symbol,
                exchange=self.exchange,
                action=action,
                quantity=quantity,
                position_size=position_size,
                price_type=price_type,
                product=self.product,
            )
            logger.info(f"Smart order: {action} {symbol} qty={quantity} pos_size={position_size}: {result}")
            return result
        except Exception as e:
            logger.error(f"Smart order failed: {e}")
            return {"status": "error", "message": str(e)}

    def get_positions(self) -> Dict:
        """Get current open positions."""
        try:
            return self.client.positionbook()
        except Exception as e:
            logger.error(f"Position fetch error: {e}")
            return {"status": "error", "data": []}

    def get_holdings(self) -> Dict:
        """Get current holdings."""
        try:
            return self.client.holdings()
        except Exception as e:
            logger.error(f"Holdings fetch error: {e}")
            return {"status": "error", "data": []}

    def get_funds(self) -> Dict:
        """Get available funds."""
        try:
            return self.client.funds()
        except Exception as e:
            logger.error(f"Funds fetch error: {e}")
            return {"status": "error", "data": {}}

    def get_orderbook(self) -> Dict:
        """Get order book."""
        try:
            return self.client.orderbook()
        except Exception as e:
            logger.error(f"Orderbook fetch error: {e}")
            return {"status": "error", "data": []}

    def get_quote(self, symbol: str) -> Dict:
        """Get real-time quote for a symbol."""
        try:
            return self.client.quotes(symbol=symbol, exchange=self.exchange)
        except Exception as e:
            logger.error(f"Quote error for {symbol}: {e}")
            return {}

    def get_historical(self, symbol: str, interval: str = "5m",
                       start_date: str = "", end_date: str = ""):
        """Get historical data via OpenAlgo."""
        try:
            return self.client.history(
                symbol=symbol, exchange=self.exchange,
                interval=interval, start_date=start_date, end_date=end_date,
            )
        except Exception as e:
            logger.error(f"Historical data error for {symbol}: {e}")
            return None

    def close_all_positions(self) -> Dict:
        """Close all open positions (square off)."""
        try:
            result = self.client.closeposition()
            logger.info(f"Closed all positions: {result}")
            return result
        except Exception as e:
            logger.error(f"Close positions error: {e}")
            return {"status": "error", "message": str(e)}

    def cancel_all_orders(self) -> Dict:
        """Cancel all pending orders."""
        try:
            result = self.client.cancelallorder()
            logger.info(f"Cancelled all orders: {result}")
            return result
        except Exception as e:
            logger.error(f"Cancel orders error: {e}")
            return {"status": "error", "message": str(e)}

    def search_symbol(self, query: str) -> Dict:
        """Search for symbols."""
        try:
            return self.client.search(query=query, exchange=self.exchange)
        except Exception as e:
            logger.error(f"Search error: {e}")
            return {"status": "error", "data": []}

    def get_open_position(self, symbol: str) -> Dict:
        """Get open position for a specific symbol."""
        try:
            return self.client.openposition(
                symbol=symbol, exchange=self.exchange, product=self.product
            )
        except Exception as e:
            logger.error(f"Open position error for {symbol}: {e}")
            return {}

    def get_order_history(self) -> List[Dict]:
        """Get local order history."""
        return self._order_history
