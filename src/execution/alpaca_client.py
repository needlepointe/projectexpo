"""
Alpaca trading client — wraps all order placement, position management,
and account queries. Handles both stocks and options.
"""

import logging
from datetime import datetime

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    GetOrdersRequest,
    ClosePositionRequest,
)
from alpaca.trading.models import Position, Order
from alpaca.trading.enums import (
    OrderSide,
    TimeInForce,
    QueryOrderStatus,
    AssetClass,
    OrderType,
)

from src.config import get_alpaca_credentials, is_paper_trading, get_config

logger = logging.getLogger(__name__)

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"


class AlpacaClient:
    def __init__(self):
        api_key, secret_key = get_alpaca_credentials()
        paper = is_paper_trading()
        url = PAPER_URL if paper else LIVE_URL
        self._client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper,
            url_override=url,
        )
        mode = "PAPER" if paper else "LIVE"
        logger.info("Alpaca client initialized [%s]", mode)

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_account(self) -> dict:
        acct = self._client.get_account()
        return {
            "equity": float(acct.equity),
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
            "portfolio_value": float(acct.portfolio_value),
            "day_trade_count": int(acct.daytrade_count or 0),
            "pattern_day_trader": acct.pattern_day_trader,
        }

    def get_cash(self) -> float:
        return float(self._client.get_account().cash)

    def get_equity(self) -> float:
        return float(self._client.get_account().equity)

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def get_positions(self) -> list[dict]:
        positions = self._client.get_all_positions()
        return [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "side": p.side.value,
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price or 0),
                "unrealized_pl": float(p.unrealized_pl or 0),
                "unrealized_plpc": float(p.unrealized_plpc or 0),
                "market_value": float(p.market_value or 0),
                "asset_class": p.asset_class.value if p.asset_class else "us_equity",
            }
            for p in positions
        ]

    def get_position(self, symbol: str) -> dict | None:
        try:
            p = self._client.get_open_position(symbol)
            return {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "side": p.side.value,
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price or 0),
                "unrealized_pl": float(p.unrealized_pl or 0),
                "unrealized_plpc": float(p.unrealized_plpc or 0),
            }
        except Exception:
            return None

    def has_position(self, symbol: str) -> bool:
        return self.get_position(symbol) is not None

    # ------------------------------------------------------------------
    # Stock orders
    # ------------------------------------------------------------------

    def buy_market(
        self,
        symbol: str,
        qty: int,
        stop_loss: float,
        take_profit: float,
        time_in_force: TimeInForce = TimeInForce.DAY,
    ) -> dict:
        """Place a market buy with bracket (stop-loss + take-profit)."""
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=time_in_force,
            order_class="bracket",
            stop_loss={"stop_price": round(stop_loss, 2)},
            take_profit={"limit_price": round(take_profit, 2)},
        )
        return self._submit(req)

    def sell_market(
        self,
        symbol: str,
        qty: int,
        stop_loss: float,
        take_profit: float,
        time_in_force: TimeInForce = TimeInForce.DAY,
    ) -> dict:
        """Place a market short-sell with bracket."""
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=time_in_force,
            order_class="bracket",
            stop_loss={"stop_price": round(stop_loss, 2)},
            take_profit={"limit_price": round(take_profit, 2)},
        )
        return self._submit(req)

    def buy_limit(
        self,
        symbol: str,
        qty: int,
        limit_price: float,
        stop_loss: float,
        take_profit: float,
        time_in_force: TimeInForce = TimeInForce.DAY,
    ) -> dict:
        req = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=time_in_force,
            limit_price=round(limit_price, 2),
            order_class="bracket",
            stop_loss={"stop_price": round(stop_loss, 2)},
            take_profit={"limit_price": round(take_profit, 2)},
        )
        return self._submit(req)

    def close_position(self, symbol: str) -> dict:
        """Immediately close an entire position at market."""
        try:
            result = self._client.close_position(symbol)
            logger.info("Closed position: %s", symbol)
            return self._order_to_dict(result)
        except Exception as exc:
            logger.error("Failed to close %s: %s", symbol, exc)
            return {"error": str(exc)}

    def close_all_positions(self) -> list[dict]:
        """Close every open position. Used by kill switch."""
        results = []
        for pos in self.get_positions():
            r = self.close_position(pos["symbol"])
            results.append(r)
        logger.warning("All positions closed.")
        return results

    def cancel_all_orders(self) -> list[str]:
        """Cancel all open orders."""
        cancelled = self._client.cancel_orders()
        ids = [str(o.id) for o in cancelled] if cancelled else []
        logger.info("Cancelled %d orders", len(ids))
        return ids

    # ------------------------------------------------------------------
    # Options orders
    # ------------------------------------------------------------------

    def buy_option(
        self,
        option_symbol: str,
        qty: int = 1,
        limit_price: float | None = None,
    ) -> dict:
        """Buy a call or put option contract."""
        if limit_price:
            req = LimitOrderRequest(
                symbol=option_symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                limit_price=round(limit_price, 2),
            )
        else:
            req = MarketOrderRequest(
                symbol=option_symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
        return self._submit(req)

    def close_option(self, option_symbol: str, qty: int) -> dict:
        req = MarketOrderRequest(
            symbol=option_symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        return self._submit(req)

    # ------------------------------------------------------------------
    # Order history
    # ------------------------------------------------------------------

    def get_recent_orders(self, limit: int = 50, status: str = "closed") -> list[dict]:
        req = GetOrdersRequest(
            status=QueryOrderStatus(status),
            limit=limit,
        )
        orders = self._client.get_orders(req)
        return [self._order_to_dict(o) for o in orders]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _submit(self, req) -> dict:
        try:
            order = self._client.submit_order(req)
            result = self._order_to_dict(order)
            logger.info(
                "Order submitted: %s %s %s qty=%s",
                result.get("side"), result.get("symbol"),
                result.get("type"), result.get("qty"),
            )
            return result
        except Exception as exc:
            logger.error("Order submission failed: %s", exc)
            raise

    def _order_to_dict(self, order: Order) -> dict:
        return {
            "id": str(order.id),
            "client_order_id": str(order.client_order_id),
            "symbol": order.symbol,
            "qty": float(order.qty or 0),
            "side": order.side.value if order.side else "",
            "type": order.type.value if order.type else "",
            "status": order.status.value if order.status else "",
            "filled_qty": float(order.filled_qty or 0),
            "filled_avg_price": float(order.filled_avg_price or 0),
            "limit_price": float(order.limit_price or 0),
            "stop_price": float(order.stop_price or 0),
            "submitted_at": str(order.submitted_at or ""),
        }
