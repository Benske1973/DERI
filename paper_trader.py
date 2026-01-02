# paper_trader.py - Paper Trading Engine
"""
Complete paper trading engine with:
- Virtual portfolio management
- Position tracking
- Order execution simulation
- P&L calculation
- Trade history and statistics
"""

import time
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

from config import config, SignalType, OrderStatus, PositionStatus

logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Order types."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"

class OrderSide(Enum):
    """Order side."""
    BUY = "BUY"
    SELL = "SELL"

@dataclass
class Order:
    """Order structure."""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    filled_quantity: float = 0
    parent_position_id: Optional[str] = None

@dataclass
class Position:
    """Trading position."""
    id: str
    symbol: str
    side: str  # LONG or SHORT
    entry_price: float
    quantity: float
    current_price: float = 0
    stop_loss: float = 0
    take_profit: float = 0
    trailing_stop: Optional[float] = None
    trailing_activation: Optional[float] = None
    status: str = "OPEN"
    opened_at: datetime = field(default_factory=datetime.now)
    closed_at: Optional[datetime] = None
    close_price: Optional[float] = None
    close_reason: Optional[str] = None
    pnl: float = 0
    pnl_percent: float = 0
    highest_price: float = 0  # For trailing stop
    lowest_price: float = float('inf')  # For trailing stop
    partial_closes: List[Dict] = field(default_factory=list)

    @property
    def value(self) -> float:
        """Current position value."""
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized P&L."""
        if self.side == "LONG":
            return (self.current_price - self.entry_price) * self.quantity
        else:  # SHORT
            return (self.entry_price - self.current_price) * self.quantity

    @property
    def unrealized_pnl_percent(self) -> float:
        """Unrealized P&L percentage."""
        if self.entry_price == 0:
            return 0
        return (self.unrealized_pnl / (self.entry_price * self.quantity)) * 100

@dataclass
class Portfolio:
    """Virtual portfolio."""
    initial_capital: float
    balance: float  # Available balance
    equity: float   # Balance + unrealized P&L
    margin_used: float = 0
    positions: Dict[str, Position] = field(default_factory=dict)
    orders: Dict[str, Order] = field(default_factory=dict)
    trade_history: List[Position] = field(default_factory=list)

    @property
    def total_unrealized_pnl(self) -> float:
        """Total unrealized P&L across all positions."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())

    @property
    def total_pnl(self) -> float:
        """Total realized P&L (from closed trades only)."""
        return sum(trade.pnl for trade in self.trade_history)

    @property
    def total_pnl_including_unrealized(self) -> float:
        """Total P&L including unrealized."""
        return self.total_pnl + self.total_unrealized_pnl

    @property
    def open_positions_count(self) -> int:
        """Number of open positions."""
        return len(self.positions)


class PaperTrader:
    """
    Paper Trading Engine.

    Simulates trading without using real funds.
    """

    def __init__(self):
        self.portfolio = Portfolio(
            initial_capital=config.risk.initial_capital,
            balance=config.risk.initial_capital,
            equity=config.risk.initial_capital
        )
        self.cfg = config.risk
        self._price_cache: Dict[str, float] = {}
        self._total_fees_paid: float = 0.0

    def calculate_entry_price_with_costs(self, price: float, side: str) -> float:
        """
        Calculate effective entry price including spread and slippage.

        For LONG: price goes UP (worse entry)
        For SHORT: price goes DOWN (worse entry)
        """
        spread_cost = price * self.cfg.spread_percent
        slippage_cost = price * self.cfg.slippage_percent

        if side == "LONG":
            # Buying: pay higher price
            return price + spread_cost + slippage_cost
        else:
            # Selling/Shorting: get lower price
            return price - spread_cost - slippage_cost

    def calculate_exit_price_with_costs(self, price: float, side: str) -> float:
        """
        Calculate effective exit price including spread and slippage.

        For closing LONG: price goes DOWN (worse exit)
        For closing SHORT: price goes UP (worse exit)
        """
        spread_cost = price * self.cfg.spread_percent
        slippage_cost = price * self.cfg.slippage_percent

        if side == "LONG":
            # Selling to close long: get lower price
            return price - spread_cost - slippage_cost
        else:
            # Buying to close short: pay higher price
            return price + spread_cost + slippage_cost

    def calculate_fee(self, value: float, is_maker: bool = False) -> float:
        """Calculate trading fee."""
        fee_rate = self.cfg.maker_fee if is_maker else self.cfg.taker_fee
        return value * fee_rate

    def update_price(self, symbol: str, price: float):
        """Update price for a symbol and check orders/positions."""
        self._price_cache[symbol] = price

        # Update positions
        if symbol in self.portfolio.positions:
            self._update_position(symbol, price)

        # Check pending orders
        self._check_orders(symbol, price)

        # Update equity
        self._update_equity()

    def update_prices(self, prices: Dict[str, float]):
        """Batch update prices."""
        for symbol, price in prices.items():
            self.update_price(symbol, price)

        self._update_equity()

    def _update_equity(self):
        """Update portfolio equity."""
        # Equity = available balance + margin used (position values) + unrealized P&L
        self.portfolio.equity = (
            self.portfolio.balance +
            self.portfolio.margin_used +
            self.portfolio.total_unrealized_pnl
        )

    def _update_position(self, symbol: str, price: float):
        """Update position with new price."""
        position = self.portfolio.positions.get(symbol)
        if not position:
            return

        position.current_price = price

        # Track highest/lowest for trailing stop
        if position.side == "LONG":
            position.highest_price = max(position.highest_price, price)
        else:
            position.lowest_price = min(position.lowest_price, price)

        # Check stop loss
        if position.stop_loss:
            if position.side == "LONG" and price <= position.stop_loss:
                self.close_position(symbol, price, "STOP_LOSS")
                return
            elif position.side == "SHORT" and price >= position.stop_loss:
                self.close_position(symbol, price, "STOP_LOSS")
                return

        # Check take profit
        if position.take_profit:
            if position.side == "LONG" and price >= position.take_profit:
                self.close_position(symbol, price, "TAKE_PROFIT")
                return
            elif position.side == "SHORT" and price <= position.take_profit:
                self.close_position(symbol, price, "TAKE_PROFIT")
                return

        # Check trailing stop
        if position.trailing_stop and position.trailing_activation:
            self._check_trailing_stop(position, price)

    def _check_trailing_stop(self, position: Position, price: float):
        """Check and update trailing stop."""
        risk = abs(position.entry_price - position.stop_loss)
        activation_price = position.entry_price + (risk * self.cfg.trailing_activation) \
            if position.side == "LONG" else \
            position.entry_price - (risk * self.cfg.trailing_activation)

        # Check if trailing is activated
        if position.side == "LONG":
            if price >= activation_price:
                # Trail from highest price
                new_stop = position.highest_price - (risk * self.cfg.trailing_distance)
                if new_stop > position.stop_loss:
                    position.stop_loss = new_stop
                    position.trailing_stop = new_stop
                    logger.info(f"Trailing stop updated for {position.symbol}: {new_stop:.4f}")
        else:  # SHORT
            if price <= activation_price:
                new_stop = position.lowest_price + (risk * self.cfg.trailing_distance)
                if new_stop < position.stop_loss:
                    position.stop_loss = new_stop
                    position.trailing_stop = new_stop
                    logger.info(f"Trailing stop updated for {position.symbol}: {new_stop:.4f}")

    def _check_orders(self, symbol: str, price: float):
        """Check and execute pending orders."""
        orders_to_fill = []

        for order_id, order in self.portfolio.orders.items():
            if order.symbol != symbol or order.status != OrderStatus.PENDING:
                continue

            should_fill = False

            if order.order_type == OrderType.MARKET:
                should_fill = True
            elif order.order_type == OrderType.LIMIT:
                if order.side == OrderSide.BUY and price <= order.price:
                    should_fill = True
                elif order.side == OrderSide.SELL and price >= order.price:
                    should_fill = True
            elif order.order_type == OrderType.STOP_LOSS:
                if order.stop_price:
                    if order.side == OrderSide.SELL and price <= order.stop_price:
                        should_fill = True
                    elif order.side == OrderSide.BUY and price >= order.stop_price:
                        should_fill = True

            if should_fill:
                orders_to_fill.append((order_id, order, price))

        for order_id, order, fill_price in orders_to_fill:
            self._fill_order(order, fill_price)

    def _fill_order(self, order: Order, fill_price: float):
        """Fill an order."""
        order.status = OrderStatus.FILLED
        order.filled_at = datetime.now()
        order.filled_price = fill_price
        order.filled_quantity = order.quantity

        logger.info(
            f"Order filled: {order.side.value} {order.quantity} {order.symbol} "
            f"@ {fill_price:.4f}"
        )

    def calculate_position_size(self, symbol: str, entry_price: float,
                                 stop_loss: float) -> float:
        """
        Calculate position size based on risk parameters.

        Args:
            symbol: Trading pair
            entry_price: Entry price
            stop_loss: Stop loss price

        Returns:
            Position size (quantity)
        """
        # Risk amount in USDT
        risk_amount = self.portfolio.equity * self.cfg.risk_per_trade

        # Distance to stop loss
        sl_distance = abs(entry_price - stop_loss)

        if sl_distance == 0:
            return 0

        # Position size based on risk
        position_size = risk_amount / sl_distance

        # Max position size constraint
        max_size = (self.portfolio.equity * self.cfg.max_position_size) / entry_price
        position_size = min(position_size, max_size)

        # Check available balance
        max_affordable = self.portfolio.balance / entry_price
        position_size = min(position_size, max_affordable)

        return position_size

    def can_open_position(self, symbol: str) -> bool:
        """Check if we can open a new position."""
        # Check max positions
        current_positions = len(self.portfolio.positions)
        logger.info(f"Position check: {current_positions}/{self.cfg.max_open_positions} positions open: {list(self.portfolio.positions.keys())}")

        if current_positions >= self.cfg.max_open_positions:
            logger.warning(f"Max open positions reached: {current_positions}/{self.cfg.max_open_positions}")
            return False

        # Check if already has position in this symbol
        if symbol in self.portfolio.positions:
            logger.warning(f"Already have position in {symbol}")
            return False

        # Check available balance
        if self.portfolio.balance < self.portfolio.equity * 0.1:  # Need at least 10%
            logger.warning(f"Insufficient balance: {self.portfolio.balance:.2f} < {self.portfolio.equity * 0.1:.2f}")
            return False

        return True

    def open_position(self, symbol: str, side: str, entry_price: float,
                      stop_loss: float, take_profit: float = None,
                      quantity: float = None) -> Optional[Position]:
        """
        Open a new position.

        Args:
            symbol: Trading pair
            side: LONG or SHORT
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price (optional)
            quantity: Position size (optional, calculated if not provided)

        Returns:
            Position object or None
        """
        if not self.can_open_position(symbol):
            return None

        # Calculate position size if not provided
        if quantity is None:
            quantity = self.calculate_position_size(symbol, entry_price, stop_loss)

        if quantity <= 0:
            logger.warning(f"Invalid position size for {symbol}")
            return None

        # Check R:R ratio
        if take_profit is None:
            risk = abs(entry_price - stop_loss)
            if side == "LONG":
                take_profit = entry_price + (risk * self.cfg.default_risk_reward)
            else:
                take_profit = entry_price - (risk * self.cfg.default_risk_reward)

        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        rr_ratio = reward / risk if risk > 0 else 0

        logger.info(f"Position sizing for {symbol}: Entry={entry_price:.4f}, SL={stop_loss:.4f}, TP={take_profit:.4f}, R:R={rr_ratio:.2f}, Qty={quantity:.4f}")

        if rr_ratio < self.cfg.min_risk_reward:
            logger.warning(f"R:R ratio too low for {symbol}: {rr_ratio:.2f} < {self.cfg.min_risk_reward}")
            return None

        # Apply spread and slippage to entry price
        effective_entry = self.calculate_entry_price_with_costs(entry_price, side)

        # Calculate entry fee
        position_value = quantity * effective_entry
        entry_fee = self.calculate_fee(position_value)

        # Create position
        position = Position(
            id=str(uuid.uuid4())[:8],
            symbol=symbol,
            side=side,
            entry_price=effective_entry,  # Use effective entry with costs
            quantity=quantity,
            current_price=effective_entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            highest_price=effective_entry,
            lowest_price=effective_entry
        )

        # Deduct from balance (position value + entry fee)
        total_cost = position_value + entry_fee
        self.portfolio.balance -= total_cost
        self.portfolio.margin_used += position_value
        self._total_fees_paid += entry_fee

        # Store position
        self.portfolio.positions[symbol] = position

        logger.info(
            f"POSITION OPENED: {side} {quantity:.4f} {symbol} @ {effective_entry:.6f} "
            f"(raw: {entry_price:.6f}) | SL: {stop_loss:.6f} | TP: {take_profit:.6f} | "
            f"R:R: {rr_ratio:.2f} | Fee: ${entry_fee:.2f}"
        )

        return position

    def close_position(self, symbol: str, close_price: float = None,
                       reason: str = "MANUAL") -> Optional[Position]:
        """
        Close a position.

        Args:
            symbol: Trading pair
            close_price: Close price (uses cached price if not provided)
            reason: Close reason

        Returns:
            Closed position or None
        """
        position = self.portfolio.positions.get(symbol)
        if not position:
            logger.warning(f"No position found for {symbol}")
            return None

        if close_price is None:
            close_price = self._price_cache.get(symbol, position.current_price)

        # Apply spread and slippage to exit price
        effective_exit = self.calculate_exit_price_with_costs(close_price, position.side)

        # Calculate exit fee
        exit_value = position.quantity * effective_exit
        exit_fee = self.calculate_fee(exit_value)

        # Calculate P&L (using effective exit price, minus exit fee)
        if position.side == "LONG":
            gross_pnl = (effective_exit - position.entry_price) * position.quantity
        else:
            gross_pnl = (position.entry_price - effective_exit) * position.quantity

        # Net P&L after exit fee
        pnl = gross_pnl - exit_fee
        pnl_percent = (pnl / (position.entry_price * position.quantity)) * 100

        # Track fees
        self._total_fees_paid += exit_fee

        # Update position
        position.status = "CLOSED"
        position.closed_at = datetime.now()
        position.close_price = effective_exit
        position.close_reason = reason
        position.pnl = pnl
        position.pnl_percent = pnl_percent

        # Update portfolio
        position_value = position.quantity * position.entry_price
        self.portfolio.balance += position_value + pnl
        self.portfolio.margin_used -= position_value

        # Move to history
        self.portfolio.trade_history.append(position)
        del self.portfolio.positions[symbol]

        result_emoji = "+" if pnl >= 0 else "-"
        logger.info(
            f"{result_emoji} POSITION CLOSED: {position.side} {symbol} | "
            f"Entry: {position.entry_price:.6f} | Exit: {effective_exit:.6f} (raw: {close_price:.6f}) | "
            f"P&L: ${pnl:.2f} ({pnl_percent:.2f}%) | Fee: ${exit_fee:.2f} | Reason: {reason}"
        )

        return position

    def partial_close(self, symbol: str, percent: float,
                      close_price: float = None) -> Optional[float]:
        """
        Partially close a position.

        Args:
            symbol: Trading pair
            percent: Percentage to close (0-1)
            close_price: Close price

        Returns:
            P&L of partial close or None
        """
        position = self.portfolio.positions.get(symbol)
        if not position:
            return None

        if close_price is None:
            close_price = self._price_cache.get(symbol, position.current_price)

        close_quantity = position.quantity * percent

        # Calculate P&L for partial close
        if position.side == "LONG":
            partial_pnl = (close_price - position.entry_price) * close_quantity
        else:
            partial_pnl = (position.entry_price - close_price) * close_quantity

        # Update position
        position.quantity -= close_quantity
        position.partial_closes.append({
            "quantity": close_quantity,
            "price": close_price,
            "pnl": partial_pnl,
            "timestamp": datetime.now().isoformat()
        })

        # Update portfolio
        self.portfolio.balance += (close_quantity * position.entry_price) + partial_pnl
        self.portfolio.margin_used -= close_quantity * position.entry_price

        logger.info(
            f"PARTIAL CLOSE: {percent*100:.0f}% of {symbol} @ {close_price:.4f} | "
            f"P&L: ${partial_pnl:.2f}"
        )

        return partial_pnl

    def check_partial_tp_levels(self, symbol: str, price: float):
        """Check if partial TP levels are hit."""
        if not self.cfg.use_partial_tp:
            return

        position = self.portfolio.positions.get(symbol)
        if not position:
            return

        risk = abs(position.entry_price - position.stop_loss)

        for level in self.cfg.partial_tp_levels:
            r_multiple = level["level"]
            close_percent = level["close_percent"]

            if position.side == "LONG":
                tp_price = position.entry_price + (risk * r_multiple)
                if price >= tp_price:
                    # Check if this level hasn't been taken yet
                    level_key = f"tp_{r_multiple}"
                    if level_key not in [pc.get("level") for pc in position.partial_closes]:
                        self.partial_close(symbol, close_percent, price)
                        position.partial_closes[-1]["level"] = level_key
            else:
                tp_price = position.entry_price - (risk * r_multiple)
                if price <= tp_price:
                    level_key = f"tp_{r_multiple}"
                    if level_key not in [pc.get("level") for pc in position.partial_closes]:
                        self.partial_close(symbol, close_percent, price)
                        position.partial_closes[-1]["level"] = level_key

    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary."""
        total_trades = len(self.portfolio.trade_history)
        winning_trades = len([t for t in self.portfolio.trade_history if t.pnl > 0])
        losing_trades = len([t for t in self.portfolio.trade_history if t.pnl < 0])

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        total_profit = sum(t.pnl for t in self.portfolio.trade_history if t.pnl > 0)
        total_loss = abs(sum(t.pnl for t in self.portfolio.trade_history if t.pnl < 0))

        profit_factor = (total_profit / total_loss) if total_loss > 0 else float('inf')

        avg_win = total_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = total_loss / losing_trades if losing_trades > 0 else 0

        return {
            "initial_capital": self.portfolio.initial_capital,
            "balance": self.portfolio.balance,
            "equity": self.portfolio.equity,
            "total_pnl": self.portfolio.total_pnl,
            "total_pnl_percent": (self.portfolio.total_pnl / self.portfolio.initial_capital) * 100 if self.portfolio.initial_capital > 0 else 0,
            "unrealized_pnl": self.portfolio.total_unrealized_pnl,
            "open_positions": len(self.portfolio.positions),
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "largest_win": max([t.pnl for t in self.portfolio.trade_history], default=0),
            "largest_loss": min([t.pnl for t in self.portfolio.trade_history], default=0),
            "total_fees_paid": self._total_fees_paid
        }

    def get_open_positions(self) -> List[Dict]:
        """Get all open positions."""
        return [
            {
                "id": pos.id,
                "symbol": pos.symbol,
                "side": pos.side,
                "entry_price": pos.entry_price,
                "current_price": pos.current_price,
                "quantity": pos.quantity,
                "value": pos.value,
                "stop_loss": pos.stop_loss,
                "take_profit": pos.take_profit,
                "unrealized_pnl": pos.unrealized_pnl,
                "unrealized_pnl_percent": pos.unrealized_pnl_percent,
                "opened_at": pos.opened_at.isoformat()
            }
            for pos in self.portfolio.positions.values()
        ]

    def get_trade_history(self, limit: int = 50) -> List[Dict]:
        """Get trade history."""
        history = sorted(
            self.portfolio.trade_history,
            key=lambda x: x.closed_at or datetime.min,
            reverse=True
        )[:limit]

        return [
            {
                "id": trade.id,
                "symbol": trade.symbol,
                "side": trade.side,
                "entry_price": trade.entry_price,
                "close_price": trade.close_price,
                "quantity": trade.quantity,
                "pnl": trade.pnl,
                "pnl_percent": trade.pnl_percent,
                "close_reason": trade.close_reason,
                "opened_at": trade.opened_at.isoformat(),
                "closed_at": trade.closed_at.isoformat() if trade.closed_at else None
            }
            for trade in history
        ]


# Singleton instance
paper_trader = PaperTrader()
