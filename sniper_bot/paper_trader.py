"""
📈 SUPERTRADER PAPER TRADING SYSTEM
====================================
Realistic paper trading with:
- Starting capital: 1000 USDT
- Fees: 0.1% maker/taker
- Slippage: 0.05% average
- Spread: 0.05% average
- Real-time P&L tracking
- Position management
- Trade history

Run: python3 paper_trader.py
"""
import sys
sys.path.insert(0, '/workspace/sniper_bot')

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import json
import time
import os

from core.indicators import Indicators


# === DATA CLASSES ===

@dataclass
class Position:
    """Open position"""
    id: str
    symbol: str
    side: str  # 'LONG'
    entry_time: str
    entry_price: float
    size: float  # In quote currency (USDT)
    quantity: float  # In base currency
    stop_loss: float
    take_profit: float
    trailing_stop: float
    trailing_activated: bool
    highest_price: float
    signal_type: str
    entry_score: float
    
    # Costs
    entry_fee: float
    slippage_cost: float
    
    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L"""
        gross_pnl = (current_price - self.entry_price) * self.quantity
        return gross_pnl
    
    def unrealized_pnl_pct(self, current_price: float) -> float:
        """Calculate unrealized P&L percentage"""
        return ((current_price - self.entry_price) / self.entry_price) * 100


@dataclass
class ClosedTrade:
    """Completed trade"""
    id: str
    symbol: str
    side: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    size: float
    quantity: float
    signal_type: str
    exit_reason: str
    
    # P&L
    gross_pnl: float
    fees: float
    slippage: float
    net_pnl: float
    pnl_pct: float
    
    # Duration
    bars_held: int


@dataclass
class AccountState:
    """Account state"""
    balance: float
    equity: float
    available: float
    unrealized_pnl: float
    realized_pnl: float
    total_fees: float
    total_slippage: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0
        return (self.winning_trades / self.total_trades) * 100


# === PAPER TRADER ===

class PaperTrader:
    """Paper trading system with realistic costs"""
    
    # Configuration
    CONFIG = {
        # Capital
        'starting_balance': 1000.0,
        
        # Costs (realistic for most exchanges)
        'maker_fee': 0.001,      # 0.1%
        'taker_fee': 0.001,      # 0.1%
        'slippage': 0.0005,      # 0.05% average slippage
        'spread': 0.0005,        # 0.05% average spread
        
        # Risk Management
        'risk_per_trade': 0.02,   # 2% risk per trade
        'max_positions': 3,       # Max concurrent positions
        'max_position_size': 0.20, # Max 20% of balance per position
        
        # Strategy (from optimized backtest)
        'min_score': 55,  # Lowered for more activity
        'atr_stop': 1.2,
        'atr_trail_activate': 0.5,
        'atr_trail_distance': 0.6,
        'take_profit_rr': 2.5,
        
        # Data
        'state_file': 'paper_trader_state.json',
    }
    
    def __init__(self):
        self.exchange = ccxt.kucoin()
        self.exchange.load_markets()
        
        # Account state
        self.balance = self.CONFIG['starting_balance']
        self.positions: List[Position] = []
        self.closed_trades: List[ClosedTrade] = []
        self.trade_counter = 0
        
        # Load saved state if exists
        self.load_state()
    
    # === STATE MANAGEMENT ===
    
    def save_state(self):
        """Save state to file"""
        state = {
            'balance': self.balance,
            'trade_counter': self.trade_counter,
            'positions': [
                {
                    'id': p.id,
                    'symbol': p.symbol,
                    'side': p.side,
                    'entry_time': p.entry_time,
                    'entry_price': p.entry_price,
                    'size': p.size,
                    'quantity': p.quantity,
                    'stop_loss': p.stop_loss,
                    'take_profit': p.take_profit,
                    'trailing_stop': p.trailing_stop,
                    'trailing_activated': p.trailing_activated,
                    'highest_price': p.highest_price,
                    'signal_type': p.signal_type,
                    'entry_score': p.entry_score,
                    'entry_fee': p.entry_fee,
                    'slippage_cost': p.slippage_cost,
                } for p in self.positions
            ],
            'closed_trades': [
                {
                    'id': t.id,
                    'symbol': t.symbol,
                    'side': t.side,
                    'entry_time': t.entry_time,
                    'exit_time': t.exit_time,
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'size': t.size,
                    'quantity': t.quantity,
                    'signal_type': t.signal_type,
                    'exit_reason': t.exit_reason,
                    'gross_pnl': t.gross_pnl,
                    'fees': t.fees,
                    'slippage': t.slippage,
                    'net_pnl': t.net_pnl,
                    'pnl_pct': t.pnl_pct,
                    'bars_held': t.bars_held,
                } for t in self.closed_trades
            ],
            'last_update': str(datetime.now()),
        }
        
        with open(self.CONFIG['state_file'], 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self):
        """Load state from file"""
        if not os.path.exists(self.CONFIG['state_file']):
            return
        
        try:
            with open(self.CONFIG['state_file'], 'r') as f:
                state = json.load(f)
            
            self.balance = state.get('balance', self.CONFIG['starting_balance'])
            self.trade_counter = state.get('trade_counter', 0)
            
            # Load positions
            self.positions = []
            for p in state.get('positions', []):
                self.positions.append(Position(
                    id=p['id'],
                    symbol=p['symbol'],
                    side=p['side'],
                    entry_time=p['entry_time'],
                    entry_price=p['entry_price'],
                    size=p['size'],
                    quantity=p['quantity'],
                    stop_loss=p['stop_loss'],
                    take_profit=p['take_profit'],
                    trailing_stop=p['trailing_stop'],
                    trailing_activated=p['trailing_activated'],
                    highest_price=p['highest_price'],
                    signal_type=p['signal_type'],
                    entry_score=p['entry_score'],
                    entry_fee=p['entry_fee'],
                    slippage_cost=p['slippage_cost'],
                ))
            
            # Load closed trades
            self.closed_trades = []
            for t in state.get('closed_trades', []):
                self.closed_trades.append(ClosedTrade(
                    id=t['id'],
                    symbol=t['symbol'],
                    side=t['side'],
                    entry_time=t['entry_time'],
                    exit_time=t['exit_time'],
                    entry_price=t['entry_price'],
                    exit_price=t['exit_price'],
                    size=t['size'],
                    quantity=t['quantity'],
                    signal_type=t['signal_type'],
                    exit_reason=t['exit_reason'],
                    gross_pnl=t['gross_pnl'],
                    fees=t['fees'],
                    slippage=t['slippage'],
                    net_pnl=t['net_pnl'],
                    pnl_pct=t['pnl_pct'],
                    bars_held=t['bars_held'],
                ))
            
            print(f"  Loaded state: Balance ${self.balance:.2f}, {len(self.positions)} positions, {len(self.closed_trades)} closed trades")
        except Exception as e:
            print(f"  Error loading state: {e}")
    
    # === DATA FETCHING ===
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = '4h', limit: int = 60) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv or len(ohlcv) < 30:
                return None
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except:
            return None
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price with spread"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker.get('last', 0)
        except:
            return None
    
    def get_pairs(self, min_volume: float = 100000) -> List[Dict]:
        """Get active pairs"""
        try:
            tickers = self.exchange.fetch_tickers()
            pairs = []
            
            for symbol, ticker in tickers.items():
                if not symbol.endswith('/USDT'):
                    continue
                if '/USDT:' in symbol or '3L' in symbol or '3S' in symbol:
                    continue
                
                vol = ticker.get('quoteVolume', 0) or 0
                price = ticker.get('last', 0) or 0
                
                if vol >= min_volume and price > 0:
                    pairs.append({
                        'symbol': symbol,
                        'price': price,
                        'volume': vol,
                        'change_24h': ticker.get('percentage', 0) or 0,
                    })
            
            return sorted(pairs, key=lambda x: x['volume'], reverse=True)
        except:
            return []
    
    # === SIGNAL GENERATION ===
    
    def analyze_coin(self, symbol: str, ticker: Dict) -> Optional[Dict]:
        """Analyze coin for signals"""
        
        df = self.fetch_ohlcv(symbol, '4h', 60)
        if df is None:
            return None
        
        price = ticker['price']
        
        # Indicators
        rsi_series = Indicators.rsi(df['close'], 14)
        rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50
        rsi_prev = float(rsi_series.iloc[-2]) if not pd.isna(rsi_series.iloc[-2]) else 50
        
        macd, signal_line, hist = Indicators.macd(df['close'])
        curr_hist = float(hist.iloc[-1]) if not pd.isna(hist.iloc[-1]) else 0
        prev_hist = float(hist.iloc[-2]) if not pd.isna(hist.iloc[-2]) else 0
        prev_hist2 = float(hist.iloc[-3]) if not pd.isna(hist.iloc[-3]) else 0
        
        macd_rising = curr_hist > prev_hist
        macd_accel = curr_hist > prev_hist > prev_hist2
        macd_cross = curr_hist > 0 and prev_hist <= 0
        
        atr = Indicators.atr(df['high'], df['low'], df['close'], 14)
        current_atr = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else price * 0.03
        
        ema20 = Indicators.ema(df['close'], 20)
        ema50 = Indicators.ema(df['close'], 50)
        trend_up = price > float(ema20.iloc[-1]) and float(ema20.iloc[-1]) > float(ema50.iloc[-1])
        
        high_20 = float(df['high'].iloc[-20:-1].max())
        
        vol_ma = float(df['volume'].rolling(20).mean().iloc[-1])
        vol_spike = float(df['volume'].iloc[-1]) / vol_ma if vol_ma > 0 else 1
        
        # Signal detection
        signal_type = None
        score = 0
        
        # Near breakout check
        dist_from_high = ((high_20 - price) / price) * 100 if price > 0 else 100
        near_breakout = dist_from_high < 2  # Within 2% of high
        
        # BOUNCE - Oversold with any positive sign
        if rsi < 35:
            signal_type = 'BOUNCE'
            score = 35
            if rsi < 25:
                score += 30
            elif rsi < 30:
                score += 20
            else:
                score += 10
            if rsi > rsi_prev:  # Recovery
                score += 20
            if macd_rising:
                score += 15
            if vol_spike > 2:
                score += 15
            elif vol_spike > 1:
                score += 5
        
        # BREAKOUT - Breaking or near breaking with momentum
        elif (price > high_20 or near_breakout) and (macd_rising or curr_hist > 0) and rsi < 75:
            signal_type = 'BREAKOUT'
            score = 35
            if price > high_20:
                score += 25
            elif near_breakout:
                score += 15
            if macd_cross:
                score += 25
            elif macd_accel:
                score += 15
            elif curr_hist > 0:
                score += 10
            if vol_spike > 3:
                score += 20
            elif vol_spike > 1.5:
                score += 10
            if trend_up:
                score += 10
            if rsi > 50 and rsi < 65:
                score += 5
        
        # MOMENTUM - Strong upward movement
        elif curr_hist > 0 and macd_rising and trend_up and rsi > 50 and rsi < 70:
            signal_type = 'MOMENTUM'
            score = 40
            if macd_accel:
                score += 20
            if vol_spike > 1.5:
                score += 15
            if 55 < rsi < 65:
                score += 10
        
        if signal_type is None or score < self.CONFIG['min_score']:
            return None
        
        return {
            'symbol': symbol,
            'price': price,
            'signal_type': signal_type,
            'score': score,
            'rsi': rsi,
            'atr': current_atr,
            'volume_spike': vol_spike,
            'trend_up': trend_up,
        }
    
    # === POSITION MANAGEMENT ===
    
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """Calculate position size based on risk"""
        risk_amount = self.balance * self.CONFIG['risk_per_trade']
        risk_per_unit = entry_price - stop_loss
        
        if risk_per_unit <= 0:
            return 0
        
        position_size = risk_amount / (risk_per_unit / entry_price)
        
        # Cap at max position size
        max_size = self.balance * self.CONFIG['max_position_size']
        position_size = min(position_size, max_size)
        
        # Cap at available balance
        available = self.balance - sum(p.size for p in self.positions)
        position_size = min(position_size, available * 0.95)  # Keep 5% buffer
        
        return max(0, position_size)
    
    def open_position(self, signal: Dict) -> Optional[Position]:
        """Open a new position"""
        
        # Check max positions
        if len(self.positions) >= self.CONFIG['max_positions']:
            return None
        
        # Check if already in this symbol
        if any(p.symbol == signal['symbol'] for p in self.positions):
            return None
        
        price = signal['price']
        atr = signal['atr']
        
        # Calculate levels
        stop_loss = price - (self.CONFIG['atr_stop'] * atr)
        risk = price - stop_loss
        take_profit = price + (risk * self.CONFIG['take_profit_rr'])
        
        # Position size
        size = self.calculate_position_size(price, stop_loss)
        if size < 10:  # Min $10
            return None
        
        # Apply costs
        spread_cost = price * self.CONFIG['spread']
        slippage_cost = price * self.CONFIG['slippage']
        effective_entry = price + spread_cost + slippage_cost
        
        fee = size * self.CONFIG['taker_fee']
        total_slippage = size * (self.CONFIG['slippage'] + self.CONFIG['spread'])
        
        quantity = size / effective_entry
        
        # Create position
        self.trade_counter += 1
        position = Position(
            id=f"T{self.trade_counter:04d}",
            symbol=signal['symbol'],
            side='LONG',
            entry_time=str(datetime.now()),
            entry_price=effective_entry,
            size=size,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop=stop_loss,
            trailing_activated=False,
            highest_price=price,
            signal_type=signal['signal_type'],
            entry_score=signal['score'],
            entry_fee=fee,
            slippage_cost=total_slippage,
        )
        
        # Deduct from balance
        self.balance -= (size + fee)
        
        self.positions.append(position)
        self.save_state()
        
        return position
    
    def close_position(self, position: Position, current_price: float, exit_reason: str) -> ClosedTrade:
        """Close a position"""
        
        # Apply exit costs
        spread_cost = current_price * self.CONFIG['spread']
        slippage_cost = current_price * self.CONFIG['slippage']
        effective_exit = current_price - spread_cost - slippage_cost
        
        exit_fee = position.size * self.CONFIG['taker_fee']
        exit_slippage = position.size * (self.CONFIG['slippage'] + self.CONFIG['spread'])
        
        # Calculate P&L
        gross_pnl = (effective_exit - position.entry_price) * position.quantity
        total_fees = position.entry_fee + exit_fee
        total_slippage = position.slippage_cost + exit_slippage
        net_pnl = gross_pnl - total_fees - total_slippage
        
        pnl_pct = (net_pnl / position.size) * 100
        
        # Create closed trade
        trade = ClosedTrade(
            id=position.id,
            symbol=position.symbol,
            side=position.side,
            entry_time=position.entry_time,
            exit_time=str(datetime.now()),
            entry_price=position.entry_price,
            exit_price=effective_exit,
            size=position.size,
            quantity=position.quantity,
            signal_type=position.signal_type,
            exit_reason=exit_reason,
            gross_pnl=gross_pnl,
            fees=total_fees,
            slippage=total_slippage,
            net_pnl=net_pnl,
            pnl_pct=pnl_pct,
            bars_held=0,  # Would need to track this
        )
        
        # Update balance
        self.balance += position.size + net_pnl - exit_fee
        
        # Remove from positions, add to closed
        self.positions.remove(position)
        self.closed_trades.append(trade)
        
        self.save_state()
        
        return trade
    
    def update_positions(self) -> List[ClosedTrade]:
        """Update all positions and check for exits"""
        
        closed = []
        
        for position in self.positions[:]:  # Copy list for iteration
            current_price = self.get_current_price(position.symbol)
            if current_price is None:
                continue
            
            # Update highest price
            if current_price > position.highest_price:
                position.highest_price = current_price
            
            # Check trailing stop activation
            if not position.trailing_activated:
                df = self.fetch_ohlcv(position.symbol, '4h', 20)
                if df is not None:
                    atr = Indicators.atr(df['high'], df['low'], df['close'], 14)
                    current_atr = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0
                    
                    if current_atr > 0:
                        profit_atr = (position.highest_price - position.entry_price) / current_atr
                        if profit_atr >= self.CONFIG['atr_trail_activate']:
                            position.trailing_activated = True
                            position.trailing_stop = position.highest_price - (self.CONFIG['atr_trail_distance'] * current_atr)
            
            # Update trailing stop
            if position.trailing_activated:
                df = self.fetch_ohlcv(position.symbol, '4h', 20)
                if df is not None:
                    atr = Indicators.atr(df['high'], df['low'], df['close'], 14)
                    current_atr = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0
                    
                    if current_atr > 0:
                        new_stop = position.highest_price - (self.CONFIG['atr_trail_distance'] * current_atr)
                        position.trailing_stop = max(position.trailing_stop, new_stop)
            
            # Check exit conditions
            exit_reason = None
            
            # Take profit
            if current_price >= position.take_profit:
                exit_reason = 'TAKE_PROFIT'
            
            # Trailing stop
            elif position.trailing_activated and current_price <= position.trailing_stop:
                exit_reason = 'TRAILING_STOP'
            
            # Stop loss
            elif current_price <= position.stop_loss:
                exit_reason = 'STOP_LOSS'
            
            if exit_reason:
                trade = self.close_position(position, current_price, exit_reason)
                closed.append(trade)
        
        self.save_state()
        return closed
    
    # === ACCOUNT STATE ===
    
    def get_account_state(self) -> AccountState:
        """Get current account state"""
        
        # Calculate unrealized P&L
        unrealized_pnl = 0
        for position in self.positions:
            current_price = self.get_current_price(position.symbol)
            if current_price:
                unrealized_pnl += position.unrealized_pnl(current_price)
        
        # Realized P&L
        realized_pnl = sum(t.net_pnl for t in self.closed_trades)
        
        # Equity
        positions_value = sum(p.size for p in self.positions)
        equity = self.balance + positions_value + unrealized_pnl
        
        # Fees and slippage
        total_fees = sum(t.fees for t in self.closed_trades)
        total_slippage = sum(t.slippage for t in self.closed_trades)
        
        # Win/loss
        wins = len([t for t in self.closed_trades if t.net_pnl > 0])
        losses = len([t for t in self.closed_trades if t.net_pnl <= 0])
        
        return AccountState(
            balance=self.balance,
            equity=equity,
            available=self.balance,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            total_fees=total_fees,
            total_slippage=total_slippage,
            total_trades=len(self.closed_trades),
            winning_trades=wins,
            losing_trades=losses,
        )
    
    # === SCANNING ===
    
    def scan_for_signals(self, max_coins: int = 100) -> List[Dict]:
        """Scan market for signals"""
        
        pairs = self.get_pairs(min_volume=100000)
        signals = []
        
        for pair in pairs[:max_coins]:
            signal = self.analyze_coin(pair['symbol'], pair)
            if signal:
                signals.append(signal)
            time.sleep(0.05)
        
        signals.sort(key=lambda x: x['score'], reverse=True)
        return signals
    
    # === DISPLAY ===
    
    def display_status(self):
        """Display full status"""
        
        state = self.get_account_state()
        
        print()
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 25 + "📈 PAPER TRADER STATUS" + " " * 31 + "║")
        print("║" + " " * 25 + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " " * 28 + "║")
        print("╠" + "═" * 78 + "╣")
        
        # Account
        pnl_total = state.equity - self.CONFIG['starting_balance']
        pnl_pct = (pnl_total / self.CONFIG['starting_balance']) * 100
        pnl_sign = "+" if pnl_total >= 0 else ""
        
        print("║  ACCOUNT:" + " " * 68 + "║")
        print(f"║  ├─ Starting Balance: ${self.CONFIG['starting_balance']:.2f}" + " " * 44 + "║")
        print(f"║  ├─ Current Equity:   ${state.equity:.2f}" + " " * (44 - len(f"{state.equity:.2f}") + 7) + "║")
        print(f"║  ├─ Available:        ${state.available:.2f}" + " " * (44 - len(f"{state.available:.2f}") + 7) + "║")
        print(f"║  ├─ Unrealized P&L:   ${state.unrealized_pnl:+.2f}" + " " * (43 - len(f"{state.unrealized_pnl:+.2f}") + 7) + "║")
        print(f"║  └─ Total P&L:        {pnl_sign}${abs(pnl_total):.2f} ({pnl_pct:+.2f}%)" + " " * (32 - len(f"{abs(pnl_total):.2f}") - len(f"{pnl_pct:+.2f}")) + "║")
        
        print("╠" + "═" * 78 + "╣")
        
        # Positions
        print(f"║  OPEN POSITIONS ({len(self.positions)}/{self.CONFIG['max_positions']}):" + " " * 52 + "║")
        
        if self.positions:
            for p in self.positions:
                current_price = self.get_current_price(p.symbol)
                if current_price:
                    pnl = p.unrealized_pnl(current_price)
                    pnl_pct = p.unrealized_pnl_pct(current_price)
                    status = "🟢" if pnl >= 0 else "🔴"
                    trail = "📍" if p.trailing_activated else ""
                    print(f"║  {status} {p.symbol:<12} Entry: ${p.entry_price:.6f} | Now: ${current_price:.6f} | P&L: {pnl_pct:+.2f}% {trail}" + " " * 5 + "║")
        else:
            print("║    No open positions" + " " * 57 + "║")
        
        print("╠" + "═" * 78 + "╣")
        
        # Statistics
        print("║  STATISTICS:" + " " * 65 + "║")
        print(f"║  ├─ Total Trades:     {state.total_trades}" + " " * (55 - len(str(state.total_trades))) + "║")
        print(f"║  ├─ Wins/Losses:      {state.winning_trades}/{state.losing_trades}" + " " * (52 - len(str(state.winning_trades)) - len(str(state.losing_trades))) + "║")
        print(f"║  ├─ Win Rate:         {state.win_rate():.1f}%" + " " * (53 - len(f"{state.win_rate():.1f}")) + "║")
        print(f"║  ├─ Total Fees:       ${state.total_fees:.2f}" + " " * (52 - len(f"{state.total_fees:.2f}")) + "║")
        print(f"║  └─ Total Slippage:   ${state.total_slippage:.2f}" + " " * (52 - len(f"{state.total_slippage:.2f}")) + "║")
        
        print("╠" + "═" * 78 + "╣")
        
        # Recent trades
        print("║  RECENT TRADES:" + " " * 62 + "║")
        
        if self.closed_trades:
            for t in self.closed_trades[-5:]:
                status = "✅" if t.net_pnl > 0 else "❌"
                print(f"║  {status} {t.symbol:<12} {t.signal_type:<10} {t.exit_reason:<15} P&L: {t.pnl_pct:+.2f}%" + " " * 10 + "║")
        else:
            print("║    No closed trades yet" + " " * 54 + "║")
        
        print("╚" + "═" * 78 + "╝")
    
    def display_signals(self, signals: List[Dict]):
        """Display found signals"""
        
        if not signals:
            print("\n  No signals found.")
            return
        
        print("\n  🎯 SIGNALS FOUND:")
        print("-" * 80)
        
        for s in signals[:10]:
            can_trade = len(self.positions) < self.CONFIG['max_positions']
            already_in = any(p.symbol == s['symbol'] for p in self.positions)
            
            status = "✅ CAN TRADE" if can_trade and not already_in else "⏸️ SKIP" if already_in else "⚠️ MAX POS"
            
            print(f"  {s['symbol']:<14} | {s['signal_type']:<10} | Score: {s['score']} | RSI: {s['rsi']:.0f} | Vol: {s['volume_spike']:.1f}x | {status}")
    
    # === MAIN LOOP ===
    
    def run_cycle(self):
        """Run one trading cycle"""
        
        print("\n" + "=" * 80)
        print(f"  🔄 TRADING CYCLE - {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 80)
        
        # 1. Update existing positions
        print("\n  Updating positions...")
        closed = self.update_positions()
        
        if closed:
            print(f"  📊 Closed {len(closed)} position(s):")
            for t in closed:
                status = "✅" if t.net_pnl > 0 else "❌"
                print(f"    {status} {t.symbol} - {t.exit_reason} - P&L: ${t.net_pnl:.2f} ({t.pnl_pct:+.2f}%)")
        
        # 2. Scan for new signals
        print("\n  Scanning for signals...")
        signals = self.scan_for_signals(max_coins=100)
        self.display_signals(signals)
        
        # 3. Open new positions
        if signals and len(self.positions) < self.CONFIG['max_positions']:
            for signal in signals:
                if len(self.positions) >= self.CONFIG['max_positions']:
                    break
                
                position = self.open_position(signal)
                if position:
                    print(f"\n  🚀 OPENED POSITION:")
                    print(f"     {position.symbol} | {position.signal_type} | Entry: ${position.entry_price:.6f}")
                    print(f"     Size: ${position.size:.2f} | SL: ${position.stop_loss:.6f} | TP: ${position.take_profit:.6f}")
        
        # 4. Display status
        self.display_status()
    
    def run_continuous(self, interval_minutes: int = 15):
        """Run continuous trading"""
        
        print("\n" + "=" * 80)
        print("  📈 PAPER TRADER - CONTINUOUS MODE")
        print(f"  Interval: {interval_minutes} minutes")
        print("=" * 80)
        
        while True:
            try:
                self.run_cycle()
            except Exception as e:
                print(f"\n  ❌ Error: {e}")
            
            print(f"\n  ⏰ Next cycle in {interval_minutes} minutes...")
            time.sleep(interval_minutes * 60)


def main():
    """Main entry point"""
    
    import sys
    
    trader = PaperTrader()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'status':
            trader.display_status()
        elif sys.argv[1] == 'run':
            interval = int(sys.argv[2]) if len(sys.argv) > 2 else 15
            trader.run_continuous(interval)
        elif sys.argv[1] == 'reset':
            if os.path.exists(trader.CONFIG['state_file']):
                os.remove(trader.CONFIG['state_file'])
            print("  State reset. Starting fresh.")
            trader = PaperTrader()
            trader.display_status()
    else:
        # Single cycle
        trader.run_cycle()


if __name__ == "__main__":
    main()
