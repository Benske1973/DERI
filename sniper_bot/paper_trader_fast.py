"""
📈 SUPERTRADER PAPER TRADING - FAST MONITORING
===============================================
- Positie check: elke 1 minuut
- Signal scan: elke 5 minuten
- Real-time P&L updates

Run: python3 paper_trader_fast.py
"""
import sys
sys.path.insert(0, '/workspace/sniper_bot')

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import json
import time
import os

from core.indicators import Indicators


@dataclass
class Position:
    """Open position"""
    id: str
    symbol: str
    side: str
    entry_time: str
    entry_price: float
    size: float
    quantity: float
    stop_loss: float
    take_profit: float
    trailing_stop: float
    trailing_activated: bool
    highest_price: float
    signal_type: str
    entry_score: float
    entry_fee: float
    slippage_cost: float
    atr_at_entry: float = 0.0
    
    def unrealized_pnl(self, current_price: float) -> float:
        return (current_price - self.entry_price) * self.quantity
    
    def unrealized_pnl_pct(self, current_price: float) -> float:
        return ((current_price - self.entry_price) / self.entry_price) * 100


@dataclass
class ClosedTrade:
    """Completed trade"""
    id: str
    symbol: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    size: float
    quantity: float
    signal_type: str
    exit_reason: str
    gross_pnl: float
    fees: float
    slippage: float
    net_pnl: float
    pnl_pct: float
    duration_minutes: int


class FastPaperTrader:
    """Fast paper trader with frequent monitoring"""
    
    CONFIG = {
        'starting_balance': 1000.0,
        'maker_fee': 0.001,
        'taker_fee': 0.001,
        'slippage': 0.0005,
        'spread': 0.0005,
        'risk_per_trade': 0.02,
        'max_positions': 3,
        'max_position_size': 0.25,
        'min_score': 55,
        'atr_stop': 1.2,
        'atr_trail_activate': 0.5,
        'atr_trail_distance': 0.6,
        'take_profit_rr': 2.5,
        'state_file': 'paper_trader_state.json',
        
        # Timing
        'position_check_seconds': 60,    # Check positions every 60 sec
        'signal_scan_minutes': 5,        # Scan for signals every 5 min
    }
    
    def __init__(self):
        self.exchange = ccxt.kucoin()
        self.exchange.load_markets()
        self.balance = self.CONFIG['starting_balance']
        self.positions: List[Position] = []
        self.closed_trades: List[ClosedTrade] = []
        self.trade_counter = 0
        self.last_signal_scan = None
        self.load_state()
    
    def save_state(self):
        state = {
            'balance': self.balance,
            'trade_counter': self.trade_counter,
            'positions': [{
                'id': p.id, 'symbol': p.symbol, 'side': p.side,
                'entry_time': p.entry_time, 'entry_price': p.entry_price,
                'size': p.size, 'quantity': p.quantity,
                'stop_loss': p.stop_loss, 'take_profit': p.take_profit,
                'trailing_stop': p.trailing_stop, 'trailing_activated': p.trailing_activated,
                'highest_price': p.highest_price, 'signal_type': p.signal_type,
                'entry_score': p.entry_score, 'entry_fee': p.entry_fee,
                'slippage_cost': p.slippage_cost, 'atr_at_entry': p.atr_at_entry,
            } for p in self.positions],
            'closed_trades': [{
                'id': t.id, 'symbol': t.symbol,
                'entry_time': t.entry_time, 'exit_time': t.exit_time,
                'entry_price': t.entry_price, 'exit_price': t.exit_price,
                'size': t.size, 'quantity': t.quantity,
                'signal_type': t.signal_type, 'exit_reason': t.exit_reason,
                'gross_pnl': t.gross_pnl, 'fees': t.fees, 'slippage': t.slippage,
                'net_pnl': t.net_pnl, 'pnl_pct': t.pnl_pct,
                'duration_minutes': t.duration_minutes,
            } for t in self.closed_trades],
            'last_update': str(datetime.now()),
        }
        with open(self.CONFIG['state_file'], 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self):
        if not os.path.exists(self.CONFIG['state_file']):
            return
        try:
            with open(self.CONFIG['state_file'], 'r') as f:
                state = json.load(f)
            self.balance = state.get('balance', self.CONFIG['starting_balance'])
            self.trade_counter = state.get('trade_counter', 0)
            self.positions = [Position(**p) for p in state.get('positions', [])]
            self.closed_trades = [ClosedTrade(**t) for t in state.get('closed_trades', [])]
        except Exception as e:
            print(f"Error loading state: {e}")
    
    def get_price(self, symbol: str) -> Optional[float]:
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker.get('last', 0)
        except:
            return None
    
    def get_prices_batch(self, symbols: List[str]) -> Dict[str, float]:
        """Get multiple prices at once"""
        prices = {}
        try:
            tickers = self.exchange.fetch_tickers()
            for symbol in symbols:
                if symbol in tickers:
                    prices[symbol] = tickers[symbol].get('last', 0)
        except:
            pass
        return prices
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = '4h', limit: int = 30) -> Optional[pd.DataFrame]:
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv or len(ohlcv) < 20:
                return None
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            return df
        except:
            return None
    
    def get_atr(self, symbol: str) -> float:
        """Get current ATR for a symbol"""
        df = self.fetch_ohlcv(symbol, '4h', 20)
        if df is None:
            return 0
        atr = Indicators.atr(df['high'], df['low'], df['close'], 14)
        return float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0
    
    def check_positions(self) -> List[ClosedTrade]:
        """Check all positions for exit conditions - FAST"""
        if not self.positions:
            return []
        
        # Get all prices at once
        symbols = [p.symbol for p in self.positions]
        prices = self.get_prices_batch(symbols)
        
        closed = []
        
        for position in self.positions[:]:
            current_price = prices.get(position.symbol)
            if not current_price:
                continue
            
            # Update highest price
            if current_price > position.highest_price:
                position.highest_price = current_price
            
            # Get current ATR for trailing
            current_atr = position.atr_at_entry
            if current_atr == 0:
                current_atr = self.get_atr(position.symbol)
                position.atr_at_entry = current_atr
            
            # Check trailing stop activation
            if not position.trailing_activated and current_atr > 0:
                profit_atr = (position.highest_price - position.entry_price) / current_atr
                if profit_atr >= self.CONFIG['atr_trail_activate']:
                    position.trailing_activated = True
                    position.trailing_stop = position.highest_price - (self.CONFIG['atr_trail_distance'] * current_atr)
                    print(f"  📍 {position.symbol}: Trailing stop activated at ${position.trailing_stop:.6f}")
            
            # Update trailing stop
            if position.trailing_activated and current_atr > 0:
                new_stop = position.highest_price - (self.CONFIG['atr_trail_distance'] * current_atr)
                if new_stop > position.trailing_stop:
                    position.trailing_stop = new_stop
            
            # Check exit conditions
            exit_reason = None
            exit_price = current_price
            
            if current_price >= position.take_profit:
                exit_reason = 'TAKE_PROFIT'
                exit_price = position.take_profit
            elif position.trailing_activated and current_price <= position.trailing_stop:
                exit_reason = 'TRAILING_STOP'
                exit_price = position.trailing_stop
            elif current_price <= position.stop_loss:
                exit_reason = 'STOP_LOSS'
                exit_price = position.stop_loss
            
            if exit_reason:
                trade = self.close_position(position, exit_price, exit_reason)
                closed.append(trade)
        
        if self.positions:
            self.save_state()
        
        return closed
    
    def close_position(self, position: Position, exit_price: float, exit_reason: str) -> ClosedTrade:
        """Close a position"""
        # Apply costs
        spread = exit_price * self.CONFIG['spread']
        slippage = exit_price * self.CONFIG['slippage']
        effective_exit = exit_price - spread - slippage
        
        exit_fee = position.size * self.CONFIG['taker_fee']
        exit_slippage = position.size * (self.CONFIG['slippage'] + self.CONFIG['spread'])
        
        # P&L
        gross_pnl = (effective_exit - position.entry_price) * position.quantity
        total_fees = position.entry_fee + exit_fee
        total_slippage = position.slippage_cost + exit_slippage
        net_pnl = gross_pnl - total_fees - total_slippage
        pnl_pct = (net_pnl / position.size) * 100
        
        # Duration
        entry_dt = datetime.fromisoformat(position.entry_time.replace('Z', '+00:00').split('.')[0])
        duration = (datetime.now() - entry_dt).total_seconds() / 60
        
        trade = ClosedTrade(
            id=position.id,
            symbol=position.symbol,
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
            duration_minutes=int(duration),
        )
        
        self.balance += position.size + net_pnl - exit_fee
        self.positions.remove(position)
        self.closed_trades.append(trade)
        self.save_state()
        
        return trade
    
    def scan_signals(self) -> List[Dict]:
        """Scan for new signals"""
        try:
            tickers = self.exchange.fetch_tickers()
        except:
            return []
        
        pairs = []
        for symbol, ticker in tickers.items():
            if not symbol.endswith('/USDT') or '/USDT:' in symbol:
                continue
            if '3L' in symbol or '3S' in symbol:
                continue
            vol = ticker.get('quoteVolume', 0) or 0
            if vol >= 100000:
                pairs.append({
                    'symbol': symbol,
                    'price': ticker.get('last', 0),
                    'volume': vol,
                })
        
        pairs.sort(key=lambda x: x['volume'], reverse=True)
        signals = []
        
        for pair in pairs[:80]:
            signal = self.analyze(pair['symbol'], pair['price'])
            if signal:
                signals.append(signal)
            time.sleep(0.03)
        
        signals.sort(key=lambda x: x['score'], reverse=True)
        return signals
    
    def analyze(self, symbol: str, price: float) -> Optional[Dict]:
        """Quick analysis"""
        df = self.fetch_ohlcv(symbol, '4h', 30)
        if df is None:
            return None
        
        # Indicators
        rsi = Indicators.rsi(df['close'], 14)
        rsi_val = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
        rsi_prev = float(rsi.iloc[-2]) if not pd.isna(rsi.iloc[-2]) else 50
        
        macd, sig, hist = Indicators.macd(df['close'])
        hist_val = float(hist.iloc[-1]) if not pd.isna(hist.iloc[-1]) else 0
        hist_prev = float(hist.iloc[-2]) if not pd.isna(hist.iloc[-2]) else 0
        macd_rising = hist_val > hist_prev
        
        atr = Indicators.atr(df['high'], df['low'], df['close'], 14)
        atr_val = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else price * 0.03
        
        high_20 = float(df['high'].iloc[-20:-1].max())
        dist_high = ((high_20 - price) / price) * 100 if price > 0 else 100
        
        ema20 = float(Indicators.ema(df['close'], 20).iloc[-1])
        ema50 = float(Indicators.ema(df['close'], 50).iloc[-1])
        trend_up = price > ema20 and ema20 > ema50
        
        vol_ma = float(df['volume'].rolling(20).mean().iloc[-1])
        vol_spike = float(df['volume'].iloc[-1]) / vol_ma if vol_ma > 0 else 1
        
        # Signals
        signal_type = None
        score = 0
        
        if rsi_val < 35:
            signal_type = 'BOUNCE'
            score = 35
            if rsi_val < 25: score += 30
            elif rsi_val < 30: score += 20
            else: score += 10
            if rsi_val > rsi_prev: score += 20
            if macd_rising: score += 15
            if vol_spike > 2: score += 15
        
        elif (price > high_20 or dist_high < 2) and (macd_rising or hist_val > 0) and rsi_val < 75:
            signal_type = 'BREAKOUT'
            score = 35
            if price > high_20: score += 25
            elif dist_high < 2: score += 15
            if hist_val > 0 and macd_rising: score += 20
            elif hist_val > 0: score += 10
            if vol_spike > 2: score += 15
            if trend_up: score += 10
        
        elif hist_val > 0 and macd_rising and trend_up and 50 < rsi_val < 70:
            signal_type = 'MOMENTUM'
            score = 40
            if vol_spike > 1.5: score += 15
            if 55 < rsi_val < 65: score += 10
        
        if signal_type is None or score < self.CONFIG['min_score']:
            return None
        
        return {
            'symbol': symbol,
            'price': price,
            'signal_type': signal_type,
            'score': score,
            'rsi': rsi_val,
            'atr': atr_val,
        }
    
    def open_position(self, signal: Dict) -> Optional[Position]:
        """Open new position"""
        if len(self.positions) >= self.CONFIG['max_positions']:
            return None
        if any(p.symbol == signal['symbol'] for p in self.positions):
            return None
        
        price = signal['price']
        atr = signal['atr']
        
        stop_loss = price - (self.CONFIG['atr_stop'] * atr)
        risk = price - stop_loss
        take_profit = price + (risk * self.CONFIG['take_profit_rr'])
        
        # Position size
        risk_amount = self.balance * self.CONFIG['risk_per_trade']
        size = risk_amount / (risk / price) if risk > 0 else 0
        size = min(size, self.balance * self.CONFIG['max_position_size'])
        
        available = self.balance - sum(p.size for p in self.positions)
        size = min(size, available * 0.95)
        
        if size < 10:
            return None
        
        # Costs
        spread = price * self.CONFIG['spread']
        slippage = price * self.CONFIG['slippage']
        entry_price = price + spread + slippage
        fee = size * self.CONFIG['taker_fee']
        slip_cost = size * (self.CONFIG['slippage'] + self.CONFIG['spread'])
        
        self.trade_counter += 1
        position = Position(
            id=f"T{self.trade_counter:04d}",
            symbol=signal['symbol'],
            side='LONG',
            entry_time=str(datetime.now()),
            entry_price=entry_price,
            size=size,
            quantity=size / entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop=stop_loss,
            trailing_activated=False,
            highest_price=price,
            signal_type=signal['signal_type'],
            entry_score=signal['score'],
            entry_fee=fee,
            slippage_cost=slip_cost,
            atr_at_entry=atr,
        )
        
        self.balance -= (size + fee)
        self.positions.append(position)
        self.save_state()
        
        return position
    
    def display_live(self, clear: bool = True):
        """Display live status"""
        if clear:
            print("\033[H\033[J", end="")  # Clear screen
        
        now = datetime.now().strftime('%H:%M:%S')
        
        # Get current prices
        prices = {}
        if self.positions:
            prices = self.get_prices_batch([p.symbol for p in self.positions])
        
        # Calculate totals
        unrealized = sum(p.unrealized_pnl(prices.get(p.symbol, p.entry_price)) for p in self.positions)
        positions_value = sum(p.size for p in self.positions)
        equity = self.balance + positions_value + unrealized
        total_pnl = equity - self.CONFIG['starting_balance']
        total_pnl_pct = (total_pnl / self.CONFIG['starting_balance']) * 100
        
        # Realized stats
        total_fees = sum(t.fees for t in self.closed_trades)
        total_slip = sum(t.slippage for t in self.closed_trades)
        wins = len([t for t in self.closed_trades if t.net_pnl > 0])
        losses = len([t for t in self.closed_trades if t.net_pnl <= 0])
        win_rate = (wins / len(self.closed_trades) * 100) if self.closed_trades else 0
        
        print("╔" + "═" * 78 + "╗")
        print(f"║  📈 LIVE PAPER TRADER          {now}                              ║")
        print("╠" + "═" * 78 + "╣")
        
        # Account
        pnl_emoji = "🟢" if total_pnl >= 0 else "🔴"
        print(f"║  {pnl_emoji} EQUITY: ${equity:>10.2f}  │  P&L: ${total_pnl:>+8.2f} ({total_pnl_pct:>+6.2f}%)              ║")
        print(f"║     Available: ${self.balance:>8.2f}   │  In Positions: ${positions_value:>8.2f}                   ║")
        
        print("╠" + "═" * 78 + "╣")
        print(f"║  POSITIONS ({len(self.positions)}/{self.CONFIG['max_positions']}):                                                         ║")
        
        if self.positions:
            for p in self.positions:
                cp = prices.get(p.symbol, p.entry_price)
                pnl = p.unrealized_pnl_pct(cp)
                emoji = "🟢" if pnl >= 0 else "🔴"
                trail = "📍" if p.trailing_activated else "  "
                
                # Distance to SL and TP
                dist_sl = ((cp - p.stop_loss) / cp) * 100
                dist_tp = ((p.take_profit - cp) / cp) * 100
                
                print(f"║  {emoji} {p.symbol:<12} ${cp:>12.6f}  P&L: {pnl:>+6.2f}%  SL:{dist_sl:>5.1f}%  TP:{dist_tp:>5.1f}% {trail}  ║")
        else:
            print("║     No open positions                                                        ║")
        
        print("╠" + "═" * 78 + "╣")
        print(f"║  STATS: {len(self.closed_trades)} trades │ {wins}W/{losses}L │ WR: {win_rate:>5.1f}% │ Fees: ${total_fees:>6.2f} │ Slip: ${total_slip:>5.2f}  ║")
        
        # Recent trades
        if self.closed_trades:
            print("╠" + "═" * 78 + "╣")
            print("║  RECENT TRADES:                                                              ║")
            for t in self.closed_trades[-3:]:
                emoji = "✅" if t.net_pnl > 0 else "❌"
                print(f"║  {emoji} {t.symbol:<12} {t.exit_reason:<15} P&L: ${t.net_pnl:>+7.2f} ({t.pnl_pct:>+5.1f}%)           ║")
        
        print("╚" + "═" * 78 + "╝")
        print(f"  [Position check: 1min | Signal scan: 5min | Press Ctrl+C to stop]")
    
    def run_live(self):
        """Run live monitoring"""
        print("\n" + "=" * 60)
        print("  🚀 STARTING LIVE PAPER TRADER")
        print("  Position check: every 1 minute")
        print("  Signal scan: every 5 minutes")
        print("=" * 60)
        
        self.last_signal_scan = datetime.now() - timedelta(minutes=10)  # Force initial scan
        
        cycle = 0
        while True:
            try:
                cycle += 1
                
                # Always check positions
                closed = self.check_positions()
                
                if closed:
                    for t in closed:
                        emoji = "✅" if t.net_pnl > 0 else "❌"
                        print(f"\n  {emoji} CLOSED: {t.symbol} - {t.exit_reason} - P&L: ${t.net_pnl:+.2f}")
                
                # Scan for signals every 5 minutes
                time_since_scan = (datetime.now() - self.last_signal_scan).total_seconds()
                if time_since_scan >= self.CONFIG['signal_scan_minutes'] * 60:
                    if len(self.positions) < self.CONFIG['max_positions']:
                        signals = self.scan_signals()
                        
                        for signal in signals:
                            if len(self.positions) >= self.CONFIG['max_positions']:
                                break
                            position = self.open_position(signal)
                            if position:
                                print(f"\n  🚀 OPENED: {position.symbol} @ ${position.entry_price:.6f}")
                    
                    self.last_signal_scan = datetime.now()
                
                # Display
                self.display_live()
                
                # Wait
                time.sleep(self.CONFIG['position_check_seconds'])
                
            except KeyboardInterrupt:
                print("\n\n  Stopping paper trader...")
                self.save_state()
                break
            except Exception as e:
                print(f"\n  Error: {e}")
                time.sleep(10)


def main():
    trader = FastPaperTrader()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'status':
            trader.display_live(clear=False)
        elif sys.argv[1] == 'reset':
            if os.path.exists(trader.CONFIG['state_file']):
                os.remove(trader.CONFIG['state_file'])
            print("  Reset complete.")
        elif sys.argv[1] == 'run':
            trader.run_live()
    else:
        trader.display_live(clear=False)
        print("\n  Commands:")
        print("    python3 paper_trader_fast.py run    - Start live trading")
        print("    python3 paper_trader_fast.py status - Show status")
        print("    python3 paper_trader_fast.py reset  - Reset to $1000")


if __name__ == "__main__":
    main()
