# backtester.py - Backtesting module voor strategie validatie
import requests
import pandas as pd
import pandas_ta as ta
import sqlite3
import time
from datetime import datetime, timedelta
from config import (
    KUCOIN_BASE_URL, HTF_TIMEFRAME, LTF_TIMEFRAME,
    EMA_LENGTH, RISK_REWARD_RATIO, API_RATE_LIMIT,
    DATABASE_PATH
)
from logger import setup_logger

log = setup_logger('backtester')


class Backtester:
    """SMC Strategy Backtester."""

    def __init__(self, symbol: str, days: int = 30):
        self.symbol = symbol
        self.days = days
        self.trades = []
        self.htf_data = None
        self.ltf_data = None

    def fetch_historical_data(self, timeframe: str, limit: int = 1500) -> pd.DataFrame:
        """Haal historische candle data op."""
        try:
            time.sleep(API_RATE_LIMIT)
            url = f"{KUCOIN_BASE_URL}/api/v1/market/candles?symbol={self.symbol}&type={timeframe}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get('code') != '200000' or not data.get('data'):
                log.error(f"Geen data voor {self.symbol} {timeframe}")
                return None

            df = pd.DataFrame(
                data['data'],
                columns=['ts', 'o', 'c', 'h', 'l', 'v', 'a']
            ).astype(float)

            # Converteer timestamp naar datetime
            df['datetime'] = pd.to_datetime(df['ts'], unit='s')
            df = df.iloc[::-1].reset_index(drop=True)  # Oudste eerst

            log.info(f"Fetched {len(df)} {timeframe} candles voor {self.symbol}")
            return df

        except Exception as e:
            log.error(f"Data fetch error: {e}")
            return None

    def detect_bullish_fvg(self, df: pd.DataFrame, idx: int) -> tuple:
        """Detecteer Bullish FVG op specifieke index."""
        if idx < 2:
            return None

        # Bullish FVG: Low[idx] > High[idx-2]
        if df['l'].iloc[idx] > df['h'].iloc[idx - 2]:
            fvg_top = df['l'].iloc[idx]
            fvg_bottom = df['h'].iloc[idx - 2]
            ob_top = df['h'].iloc[idx - 2]
            ob_bottom = df['l'].iloc[idx - 2]
            return ('BULLISH', fvg_top, fvg_bottom, ob_top, ob_bottom)

        return None

    def detect_bearish_fvg(self, df: pd.DataFrame, idx: int) -> tuple:
        """Detecteer Bearish FVG op specifieke index."""
        if idx < 2:
            return None

        # Bearish FVG: High[idx] < Low[idx-2]
        if df['h'].iloc[idx] < df['l'].iloc[idx - 2]:
            fvg_top = df['l'].iloc[idx - 2]
            fvg_bottom = df['h'].iloc[idx]
            ob_top = df['h'].iloc[idx - 2]
            ob_bottom = df['l'].iloc[idx - 2]
            return ('BEARISH', fvg_top, fvg_bottom, ob_top, ob_bottom)

        return None

    def simulate_trade(self, setup: dict, future_data: pd.DataFrame) -> dict:
        """Simuleer een trade met toekomstige data."""
        direction = setup['direction']
        entry = setup['entry']
        sl = setup['sl']
        tp = setup['tp']

        for idx, row in future_data.iterrows():
            high = row['h']
            low = row['l']

            if direction == 'BULLISH':
                # Check SL eerst (worst case)
                if low <= sl:
                    return {
                        'result': 'LOSS',
                        'exit_price': sl,
                        'pnl_percent': ((sl - entry) / entry) * 100,
                        'bars_held': idx + 1
                    }
                # Check TP
                if high >= tp:
                    return {
                        'result': 'WIN',
                        'exit_price': tp,
                        'pnl_percent': ((tp - entry) / entry) * 100,
                        'bars_held': idx + 1
                    }

            else:  # BEARISH
                # Check SL eerst
                if high >= sl:
                    return {
                        'result': 'LOSS',
                        'exit_price': sl,
                        'pnl_percent': ((entry - sl) / entry) * 100,
                        'bars_held': idx + 1
                    }
                # Check TP
                if low <= tp:
                    return {
                        'result': 'WIN',
                        'exit_price': tp,
                        'pnl_percent': ((entry - tp) / entry) * 100,
                        'bars_held': idx + 1
                    }

        # Trade nog open na alle data
        last_close = future_data['c'].iloc[-1]
        if direction == 'BULLISH':
            pnl = ((last_close - entry) / entry) * 100
        else:
            pnl = ((entry - last_close) / entry) * 100

        return {
            'result': 'OPEN',
            'exit_price': last_close,
            'pnl_percent': pnl,
            'bars_held': len(future_data)
        }

    def run_backtest(self) -> dict:
        """Voer de backtest uit."""
        log.info(f"Starting backtest voor {self.symbol}...")

        # Haal HTF data op
        self.htf_data = self.fetch_historical_data(HTF_TIMEFRAME)
        if self.htf_data is None or len(self.htf_data) < EMA_LENGTH + 50:
            return {'error': 'Onvoldoende HTF data'}

        # Bereken EMA
        self.htf_data['ema'] = ta.ema(self.htf_data['c'], length=EMA_LENGTH)

        # Loop door data en zoek setups
        for i in range(EMA_LENGTH + 10, len(self.htf_data) - 20):
            current_close = self.htf_data['c'].iloc[i]
            current_ema = self.htf_data['ema'].iloc[i]

            setup = None

            # Bullish setup (prijs boven EMA)
            if current_close > current_ema:
                fvg = self.detect_bullish_fvg(self.htf_data, i)
                if fvg:
                    direction, fvg_top, fvg_bottom, ob_top, ob_bottom = fvg
                    entry = ob_top
                    sl = ob_bottom
                    risk = entry - sl
                    tp = entry + (risk * RISK_REWARD_RATIO)

                    setup = {
                        'direction': direction,
                        'entry': entry,
                        'sl': sl,
                        'tp': tp,
                        'fvg_top': fvg_top,
                        'fvg_bottom': fvg_bottom,
                        'timestamp': self.htf_data['datetime'].iloc[i]
                    }

            # Bearish setup (prijs onder EMA)
            elif current_close < current_ema:
                fvg = self.detect_bearish_fvg(self.htf_data, i)
                if fvg:
                    direction, fvg_top, fvg_bottom, ob_top, ob_bottom = fvg
                    entry = ob_bottom
                    sl = ob_top
                    risk = sl - entry
                    tp = entry - (risk * RISK_REWARD_RATIO)

                    setup = {
                        'direction': direction,
                        'entry': entry,
                        'sl': sl,
                        'tp': tp,
                        'fvg_top': fvg_top,
                        'fvg_bottom': fvg_bottom,
                        'timestamp': self.htf_data['datetime'].iloc[i]
                    }

            # Simuleer trade als setup gevonden
            if setup:
                # Check eerst of prijs de zone tapt
                future_data = self.htf_data.iloc[i+1:i+21].copy()

                zone_tapped = False
                for idx, row in future_data.iterrows():
                    if setup['fvg_bottom'] <= row['l'] <= setup['fvg_top'] or \
                       setup['fvg_bottom'] <= row['h'] <= setup['fvg_top']:
                        zone_tapped = True
                        tap_idx = idx
                        break

                if zone_tapped:
                    # Simuleer trade vanaf tap
                    trade_data = self.htf_data.iloc[tap_idx+1:].head(50)
                    if len(trade_data) > 0:
                        result = self.simulate_trade(setup, trade_data)
                        result['setup'] = setup
                        self.trades.append(result)

        return self.calculate_stats()

    def calculate_stats(self) -> dict:
        """Bereken backtest statistieken."""
        if not self.trades:
            return {
                'symbol': self.symbol,
                'total_trades': 0,
                'message': 'Geen trades gevonden in backtest periode'
            }

        wins = [t for t in self.trades if t['result'] == 'WIN']
        losses = [t for t in self.trades if t['result'] == 'LOSS']
        opens = [t for t in self.trades if t['result'] == 'OPEN']

        total = len(self.trades)
        closed = len(wins) + len(losses)
        win_rate = (len(wins) / closed * 100) if closed > 0 else 0

        total_pnl = sum(t['pnl_percent'] for t in self.trades)
        avg_pnl = total_pnl / total if total > 0 else 0

        avg_win = sum(t['pnl_percent'] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t['pnl_percent'] for t in losses) / len(losses) if losses else 0

        avg_bars = sum(t['bars_held'] for t in self.trades) / total if total > 0 else 0

        return {
            'symbol': self.symbol,
            'total_trades': total,
            'wins': len(wins),
            'losses': len(losses),
            'open': len(opens),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_bars_held': avg_bars,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0
        }

    def save_results(self, stats: dict):
        """Sla backtest resultaten op in database."""
        conn = sqlite3.connect(DATABASE_PATH)
        c = conn.cursor()

        for trade in self.trades:
            setup = trade['setup']
            c.execute("""
                INSERT INTO backtest_results
                (symbol, direction, entry_price, sl, tp, result, pnl_percent, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.symbol,
                setup['direction'],
                setup['entry'],
                setup['sl'],
                setup['tp'],
                trade['result'],
                trade['pnl_percent'],
                setup['timestamp'].isoformat()
            ))

        conn.commit()
        conn.close()
        log.info(f"Saved {len(self.trades)} backtest trades to database")


def run_multi_backtest(symbols: list) -> list:
    """Run backtest op meerdere symbols."""
    results = []

    for symbol in symbols:
        log.info(f"\n{'='*50}")
        log.info(f"Backtesting: {symbol}")
        log.info('='*50)

        bt = Backtester(symbol)
        stats = bt.run_backtest()

        if 'error' not in stats:
            bt.save_results(stats)

        results.append(stats)

        # Rate limiting tussen symbols
        time.sleep(1)

    return results


def print_results(results: list):
    """Print backtest resultaten."""
    print("\n" + "="*70)
    print("BACKTEST RESULTATEN")
    print("="*70)

    total_trades = 0
    total_wins = 0
    total_losses = 0
    total_pnl = 0

    for r in results:
        if 'error' in r:
            print(f"\n{r['symbol']}: {r.get('message', r['error'])}")
            continue

        print(f"\n{r['symbol']}:")
        print(f"  Trades: {r['total_trades']} (W:{r['wins']} L:{r['losses']} O:{r['open']})")
        print(f"  Win Rate: {r['win_rate']:.1f}%")
        print(f"  Total PnL: {r['total_pnl']:+.2f}%")
        print(f"  Avg PnL: {r['avg_pnl']:+.2f}%")
        print(f"  Profit Factor: {r['profit_factor']:.2f}")

        total_trades += r['total_trades']
        total_wins += r['wins']
        total_losses += r['losses']
        total_pnl += r['total_pnl']

    print("\n" + "="*70)
    print("TOTAAL OVER ALLE SYMBOLS:")
    print(f"  Trades: {total_trades}")
    print(f"  Wins: {total_wins}")
    print(f"  Losses: {total_losses}")
    if total_wins + total_losses > 0:
        print(f"  Overall Win Rate: {total_wins/(total_wins+total_losses)*100:.1f}%")
    print(f"  Total PnL: {total_pnl:+.2f}%")
    print("="*70)


if __name__ == "__main__":
    log.info("SMC Strategy Backtester")
    log.info("="*50)

    # Test symbols
    test_symbols = [
        'BTC-USDT',
        'ETH-USDT',
        'SOL-USDT',
        'AVAX-USDT',
        'LINK-USDT'
    ]

    results = run_multi_backtest(test_symbols)
    print_results(results)
