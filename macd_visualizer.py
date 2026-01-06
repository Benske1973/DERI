"""
MACD Money Map Visualizer
=========================
Visual representation of all 3 MACD systems
- System 1: Trend (Zero Line, Crossovers)
- System 2: Reversal (Divergence, Histogram patterns)
- System 3: Confirmation (Multi-timeframe)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from macd_money_map import MACDMoneyMap, CONFIG, calculate_macd, find_swing_points


def plot_macd_analysis(symbol: str, save_file: bool = False):
    """
    Create a comprehensive MACD analysis chart
    """
    scanner = MACDMoneyMap()
    
    # Fetch data for all timeframes
    print(f"Fetching data for {symbol}...")
    
    df_daily = scanner.fetch_data(symbol, '1d', limit=100)
    df_4h = scanner.fetch_data(symbol, '4h', limit=200)
    df_1h = scanner.fetch_data(symbol, '1h', limit=200)
    
    if df_4h is None:
        print(f"Could not fetch data for {symbol}")
        return
    
    # Get analysis
    analysis = scanner.analyze_symbol(symbol)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(f'MACD Money Map Analysis: {symbol}', fontsize=16, fontweight='bold')
    
    # Define grid
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.2)
    
    # ==========================================
    # MAIN CHART: 4H Price + Signals
    # ==========================================
    ax1 = fig.add_subplot(gs[0, :])
    
    # Plot price
    ax1.plot(df_4h['timestamp'], df_4h['close'], color='#2196F3', linewidth=1.5, label='Price')
    
    # Mark signals
    if analysis['final_signal'] == 'BUY':
        ax1.axhline(y=analysis['entry'], color='green', linestyle='--', alpha=0.7, label=f"Entry: {analysis['entry']:.4f}")
        ax1.axhline(y=analysis['stop_loss'], color='red', linestyle=':', alpha=0.7, label=f"SL: {analysis['stop_loss']:.4f}")
        ax1.axhline(y=analysis['take_profit'], color='#4CAF50', linestyle='--', alpha=0.7, label=f"TP: {analysis['take_profit']:.4f}")
        ax1.scatter([df_4h['timestamp'].iloc[-1]], [analysis['entry']], color='green', s=200, marker='^', zorder=5)
    elif analysis['final_signal'] == 'SELL':
        ax1.axhline(y=analysis['entry'], color='red', linestyle='--', alpha=0.7, label=f"Entry: {analysis['entry']:.4f}")
        ax1.axhline(y=analysis['stop_loss'], color='red', linestyle=':', alpha=0.7, label=f"SL: {analysis['stop_loss']:.4f}")
        ax1.axhline(y=analysis['take_profit'], color='#4CAF50', linestyle='--', alpha=0.7, label=f"TP: {analysis['take_profit']:.4f}")
        ax1.scatter([df_4h['timestamp'].iloc[-1]], [analysis['entry']], color='red', s=200, marker='v', zorder=5)
    
    # Signal box
    signal_text = f"Signal: {analysis['final_signal'] or 'NONE'}"
    if analysis['confidence']:
        signal_text += f" ({analysis['confidence']}%)"
    
    ax1.text(0.02, 0.95, signal_text, transform=ax1.transAxes, fontsize=12, 
             fontweight='bold', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax1.set_title('4H Price Chart with Entry/SL/TP', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # ==========================================
    # SYSTEM 1: MACD with Zero Line & Crossovers (4H)
    # ==========================================
    ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
    
    # MACD Lines
    ax2.plot(df_4h['timestamp'], df_4h['macd_line'], color='#2196F3', linewidth=1.5, label='MACD')
    ax2.plot(df_4h['timestamp'], df_4h['signal_line'], color='#FF9800', linewidth=1.5, label='Signal')
    
    # Zero Line (THE FOUNDATION)
    ax2.axhline(y=0, color='white', linewidth=2, label='Zero Line')
    ax2.fill_between(df_4h['timestamp'], 0, df_4h['macd_line'], 
                     where=(df_4h['macd_line'] > 0), color='#4CAF50', alpha=0.3, label='Bullish Zone')
    ax2.fill_between(df_4h['timestamp'], 0, df_4h['macd_line'], 
                     where=(df_4h['macd_line'] < 0), color='#F44336', alpha=0.3, label='Bearish Zone')
    
    # Distance threshold lines
    threshold = CONFIG['trend_system']['distance_threshold']
    ax2.axhline(y=threshold, color='#E91E63', linestyle='--', alpha=0.7, label=f'Distance Rule (+{threshold})')
    ax2.axhline(y=-threshold, color='#E91E63', linestyle='--', alpha=0.7, label=f'Distance Rule (-{threshold})')
    
    # Mark crossovers
    for i in range(1, len(df_4h)):
        macd_prev = df_4h['macd_line'].iloc[i-1]
        macd_curr = df_4h['macd_line'].iloc[i]
        signal_prev = df_4h['signal_line'].iloc[i-1]
        signal_curr = df_4h['signal_line'].iloc[i]
        
        # Bullish crossover
        if macd_prev <= signal_prev and macd_curr > signal_curr:
            color = 'lime' if macd_curr > threshold else 'gray'
            ax2.scatter([df_4h['timestamp'].iloc[i]], [macd_curr], color=color, s=100, marker='^', zorder=5)
        
        # Bearish crossover
        if macd_prev >= signal_prev and macd_curr < signal_curr:
            color = 'red' if macd_curr < -threshold else 'gray'
            ax2.scatter([df_4h['timestamp'].iloc[i]], [macd_curr], color=color, s=100, marker='v', zorder=5)
    
    # Trend bias indicator
    bias = analysis['timeframes'].get('setup', {}).get('trend', {}).get('bias', 'NEUTRAL')
    bias_color = 'green' if bias == 'BULLISH' else 'red' if bias == 'BEARISH' else 'gray'
    ax2.text(0.02, 0.95, f"System 1 - Trend Bias: {bias}", transform=ax2.transAxes, fontsize=11, 
             fontweight='bold', verticalalignment='top', color=bias_color,
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    ax2.set_title('SYSTEM 1: Trend System - MACD Zero Line & Crossovers', fontsize=12)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # ==========================================
    # SYSTEM 2: Histogram with Patterns (4H)
    # ==========================================
    ax3 = fig.add_subplot(gs[2, 0])
    
    # Histogram with colors
    colors = []
    for i in range(len(df_4h)):
        hist = df_4h['histogram'].iloc[i]
        hist_prev = df_4h['histogram'].iloc[i-1] if i > 0 else hist
        
        if hist >= 0:
            colors.append('#26A69A' if hist > hist_prev else '#B2DFDB')  # Growing green or fading green
        else:
            colors.append('#EF5350' if hist < hist_prev else '#FFCDD2')  # Growing red or fading red
    
    ax3.bar(df_4h['timestamp'], df_4h['histogram'], color=colors, width=0.05)
    ax3.axhline(y=0, color='white', linewidth=1)
    
    # Mark histogram patterns
    hist_pattern = analysis['timeframes'].get('setup', {}).get('reversal', {}).get('histogram', {})
    pattern_text = f"Pattern: {hist_pattern.get('pattern', 'None')}"
    if hist_pattern.get('direction'):
        pattern_text += f" ({hist_pattern['direction']})"
    
    ax3.text(0.02, 0.95, f"System 2 - {pattern_text}", transform=ax3.transAxes, fontsize=10, 
             fontweight='bold', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    ax3.set_title('SYSTEM 2: Histogram Patterns (Flip/Shrinking/Bounce)', fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # ==========================================
    # SYSTEM 2: Divergence Detection
    # ==========================================
    ax4 = fig.add_subplot(gs[2, 1])
    
    # Price for divergence
    ax4_price = ax4
    ax4_macd = ax4.twinx()
    
    ax4_price.plot(df_4h['timestamp'].tail(50), df_4h['close'].tail(50), color='#2196F3', linewidth=1.5, label='Price')
    ax4_macd.plot(df_4h['timestamp'].tail(50), df_4h['macd_line'].tail(50), color='#FF9800', linewidth=1.5, label='MACD')
    
    # Check for divergence
    divergence = analysis['timeframes'].get('setup', {}).get('reversal', {}).get('divergence', {})
    if divergence.get('has_divergence'):
        div_type = divergence['type']
        ax4.text(0.02, 0.95, f"System 2 - ⚠️ {div_type} DIVERGENCE DETECTED!", transform=ax4.transAxes, 
                 fontsize=10, fontweight='bold', verticalalignment='top', color='yellow',
                 bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    else:
        ax4.text(0.02, 0.95, "System 2 - No Divergence", transform=ax4.transAxes, 
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    ax4.set_title('SYSTEM 2: Divergence Detection', fontsize=11)
    ax4_price.set_ylabel('Price', color='#2196F3')
    ax4_macd.set_ylabel('MACD', color='#FF9800')
    ax4.grid(True, alpha=0.3)
    
    # ==========================================
    # SYSTEM 3: Triple Timeframe Confirmation
    # ==========================================
    ax5 = fig.add_subplot(gs[3, :])
    
    # Create a visual representation of timeframe alignment
    timeframes = ['Daily (Trend)', '4H (Setup)', '1H (Entry)']
    tf_data = ['trend', 'setup', 'entry']
    
    positions = [0.2, 0.5, 0.8]
    
    for i, (tf_name, tf_key) in enumerate(zip(timeframes, tf_data)):
        tf_analysis = analysis['timeframes'].get(tf_key, {})
        trend_bias = tf_analysis.get('trend', {}).get('bias', 'NEUTRAL')
        
        # Determine color
        if trend_bias == 'BULLISH':
            color = '#4CAF50'
            symbol_marker = '▲'
        elif trend_bias == 'BEARISH':
            color = '#F44336'
            symbol_marker = '▼'
        else:
            color = '#9E9E9E'
            symbol_marker = '●'
        
        # Draw circle
        circle = plt.Circle((positions[i], 0.5), 0.12, color=color, alpha=0.8)
        ax5.add_patch(circle)
        
        # Add text
        ax5.text(positions[i], 0.5, symbol_marker, ha='center', va='center', fontsize=24, color='white', fontweight='bold')
        ax5.text(positions[i], 0.15, tf_name, ha='center', va='center', fontsize=11, fontweight='bold')
        ax5.text(positions[i], 0.02, trend_bias, ha='center', va='center', fontsize=10, color=color)
        
        # Draw connecting arrows
        if i < 2:
            ax5.annotate('', xy=(positions[i+1]-0.13, 0.5), xytext=(positions[i]+0.13, 0.5),
                        arrowprops=dict(arrowstyle='->', color='white', lw=2))
    
    # Alignment status
    alignment = analysis.get('alignment', {})
    if alignment.get('aligned'):
        status_text = f"✅ ALL TIMEFRAMES ALIGNED - {alignment['direction']} SIGNAL"
        status_color = '#4CAF50' if alignment['direction'] == 'BUY' else '#F44336'
    else:
        status_text = "❌ TIMEFRAMES NOT ALIGNED - NO TRADE"
        status_color = '#9E9E9E'
    
    ax5.text(0.5, 0.85, status_text, ha='center', va='center', fontsize=14, 
             fontweight='bold', color=status_color,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8))
    
    ax5.set_xlim(0, 1)
    ax5.set_ylim(-0.1, 1)
    ax5.set_title('SYSTEM 3: Triple Timeframe Confirmation', fontsize=12)
    ax5.axis('off')
    
    # Style
    fig.patch.set_facecolor('#1a1a2e')
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#4a4a6a')
    
    plt.tight_layout()
    
    if save_file:
        filename = f"macd_analysis_{symbol.replace('/', '_')}.png"
        plt.savefig(filename, dpi=150, facecolor=fig.get_facecolor())
        print(f"Saved chart to {filename}")
    
    plt.show()


def plot_quick_overview(symbols: list = None):
    """
    Quick overview of multiple symbols showing their MACD status
    """
    if symbols is None:
        symbols = CONFIG['symbols'][:6]  # Top 6 symbols
    
    scanner = MACDMoneyMap()
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('MACD Money Map - Quick Overview', fontsize=16, fontweight='bold')
    
    for idx, symbol in enumerate(symbols):
        ax = axes[idx // 3, idx % 3]
        
        df = scanner.fetch_data(symbol, '4h', limit=100)
        if df is None:
            continue
        
        analysis = scanner.analyze_symbol(symbol)
        
        # Plot histogram
        colors = ['#26A69A' if h >= 0 else '#EF5350' for h in df['histogram']]
        ax.bar(range(len(df)), df['histogram'], color=colors, width=0.8)
        ax.axhline(y=0, color='white', linewidth=1)
        
        # Signal indicator
        signal = analysis['final_signal']
        if signal == 'BUY':
            ax.set_facecolor('#1a3d1a')
        elif signal == 'SELL':
            ax.set_facecolor('#3d1a1a')
        else:
            ax.set_facecolor('#16213e')
        
        bias = analysis['timeframes'].get('setup', {}).get('trend', {}).get('bias', 'N/A')
        title = f"{symbol}\nBias: {bias}"
        if signal:
            title += f" | Signal: {signal}"
        
        ax.set_title(title, fontsize=10, color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#4a4a6a')
    
    fig.patch.set_facecolor('#1a1a2e')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        symbol = sys.argv[1]
        plot_macd_analysis(symbol, save_file=True)
    else:
        # Default: show BTC analysis
        plot_macd_analysis('BTC/USDT')
