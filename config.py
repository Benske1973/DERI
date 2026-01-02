# config.py - KuCoin Multiscanner Papertrader Configuration
"""
Central configuration for the KuCoin Multiscanner Papertrader.
All settings can be adjusted here without modifying core code.
"""

from dataclasses import dataclass, field
from typing import List, Dict
from enum import Enum

class TimeFrame(Enum):
    """Supported timeframes for analysis."""
    M1 = "1min"
    M3 = "3min"
    M5 = "5min"
    M15 = "15min"
    M30 = "30min"
    H1 = "1hour"
    H2 = "2hour"
    H4 = "4hour"
    H6 = "6hour"
    H8 = "8hour"
    H12 = "12hour"
    D1 = "1day"
    W1 = "1week"

class TrendDirection(Enum):
    """Market trend direction."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"

class SignalType(Enum):
    """Signal types for trading."""
    LONG = "LONG"
    SHORT = "SHORT"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"

class OrderStatus(Enum):
    """Order status for paper trading."""
    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"
    CLOSED = "CLOSED"

class PositionStatus(Enum):
    """Position status."""
    SCANNING = "SCANNING"
    TAPPED = "TAPPED"
    CONFIRMED = "CONFIRMED"
    IN_TRADE = "IN_TRADE"
    CLOSED_TP = "CLOSED_TP"
    CLOSED_SL = "CLOSED_SL"
    CLOSED_MANUAL = "CLOSED_MANUAL"

@dataclass
class ScannerConfig:
    """Scanner configuration."""
    # Symbols to scan (empty = scan all USDT pairs)
    symbols: List[str] = field(default_factory=lambda: [])

    # Auto-discover top volume pairs
    auto_discover: bool = True
    top_pairs_count: int = 50
    min_volume_24h: float = 1_000_000  # Minimum 24h volume in USDT

    # Timeframes for analysis
    htf_timeframe: TimeFrame = TimeFrame.H4  # Higher timeframe for trend/POI
    ltf_timeframe: TimeFrame = TimeFrame.M15  # Lower timeframe for entry

    # Scan intervals (seconds)
    htf_scan_interval: int = 900  # 15 minutes
    ltf_scan_interval: int = 30   # 30 seconds

    # Filters
    exclude_stablecoins: bool = True
    exclude_leveraged: bool = True
    stablecoins: List[str] = field(default_factory=lambda: [
        "USDT", "USDC", "DAI", "BUSD", "TUSD", "USDP", "USDD", "FRAX"
    ])

@dataclass
class IndicatorConfig:
    """Technical indicator settings."""
    # EMAs
    ema_fast: int = 9
    ema_medium: int = 21
    ema_slow: int = 50
    ema_trend: int = 200

    # RSI
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0

    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0

    # ATR
    atr_period: int = 14

    # Volume
    volume_ma_period: int = 20
    volume_spike_multiplier: float = 2.0

    # SMC Settings
    swing_lookback: int = 10
    fvg_min_size_atr: float = 0.5  # FVG minimum size in ATR
    ob_validity_bars: int = 50     # Order block validity

@dataclass
class RiskConfig:
    """Risk management settings."""
    # Capital
    initial_capital: float = 10000.0  # Paper trading capital

    # Position sizing
    risk_per_trade: float = 0.01      # 1% risk per trade
    max_position_size: float = 0.10   # Max 10% of capital per position
    max_open_positions: int = 5       # Maximum concurrent positions

    # Risk/Reward
    min_risk_reward: float = 2.0      # Minimum R:R ratio
    default_risk_reward: float = 3.0  # Default R:R if not specified

    # Stop Loss
    use_atr_stops: bool = True
    atr_sl_multiplier: float = 1.5
    max_sl_percent: float = 0.05      # Maximum 5% SL

    # Take Profit
    use_partial_tp: bool = True
    partial_tp_levels: List[Dict] = field(default_factory=lambda: [
        {"level": 1.0, "close_percent": 0.33},  # Close 33% at 1R
        {"level": 2.0, "close_percent": 0.33},  # Close 33% at 2R
        {"level": 3.0, "close_percent": 0.34},  # Close 34% at 3R
    ])

    # Trailing Stop
    use_trailing_stop: bool = True
    trailing_activation: float = 1.5  # Activate at 1.5R profit
    trailing_distance: float = 0.5    # Trail by 0.5R

@dataclass
class StrategyConfig:
    """Trading strategy settings."""
    # SMC Strategy
    require_trend_alignment: bool = True
    require_fvg: bool = True
    require_ob: bool = True
    require_choch: bool = True        # Change of Character confirmation
    require_bos: bool = False         # Break of Structure

    # Entry filters
    min_score: float = 70.0           # Minimum signal score (0-100)
    require_volume_confirmation: bool = True
    require_momentum_confirmation: bool = True

    # Session filters
    use_session_filter: bool = True
    allowed_sessions: List[str] = field(default_factory=lambda: [
        "LONDON",    # 08:00-17:00 UTC
        "NEW_YORK",  # 13:00-22:00 UTC
        "OVERLAP"    # 13:00-17:00 UTC (London/NY overlap)
    ])

    # Day filters
    trade_weekends: bool = False

@dataclass
class WebSocketConfig:
    """WebSocket configuration."""
    reconnect_delay: int = 5
    ping_interval: int = 20
    ping_timeout: int = 10
    max_reconnect_attempts: int = 10

@dataclass
class DatabaseConfig:
    """Database configuration."""
    db_path: str = "kucoin_papertrader.db"
    backup_enabled: bool = True
    backup_interval: int = 3600  # Every hour

@dataclass
class LogConfig:
    """Logging configuration."""
    log_level: str = "INFO"
    log_to_file: bool = True
    log_file: str = "papertrader.log"
    max_log_size: int = 10_000_000  # 10MB
    backup_count: int = 5

@dataclass
class Config:
    """Main configuration container."""
    scanner: ScannerConfig = field(default_factory=ScannerConfig)
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    websocket: WebSocketConfig = field(default_factory=WebSocketConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    logging: LogConfig = field(default_factory=LogConfig)

    # KuCoin API (public endpoints only for paper trading)
    kucoin_base_url: str = "https://api.kucoin.com"
    kucoin_ws_public: str = "https://api.kucoin.com/api/v1/bullet-public"

# Global configuration instance
config = Config()

def load_config(path: str = None) -> Config:
    """Load configuration from file if exists, otherwise use defaults."""
    import json
    import os

    if path and os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
            # Update config with loaded values
            # Implementation for config loading
            pass

    return config

def save_config(cfg: Config, path: str = "config.json"):
    """Save configuration to file."""
    import json
    from dataclasses import asdict

    with open(path, 'w') as f:
        json.dump(asdict(cfg), f, indent=2, default=str)
