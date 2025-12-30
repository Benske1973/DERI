# config.py - Centrale configuratie voor de trading bot
import os

# === API CONFIGURATIE ===
KUCOIN_BASE_URL = "https://api.kucoin.com"
KUCOIN_WS_PUBLIC = "/api/v1/bullet-public"

# === TRADING PARAMETERS ===
MIN_VOLUME_USDT = 50000  # Minimum 24h volume in USDT
RISK_REWARD_RATIO = 3    # 1:3 R:R ratio
HTF_TIMEFRAME = "4hour"  # Higher timeframe voor setups
LTF_TIMEFRAME = "15min"  # Lower timeframe voor confirmatie
EMA_LENGTH = 50          # EMA periode voor trend filter

# === TIMING ===
SCANNER_INTERVAL = 900   # 15 minuten
EXECUTOR_INTERVAL = 30   # 30 seconden
MONITOR_INTERVAL = 10    # 10 seconden voor TP/SL check

# === RATE LIMITING ===
API_RATE_LIMIT = 0.2     # Seconden tussen API calls (5 per seconde max)
MAX_RETRIES = 3          # Max aantal retries bij API failure
RETRY_DELAY = 2          # Seconden wachten voor retry

# === DATABASE ===
DATABASE_PATH = os.path.join(os.path.dirname(__file__), 'trading_bot.db')

# === LOGGING ===
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR

# === TELEGRAM (optioneel) ===
TELEGRAM_ENABLED = False
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# === WEB DASHBOARD ===
DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_PORT = 5000
