# telegram_alerts.py - Telegram notificaties voor trades
import requests
import sqlite3
import time
from datetime import datetime, timedelta
from config import (
    TELEGRAM_ENABLED, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    DATABASE_PATH
)
from logger import get_telegram_logger

log = get_telegram_logger()


class TelegramBot:
    """Telegram bot voor trade alerts."""

    def __init__(self):
        self.enabled = TELEGRAM_ENABLED and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID
        self.token = TELEGRAM_BOT_TOKEN
        self.chat_id = TELEGRAM_CHAT_ID
        self.base_url = f"https://api.telegram.org/bot{self.token}"

        if self.enabled:
            log.info("Telegram alerts ingeschakeld")
        else:
            log.info("Telegram alerts uitgeschakeld (geen token/chat_id)")

    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Stuur een bericht naar Telegram."""
        if not self.enabled:
            log.debug(f"Telegram disabled, zou sturen: {text[:50]}...")
            return False

        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True
            }

            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()

            result = response.json()
            if result.get("ok"):
                log.debug("Telegram bericht verzonden")
                return True
            else:
                log.warning(f"Telegram error: {result}")
                return False

        except Exception as e:
            log.error(f"Telegram send error: {e}")
            return False

    def format_setup_alert(self, symbol: str, trend: str, fvg_top: float,
                          fvg_bottom: float) -> str:
        """Format een setup alert bericht."""
        emoji = "🟢" if trend == "BULLISH" else "🔴"

        return f"""
{emoji} <b>NIEUWE SETUP</b> {emoji}

<b>Symbol:</b> {symbol}
<b>Trend:</b> {trend}
<b>FVG Zone:</b> {fvg_bottom:.6f} - {fvg_top:.6f}

⏳ Wachten op zone tap...
"""

    def format_tap_alert(self, symbol: str, trend: str, price: float) -> str:
        """Format een zone tap alert."""
        emoji = "🎯"

        return f"""
{emoji} <b>ZONE TAP!</b> {emoji}

<b>Symbol:</b> {symbol}
<b>Trend:</b> {trend}
<b>Tap Price:</b> {price:.6f}

⏳ Wachten op LTF confirmatie...
"""

    def format_trade_alert(self, symbol: str, direction: str, entry: float,
                          sl: float, tp: float) -> str:
        """Format een nieuwe trade alert."""
        emoji = "🚀" if direction == "BULLISH" else "📉"
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        rr_ratio = reward / risk if risk > 0 else 0

        return f"""
{emoji} <b>TRADE GEACTIVEERD</b> {emoji}

<b>Symbol:</b> {symbol}
<b>Direction:</b> {direction}
<b>Entry:</b> {entry:.6f}
<b>Stop Loss:</b> {sl:.6f}
<b>Take Profit:</b> {tp:.6f}
<b>R:R Ratio:</b> 1:{rr_ratio:.1f}
"""

    def format_close_alert(self, symbol: str, direction: str, entry: float,
                          exit_price: float, result: str, pnl: float) -> str:
        """Format een trade close alert."""
        if result == "WIN":
            emoji = "🎯💰"
            result_text = "WINST"
        else:
            emoji = "❌"
            result_text = "VERLIES"

        return f"""
{emoji} <b>TRADE GESLOTEN</b> {emoji}

<b>Symbol:</b> {symbol}
<b>Direction:</b> {direction}
<b>Result:</b> {result_text}
<b>Entry:</b> {entry:.6f}
<b>Exit:</b> {exit_price:.6f}
<b>PnL:</b> {pnl:+.2f}%
"""

    def format_daily_summary(self, stats: dict) -> str:
        """Format een dagelijkse samenvatting."""
        return f"""
📊 <b>DAGELIJKSE SAMENVATTING</b> 📊

<b>Datum:</b> {datetime.now().strftime('%Y-%m-%d')}

<b>Trades vandaag:</b> {stats.get('today_trades', 0)}
<b>Wins:</b> {stats.get('wins', 0)}
<b>Losses:</b> {stats.get('losses', 0)}
<b>Win Rate:</b> {stats.get('win_rate', 0):.1f}%

<b>PnL vandaag:</b> {stats.get('today_pnl', 0):+.2f}%
<b>Totale PnL:</b> {stats.get('total_pnl', 0):+.2f}%

<b>Open posities:</b> {stats.get('open_trades', 0)}
<b>Actieve setups:</b> {stats.get('active_setups', 0)}
"""


# Global bot instance
bot = TelegramBot()


def send_setup_alert(symbol: str, trend: str, fvg_top: float, fvg_bottom: float):
    """Stuur setup alert."""
    msg = bot.format_setup_alert(symbol, trend, fvg_top, fvg_bottom)
    bot.send_message(msg)


def send_tap_alert(symbol: str, trend: str, price: float):
    """Stuur tap alert."""
    msg = bot.format_tap_alert(symbol, trend, price)
    bot.send_message(msg)


def send_trade_alert(symbol: str, direction: str, entry: float, sl: float, tp: float):
    """Stuur trade alert."""
    msg = bot.format_trade_alert(symbol, direction, entry, sl, tp)
    bot.send_message(msg)


def send_close_alert(symbol: str, direction: str, entry: float,
                    exit_price: float, result: str, pnl: float):
    """Stuur close alert."""
    msg = bot.format_close_alert(symbol, direction, entry, exit_price, result, pnl)
    bot.send_message(msg)


def get_daily_stats() -> dict:
    """Haal dagelijkse statistieken op."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row

    try:
        c = conn.cursor()
        today = datetime.now().strftime('%Y-%m-%d')

        # Trades vandaag
        c.execute("""
            SELECT
                COUNT(*) as today_trades,
                SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END) as losses,
                COALESCE(SUM(pnl_percent), 0) as today_pnl
            FROM trade_log
            WHERE date(timestamp) = ? AND status = 'CLOSED'
        """, (today,))
        today_stats = c.fetchone()

        # Totale PnL
        c.execute("SELECT COALESCE(SUM(pnl_percent), 0) as total FROM trade_log WHERE status = 'CLOSED'")
        total_pnl = c.fetchone()['total']

        # Open trades
        c.execute("SELECT COUNT(*) as count FROM trade_log WHERE status = 'OPEN'")
        open_trades = c.fetchone()['count']

        # Active setups
        c.execute("SELECT COUNT(*) as count FROM signals WHERE status IN ('SCANNING', 'TAPPED')")
        active_setups = c.fetchone()['count']

        wins = today_stats['wins'] or 0
        losses = today_stats['losses'] or 0
        total_today = wins + losses
        win_rate = (wins / total_today * 100) if total_today > 0 else 0

        return {
            'today_trades': today_stats['today_trades'] or 0,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'today_pnl': today_stats['today_pnl'] or 0,
            'total_pnl': total_pnl or 0,
            'open_trades': open_trades,
            'active_setups': active_setups
        }

    finally:
        conn.close()


def send_daily_summary():
    """Stuur dagelijkse samenvatting."""
    stats = get_daily_stats()
    msg = bot.format_daily_summary(stats)
    bot.send_message(msg)


if __name__ == "__main__":
    log.info("Telegram Alert Module")
    log.info("=" * 40)

    if bot.enabled:
        # Test bericht
        bot.send_message("🤖 <b>Trading Bot gestart!</b>\n\nAlle systemen operationeel.")

        # Toon stats
        send_daily_summary()
    else:
        log.info("Telegram is niet geconfigureerd.")
        log.info("Stel TELEGRAM_BOT_TOKEN en TELEGRAM_CHAT_ID in config.py")
        log.info("of als environment variables.")
