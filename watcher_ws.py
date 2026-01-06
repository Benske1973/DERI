# watcher_ws.py
import asyncio
import websockets
import json
import sqlite3
import os
import requests
import time
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "trading_bot.db"
WATCHLIST_REFRESH_SECONDS = int(os.getenv("WATCHLIST_REFRESH_SECONDS", "30"))

def load_scanning_zones(conn: sqlite3.Connection) -> dict[str, tuple[float, float]]:
    """
    Returns a mapping: symbol -> (fvg_top, fvg_bottom) for signals in SCANNING state.
    """
    c = conn.cursor()
    c.execute("SELECT symbol, fvg_top, fvg_bottom FROM signals WHERE status='SCANNING'")
    rows = c.fetchall()
    zones: dict[str, tuple[float, float]] = {}
    for symbol, fvg_top, fvg_bottom in rows:
        zones[str(symbol)] = (float(fvg_top), float(fvg_bottom))
    return zones

async def handle_ws():
    while True: # Blijf proberen bij verbreking
        try:
            print(f"[{time.strftime('%H:%M:%S')}] WebSocket verbinding maken met KuCoin...")
            
            # Token & URL ophalen
            r = requests.post("https://api.kucoin.com/api/v1/bullet-public").json()
            ws_url = r['data']['instanceServers'][0]['endpoint'] + "?token=" + r['data']['token']
            
            async with websockets.connect(ws_url, ping_interval=20, ping_timeout=10) as ws:
                conn = sqlite3.connect(DB_PATH)
                zones = load_scanning_zones(conn)
                last_refresh = time.time()

                # We subscriben op alle tickers om te zien wanneer onze POI's geraakt worden
                await ws.send(json.dumps({
                    "id": str(int(time.time())),
                    "type": "subscribe", 
                    "topic": "/market/ticker:all",
                    "privateChannel": False,
                    "response": True
                }))
                
                print(f"[{time.strftime('%H:%M:%S')}] Verbonden en aan het monitoren...")

                while True:
                    try:
                        # Periodiek watchlist refresh (nieuwe scans oppakken)
                        now = time.time()
                        if now - last_refresh >= WATCHLIST_REFRESH_SECONDS:
                            zones = load_scanning_zones(conn)
                            last_refresh = now

                        msg = await asyncio.wait_for(ws.recv(), timeout=30)
                        data = json.loads(msg)
                        
                        if data.get("type") == "message" and "data" in data:
                            symbol = data['topic'].split(":")[-1]
                            price = float(data['data']['price'])

                            zone = zones.get(symbol)
                            if zone is None:
                                continue

                            fvg_top, fvg_bottom = zone
                            # De "Tap" check
                            if price <= fvg_top and price >= fvg_bottom:
                                c = conn.cursor()
                                c.execute("UPDATE signals SET status='TAPPED' WHERE symbol=? AND status='SCANNING'", (symbol,))
                                conn.commit()
                                zones.pop(symbol, None)
                                print(f"🎯 ALERT: {symbol} tapt de zone op {price}! Executor pakt het over.")
                            
                    except asyncio.TimeoutError:
                        # Stuur een ping om de verbinding levend te houden
                        await ws.send(json.dumps({"type": "ping"}))
                    except Exception as e:
                        print(f"Fout in berichtverwerking: {e}")
                        break # Breekt de inner loop om opnieuw te verbinden
                try:
                    conn.close()
                except Exception:
                    pass

        except Exception as e:
            print(f"Verbindingsfout: {e}. Opnieuw proberen in 5 seconden...")
            await asyncio.sleep(5)

if __name__ == "__main__":
    try:
        asyncio.run(handle_ws())
    except KeyboardInterrupt:
        print("\nWatcher gestopt door gebruiker.")