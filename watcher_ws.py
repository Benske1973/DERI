# watcher_ws.py
import asyncio
import websockets
import json
import sqlite3
import requests
import time

async def handle_ws():
    while True: # Blijf proberen bij verbreking
        try:
            print(f"[{time.strftime('%H:%M:%S')}] WebSocket verbinding maken met KuCoin...")
            
            # Token & URL ophalen
            r = requests.post("https://api.kucoin.com/api/v1/bullet-public").json()
            ws_url = r['data']['instanceServers'][0]['endpoint'] + "?token=" + r['data']['token']
            
            async with websockets.connect(ws_url, ping_interval=20, ping_timeout=10) as ws:
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
                        msg = await asyncio.wait_for(ws.recv(), timeout=30)
                        data = json.loads(msg)
                        
                        if data.get("type") == "message" and "data" in data:
                            symbol = data['topic'].split(":")[-1]
                            price = float(data['data']['price'])

                            # Check in SQLite
                            conn = sqlite3.connect('trading_bot.db')
                            c = conn.cursor()
                            c.execute("SELECT fvg_top, fvg_bottom, status FROM signals WHERE symbol=?", (symbol,))
                            row = c.fetchone()

                            if row and row[2] == 'SCANNING':
                                fvg_top, fvg_bottom = row[0], row[1]
                                # De "Tap" check
                                if price <= fvg_top and price >= fvg_bottom:
                                    c.execute("UPDATE signals SET status='TAPPED' WHERE symbol=?", (symbol,))
                                    conn.commit()
                                    print(f"🎯 ALERT: {symbol} tapt de zone op {price}! Executor pakt het over.")
                            
                            conn.close()
                            
                    except asyncio.TimeoutError:
                        # Stuur een ping om de verbinding levend te houden
                        await ws.send(json.dumps({"type": "ping"}))
                    except Exception as e:
                        print(f"Fout in berichtverwerking: {e}")
                        break # Breekt de inner loop om opnieuw te verbinden

        except Exception as e:
            print(f"Verbindingsfout: {e}. Opnieuw proberen in 5 seconden...")
            await asyncio.sleep(5)

if __name__ == "__main__":
    try:
        asyncio.run(handle_ws())
    except KeyboardInterrupt:
        print("\nWatcher gestopt door gebruiker.")