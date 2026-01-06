import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run_init_db() -> None:
    subprocess.check_call([sys.executable, str(ROOT / "init_db.py")])


def start_process(script_name: str) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, str(ROOT / script_name)])


def main() -> int:
    parser = argparse.ArgumentParser(description="Start scanner + watcher + executor.")
    parser.add_argument("--init-db", action="store_true", help="Initialiseer SQLite DB (dropt signals tabel).")
    args = parser.parse_args()

    if args.init_db:
        run_init_db()

    procs: list[subprocess.Popen] = []
    try:
        print("Starting HTF scanner...")
        procs.append(start_process("scanner_htf.py"))
        time.sleep(0.5)

        print("Starting WebSocket watcher...")
        procs.append(start_process("watcher_ws.py"))
        time.sleep(0.5)

        print("Starting LTF executor...")
        procs.append(start_process("executor_ltf.py"))

        # Wacht tot één proces stopt (of Ctrl+C)
        while True:
            for p in procs:
                code = p.poll()
                if code is not None:
                    raise RuntimeError(f"Process exited ({code}).")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"\nBot stopped: {e}")
    finally:
        for p in procs:
            if p.poll() is None:
                try:
                    p.send_signal(signal.SIGTERM)
                except Exception:
                    pass
        # korte grace period
        time.sleep(1)
        for p in procs:
            if p.poll() is None:
                try:
                    p.kill()
                except Exception:
                    pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

