import os
import time
import ccxt
import pandas as pd
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# load API keys
load_dotenv()
API_KEY    = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")

# init Bybit future via CCXT
exchange = ccxt.bybit({
    'apiKey':    API_KEY,
    'secret':    API_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}
})

# Parameters
SYMBOL      = 'BTC/USDT:USDT'
TIMEFRAME   = '1h'
LOOKBACK    = 25   # fetch 25 bars to be safe for 24h+1h
THRESHOLD     = 0.002    # 0.2%
STOP_LOSS_pct = 0.03     # 3%
TAKE_PROFIT_pct = 0.04   # 4%
LEVERAGE      = 6
DUMMY_CAPITAL = 1000     # USDT, for testing qty calc


def fetch_hourly_ohlcv():
    """
    Fetch the most recent LOOKBACK hourly bars and return a pandas DataFrame
    with a UTC datetime index.
    """
    bars = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=LOOKBACK)
    df   = pd.DataFrame(bars, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    return df

def on_entry_trigger():
    print(f"\n[ENTRY @ {pd.Timestamp.utcnow()}]")

    # 1) Get yesterday’s calendar‐day return from Bybit’s 1d bars
    daily_bars = exchange.fetch_ohlcv(SYMBOL, '1d', limit=2)
    prev_prev_close = daily_bars[-2][4]   # close 2 days ago
    prev_close      = daily_bars[-1][4]   # close yesterday
    r = prev_close / prev_prev_close - 1
    print(f"  ↳ Yesterday’s return = {r:.4%}")

    # 2) Check threshold
    if abs(r) < THRESHOLD:
        print(f"  ↳ |{r:.4%}| < {THRESHOLD:.4%} → no trade today.")
        return

    # 3) Get the current 1h bar’s open for our entry price
    hr_bars = exchange.fetch_ohlcv(SYMBOL, '1h', limit=2)
    entry_price = hr_bars[-1][1]  # the “open” of the current in‐progress hour
    direction   = 1 if r > 0 else -1

    # 4) Compute SL/TP and position size
    stop_price = entry_price * (1 - direction * STOP_LOSS_pct)
    take_price = entry_price * (1 + direction * TAKE_PROFIT_pct)
    qty         = (DUMMY_CAPITAL * LEVERAGE) / entry_price
    side        = 'buy' if direction == 1 else 'sell'

    # 5) Print your planned trade
    print(f"  ↳ Signal:    {'LONG' if direction>0 else 'SHORT'}")
    print(f"  ↳ Entry:     {side.upper()} @ {entry_price:.2f}")
    print(f"  ↳ Qty:       {qty:.6f} contracts (on {DUMMY_CAPITAL} USDT at {LEVERAGE}×)")
    print(f"  ↳ Stop‐loss: {stop_price:.2f}")
    print(f"  ↳ Take‐profit: {take_price:.2f}")

    # — here you will place your bracket order via ccxt —

def on_exit_trigger():
    df = fetch_hourly_ohlcv()
    print(f"\n[ EXIT  @ {pd.Timestamp.utcnow()} ]")
    # You can re-use df to decide if a forced exit is needed,
    # or just close any open trade here unconditionally.

if __name__ == "__main__":
    scheduler = BackgroundScheduler(timezone='UTC')

    # schedule entry at 00:00 UTC daily
    scheduler.add_job(
        on_entry_trigger,
        CronTrigger(hour=0, minute=0, timezone='UTC'),
        name="DailyEntry00UTC"
    )

    # schedule exit at 12:00 UTC daily
    scheduler.add_job(
        on_exit_trigger,
        CronTrigger(hour=12, minute=0, timezone='UTC'),
        name="DailyExit12UTC"
    )

    scheduler.start()
    print("Scheduler started—waiting for 00:00 and 12:00 UTC triggers...")

    try:
        # keep the script alive
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        print("Scheduler stopped.")
