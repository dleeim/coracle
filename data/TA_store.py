#!/usr/bin/env python3
"""
Download OHLCV from Bybit via CCXT  ➜  compute TA‑Lib features  ➜  MySQL
One table per timeframe:  ta_features_<tf>
"""
import os, warnings, math, urllib.parse, argparse
import numpy as np, pandas as pd
import ccxt, talib
from talib import abstract
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# ────────────────────────────────────────────────────────────────────────────
# 1.  MySQL helpers
# ────────────────────────────────────────────────────────────────────────────
def get_mysql_engine():
    load_dotenv()
    pwd = urllib.parse.quote_plus(os.getenv('MYSQL_PASSWORD'))
    url = f"mysql+pymysql://{os.getenv('MYSQL_USER')}:{pwd}" \
          f"@{os.getenv('MYSQL_HOST')}:{os.getenv('MYSQL_PORT')}/" \
          f"{os.getenv('MYSQL_DATABASE')}"
    return create_engine(url, pool_pre_ping=True, pool_recycle=3600)

engine = get_mysql_engine()
DBNAME = os.getenv('MYSQL_DATABASE')

def ensure_table_and_columns(table, frame):
    """
    • Creates the table if missing, then adds any new columns.
    Works with SQLAlchemy ≥ 1.4 and 2.x.
    """
    with engine.begin() as conn:                    # ← opens + commits tx
        # 1) create skeleton if absent
        conn.exec_driver_sql(f"""
            CREATE TABLE IF NOT EXISTS {table} (
                datetime DATETIME NOT NULL PRIMARY KEY
            ) ENGINE=InnoDB
        """)
        # 2) collect existing cols
        existing = pd.read_sql(text("""
            SELECT COLUMN_NAME
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = :db AND TABLE_NAME = :tbl
        """), conn, params={'db': DBNAME, 'tbl': table})['COLUMN_NAME'].tolist()

        # 3) add missing
        for col in [c for c in frame.columns if c not in existing]:
            conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN `{col}` DOUBLE NULL")

# ------------------------------------------------------------------
# 2) bulk loader  — NaN‑safe version
# ------------------------------------------------------------------
def bulk_upsert(table, frame, chunk=20_000):
    """
    UPSERT DataFrame → MySQL
    • ±inf  → NaN
    • NaN   → Python None → SQL NULL
    Works with SQLAlchemy ≥ 2.0
    """
    # --- 2.1  sanitise dataframe -------------------------------------------
    clean = (frame
             .replace([np.inf, -np.inf], np.nan)      # kill ±inf first
             .astype(object))                         # allow mixed types

    # --- 2.2  SQL parts -----------------------------------------------------
    cols = ", ".join(f"`{c}`" for c in clean.columns)
    ph   = ", ".join(["%s"] * len(clean.columns))
    upd  = ", ".join(f"`{c}`=VALUES(`{c}`)" for c in clean.columns)
    sql  = f"INSERT INTO {table} ({cols}) VALUES ({ph}) " \
           f"ON DUPLICATE KEY UPDATE {upd}"

    # --- 2.3  stream in chunks ---------------------------------------------
    conn = engine.raw_connection()
    cur  = conn.cursor()
    try:
        for start in range(0, len(clean), chunk):
            part = clean.iloc[start:start+chunk]
            # → convert every float‑nan to None
            rows = [
                [None if (isinstance(x, float) and math.isnan(x)) else x
                 for x in row]
                for row in part.itertuples(index=False, name=None)
            ]
            cur.executemany(sql, rows)
            conn.commit()
    finally:
        cur.close()
        conn.close()

# ────────────────────────────────────────────────────────────────────────────
# 3.  TA‑Lib helpers
# ────────────────────────────────────────────────────────────────────────────
def flat_names(d):    # flatten TA‑Lib OrderedDict of names
    out = []; [out.extend(v if isinstance(v,list) else [v]) for v in d.values()]
    return out

def calc_indicators(df):
    res = {}
    for ind in talib.get_functions():
        try:
            fn  = abstract.Function(ind)
            arr = fn(*[df[c] for c in flat_names(fn.input_names)])
            if isinstance(arr, (list, tuple)):
                for n, s in zip(fn.output_names, arr):
                    res[f"{ind}_{n}"] = s
            else:
                res[f"{ind}_{fn.output_names[0]}"] = arr
        except Exception as e:
            warnings.warn(f"{ind} skipped: {e}")
    return pd.DataFrame(res, index=df.index)

# ────────────────────────────────────────────────────────────────────────────
# 4.  CCXT downloader
# ────────────────────────────────────────────────────────────────────────────
def fetch_ohlcv(symbol, timeframe, perp):
    exch = ccxt.bybit({
        'enableRateLimit': True,
        'options': {'defaultType': 'future' if perp else 'spot'}
    })
    since = exch.parse8601('2015-01-01T00:00:00Z')
    limit = 1000
    data  = []
    while True:
        chunk = exch.fetch_ohlcv(symbol, timeframe, since, limit)
        if not chunk: break
        data.extend(chunk)
        since = chunk[-1][0] + 1
        print(f"{timeframe}: pulled {len(data):,} rows …", end="\r")
    if not data:
        raise RuntimeError("No data fetched.")
    df = pd.DataFrame(data, columns=['timestamp','open','high','low','close','volume'])
    df['datetime']  = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('datetime', inplace=True)
    return df.drop(columns='timestamp')

# ────────────────────────────────────────────────────────────────────────────
# 5.  Pipeline per timeframe
# ────────────────────────────────────────────────────────────────────────────
def process_timeframe(symbol, tf, perp):
    print(f"\n▶ {tf}  fetching …")
    raw = fetch_ohlcv(symbol, tf, perp)

    print(f"  computing TA‑Lib ({len(raw):,} rows)…")
    indi = calc_indicators(raw)
    merged = pd.concat([raw, indi], axis=1)

    tbl = f"ta_features_{tf}"
    ensure_table_and_columns(tbl, merged.reset_index())
    print(f"  upserting → {tbl}")
    bulk_upsert(tbl, merged.reset_index())
    print(f"  done {tf}")

# ────────────────────────────────────────────────────────────────────────────
# 6.  CLI entry‑point
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch OHLCV, compute TA, store per‑TF.")
    parser.add_argument('--symbol', default='BTCUSDT')
    parser.add_argument('--timeframes', nargs='+',
                        default=['1m', '3m', '5m', '15m', '30m', '1h', '4h', '6h', '12h', '1d'])
    parser.add_argument('--perp', action='store_true')
    args = parser.parse_args()

    for tf in args.timeframes:
        try:
            process_timeframe(args.symbol, tf, args.perp)
        except Exception as e:
            print(f"⚠️  {tf} failed: {e}")

