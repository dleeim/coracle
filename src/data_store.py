import ccxt
import pandas as pd
from os.path import join
import os
from sqlalchemy import create_engine, text
import urllib.parse
from dotenv import load_dotenv
import argparse

def download_data(symbol: str, timeframe: str, perp: bool):
    # Initialize the exchange based on perp flag
    if perp:
        exchange = ccxt.bybit({
            'enableRateLimit': True,  # Respect rate limits
            'options': {
                'defaultType': 'future'  # Set to futures (perpetual contracts)
            }
        })
    else:
        exchange = ccxt.bybit({
            'enableRateLimit': True,  # Respect rate limits
        })

    # Set the start date and limit
    since = exchange.parse8601('2015-01-01T00:00:00Z')  # Start date in milliseconds
    limit = 1000  # Max candles per request
    all_ohlcv = []  # List to store all fetched OHLCV data

    # Fetch data with pagination
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        if not ohlcv:  # Break if no more data is returned
            break
        all_ohlcv.extend(ohlcv)  # Add fetched candles to the list
        since = ohlcv[-1][0] + 1  # Update since to the timestamp after the last candle
        print(f"Fetched {len(ohlcv)} candles. Total so far: {len(all_ohlcv)}")

    # Check if any data was fetched
    if not all_ohlcv:
        raise ValueError("No data fetched. Check symbol, timeframe, or exchange connectivity.")

    # Convert to DataFrame
    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df['timeframe'] = [timeframe] * len(df)
    df['symbol'] = [symbol] * len(df)
    df['perp'] = [perp] * len(df)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Save to CSV
    save_to_mysql(df, table='ohlcv')

def get_mysql_engine(user, password, host, port, database):
    # URL‑encode your password in case it has special chars
    pwd = urllib.parse.quote_plus(password)
    url = f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{database}"
    return create_engine(url, echo=False)

def save_to_mysql(df: pd.DataFrame, table: str):
    """
    Append a DataFrame to a MySQL table.
    if_exists: 'fail', 'replace', or 'append'
    """
    # you can also pull these from env vars or a config file
    engine = get_mysql_engine(
        user= os.getenv('MYSQL_USER'),
        password=os.getenv('MYSQL_PASSWORD'),
        host=os.getenv('MYSQL_HOST'),
        port=os.getenv('MYSQL_PORT'),
        database=os.getenv('MYSQL_DATABASE'),
    )

    # write in chunks so you don’t overload memory or hit packet size limits
    df.to_sql(
        name=table,
        con=engine,
        if_exists="append",
        index=False,
        chunksize=500,        # adjust per your needs
        method='multi',       # uses executemany()
    )
    print(f"{'Appended'} {len(df)} rows to `{table}`")

if __name__ == '__main__':
    load_dotenv() 

    parser = argparse.ArgumentParser(description="Download OHLCV and load into MySQL.")
    parser.add_argument('--symbol',   default='BTCUSDT', help='Trading pair symbol')
    parser.add_argument('--timeframes', nargs='+', default=['5m','1h','1d'], help='List of timeframes')
    parser.add_argument('--perp', action='store_true', help='Fetch perpetual futures data')
    args = parser.parse_args()
    
    for tf in args.timeframes:
        download_data(
            symbol=args.symbol,
            timeframe=tf,
            perp=args.perp,
        )