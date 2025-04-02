import ccxt
import pandas as pd
from os.path import join
import os

def download_data(symbol: str, timeframe: str, exchange_name: str, perp: bool, data_dir: str):
    # Initialize the exchange based on perp flag
    if perp:
        exchange = ccxt.binance({
            'enableRateLimit': True,  # Respect rate limits
            'options': {
                'defaultType': 'future'  # Set to futures (perpetual contracts)
            }
        })
    else:
        exchange = ccxt.binance({
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
    # df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    
    # Ensure the data directory exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")

    # Save to CSV
    file_path = filename(symbol, exchange_name, perp, timeframe, data_dir)
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

def filename(symbol: str, exchange: str, perp: bool, timeframe: str, data_dir: str) -> str:
    market = 'perp' if perp else 'spot'
    file = f"{symbol}-{exchange}-{market}-{timeframe}.csv"
    return join(data_dir, file)

if __name__ == '__main__':
    download_data('BTCUSDT', '1d', 'binance', perp=True, data_dir='./data')