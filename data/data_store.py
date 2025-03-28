from requests import get
from os.path import join
from pandas import DataFrame, read_csv, concat, Series
from numpy import ndarray, zeros

def download_data(symbol: str, timeframe: str, exchange: str, data_dir: str):
    url = f"https://www.cryptodatadownload.com/cdd/{exchange.capitalize()}_{symbol.upper()}_{timeframe}.csv"
    data = get(url)

    if not data.ok:
        raise RuntimeError(f"Unable to download data for {symbol} at {exchange} with timeframe {timeframe}")

    data = data.text.split('\n', 1)
    file = filename(symbol, exchange, timeframe, data_dir)

    with open(file, 'w') as file:
        file.write(data[1])

def filename(symbol: str, exchange: str, timeframe: str, data_dir: str) -> str:
    file = f"{symbol}-{exchange}-{timeframe}.csv"
    return join(data_dir, file)

if __name__ == '__main__':
    download_data('BTCUSDT','d','binance','./data')