import talib
from talib import abstract
import pandas as pd
import numpy as np

data = pd.read_csv("data/BTCUSDT-binance-spot-1d.csv")
data["timestamp"] = pd.to_datetime(data["timestamp"], unit="ms")

# Test
output = abstract.Function('CDLDRAGONFLYDOJI')(data['open'],data['high'],data['low'],data['close'])
print(output[100:110])
# indicators = talib.get_functions()
# for indicator in indicators:
#     print(f"{indicator}: {abstract.Function(indicator).parameters}")


