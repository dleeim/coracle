import talib
from talib import abstract
import pandas as pd
import numpy as np

indicators = talib.get_functions()
data = pd.read_csv("data/BTCUSDT-binance-spot-1d.csv")
data["timestamp"] = pd.to_datetime(data["timestamp"], unit="ms")

# Test the CDLTAKURI indicator
func = abstract.Function("CDLTAKURI")
output = func.run(data)
print("Output type:", type(output))
print("Output sample:", output.head())

# # Dictionary to store any errors (optional)
# errors = {}

# for indicator in indicators:
#     if indicator == 'CDLTAKURI':
#         try:
#             # Create a Function object for the indicator
#             func = abstract.Function(indicator)
            
#             # Run the indicator on the DataFrame
#             output = func.run(data)
            
#             # Handle single-output indicators (returns a Series)
#             if isinstance(output, pd.Series):
#                 data[indicator] = output
            
#             # Handle multi-output indicators (returns a DataFrame)
#             elif isinstance(output, pd.DataFrame):
#                 for col in output.columns:
#                     data[f"{indicator}_{col}"] = output[col]
                    
#         except Exception as e:
#             # Log errors and continue (optional)
#             errors[indicator] = str(e)
#             print(f"Error with {indicator}: {e}")
#             continue