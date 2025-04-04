import os
import talib
from talib import abstract
import pandas as pd
import numpy as np

def flatten_names(names):
    """
    Description:
        - Flattens the names OrderedDict into a list of column names.
    Args:
        - names (OrderedDict): The names attribute from a TA-Lib indicator.
    Returns:
        - list: A flat list of column names.
    """
    flat_list = []
    for value in names.values():
        if isinstance(value, list):
            flat_list.extend(value)
        else:
            flat_list.append(value)
    return flat_list

def apply_indicator(indicator, required_inputs, dataframe):
    """
    Description:
        - Applies a TA-Lib indicator to a pandas DataFrame dynamically.
    Args:
        - indicator (str): The name of the TA-Lib indicator (e.g., 'ADOSC').
        - dataframe (pd.DataFrame): The DataFrame containing the required input columns.
    Returns:
        - The output of the TA-Lib indicator function.
    Raises:
        - ValueError: If the indicator or required inputs are not found.
    """
    missing_inputs = [inp for inp in required_inputs if inp not in dataframe.columns]
    if missing_inputs:
        raise ValueError(f"Missing required inputs in DataFrame: {missing_inputs}")
    
    input_data = [dataframe[inp] for inp in required_inputs]
    func = abstract.Function(indicator)
    output = func(*input_data)  # Unpack the list into individual arguments
    
    return output

def main():
    # Step 1: Collect Data
    data = pd.read_csv("data/BTCUSDT-binance-spot-1d.csv")

    # Step 2: Get all TA-Lib indicators and their flattened input requirements
    indicators = talib.get_functions()

    # Step 3: Calculate every indicators
    result_dict = {}
    for indicator in indicators:
        try:
            func = abstract.Function(indicator)

            # Save input names
            input_names = func.input_names
            indicator_inputs = flatten_names(input_names)
            
            # Save output names
            indicator_outputs = func.output_names

            # Calculate indicator and save it to ddata
            result = apply_indicator(indicator, indicator_inputs, data)

            if len(indicator_outputs) == 1:
                output_name = indicator_outputs[0]
                result_dict[f"{indicator}_{output_name}"] = result
            else:
                for i in range(len(indicator_outputs)):
                    result_dict[f"{indicator}_{indicator_outputs[i]}"] = result[i]

        except Exception as e:
            print(f"Could not calculate {indicator}: {e}")

    new_columns = pd.DataFrame(result_dict, index=data.index)
    data = pd.concat([data, new_columns], axis=1)
    
    # Step 4: Save data with all indicators and price data in data folder
    file_path = os.path.join("data","BTCUSDT-binance-spot-1d-indicators.csv")
    data.to_csv(file_path,index=False)
    print(f"Dataframe with all indicators saved to {file_path}")

if __name__ == "__main__":
    main()
