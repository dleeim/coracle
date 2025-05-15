# coracle
Use XGBoost with large dimensional feature (technical analysis and on-chain data) to predict crypto assets in perpetual market
Trading with XGBOOST: https://github.com/dleeim/coracle/blob/main/xgboost/EDA_1d.ipynb 

Due list
1. Create more features: 
    - rolling statistics: compute rolling-window features on price and volume (ie 3,6,12,24); calculate rolling mean, standard deviation, min/max, percentile of the close price or returns
    - price-action featurs: derive features like (Close-Open), (High-Low), candle body size, MACD in various window size
    - Time features: encode cyclical time effects. For hourly data, you might include hour of day, day of week to capture recurring intraday patterns.
2. Modify xgboost EDA:
    - Boruta finds important features but from too many candidates. There needs to be more robust methods of pre-filtering out highly correlated features. (To what extent should we consider features being highly correlated?)
    - Currently, lots of same features but in different lags are used as features. We should use 
2. Need to use data from binance spot market instead of bybit perp market.
3. Find robust method for window sizing for train, validation test data. Maybe related to out of distribution or distributional shift problem?
4. Need to add k-cross validation methods. Again, think about the window sizing for train, validation and test data. 
5. Need to add parameter optimization for XGBOOST: either gridsearch CV or evolutionary algorithms
6. Tune decision thresholds: By default clf.predict() picks the argmax of probabilities. But you can adjust the threshold for class 1 to favor recall.