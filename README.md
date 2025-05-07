# coracle
Use XGBoost with large dimensional feature (technical analysis and on-chain data) to predict crypto assets in perpetual market

Due list
1. Create Train; Validation; Test data
2. Find robust method for window sizing for train, validation test data. Maybe related to out of distribution or distributional shift problem?
3. Need to add k-cross validation methods. Again, think about the window sizing for train, validation and test data. 
4. Need to add parameter optimization for XGBOOST: either gridsearch CV or evolutionary algorithms
5. Class Imbalance: Currently, data is classified using threshold where each class (0,1,2) are evenly splitted. This may be improved by 1) create imbalanced class data and then use methods that overcome class imbalance. 
6. Tune decision thresholds: By default clf.predict() picks the argmax of probabilities. But you can adjust the threshold for class 1 to favor recall.