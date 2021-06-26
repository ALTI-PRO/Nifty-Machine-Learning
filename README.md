# Nifty-Machine-Learning
The projet predicts next day closing price and compares performance of two machine learning models

**Models Compared:** Multiple Linear Regression and Random Forest Regression
R2 Results: Multiple Linear Regression - 0.9683650083879028, Random Forest Regression - 0.9635727971342791

**Data:** NIFTY index data form Yahoo.

Independent variables (features): RSI indicator, Moving Average, High-Low Range percentage, Close-open percentage
Dependent Variable: Next Day closing Price 

**Limitations and Improvements:**

This training does not consider the data as a time series. Further performance comparison can be done with LSTM which is more suitable for timeseries data. The features used in the project are also very limite. A robust feature selection method can be used to improve the performance of the models.
