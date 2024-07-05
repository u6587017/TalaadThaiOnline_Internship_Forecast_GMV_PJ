# TalaadThaiOnline_Internship_Forecast_GMV_PJ

## About Project
The project involves creating a Machine Learning model to predict GMV (Gross Merchandise Value). It will be carried out in two phases:
### Files
#### PJ_forecast_GMV_phase1.ipynb: Development of a Machine Learning Model using Darts Library
#### PJ_forecast_GMV_phase2.ipynb: Development of a Deep Learning Model using TensorFlow and Keras
#### forecast_gmv_06_12_2024.csv: CSV file includes features for GMV forecasting
- date (Day)
- gmv (per day)
- quantity (per day)
- total_male (per day)
- total_female (per day)
- total_unknown (per day)
- total_order(per day)
## Prerequisites
- Python 3.9-3.12
- Understanding in basic machine learning, lags featuring
## Basic knowledge
### Lags feature
Lag features are a fundamental concept in time series analysis and forecasting. They involve using previous time points in the series as input features for predicting future values. Lag features are created by shifting the time series data backward by a certain number of periods (lags). Each lag represents the value of the time series at a previous time step.
![Screenshot 2024-07-05 131315](https://github.com/u6587017/TalaadThaiOnline_Internship_Forecast_GMV_PJ/assets/108443663/24aea135-89bb-40fc-9639-aa919a3edf45)
### Histoical Forecast
![historical_forecast](https://github.com/u6587017/TalaadThaiOnline_Internship_Forecast_GMV_PJ/assets/108443663/d5a89563-f728-4f82-9d9f-3a0ef9153c39)
### Library
Pandas, Numpy, Matplotlib 
#### Phase 1 : Darts 0.13.0, Scikit-learn 3.8.1 
#### Phase 2 : TensorFlow 2.15.0

## Phase 1 
### Import Library
```
import pandas as pd
from darts import TimeSeries
import numpy as np
import matplotlib.pyplot as plt
from darts.metrics.metrics import mape
from darts.metrics.metrics import mae
from darts.metrics.metrics import rmse
```
#### Read CSV File using Pandas
```
df =pd.read_csv('./forecast_gmv_06_12_2024.csv')
df.head()
```
#### Update GMV (Every day)
- Add new date into 'date' column and new GMV into 'gmv' column
- We have to update GMV on 2024-06-12 since it was the day we query, so we want to ensure that the GMV reflects the complete data for that day.
```
# Update gmv each day after query

new_rows = pd.DataFrame({
    'date': [pd.to_datetime('2024-06-13'), pd.to_datetime('2024-06-14'), pd.to_datetime('2024-06-15'), pd.to_datetime('2024-06-16'), pd.to_datetime('2024-06-17'), pd.to_datetime('2024-06-18'),
             pd.to_datetime('2024-06-19'), pd.to_datetime('2024-06-20'), pd.to_datetime('2024-06-21'), pd.to_datetime('2024-06-22'), pd.to_datetime('2024-06-23'), pd.to_datetime('2024-06-24'),
             pd.to_datetime('2024-06-25'),pd.to_datetime('2024-06-26'),pd.to_datetime('2024-06-27'),pd.to_datetime('2024-06-28'),pd.to_datetime('2024-06-29'),pd.to_datetime('2024-06-30'),pd.to_datetime('2024-07-01'),
             pd.to_datetime('2024-07-02'), pd.to_datetime('2024-07-03')
             ],
    'gmv': [211619, 210396, 232173, 208262, 162534, 200252, 208692, 190115, 232990, 207749, 196004, 229341, 225226 ,246038, 214500 , 218507, 193665, 242967, 218320,229796 ,251898]
})
df['gmv'][df['date'] == '2024-06-12'] = 212833

# Add the new rows using concat
df = pd.concat([df, new_rows], ignore_index=True)
```
#### Plot GMV
```
df['gmv'].plot()
```
![ts](https://github.com/u6587017/TalaadThaiOnline_Internship_Forecast_GMV_PJ/assets/108443663/5f357356-30a4-4b12-895a-c1bf96d74ff4)
### Data Cleaning
#### Convert Data Types
```
#Covert data types
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by='date')
df = df[(df['date'] >= '2023-01-01') & (df['date'] <= '2024-07-04')]
df.shape
```
#### Check Data Types
#### Outlier
Identify and replace outliers based on the Z-score. A common threshold for identifying outliers is a Z-score greater than 3 or less than -3. np.abs(z_scores) > 3 returns a boolean array where True indicates an outlier
```
# Function to replace outliers with the mean of the rest of the values

from scipy.stats import zscore
df1 = df.copy()


def replace_outliers_with_mean(df, column_name):
    # Calculate Z-scores
    z_scores = zscore(df[column_name])
    
    # Identify outliers (using a threshold of 3 for Z-score)
    outliers = np.abs(z_scores) > 3
    
    # Calculate mean of non-outliers
    mean_non_outliers = df.loc[~outliers, column_name].mean()
    
    # Replace outliers with the mean of non-outliers
    df.loc[outliers, column_name] = mean_non_outliers
    
    return df

# Replace outliers in 'gmv' column
df = replace_outliers_with_mean(df, 'gmv')

# Display the DataFrame
df[['date', 'gmv']][df['date'] == '2024-06-05']
```
#### Replace Null data with mean of data
```
# แทนที่ null ด้วยค่าเฉลี่ย
df.fillna(df.mean(), inplace=True)
```
#### Check Null
```
# เช็ค Null
df.isnull().sum()
```
#### Check 0's 
```
# เช็ค 0s
df.eq(0).sum()
```
#### Fill null (missing) values with the mean of the data
```
# แทนที่ null ด้วยค่าเฉลี่ย
df.fillna(df.mean(), inplace=True)
```
### Feature Engineering
Create new features or transform existing ones to improve the performance of machine learning models
From below code, we create new features from date such as Marketing Campaign
```
df['Quarter'] = df['date'].dt.quarter
df['Month'] = df['date'].dt.month
df['Weekday'] = df['date'].dt.weekday + 1  # Adding 1 to make it 1-7 (Mon-Sun)
df['Dayofyear'] = df['date'].dt.dayofyear
df['double_date'] = df['date'].apply(lambda x: 1 if x.month == x.day else 0)
df['mid_month'] = df['date'].apply(lambda x: 1 if x.day == 15 else 0)
df['payday'] = df['date'].apply(lambda x: 1 if x.day>=25 else 0)
```
In this code, we will make df2 with copying of only extracted feature from df for an easier to use. We will see it shortly 
```
date_range = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')

# Create a DataFrame with the date range
df2 = pd.DataFrame(date_range, columns=['date'])

# Feature engineering from Date
df2['Quarter'] = df2['date'].dt.quarter
df2['Month'] = df2['date'].dt.month
df2['Weekday'] = df2['date'].dt.weekday + 1  # Adding 1 to make it 1-7 (Mon-Sun)
df2['Dayofyear'] = df2['date'].dt.dayofyear
df2['double_date'] = df['date'].apply(lambda x: 1 if x.month == x.day else 0)
df2['mid_month'] = df['date'].apply(lambda x: 1 if x.day == 15 else 0)
df2['payday'] = df['date'].apply(lambda x: 1 if x.day>=25 else 0)
```
### Convert Dataframe type of Pandas into Timeseries type of Darts Library
```
data_ts = TimeSeries.from_dataframe(df, time_col="date")
# data_ts2 = TimeSeries.from_dataframe(df1, time_col="date")
# future_cov = TimeSeries.from_dataframe(df2, 'date', ['double_date', 'mid_month', 'payday'])
# future_cov = TimeSeries.from_dataframe(df2, 'date', ['Quarter',	'Month','Weekday','Dayofyear'])
# future_cov = TimeSeries.from_dataframe(df2, 'date', ['Month','Weekday','Dayofyear'])
# future_cov = TimeSeries.from_dataframe(df2, 'date', ['double_date'])
future_cov = TimeSeries.from_dataframe(df2, 'date', ['Weekday','Dayofyear','payday'])
# future_cov = TimeSeries.from_dataframe(df2, 'date', ['Quarter',	'Month','Weekday','Dayofyear','double_date', 'mid_month', 'payday'])
```
You can also fillna with Timeseries data type in Darts
```
from darts.utils.missing_values import fill_missing_values
future_cov = fill_missing_values(future_cov, fill='auto')
```
## Scaler (Optional)
Scaler in Darts is MinMaxScaler. MinMaxScaler is a feature scaling technique in data preprocessing that transforms features by scaling each feature to a given range, typically between 0 and 1.
MinMaxScaler is used to normalize the features in a dataset, which ensures that each feature contributes equally to the result and prevents features with larger ranges from dominating the model.
```
from darts.dataprocessing.transformers import Scaler

scaler_gmv = Scaler()
scaler_cov = Scaler()

rescaled = scaler_gmv.fit_transform(data_ts['gmv'])
future_cov = scaler_cov.fit_transform(future_cov)
future_cov
```
#### Correlation
Highly correlated features can provide redundant information to the model
The Pearson correlation coefficient (r) ranges from -1 to 1:
- 1 to -0.7 or 0.7 to 1: Strong correlation
- 0.7 to -0.5 or 0.5 to 0.7: Moderate correlation
- 0.5 to -0.3 or 0.3 to 0.5: Weak correlation
- 0.3 to 0.3: Negligible or no correlation
```
df.corr()
```
### Check for stationary data using ACF and PACF
Stationary data is a type of time series data whose statistical properties, such as mean, variance, and autocorrelation, do not change over time. In other words, the data does not exhibit trends, seasonality, or other time-dependent structures that could affect the consistency of these properties.
Models typically perform better with stationary data rather than non-stationary data. So, we could try to make the data non-stationary and we can we can see which one perform better
```
from darts.utils.statistics import plot_acf, plot_pacf
acf = plot_acf(rescaled['gmv'], max_lag=80)
pacf =plot_pacf(rescaled['gmv'], max_lag=80)
```
- Stationary Time Series:
ACF: The autocorrelations drop to zero relatively quickly.
PACF: The partial autocorrelations also drop to zero quickly.

- Non-Stationary Time Series:
ACF: The autocorrelations decrease slowly and may remain significant for many lags.
PACF: The partial autocorrelations can be significant for many lags.

We could see from the below attached images
##### ACF
![acf](https://github.com/u6587017/TalaadThaiOnline_Internship_Forecast_GMV_PJ/assets/108443663/d548e23a-3794-4ee5-8af5-fef2a5576999)
##### PACF
![pacf](https://github.com/u6587017/TalaadThaiOnline_Internship_Forecast_GMV_PJ/assets/108443663/3498a4b8-d626-4359-b782-97ec2812a297)
From the images, we could see that they are in the case of stationary.
#### Make data from Stationary to Non-Stationary
```
# สมมติว่ามี DataFrame ชื่อ data ที่มีคอลัมน์ 'value'
df['gmv_diff'] = df['gmv'].diff()
df = df.dropna().reset_index(drop=True)
```
Use Adfuller
```
from statsmodels.tsa.stattools import adfuller

result = adfuller(df['gmv_diff'].dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

```

### Evaluation (Historical_Forecast)
We could evaluate from 3 metrics
- MAE calculates the average absolute difference between the actual values and the predicted values. It gives an idea of how much the predictions deviate from the actual values on average. Interpretation: A lower MAE value indicates better model performance. It is easy to understand and interpret.
<br />![mean-absolute-error-equation](https://github.com/u6587017/TalaadThaiOnline_Internship_Forecast_GMV_PJ/assets/108443663/70a2a27c-40a2-4a47-a0a4-4fc1a789af89)
- MAPE measures the average absolute percentage difference between the actual values and the predicted values. It is scale-independent and gives a percentage error, which is useful for comparing forecast accuracy across different datasets. Interpretation: A lower MAPE value indicates better model performance. It is intuitive but can be problematic when actual values are close to zero, leading to extremely high percentage errors.
<br />![mape](https://github.com/u6587017/TalaadThaiOnline_Internship_Forecast_GMV_PJ/assets/108443663/8f8d1ae8-d045-4cdd-a049-284a43617870)
- RMSE calculates the square root of the average squared differences between the actual values and the predicted values.
It is sensitive to outliers because it squares the errors, which can disproportionately affect the RMSE if large errors are present.
Interpretation: A lower RMSE value indicates better model performance. It provides a good measure of how accurately the model predicts the target variable.
<br />![RMSE-equation](https://github.com/u6587017/TalaadThaiOnline_Internship_Forecast_GMV_PJ/assets/108443663/5b15ff11-cea4-4c45-82a6-662fb26979e5)<br />
Select models to do Historical_Forecast, it will plot actual data (black line), validation data (blue line), forecast data (pink line) and the metrics at the title of each model figure
```
from darts.models.forecasting.catboost_model import CatBoostModel
from darts.models.forecasting.lgbm import LightGBMModel
from sklearn.ensemble import RandomForestRegressor
from darts.models.forecasting.regression_model import RegressionModel
from darts.models.forecasting.exponential_smoothing import ExponentialSmoothing
from darts.models.forecasting.random_forest import RandomForest
from sklearn.linear_model import LinearRegression
from darts.models.forecasting.xgboost import XGBModel
from darts.utils.utils import ModelMode, SeasonalityMode
from darts.models.forecasting.arima import ARIMA
from sklearn.linear_model import ElasticNetCV

# List of models configurations
models = [
    LightGBMModel(random_state=42, lags=7, lags_future_covariates=(0,7), output_chunk_length=7,show_warnings=False,verbose=-1),
    # CatBoostModel(random_state=42, lags=7, output_chunk_length=7, lags_future_covariates=(0,7)),
    RegressionModel(model=LinearRegression() ,lags=3, lags_future_covariates=(0,7),output_chunk_length=7,),
    RegressionModel(model=RandomForestRegressor(max_depth=9, n_estimators=300, random_state=42),lags=7, lags_future_covariates=(0,7),output_chunk_length=7,),
    ExponentialSmoothing(trend=ModelMode.NONE, damped=False, seasonal=SeasonalityMode.ADDITIVE, seasonal_periods=7,),
    XGBModel(random_state=42, lags=14, lags_future_covariates=(0,7), output_chunk_length=7, ),
    ARIMA(p=0, d=0, q=1, seasonal_order=(1, 0, 2, 12), trend=None, random_state=42),
        ]
model_name = [  'LightGBMModel',
                'LinearRegression',
                  'RandomForestRegressor',
                    'ExponentialSmoothing', 'XGBModel',
                    'ARIMA'
                  ]
i = 0

# Setting up the plot
fig, axes = plt.subplots(nrows=len(models), ncols=1, figsize=(12, 6 * len(models)))
if len(models) == 1:
    axes = [axes]  # Make axes iterable if there's only one model

# Loop through models and plot each one
for model, ax in zip(models, axes):
    if (i == 3):
        historical_forecasts = model.historical_forecasts(
        series=rescaled['gmv'],
        start=0.8,  # Start generating historical forecasts after 80% of the data
        forecast_horizon=1,  # Forecast horizon (number of steps to forecast)
        stride=1,  # Make a forecast every time step
        retrain=True,  # Retrain the model at each step
        verbose=True,
    )
    else:
    # Generate historical forecasts
        historical_forecasts = model.historical_forecasts(
            series=rescaled['gmv'],
            start=0.8,  # Start generating historical forecasts after 80% of the data
            forecast_horizon=1,  # Forecast horizon (number of steps to forecast)
            stride=1,  # Make a forecast every time step
            retrain=True,  # Retrain the model at each step
            verbose=True,
            future_covariates=future_cov
        )

    # last_known_gmv = df['gmv'].iloc[-1]

    original = scaler_gmv.inverse_transform(historical_forecasts['gmv'])
    # original = historical_forecasts['gmv_diff']
#     original = original.pd_dataframe().reset_index()
#     original.columns = ['date', 'gmv_diff']

#     original['gmv_diff'] = np.cumsum(original['gmv_diff']) + last_known_gmv

#     # Combine Date and Forecast into the final DataFrame
#     forecast_gmv = pd.DataFrame({
#         'date': original['date'],
#         'gmv': original['gmv_diff']
# })
#     forecast_gmv = TimeSeries.from_dataframe(forecast_gmv, time_col="date")

    # Calculate metrics
    hf_mae = mae(validation_ori['gmv'], original)
    hf_mape = mape(validation_ori['gmv'], original)
    hf_rmse = rmse(validation_ori['gmv'], original)

    # Plot the results
    # training['gmv_diff'].plot(label='Actual Data', ax=ax)
    data_ts['gmv'].plot(label='GMV', ax=ax)
    validation_ori['gmv'].plot(label='Validation Data', ax=ax)
    original.plot(label=f'Historical Forecasts (MAE: {hf_mae:.2f}, MAPE: {hf_mape:.2f}%, RMSE: {hf_rmse:.2f})', ax=ax)
    ax.legend()
    ax.set_title(f'{model_name[i]} Accuracy: {100-hf_mape:.2f}')
    i += 1
plt.tight_layout()
plt.show()
```
### Result
![18_Jun_Historical_Forecast](https://github.com/u6587017/TalaadThaiOnline_Internship_Forecast_GMV_PJ/assets/108443663/120a0ce0-c789-4d7a-94cb-82c07aebfcea)
### Hyperparameter Tuning
Hyperparameter Tuning is the process of finding the most effective set of hyperparameters for a machine learning model. Hyperparameters are the parameters that are not learned from the data but set prior to the training process. They control the overall behavior of the training process and the structure of the model. We will use GridSearch. Darts Library also provide GridSearch, so we could find the best params like lags.
```
# from darts.models.forecasting.lgbm import LightGBMModel
# from darts.models.forecasting.catboost_model import CatBoostModel
# from sklearn.ensemble import RandomForestRegressor
# from darts.models.forecasting.regression_model import RegressionModel
# from darts.models.forecasting.exponential_smoothing import ExponentialSmoothing
# from darts.models.forecasting.linear_regression_model import LinearRegressionModel
# from darts.models.forecasting.random_forest import RandomForest
# from darts.models.forecasting.xgboost import XGBModel
# from darts.utils.utils import ModelMode, SeasonalityMode
# from sklearn.ensemble import RandomForestRegressor
# from darts.models.forecasting.arima import ARIMA
#LightGBM
# parameters = {
#     'lags': [1,3,7,14,21,28,30],
#     'lags_future_covariates': [(0,7)],
#     'output_chunk_length': [7],
#     'show_warnings': [False],
#     'verbose': [-1],
#     'likelihood':['quantile', 'poisson'],
#     'output_chunk_shift': [0,1,2,3]
# }

#RNDForest
# parameters = {
#     'lags': [1,2,3,7,14,21,30],
#     'lags_future_covariates': [(0,7)],
#     'output_chunk_length': [7],
#     'output_chunk_shift' : [0,1,3,5],
#     'max_depth': [3,5,7,9,11],
#     'n_estimators': [100,200, 300]
# }

#Catboost
# parameters = {
#     'lags': [7,14,21,28,30,31],
#     'lags_future_covariates': [(0,7)],
#     'output_chunk_length': [7]
# }

#XGBoost
# parameters = {
#     'lags': [3,7,14,21,28,30,31],
#     'lags_future_covariates': [(0,7)],
#     'output_chunk_length': [7],
#     'random_state': [42]
# } #Lags=14

# ExponentialSmoothing
# parameters = {
#     'trend': [ModelMode.ADDITIVE, ModelMode.NONE],
#     'seasonal': [SeasonalityMode.ADDITIVE, SeasonalityMode.NONE],
#     'seasonal_periods': [7],
#     'random_state': [42]
# } # ADDITIVE 40.86087049269774 |

#Regression
parameters = {
    'model': [linear_model.Ridge(random_state=42)
     ],
    'lags': [3,7,14,21,28,30,31],
    'lags_future_covariates': [(0,7)],
    'output_chunk_length': [7]
}

#ARIMA #{'p': 0, 'd': 0, 'q': 1}, #
# parameters = {
#     'p': [0],
#     'd':[0],
#     'q': [1],
#     'seasonal_order': [(0, 0, 0, 12), (0, 0, 1, 12), (0, 0, 2, 12), (0, 1, 0, 12), (0, 1, 1, 12), (0, 1, 2, 12), (0, 2, 0, 12), (0, 2, 1, 12), (0, 2, 2, 12),
#  (1, 0, 0, 12), (1, 0, 1, 12), (1, 0, 2, 12), (1, 1, 0, 12), (1, 1, 1, 12), (1, 1, 2, 12), (1, 2, 0, 12), (1, 2, 1, 12), (1, 2, 2, 12),
#  (2, 0, 0, 12), (2, 0, 1, 12), (2, 0, 2, 12), (2, 1, 0, 12), (2, 1, 1, 12), (2, 1, 2, 12), (2, 2, 0, 12), (2, 2, 1, 12), (2, 2, 2, 12)]

    # 'random_state': [42]
# }

best_param = RegressionModel.gridsearch(
    parameters=parameters,
    series=target,
    start=0.8,  # Start generating historical forecasts after 80% of the data
    forecast_horizon=7,  # Forecast horizon (number of steps to forecast)
    stride=7,  # Make a forecast every time step
    metric = mape,
    future_covariates=future_cov,
    reduction=np.mean,
    verbose=1,
    )

```
### Train and use model to Forecast
- To train model, we will select the best model in each day to fit with the time-series data
- To forecast, we will use model.predict(length of days we're going to forecast, future_covariates)
- I also commented the code in case we use gmv_diff which is more non-stationary data to forecast
```
from darts.models.forecasting.lgbm import LightGBMModel

model1= RegressionModel(model=linear_model.Ridge(random_state=42) ,lags=3, lags_future_covariates=(0,7),output_chunk_length=7)
model1.fit(rescaled['gmv'], future_covariates=future_cov)
forecast_diff = model1.predict(7, future_covariates=future_cov) # Forecast 
forecast_diff = scaler_gmv.inverse_transform(forecast_diff)

# For use GMV_diff 

# forecast_diff_df = forecast_diff.pd_dataframe().reset_index()
# forecast_diff_df.columns = ['Date', 'Forecast']

# Invert differencing to get the original GMV forecast
# last_known_gmv = df['gmv'].iloc[-1]
# forecast_diff_df['Forecast'] = np.cumsum(forecast_diff_df['Forecast']) + last_known_gmv

# Combine Date and Forecast into the final DataFrame
# forecast_gmv = pd.DataFrame({
#     'Date': forecast_diff_df['Date'],
#     'Forecast': forecast_diff_df['Forecast']
# })

# Display the forecasted GMV
# forecast_gmv
forecast_diff.pd_dataframe()
```
