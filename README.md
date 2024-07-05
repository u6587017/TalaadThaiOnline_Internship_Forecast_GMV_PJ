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
Python 3.9-3.12
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
### Evaluation (Historical_Forecast)
### Result
