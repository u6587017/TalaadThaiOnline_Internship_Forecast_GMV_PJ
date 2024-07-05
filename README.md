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
### Data Cleaning
### Evaluation (Historical_Forecast)
### Result
