# TalaadThaiOnline_Internship_Forecast_GMV_PJ

## About Project
Project เกี่ยวข้องกับการสร้างโมเดล Machine Learning เพื่อทำนาย GMV (Gross Merchandise Value) ซึ่งจะดำเนินการเป็น 2 phases

## Table of contents
- [Files](#files)
- [Features inside CSV File](#features)
- [Basic knowledge](#basic_knowledge)
- [Phase 1](#phase_1)
  - [Import Library](#library)
  - [Read CSV File](#read_csv)
  - [Data Cleaning](#data_cleaning)
  - [Feature Engineering](#feature_engineering)
  - [Scaling](#scaler)
  - [Model Evaluation](#evaluation)
  - [HyperParameter Tuning](#hyper)
  - [Result](#result)
  - [Model selection, training and forecasting](#train)
- [Phase 2](#phase_2)
  - [Import Library](#lib2)
  - [Convert DataFrame to Numpy Array (Use 1 feature for trainng)](1_feature)
  - [Convert DataFrame to Numpy Array (Use multiple features for trainng)](multiple_feature)
  - [Scaling](#standardize)
  - [Define sequential model](#seq_model)
  - [Model Training](#train_2)
  - [Evaluation](#eval_2)
  - [Result](#result_2)
### <a name="files"></a>Files
#### PJ_forecast_GMV_phase1.ipynb: Development of a Machine Learning Model using Darts Library<br />[Clickhere to Phase 1](#phase_1)
#### PJ_forecast_GMV_phase2.ipynb: Development of a Deep Learning Model using TensorFlow and Keras<br />[Clickhere to Phase 2](#phase_2)
### Note!!!
- Project นี้ต้องการความรู้พื้นฐานเกี่ยวกับ Machine Learning และ Time Series Lag Feature
- เพื่อให้เข้าใจเกี่ยวกับคุณลักษณะของ Lag สามารถศึกษาเบื้องต้นได้ในส่วน [Basic knowledge](#basic_knowledge)
- ใน Phase ที่ 2 ผมจะใช้กระบวนการทำความสะอาดข้อมูล (Data Cleaning) และการสร้างคุณลักษณะ (Feature Engineering) เช่นเดียวกับใน phase ที่ 1 ดังนั้นควรดู phase ที่ 1 ก่อน
- หากมีปัญหาเกี่ยวกับ Version ในเฟสที่ 2 สามารถเปลี่ยนไปใช้ Google Colab ซึ่งมี Library ที่จำเป็นพร้อมแล้วได้
#### <a name="features"></a>forecast_gmv_06_12_2024.csv: CSV file includes features for GMV forecasting
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
- <a href="https://unit8co.github.io/darts/quickstart/00-quickstart.html">Darts Library Install</a>
- <a href="https://keras.io/getting_started/">Keras Install</a>
## <a name="basic_knowledge"></a>Basic knowledge
### Darts
Darts เป็นไลบรารีของ Python ที่ใช้งานง่ายสำหรับการพยากรณ์และการตรวจจับความผิดปกติใน Time-series data มีโมเดลหลากหลาย ตั้งแต่โมเดลคลาสสิกอย่าง ARIMA ไปจนถึงเครือข่ายประสาทลึก (Deep Neural Networks) โมเดลการพยากรณ์ทั้งหมดสามารถใช้งานได้ในลักษณะเดียวกัน โดยใช้ฟังก์ชัน fit() และ predict() คล้ายกับ scikit-learn ไลบรารียังทำให้การทดสอบย้อนกลับของโมเดล (Backtesting) ง่ายขึ้น รวมทั้งการรวมการทำนายของโมเดลหลายตัว และการคำนึงถึงการใช้ข้อมูลภายนอก<br />
![darts_unit8](https://github.com/u6587017/TalaadThaiOnline_Internship_Forecast_GMV_PJ/assets/108443663/9970cfd5-23bd-4f1d-8691-aead61ef8ebb)
![darts](https://github.com/u6587017/TalaadThaiOnline_Internship_Forecast_GMV_PJ/assets/108443663/d57bbe15-013c-4433-9807-61fe1b98c6a5)
### Lags feature
Lags feature เป็นแนวคิดพื้นฐานในการวิเคราะห์และพยากรณ์ Time-series data โดยการใช้ค่าของ Time-series data ในช่วงเวลาก่อนหน้าเป็น Feature ในการทำนายค่าของ Time-series ในอนาคต Lag feature ถูกสร้างขึ้นโดยการเลื่อนข้อมูลซีรีส์เวลาย้อนหลังตามจำนวนช่วงเวลาที่กำหนด (lags) แต่ละ lag แทนค่าของซีรีส์เวลาในขั้นเวลาก่อนหน้า ดูตัวอย่างจากภาพด้านล่าง
![Screenshot 2024-07-05 131315](https://github.com/u6587017/TalaadThaiOnline_Internship_Forecast_GMV_PJ/assets/108443663/24aea135-89bb-40fc-9639-aa919a3edf45)
### Covariates in Darts
ใน Darts, covariates หมายถึงข้อมูลภายนอกที่สามารถใช้เป็น Feature ให้กับโมเดลเพื่อช่วยปรับปรุงการพยากรณ์ 
- Past covariates (ตามความหมาย) หมายถึง covariates ที่รู้เฉพาะในอดีต (เช่น การวัดค่าต่าง ๆ) 
- Future covariates (ตามความหมาย) หมายถึง covariates ที่รู้ล่วงหน้าในอนาคต (เช่น การพยากรณ์อากาศ, ข้อมูลปฏิทิน, Marketing Campaign date)
![cov](https://github.com/u6587017/TalaadThaiOnline_Internship_Forecast_GMV_PJ/assets/108443663/b45e999c-afb3-43d6-94ad-421c3ce59210)<br />
หากฝึกโมเดลโดยใช้ past_covariates คุณจะต้องใส่ past_covariates ในเวลาทำนายกับฟังก์ชัน predict() ด้วย รวมทั้ง future_covariates เช่นกัน แต่มีข้อแม้ว่าต้องให้ future_covariates ขยายไปถึงอนาคตในเวลาทำนาย (จนถึงขอบเขตการพยากรณ์ n) ดังที่เห็นในกราฟด้านล่าง past_covariates จะต้องรวมช่วงเวลาเดียวกันอย่างน้อยกับ target และ future_covariates ต้องรวมช่วงเวลาเดียวกันอย่างน้อยบวกกับช่วงเวลา n ของขอบเขตการพยากรณ์

### Historical Forecast
Historical Forecast หมายถึงการสร้างการพยากรณ์สำหรับหลายจุดในเวลา โดยมักใช้วิธีการ rolling หรือ expanding window วิธีนี้ใช้เพื่อจำลองว่าโมเดลจะทำงานอย่างไรในสถานการณ์การพยากรณ์จริงโดยการทำนายในหลายขั้นเวลาในอดีตและเปรียบเทียบกับค่าจริง
ซึ่งช่วยอำนวยความสะดวกทำให้ไม่จำเป็นต้อง Train/Test Split แยกเพราะสามารถกำหนดสัดส่วนข้อมูลภายใน Parameter ได้เลย
![historical_forecast](https://github.com/u6587017/TalaadThaiOnline_Internship_Forecast_GMV_PJ/assets/108443663/d5a89563-f728-4f82-9d9f-3a0ef9153c39)
### Backtesting
Backtesting เกี่ยวข้องกับการแบ่งข้อมูลในอดีตเป็น Training set และ Testing set จากนั้นฝึกโมเดลด้วยข้อมูลการฝึกสอนและทดสอบความสามารถของโมเดลในการทำนายค่าของข้อมูลในชุดการทดสอบ กระบวนการนี้สามารถทำได้หลายวิธี เช่น การใช้ rolling window หรือ expanding window approach เพื่อทำการพยากรณ์ในหลายจุดเวลา ผลคือเราจะได้ผลลัพธ์โมเดลที่เป็นจริงมากกว่าเนื่องจากได้มีการทดสอบกับข้อมูลในทุก ๆ ช่วง
![backtest](https://github.com/u6587017/TalaadThaiOnline_Internship_Forecast_GMV_PJ/assets/108443663/68d633ef-ac96-4ecf-bdbe-b4597b235170)<br />
### <a href="https://unit8co.github.io/darts/userguide.html">Read More about Darts</a>
### Library
Pandas, Numpy, Matplotlib 
#### Phase 1 : Darts 0.13.0, Scikit-learn 3.8.1 
#### Phase 2 : TensorFlow 2.15.0

## <a name="phase_1"></a>Phase 1
### <a name="library"></a>Import Library
```
import pandas as pd
from darts import TimeSeries
import numpy as np
import matplotlib.pyplot as plt
from darts.metrics.metrics import mape
from darts.metrics.metrics import mae
from darts.metrics.metrics import rmse
```
#### <a name="read_csv"></a>Read CSV File using Pandas
```
df =pd.read_csv('./forecast_gmv_06_12_2024.csv')
df.head()
```
#### Update GMV (Every day)
- เพิ่มข้อมูล date ลง 'date' column and GMV ลง 'gmv' column
- ต้องทำการอัปเดต GMV ในวันที่ 12 มิถุนายน 2024 เนื่องจากเป็นวันที่เราทำการ Query ข้อมูลออกมาระหว่างวัน ทำให้ยอด GMV ณ ตอน Query ยังไม่ใช่ยอดที่จบวัน
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
### <a name="data_cleaning"></a>Data Cleaning
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
ระบุและแทนที่ค่า Outlier โดยใช้ค่า Z-score Standardization สำหรับการระบุค่า Outlier คือ Z-score ที่มากกว่า 3 หรือน้อยกว่า -3 ฟังก์ชัน np.abs(z_scores) > 3 จะคืนค่าเป็นอาเรย์ของบูลีนที่มีค่า True ซึ่งระบุค่าผิดปกติ
จากนั้นจะแทนที่ Outlier ด้วยค่าเฉลี่ยของข้อมูลที่ไม่ใช่ Outlier
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
### <a name="feature_engineering"></a>Feature Engineering
สร้าง Feature ใหม่หรือแปลง Feature ที่มีอยู่เพื่อปรับปรุงประสิทธิภาพของโมเดล
จากโค้ดด้านล่าง เราสร้างคุณลักษณะใหม่จากวันที่ เช่น การแคมเปญการตลาด (Marketing Campaign)
```
df['Quarter'] = df['date'].dt.quarter
df['Month'] = df['date'].dt.month
df['Weekday'] = df['date'].dt.weekday + 1  # Adding 1 to make it 1-7 (Mon-Sun)
df['Dayofyear'] = df['date'].dt.dayofyear
df['double_date'] = df['date'].apply(lambda x: 1 if x.month == x.day else 0)
df['mid_month'] = df['date'].apply(lambda x: 1 if x.day == 15 else 0)
df['payday'] = df['date'].apply(lambda x: 1 if x.day>=25 else 0)
```
ในโค้ดนี้ เราจะสร้าง DataFrame ชื่อ df2 โดยคัดลอกเฉพาะ Feature ที่เป็น Covariates จาก df เพื่อให้ใช้งานได้ง่ายขึ้น
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
## <a name="scaler"></a>Scaler (Optional)
ใน Darts, Scaler คือ MinMaxScaler ซึ่งเป็นเทคนิคการปรับ Feature ในกระบวนการเตรียมข้อมูล โดยทำการปรับขนาด Feature แต่ละตัวให้อยู่ในช่วงที่กำหนด โดยทั่วไปจะอยู่ระหว่าง 0 และ 1
MinMaxScaler ถูกใช้เพื่อทำให้ Feature ในชุดข้อมูล ซึ่งจะช่วยให้มั่นใจว่า Feature แต่ละตัวป้องกันไม่ให้คุณลักษณะที่มีช่วงกว้างกว่าส่งผลต่อโมเดลมากเกินไป
```
from darts.dataprocessing.transformers import Scaler

scaler_gmv = Scaler()
scaler_cov = Scaler()

rescaled = scaler_gmv.fit_transform(data_ts['gmv'])
future_cov = scaler_cov.fit_transform(future_cov)
future_cov
```
#### Correlation
คุณลักษณะที่มีความสัมพันธ์สูงสามารถให้ข้อมูลที่เป็นประโยชน์ในการเลือก Feature ไปใช้ Train โมเดลได้
The Pearson correlation coefficient (r) ranges from -1 to 1:
- 1 to -0.7 or 0.7 to 1: Strong correlation
- 0.7 to -0.5 or 0.5 to 0.7: Moderate correlation
- 0.5 to -0.3 or 0.3 to 0.5: Weak correlation
- 0.3 to 0.3: Negligible or no correlation
```
df.corr()
```
### Check for stationary data using ACF and PACF
ข้อมูลที่อยู่ในสถานะคงที่ (Stationary data) เป็นประเภทของ Time-series data ที่คุณสมบัติทางสถิติเช่น ค่าเฉลี่ย, ความแปรปรวน, และการหาความสัมพันธ์ตัวเอง (autocorrelation) ไม่เปลี่ยนแปลงไปตามเวลา พูดอีกอย่างหนึ่งคือ ข้อมูลไม่แสดงแนวโน้ม, ฤดูกาล หรือโครงสร้างที่ขึ้นอยู่กับเวลาซึ่งสามารถส่งผลต่อความคงที่ของคุณสมบัติเหล่านี้
โมเดลมักทำงานได้ดีขึ้นกับข้อมูลที่อยู่ในสถานะคง (Stationary data) ที่มากกว่าข้อมูลที่ไม่อยู่ในสถานะคงที่ (Non-Stationary) ดังนั้น เราอาจลองทำให้ข้อมูลเป็นสถานะไม่คงที่แล้วดูว่าโมเดลไหนทำงานได้ดีกว่ากัน ซึ่งเราสามารถใช้ค่า ACF และ PACF เพื่อเช็คได้ว่าข้อมูลเป็น Stationary หรือ Non-Stationary
```
from darts.utils.statistics import plot_acf, plot_pacf
acf = plot_acf(rescaled['gmv'], max_lag=80)
pacf =plot_pacf(rescaled['gmv'], max_lag=80)
```
- Stationary Time Series:
  - ACF: Autocorrelations ลดลงเป็นศูนย์อย่างรวดเร็ว
  - PACF: Partial autocorrelations ก็ลดลงเป็นศูนย์อย่างรวดเร็วเช่นกัน

- Non-Stationary Time Series:
  - ACF: Autocorrelations ลดลงอย่างช้า ๆ และอาจคงมีนัยสำคัญสำหรับหลาย lags
  - PACF: Partial autocorrelations อาจมีนัยสำคัญสำหรับหลาย lags

เราสามารถดูได้จากภาพด้านล่าง
##### ACF
![acf](https://github.com/u6587017/TalaadThaiOnline_Internship_Forecast_GMV_PJ/assets/108443663/d548e23a-3794-4ee5-8af5-fef2a5576999)
##### PACF
![pacf](https://github.com/u6587017/TalaadThaiOnline_Internship_Forecast_GMV_PJ/assets/108443663/3498a4b8-d626-4359-b782-97ec2812a297)<br />
จากรูป เราจะเห็นได้ว่าข้อมูลยังเป็น Non-Stationary
#### Make data from Stationary to Non-Stationary
```
# สมมติว่ามี DataFrame ชื่อ data ที่มีคอลัมน์ 'value'
df['gmv_diff'] = df['gmv'].diff()
df = df.dropna().reset_index(drop=True)
```
ใช้ Adfuller <br />
การตีความผลลัพธ์
หาก p-value น้อยกว่า 0.05 เราปฏิเสธสมมติฐานศูนย์ (null hypothesis)
ซึ่งหมายความว่าข้อมูลอยู่ในสถานะคงที่ (Stationary)
```
from statsmodels.tsa.stattools import adfuller

result = adfuller(df['gmv_diff'].dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

```

### <a name="evaluation"></a>Evaluation
เราสามารถประเมินผลของโมเดลได้ด้วย Metrics ดังต่อไปนี้
- MAE คำนวณค่าเฉลี่ยของความแตกต่างสัมบูรณ์ระหว่างค่าจริงและค่าที่ทำนาย คิดว่าการทำนายเบี่ยงเบนจากค่าจริงโดยเฉลี่ยเท่าใด
  - การตีความ: ค่า MAE ที่ต่ำกว่าบ่งบอกถึงประสิทธิภาพของโมเดลที่ดีกว่า ข้อดีของ MAE คือง่ายต่อการเข้าใจและตีความ
<br />![mean-absolute-error-equation](https://github.com/u6587017/TalaadThaiOnline_Internship_Forecast_GMV_PJ/assets/108443663/70a2a27c-40a2-4a47-a0a4-4fc1a789af89)
- MAPE วัดค่าเฉลี่ยของความแตกต่างสัมบูรณ์เป็นเปอร์เซ็นต์ระหว่างค่าจริงและค่าที่ทำนาย ไม่ขึ้นอยู่กับขนาดของข้อมูลและให้ค่าเป็นเปอร์เซ็นต์ ซึ่งเป็นประโยชน์ในการเปรียบเทียบความแม่นยำของการพยากรณ์ระหว่างชุดข้อมูลต่างๆ
  - การตีความ: ค่า MAPE ที่ต่ำกว่าบ่งบอกถึงประสิทธิภาพของโมเดลที่ดีกว่า เข้าใจง่ายแต่สามารถเป็นปัญหาได้เมื่อค่าจริงใกล้ศูนย์ ซึ่งจะทำให้เกิดข้อผิดพลาดเป็นเปอร์เซ็นต์ที่สูงมาก
<br />![mape](https://github.com/u6587017/TalaadThaiOnline_Internship_Forecast_GMV_PJ/assets/108443663/8f8d1ae8-d045-4cdd-a049-284a43617870)
- RMSE คำนวณรากที่สองของค่าเฉลี่ยของความแตกต่างที่ยกกำลังสองระหว่างค่าจริงและค่าที่ทำนาย มันไวต่อค่าผิดปกติเพราะมันยกกำลังสองข้อผิดพลาด ซึ่งสามารถส่งผลกระทบอย่างมากต่อ RMSE หากมีข้อผิดพลาดขนาดใหญ่เกิดขึ้น
  - การตีความ: ค่า RMSE ที่ต่ำกว่าบ่งบอกถึงประสิทธิภาพของโมเดลที่ดีกว่า มันให้การวัดที่ดีว่าโมเดลทำนายตัวแปรเป้าหมายได้แม่นยำเพียงใด
<br />![RMSE-equation](https://github.com/u6587017/TalaadThaiOnline_Internship_Forecast_GMV_PJ/assets/108443663/5b15ff11-cea4-4c45-82a6-662fb26979e5)<br />
การเลือกโมเดลเพื่อทำ Historical Forecast จะมีการพล็อตข้อมูลจริง (เส้นสีดำ), ข้อมูลการตรวจสอบ (เส้นสีน้ำเงิน), ข้อมูลการพยากรณ์ (เส้นสีชมพู) และค่า Metrics ที่ใช้
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
## <a name="backtest"></a>Backtesting
ทำ Backtesting เพื่อลองให้โมเดลลองทำนายข้อมูลเป็นช่วง ๆ ทั้งหมด ทำให้เราเห็นประสิทธิภาพจริง ๆ ว่าโมเดลไหนควรนำมาใช้
![backtest_](https://github.com/u6587017/TalaadThaiOnline_Internship_Forecast_GMV_PJ/assets/108443663/0fc70118-71cd-4e9f-bdd9-151190526e7a)<br />
### <a name="result"></a>Result
![18_Jun_Historical_Forecast](https://github.com/u6587017/TalaadThaiOnline_Internship_Forecast_GMV_PJ/assets/108443663/120a0ce0-c789-4d7a-94cb-82c07aebfcea)
### <a name="hyper"></a>Hyperparameter Tuning
Hyperparameter Tuning เป็นกระบวนการหาชุด Hyperparameter ที่มีประสิทธิภาพที่สุดสำหรับโมเดล Hyperparameter เป็นพารามิเตอร์ที่ไม่ได้เรียนรู้จากข้อมูลแต่ถูกตั้งค่าก่อน Training มันควบคุมพฤติกรรมโดยรวมของกระบวนการ Training และโครงสร้างของโมเดล เราจะใช้ GridSearch ซึ่งไลบรารี Darts ก็มีฟังก์ชัน GridSearch ดังนั้นเราสามารถหาค่าพารามิเตอร์ที่ดีที่สุดได้ เช่น lags
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
### <a name="train"></a>Train and use model to Forecast
- ในการฝึกโมเดล เราจะเลือกโมเดลที่ดีที่สุดโดยอิงจากค่า MAPE ที่ต่ำที่สุดเพื่อให้เข้ากับ Time-series data
- ในการพยากรณ์ เราจะใช้ model.predict (จำนวนวันที่จะพยากรณ์, future_covariates)
- ผมยังได้คอมเมนต์โค้ดในกรณีที่เราใช้ gmv_diff ซึ่งเป็นข้อมูลที่ไม่อยู่ในสถานะคงที่ (Non-stationary) มากขึ้นในการพยากรณ์ โดยทำการลองฝึกทั้งข้อมูล 2 รูปแบบ และบันทึกผล
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
![forecast](https://github.com/u6587017/TalaadThaiOnline_Internship_Forecast_GMV_PJ/assets/108443663/9c0c56a4-79e4-4bd6-a011-c51207e3e1d5)
## <a name="phase_2"></a>Phase 2
ใน Phase นี้ จะนำเสนอการใช้งานโมเดลการเรียนรู้เชิงลึก (Deep Learning) โดยใช้ Tensorflow และ Keras
### <a name="lib2"></a>Import Library
```
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import math

import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import mean_squared_error
```
โดยเราจะใช้โค้ดตั้งแต่การอ่านไฟล์ CSV ไปจนถึงการสร้างคุณลักษณะ (Feature Engineering) เหมือนใน Phase ที่ 1
### Deep Learning
หลังจากที่ DataFrame พร้อมแล้ว เราจะสร้างฟังก์ชันเพื่อแปลง DataFrame เป็น Numpy Array สำหรับการทำ Deep Learning
#### <a name="1_feature"></a>Function to convert DataFrame to Numpy array (1 feature)
```
# [ [ [g1],[g2],[g3],[g4],[g5] ] ] [g6]
# [ [ [g2],[g3],[g4],[g5],[g6] ] ] [g7]

def df_to_X_y (df, window_size=5):
  df_as_np = df.to_numpy()
  X = []
  y = []
  for i in range(len(df_as_np)-window_size):
    row = [[a] for a in df_as_np[i:i+window_size]]
    X.append(row)
    label = df_as_np[i+window_size]
    y.append(label)
  return np.array(X), np.array(y)
```
#### Function Purpose
เป็นฟังก์ชันที่เราจะใส่ DataFrame และแปลงออกมาเป็น Numpy โดยใช้ความเข้าใจเรื่อง Lag Features ซึ่งผลลัพธ์จะได้เป็น eg. ((690, 5, 1), (690,)) โดย 690 คือจำนวน Row ของข้อมูลทั้งหมด, 5 คือจำนวน lags และ 1 คือ feature ที่ใช้ในที่นี้คือ gmv<br />
[ [ [g1],[g2],[g3],[g4],[g5] ] ] ==> [g6] ; เมื่อ g1 คือ lag feature ที่ย้อนกลับไป 5 lags<br /> 
Input: A pandas DataFrame df and a window_size (default is 5).
Output: Two numpy arrays, X and y, where X contains sequences of data points and y contains the corresponding labels (the next data point following each sequence).

#### Use convert function Check numpy array shape
```
WINDOW_SIZE = 5
X, y = df_to_X_y(df['gmv'], WINDOW_SIZE)
X.shape, y.shape
```
#### Split Training/Testing data set as 80:20 for 1 feature
```
X_train, y_train = X[:432], y[:432]
X_test, y_test = X[432:], y[432:]
X_train.shape, y_train.shape, X_test.shape, y_test.shape
```
#### <a name="multiple_features"></a>Function to convert DataFrame to Numpy array (multiple features)
```
# [ [ [g1, q1, g3],[g2, q1],[g3, q1],[g4, q1],[g5, q1] ] ] ==> [g6]

def df_to_X_y2 (df, window_size=7):
  df_as_np = df.to_numpy()
  X = []
  y = []
  for i in range(len(df_as_np)-window_size):
    row = [r for r in df_as_np[i:i+window_size]]
    X.append(row)
    label = df_as_np[i+window_size][0]
    y.append(label)
  return np.array(X), np.array(y)
```
#### Function Purpose
เป็นฟังก์ชันที่เปลี่ยนจาก DataFrame เป็น Numpy array คล้ายกับด้านบน ต่างกันที่จะเป็นการใช้หลาย ๆ features มาช่วยในการ train <br />
[ [ [g1, q1],[g2, q1],[g3, q1],[g4, q1],[g5, q1] ] ] ==> [g6] ; เมื่อ g, q คือแต่ละ feature<br />
- Input: DataFrame และ window_size (ค่าเริ่มต้นคือ 7)
- Output: Numpy: X และ y โดยที่ X คือ array ของ feature (rows, lags, features) และ y เป็น label เป้าหมายที่เราจะทำนาย
#### Use convert function Check numpy array shape
```
X2, y2 = df_to_X_y2(df)
X2.shape, y2.shape
```
``` Split Training/Testing data set as 80:20 for multiple features
X2_train, y2_train = X2[:int(len(df)*0.8)], y2[:int(len(df)*0.8)]
X2_test, y2_test = X2[int(len(df)*0.8):], y2[int(len(df)*0.8):]
X2_train.shape, y2_train.shape, X2_test.shape, y2_test.shape
```
#### <a name="standardize"></a>Standardization
Standardization คือกระบวนการในขั้นตอนการเตรียมข้อมูลที่ใช้ปรับขนาดคุณลักษณะ (features) ให้มีค่าเฉลี่ยเป็น 0 และส่วนเบี่ยงเบนมาตรฐานเป็น 1 การทำเช่นนี้เพื่อให้ feature แต่ละตัวมีส่วนร่วมใน model อย่างเท่าเทียมกันและเพื่อปรับปรุงประสิทธิภาพและความเสถียรในการฝึกของอัลกอริทึม Machine Learning<br />
<br />
```
gmv_training_mean = np.mean(X2_train[:, :, 0])
gmv_training_std = np.std(X2_train[:, :, 0])

def preprocess(X):
  X[:, :, 0] = (X[:, :, 0] - gmv_training_mean) / gmv_training_std
  return X

def preprocess_output(y):
  y = (y - gmv_training_mean) / gmv_training_std
  return y
```
#### <a name="seq_model"></a>Compiles a Sequential model using Keras for time series forecasting
```
model4 = Sequential()
model4.add(InputLayer(input_shape=(7, 13)))
model4.add(LSTM(50, return_sequences = True,))
model4.add(LSTM(50, ))
model4.add(Dense(25, 'relu'))
model4.add(Dense(1, 'linear'))

model4.summary()
model4.compile(loss='mse', optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
```
#### <a name="train_2"></a>Training model
ฝึกโมเดล model4 โดยใช้เมธอด fit กับข้อมูลการฝึก X2_train และ y2_train เป็นเวลา 20 รอบการฝึก (epochs) เมธอด fit จะคืนค่า history ซึ่งประกอบด้วยรายละเอียดเกี่ยวกับกระบวนการฝึก<br />
<br />
Trains the model4 using the fit method with the training data X2_train and y2_train for 20 epochs. The fit method returns a history object which contains details about the training process.

- model4.fit(data): Function ฝึกโมเดลโดยใช้ข้อมูลการฝึกที่จัดเตรียมไว้
- X2_train: Input features array
- y2_train: Target array
- epochs=20: โมเดลจะทำการวนรอบผ่านชุดข้อมูลการฝึกทั้งหมด 20 ครั้ง
```
history = model4.fit(X2_train, y2_train, epochs=20)
```
#### Function to convert standardized GMV back to actual GMV
Standardized จะทำให้ข้อมูลมีค่าอยู่ระหว่าง -3 ถึง 3 เมื่อททำนายออกมาก็จะได้ค่าใน Range นี้ ก่อนทำการเปรียบเทียบจึงต้องทำการแปลงค่ากลับเป็นค่าจริง
```
def post_process_gmv(arr):
  arr = (arr*gmv_training_std) + gmv_training_mean
  return arr
```
```
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from math import sqrt
df2 = df.copy()
```
### <a name="eval_2"></a>Evaluation
#### Function to predict the validate data
```
def plot_predictions2(model, X, y, l):
    predictions = model.predict(X)
    gmv_pred = post_process_gmv(predictions[:, 0])
    gmv_actual = post_process_gmv(y)
    df = pd.DataFrame(data={'Predictions':gmv_pred, 'Actuals':gmv_actual}, index= df2[int(len(df2)*0.8) + 7:].index)
    rmse = sqrt(mean_squared_error(df['Actuals'], df['Predictions']))
    mae = mean_absolute_error(df['Actuals'], df['Predictions'])
    mape = mean_absolute_percentage_error(df['Actuals'], df['Predictions'])
    plt.plot(df2['gmv'], label='Original GMV')
    plt.plot(df['Predictions'], label='Predictions')
    plt.plot(df['Actuals'], label='Actuals')
    plt.title(label=l+f' mae: {mae} - mape: {mape} - rmse: {rmse}')
    plt.legend()
    return df.tail(7)
```
#### <a name="result_2"></a>Call plot_prediction function
Function จะแสดง DataFrame ค่าจริงและค่าที่พยากรณ์และจะแสดงเป็นกราฟด้วย
```
plot_predictions2(model4, X2_test, y2_test, 'LSTM')
```
![prediction](https://github.com/u6587017/TalaadThaiOnline_Internship_Forecast_GMV_PJ/assets/108443663/371ebf92-4ee9-4717-9eaa-1f05cf76e6a0)
![latest](https://github.com/u6587017/TalaadThaiOnline_Internship_Forecast_GMV_PJ/assets/108443663/81cd42b0-be71-42f3-a22d-46ac1bd58830)
