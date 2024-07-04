# Forecast_GMV_PJ

## About Project
The project involves creating a Machine Learning model to predict GMV (Gross Merchandise Value). It will be carried out in two phases:
### Files
#### PJ_forecast_GMV_phase1.ipynb: Development of a Machine Learning Model using Darts Library
#### PJ_forecast_GMV_phase2.ipynb: Development of a Deep Learning Model using TensorFlow and Keras
## Prerequisites
Python 3.9-3.12
### Library
Pandas, Numpy, Matplotlib 
#### Phase 1 : Darts 0.13.0, Scikit-learn 3.8.1 
#### Phase 2 : TensorFlow 2.15.0

## Phase 1
Import required library
```
import pandas as pd
from darts import TimeSeries
import numpy as np
import matplotlib.pyplot as plt
from darts.metrics.metrics import mape
from darts.metrics.metrics import mae
from darts.metrics.metrics import rmse
```
