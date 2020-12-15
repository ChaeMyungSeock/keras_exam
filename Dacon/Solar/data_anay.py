import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import time
import statsmodels.api as sm
from sklearn import linear_model
from tqdm import tnrange, tqdm_notebook
from sklearn.metrics import mean_squared_error
from statsmodels.regression.quantile_regression import QuantReg
import tensorflow as tf

train = pd.read_csv('F:/Data/soloar_data/data/train/train.csv')
sub = pd.read_csv('F:/Data/soloar_data/data/sample_submission.csv')


for i in range(0,81):
    s1 = "test_%d = pd.read_csv('F:/Data/soloar_data/data/test/%d.csv')"%(i,i)
    exec(s1)


print(train.head(49))

print(test_0.head(49))

# Hour - 시간
# Minute - 분
# DHI - 수평면 산란일사량(Diffuse Horizontal Irradiance (W/m2))
# DNI - 직달일사량(Direct Normal Irradiance (W/m2))
# WS - 풍속(Wind Speed (m/s))
# RH - 상대습도(Relative Humidity (%))
# T - 기온(Temperature (Degree C))
# Target - 태양광 발전량 (kW)


tf.test.gpu_device_name() # 결과로 나오는 GPU는 본인 pc 설정에 따라 다를 수 있습니다.

