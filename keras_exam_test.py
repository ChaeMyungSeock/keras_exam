from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv2D
from keras.layers import Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


# 1. 데이터
data_kospi200_load = np.load('./data/kospi200.npy', allow_pickle= True)
data_samsung_load = np.load('./data/samsung.npy', allow_pickle= True)


print(data_kospi200_load.__class__)
print(data_samsung_load.__class__)
