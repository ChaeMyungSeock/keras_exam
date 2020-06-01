import numpy as np
import pandas as pd

kospi200_data = pd.read_csv('./data/csv/kospi200.csv', index_col=0, header=0, sep=',', encoding='cp949')
samsung_data = pd.read_csv('./data/csv/samsung.csv', index_col=0, header=0, sep=',', encoding = 'cp949')

print(kospi200_data)
print(samsung_data)

print(len(kospi200_data.index))


# # kospi200_data의 거래량 str -> int
for i in range(len(kospi200_data.index)):
    kospi200_data.iloc[i,4] = int(kospi200_data.iloc[i,4].replace(',',''))


# samsung의 거래량 str -> int
for i in range(len(samsung_data.index)):
    for j in range(len(samsung_data.iloc[i])):
        samsung_data.iloc[i,j] = int(samsung_data.iloc[i,j].replace(',',''))

kospi200_data = kospi200_data.sort_values(['일자'], ascending = [True])
samsung_data = samsung_data.sort_values(['일자'], ascending = [True])



print(kospi200_data)
print(samsung_data)


kospi200_data = kospi200_data.values
samsung_data = samsung_data.values

np.save('./data/kospi200.npy',arr=kospi200_data)
np.save('./data/samsung.npy',arr=samsung_data)
