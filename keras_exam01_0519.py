# 1.데이터

import numpy as np
dataset = np.array([1,2,3,4,5,6,7,8,9,10])

def split_xy1(dataset, time_step):
    x,y = list(), list()
    
    for i in range(len(dataset)):
        end_number = i+time_step
        # 추가
        # if end_number > len(dataset)-1:
            # break

        if end_number > len(dataset)-1:
            break
            
        tmp_x, tmp_y = dataset[i : end_number], dataset[end_number]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

time_step = 4

x,y = split_xy1(dataset,time_step)

print(x, "\n" ,y)
print("x.shape : ",x.shape)
print("y.shape : ",y.shape)


x = x.reshape(x.shape[0], x.shape[1], 1) # RNN을 하기 위해서는 shape를 (행, 열, 특성)으로 변환할 필요성이 있다
print(x.shape) 


# RNN은 순차형(sequential) 데이터를 모델링하는데 최적화된 구조

# print(x_train.shape) (3,5,1)
# print(y_train.shape) (3,)

# print(x_train)

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, SimpleRNN
# from keras.layers import Dense, SimpleRNN

model = Sequential()
model.add(LSTM(64, input_shape = (4,1))) # 4개의 칼럼을 1개씩 잘라서 쓰는 LSTM을 구성하겠다는 의미
model.add(Dense(100))
model.add(Dense(150))
model.add(Dense(250))
model.add(Dense(200))
model.add(Dense(150))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))



model.summary() 


# 3. 훈련
model.compile(loss = 'mse', optimizer = 'adam')
from keras.callbacks import EarlyStopping, TensorBoard # 웹에서 우리의 모델을 볼 수 있는 TensorBoard를 사용해보자
# tb_hist = TensorBoard(log_dir = "D:\Study\keras\graph", histogram_freq =  0, write_graph = True, write_images=True) # 이 코드를 model.fit전에 삽입한다 그리고 fit의 callbacks를 수정해준다
early_stopping = EarlyStopping(monitor = 'loss', patience = 3, mode = 'min')
model.fit(x, y, epochs=1000, batch_size = 1, verbose=1, callbacks=[early_stopping] )


# 4. 예측 , 평가
loss = model.evaluate(x,y)
print("loss : ", loss)

x_predict = np.array([7,8,9,10])


x_predict = x_predict.reshape(1, x_predict.shape[0],1) # 위에서 데이터를 fit하는 과정의 데이터 set이 (1, 4, 1)이 였으므로 predict 또한 그에 맞춰줘야 합니다
print("x_predict.shape : ", x_predict.shape)
y_predict = model.predict(x_predict)
print(y_predict)