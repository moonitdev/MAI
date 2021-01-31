# [LSTM과 FinanceDataReader API를 활용한 삼성전자 주가 예측](https://teddylee777.github.io/tensorflow/lstm-stock-forecast)
​
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import FinanceDataReader as fdr
​
%matplotlib inline
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'NanumGothic'

# 삼성전자 주식코드: 005930
STOCK_CODE = '005930'
stock = fdr.DataReader(STOCK_CODE)

stock['Year'] = stock.index.year
stock['Month'] = stock.index.month
stock['Day'] = stock.index.day

# plt.figure(figsize=(16, 9))
# sns.lineplot(y=stock['Close'], x=stock.index)
# plt.xlabel('time')
# plt.ylabel('price')

# time_steps = [['1990', '2000'], 
#               ['2000', '2010'], 
#               ['2010', '2015'], 
#               ['2015', '2020']]
# ​
# fig, axes = plt.subplots(2, 2)
# fig.set_size_inches(16, 9)

# for i in range(4):
#     ax = axes[i//2, i%2]
#     df = stock.loc[(stock.index > time_steps[i][0]) & (stock.index < time_steps[i][1])]
#     sns.lineplot(y=df['Close'], x=df.index, ax=ax)
#     ax.set_title(f'{time_steps[i][0]}~{time_steps[i][1]}')
#     ax.set_xlabel('time')
#     ax.set_ylabel('price')

# plt.tight_layout()
# plt.show()

## 데이터 전처리
# 주가 데이터에 대하여 딥러닝 모델이 더 잘 학습할 수 있도록 정규화(Normalization)를 해주도록 하겠습니다.
​
# 표준화 (Standardization)와 정규화(Normalization)에 대한 내용은 아래 링크에서 더 자세히 다루니, 참고해 보시기 바랍니다.
​
# [데이터 전처리에 관하여](https://teddylee777.github.io/scikit-learn/scikit-learn-preprocessing)
​
from sklearn.preprocessing import MinMaxScaler
​
scaler = MinMaxScaler()
# 스케일을 적용할 column을 정의합니다.
scale_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
# 스케일 후 columns
scaled = scaler.fit_transform(stock[scale_cols])

df = pd.DataFrame(scaled, columns=scale_cols)
#  train / test 분할
​
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df.drop('Close', 1), df['Close'], test_size=0.2, random_state=0, shuffle=False)
x_train.shape, y_train.shape
x_test.shape, y_test.shape

## TensroFlow Dataset을 활용한 시퀀스 데이터셋 구성
import tensorflow as tf
​
def windowed_dataset(series, window_size, batch_size, shuffle):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.map(lambda w: (w[:-1], w[-1]))
    return ds.batch(batch_size).prefetch(1)

## Hyperparameter를 정의합니다.
​
WINDOW_SIZE=20
BATCH_SIZE=32

## trian_data는 학습용 데이터셋, test_data는 검증용 데이터셋 입니다.
train_data = windowed_dataset(y_train, WINDOW_SIZE, BATCH_SIZE, True)
test_data = windowed_dataset(y_test, WINDOW_SIZE, BATCH_SIZE, False)

## 아래의 코드로 데이터셋의 구성을 확인해 볼 수 있습니다.
# X: (batch_size, window_size, feature)
# Y: (batch_size, feature)
for data in train_data.take(1):
    print(f'데이터셋(X) 구성(batch_size, window_size, feature갯수): {data[0].shape}')
    print(f'데이터셋(Y) 구성(batch_size, window_size, feature갯수): {data[1].shape}')

## 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Lambda
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
​
​
model = Sequential([
    # 1차원 feature map 생성
    Conv1D(filters=32, kernel_size=5,
           padding="causal",
           activation="relu",
           input_shape=[WINDOW_SIZE, 1]),
    # LSTM
    LSTM(16, activation='tanh'),
    Dense(16, activation="relu"),
    Dense(1),
])

## Sequence 학습에 비교적 좋은 퍼포먼스를 내는 Huber()를 사용합니다.
loss = Huber()
optimizer = Adam(0.0005)
model.compile(loss=Huber(), optimizer=optimizer, metrics=['mse'])
# earlystopping은 10번 epoch통안 val_loss 개선이 없다면 학습을 멈춥니다.
earlystopping = EarlyStopping(monitor='val_loss', patience=10)

## val_loss 기준 체크포인터도 생성합니다.
filename = os.path.join('tmp', 'ckeckpointer.ckpt')
checkpoint = ModelCheckpoint(filename, 
                             save_weights_only=True, 
                             save_best_only=True, 
                             monitor='val_loss', 
                             verbose=1)
history = model.fit(train_data, 
                    validation_data=(test_data), 
                    epochs=50, 
                    callbacks=[checkpoint, earlystopping])

## 저장한 ModelCheckpoint 를 로드합니다.
​
model.load_weights(filename)
# test_data를 활용하여 예측을 진행합니다.
​
pred = model.predict(test_data)

plt.figure(figsize=(12, 9))
plt.plot(np.asarray(y_test)[20:], label='actual')
plt.plot(pred, label='prediction')
plt.legend()
plt.show()

plt.figure(figsize=(12, 9))
plt.plot(np.asarray(y_test)[20:], label='actual')
plt.plot(pred, label='prediction')
plt.legend()
plt.show()
​