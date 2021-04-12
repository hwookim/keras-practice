import keras
import matplotlib.pyplot as plt
import numpy as np


# 데이터셋 생성
def seq2dataset(seq, window_size, code2idx):
    dataset = []
    for i in range(len(seq) - window_size):
        subset = seq[i:(i + window_size + 1)]
        dataset.append([code2idx[item] for item in subset])
    return np.array(dataset)


# 손실 이력 클래스 정의
class LossHistory(keras.callbacks.Callback):
    def init(self):
        self.losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


# 학습 과정 출력
def print_learning(history):
    # 학습 과정 출력
    plt.plot(history.losses)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()