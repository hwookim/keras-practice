import keras
import numpy as np
import matplotlib.pyplot as plt

# COLOR = 'white'
# plt.rcParams['axes.labelcolor'] = COLOR
# plt.rcParams['xtick.color'] = COLOR
# plt.rcParams['ytick.color'] = COLOR


# 데이터셋 생성
def create_dataset(signal_data, look_back=1):
    dataX = []
    dataY = []
    for i in range(len(signal_data) - look_back):
        dataX.append(signal_data[i:(i + look_back), 0])
        dataY.append(signal_data[i + look_back, 0])

    return np.array(dataX), np.array(dataY)


# 학습과정 출력
def show_model_learning_process(loss, val_loss):
    plt.plot(loss)
    plt.plot(val_loss)
    plt.ylim(0.0, 0.15)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


# 모델 평가
def evaluate_model(model, x_train, y_train, x_val, y_val, x_test, y_test, batch_size=None):
    trainScore = model.evaluate(x_train, y_train, batch_size, verbose=0)
    model.reset_states()
    print('Train Score: ', trainScore)
    valScore = model.evaluate(x_val, y_val, batch_size, verbose=0)
    model.reset_states()
    print('Validataion Score: ', valScore)
    testScore = model.evaluate(x_test, y_test, batch_size, verbose=0)
    model.reset_states()
    print('Test Score: ', testScore)


# 모델 결과 출력
def print_used_model(y_test, look_ahead, predictions):
    plt.figure(figsize=(12, 5))
    plt.plot(np.arange(look_ahead), predictions, 'r', label="prediction")
    plt.plot(np.arange(look_ahead), y_test[:look_ahead], label="test function")
    plt.legend()
    plt.show()


class CustomHistory(keras.callbacks.Callback):
    def init(self):
        self.train_loss = []
        self.val_loss = []

    def on_epoch_end(self, batch, logs={}):
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))