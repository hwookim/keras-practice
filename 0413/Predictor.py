import numpy as np
import matplotlib.pyplot as plt


def run(model):
    # 데이터셋 생성
    x_train, y_train, x_test, y_test = set_data()

    # 모델 학습과정 설정
    model.compile(optimizer='rmsprop', loss='mse')

    # 학습
    hist = model.fit(x_train, y_train, epochs=50, batch_size=64)

    # 과정 출력 및 평가
    print_learning(hist)
    evaluate_model(model, x_test, y_test)


def set_data():
    x_train = np.random.random((1000, 1))
    y_train = x_train * 2 + np.random.random((1000, 1)) / 3.0
    x_test = np.random.random((100, 1))
    y_test = x_test * 2 + np.random.random((100, 1)) / 3.0
    return x_train, y_train, x_test, y_test


def print_learning(hist):
    plt.plot(hist.history['loss'])
    plt.ylim(0.0, 1.5)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()


def evaluate_model(model, x_test, y_test, batch_size=32):
    loss = model.evaluate(x_test, y_test, batch_size)
    print('loss : ' + str(loss))