import numpy as np
import matplotlib.pyplot as plt


# 데이터셋 생성
def create_dataset(signal_data, look_back=1):
    dataX = []
    dataY = []
    for i in range(len(signal_data) - look_back):
        dataX.append(signal_data[i:(i + look_back), 0])
        dataY.append(signal_data[i + look_back, 0])

    return np.array(dataX), np.array(dataY)


# 학습과정 출력
def show_model_learning_process(hist):
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
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


# 모델 사용
def use_model(model, x_test, y_test, batch_size):
    look_ahead = 250
    xhat = x_test[0, None]
    predictions = np.zeros((look_ahead, 1))
    for i in range(look_ahead):
        prediction = model.predict(xhat, batch_size)
        predictions[i] = prediction
        xhat = np.hstack([xhat[:, 1:], prediction])

    plt.figure(figsize=(12, 5))
    plt.plot(np.arange(look_ahead), predictions, 'r', label="prediction")
    plt.plot(np.arange(look_ahead), y_test[:look_ahead], label="test function")
    plt.legend()
    plt.show()
