import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical


def run(model):
    x_train, y_train, x_test, y_test = set_data()

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    hist = model.fit(x_train, y_train, epochs=1000, batch_size=64)

    print_learning(hist)
    evaluate_model(model, x_test, y_test)


def set_data():
    x_train = np.random.random((1000, 12))
    y_train = np.random.randint(10, size=(1000, 1))
    y_train = to_categorical(y_train, num_classes=10)
    x_test = np.random.random((100, 12))
    y_test = np.random.randint(10, size=(100, 1))
    y_test = to_categorical(y_test, num_classes=10)
    return x_train, y_train, x_test, y_test


def print_learning(hist):
    fig, loss_ax = plt.subplots()

    acc_ax = loss_ax.twinx()

    loss_ax.set_ylim([0.0, 3.0])
    acc_ax.set_ylim([0.0, 1.0])

    loss_ax.plot(hist.history['loss'], 'y', label='train loss')
    acc_ax.plot(hist.history['acc'], 'b', label='train acc')

    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_ylabel('accuray')

    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')

    plt.show()


def evaluate_model(model, x_test, y_test, batch_size=32):
    loss_and_metrics = model.evaluate(x_test, y_test, batch_size)
    print('loss_and_metrics : ' + str(loss_and_metrics))
