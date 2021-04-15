import matplotlib.pyplot as plt


def print_learning(hist):
    plt.plot(hist.history['loss'])
    plt.ylim(0.0, 1.5)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()
