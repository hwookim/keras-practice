import numpy as np
import matplotlib.pyplot as plt

COLOR = 'white'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR
plt.rcParams['xtick.color'] = COLOR
plt.rcParams['ytick.color'] = COLOR


# 데이터셋 생성
def generate_dataset(samples, width=16, height=16):
    ds_x = []
    ds_y = []

    for it in range(samples):
        num_pt = np.random.randint(0, width * height)
        img = generate_image(num_pt, width, height)

        ds_y.append(num_pt)
        ds_x.append(img)

    return np.array(ds_x), np.array(ds_y).reshape(samples, 1)


# 이미지 생성
def generate_image(points, width=16, height=16):
    img = np.zeros((width, height))
    pts = np.random.random((points, 2))

    for ipt in pts:
        img[int(ipt[0] * width), int(ipt[1] * height)] = 1

    return img.reshape(width, height, 1)


# 학습 과정 출력
def show_model_learning_process(hist):
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.ylim(0.0, 300.0)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()