import numpy as np
import matplotlib.pyplot as plt


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


# 데이터셋 가시화
def visualize_dataset(x_train, y_train, row=5, col=5, width=16, height=16):
    plt.rcParams["figure.figsize"] = (10, 10)

    f, axarr = plt.subplots(row, col)

    for i in range(row * col):
        sub_plt = axarr[i / row, i % col]
        sub_plt.axis('off')
        sub_plt.imshow(x_train[i].reshape(width, height))
        sub_plt.set_title('R ' + str(y_train[i][0]))

    plt.show()


# 학습 과정 출력
def show_model_learning_process(hist):
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.ylim(0.0, 300.0)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()