import numpy as np


# 데이터셋 생성
def seq2dataset(seq, window_size, code2idx):
    dataset = []
    for i in range(len(seq) - window_size):
        subset = seq[i:(i + window_size + 1)]
        dataset.append([code2idx[item] for item in subset])
    return np.array(dataset)
