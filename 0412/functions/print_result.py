import numpy as np


# 한 스텝 예측
def print_one_step_prediction(model, x_train, idx2code, pred_count=50):
    seq_out = ['g8', 'e8', 'e4', 'f8']
    pred_out = model.predict(x_train)

    for i in range(pred_count):
        idx = np.argmax(pred_out[i])  # one-hot 인코딩을 인덱스 값으로 변환
        seq_out.append(idx2code[idx])  # seq_out는 최종 악보이므로 인덱스 값을 코드로 변환하여 저장

    print("one step prediction : ", seq_out)


# 곡 전체 예측
def print_full_song_prediction(model, code2idx, idx2code, max_idx_value, pred_count=50):
    seq_in = ['g8', 'e8', 'e4', 'f8']
    seq_out = seq_in
    seq_in = [code2idx[it] / float(max_idx_value) for it in seq_in]  # 코드를 인덱스값으로 변환

    for i in range(pred_count):
        sample_in = np.array(seq_in)
        sample_in = np.reshape(sample_in, (1, 4))  # batch_size, feature
        pred_out = model.predict(sample_in)
        idx = np.argmax(pred_out)
        seq_out.append(idx2code[idx])
        seq_in.append(idx / float(max_idx_value))
        seq_in.pop(0)

    print("full song prediction : ", seq_out)