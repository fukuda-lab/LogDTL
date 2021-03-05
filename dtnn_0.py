#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 11:20, 05/03/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from time import time
from pathlib import Path
from models.transfer.cnn_dgru_crf import Model
from models.utils.dataset_util import create_char_index, create_word_index, make_dataset_based_on_word, make_dataset_based_on_char
from models.utils.measurement_util import measurement
from config import Config


DATASET = {
    "source": [Config.LINUX_TRAIN, Config.LINUX_TEST],
    "target": [Config.WINDOWS_TRAIN, Config.WINDOWS_TEST]
}
DATASET_GROUNDTRUTH = {
    "linux": Config.LINUX_TEST,
    "windows": Config.WINDOWS_TEST
}

BATCH_SIZE_TRAIN = 100
MAX_EPOCHS = [10, 1000]             # Model Source , Model Target
BATCH_SIZE_TESTS = [100, 100]       # Model Source, Model Target
PREDICTED_BATCH = 10                # Predicted Batch in Testing Set of Model Target


def running(trial):
    performance_results = []

    LIST_TRAIN_TEST = [
        [DATASET["source"][0], DATASET["source"][1]],
        [DATASET["target"][1]]
    ]
    char_set, word_set = set(), set()
    for idx_task, task in enumerate(DATASET.items()):
        char_index, _ = create_char_index(LIST_TRAIN_TEST[idx_task])  # char_index and char_count
        for k, v in char_index.items():
            char_set.add(k)
        word_index, _ = create_word_index(LIST_TRAIN_TEST[idx_task])  # word_index and word_count
        for k, v in word_index.items():
            word_set.add(k)

    char_index, char_cnt = {}, 0
    for char in char_set:
        char_index[char] = char_cnt
        char_cnt += 1
    word_index, word_cnt = {}, 0
    for word in word_set:
        word_index[word] = word_cnt
        word_cnt += 1

    ind2word_full = {}
    for k, v in word_index.items():  ## All words {index: word, index: word,....}
        ind2word_full[v] = k

    datasets = {}
    # ## Make source task model
    s_wx, s_y, s_m = make_dataset_based_on_word(LIST_TRAIN_TEST[0][0], word_index, Config.MAX_SENTENCE_LEN, Config.LABEL_INDEX)
    s_twx, s_ty, s_tm = make_dataset_based_on_word(LIST_TRAIN_TEST[0][1], word_index, Config.MAX_SENTENCE_LEN, Config.LABEL_INDEX)
    s_x, s_cm = make_dataset_based_on_char(LIST_TRAIN_TEST[0][0], char_index, Config.MAX_WORD_LEN, Config.MAX_SENTENCE_LEN)
    s_tx, s_tcm = make_dataset_based_on_char(LIST_TRAIN_TEST[0][1], char_index, Config.MAX_WORD_LEN, Config.MAX_SENTENCE_LEN)
    # word2vec_embedding = create_word2vec_embedding(LIST_TRAIN_TEST[idx_task], ind2word_list[idx_task])
    word2vec_embedding = None
    model = Model(char_cnt, len(Config.LABEL_INDEX), word_cnt,
                  batch_size=BATCH_SIZE_TRAIN,
                  test_batch_size=BATCH_SIZE_TESTS[0],
                  max_epoch=MAX_EPOCHS[0],
                  char_double_layer=False, word_double_layer=False, very_top_joint=True)
    model.build(s_x, s_y, s_m, s_wx, s_cm, word2vec_embedding)
    model.step_train_init()
    datasets[0] = {"x": s_x, "y": s_y, "m": s_m, "wx": s_wx, "cm": s_cm, "tx": s_tx, "ty": s_ty, "tm": s_tm, "twx": s_twx, "tcm": s_tcm}

    # ## Make target task model
    t_twx, t_ty, t_tm = make_dataset_based_on_word(LIST_TRAIN_TEST[1][0], word_index, Config.MAX_SENTENCE_LEN, Config.LABEL_INDEX)
    t_tx, t_tcm = make_dataset_based_on_char(LIST_TRAIN_TEST[1][0], char_index, Config.MAX_WORD_LEN, Config.MAX_SENTENCE_LEN)
    datasets[1] = {"tx": t_tx, "ty": t_ty, "tm": t_tm, "twx": t_twx, "tcm": t_tcm}

    time_train = time()
    file_prediction = f"{0}-{MAX_EPOCHS}-{trial}-{Config.FILE_SAVE_PREDICTION}.csv"
    for idx_epoch in range(MAX_EPOCHS[1]):
        py, loss_epoch = model.step_train(datasets[0]["tx"], datasets[0]["ty"], datasets[0]["tm"], datasets[0]["twx"], datasets[0]["tcm"])
        if py is not None:
            acc, f1, prec, recall, rand_score = model.step_evaluate(py, datasets[0]["ty"], datasets[0]["tm"])
            print(f"Task: Target, Epoch: {model.epoch}, Iter: {model.iter_batch}, Acc: {acc}, F1: {f1}, Loss: {loss_epoch}")
        if model.iter_batch == Config.PREDICTED_BATCH:
            model.predict(datasets[1]["tx"], datasets[1]["ty"], datasets[1]["tm"], datasets[1]["twx"], datasets[1]["tcm"],
                          ind2word_full, f"{pathsave}/{file_prediction}")
            WA, TWA, rand_score, precision, recall, f_score, PA, CA, ESM, dist_mean, dist_std, TA = \
                measurement(DATASET_GROUNDTRUTH[1], f"{pathsave}/{file_prediction}")
            print(f"Task: Target, Epoch: {idx_epoch}, WA: {WA:.3f}, TWA: {TWA:.3f}, F: {f_score:.3f}, CA: {CA:.3f}, Line_Acc: {ESM:.3f}, TA: {TA:.3f}")
    time_train = time() - time_train

    time_test = time()
    model.predict(datasets[1]["tx"], datasets[1]["ty"], datasets[1]["tm"], datasets[1]["twx"], datasets[1]["tcm"],
            ind2word_full, f"{pathsave}/{file_prediction}")
    WA, TWA, rand_score, precision, recall, f_score, PA, CA, ESM, dist_mean, dist_std, TA = \
        measurement(DATASET_GROUNDTRUTH["windows"], f"{pathsave}/{file_prediction}")
    time_test = time() - time_test

    performance_results.append([0, MAX_EPOCHS, trial, WA, TWA, rand_score, precision,
                                recall, f_score, PA, CA, ESM, dist_mean, dist_std, TA, time_train, time_test])
    return performance_results


if __name__ == '__main__':
    pathsave = f"{Config.RESULTS_DATA}/windows/dtnn_0"
    Path(pathsave).mkdir(parents=True, exist_ok=True)
    with open(f'{pathsave}/{Config.FILE_METRICS_NAME}', 'a') as file:
        file.write(f"{', '.join(Config.FILE_METRICS_HEADER)}\n")
        for trial in range(Config.N_TRIALS):
            temp = running(trial)
            file.write(f"{', '.join([str(n) for n in temp])}\n")
            file.flush()
