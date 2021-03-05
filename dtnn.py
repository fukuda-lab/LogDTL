#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 08:29, 05/03/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import lasagne
from time import time
from pathlib import Path
from pandas import DataFrame, read_csv
from models.transfer.cnn_dgru_crf import Model
from models.utils import sample
from models.utils.dataset_util import create_word2vec_embedding, create_char_index
from models.utils.dataset_util import create_word_index, make_dataset_based_on_word, make_dataset_based_on_char
from models.utils.measurement_util import measurement
from config import Config

# =========================== Making training and testing dataset =============================

# =========================== Making model ====================================================

DATASET = {
    "source": [Config.LINUX_TRAIN, Config.LINUX_TEST],
    "target": [Config.WINDOWS_TRAIN, Config.WINDOWS_TEST]
}
DATASET_GROUNDTRUTH = {
    "linux": Config.LINUX_TEST,
    "windows": Config.WINDOWS_TEST
}

def running(transfer_paras, trial):
    performance_results = []

    ## Create the actual training dataset
    ACTUAL_TARGET_TRAIN = DATASET["target"][0].split(".")[0] + "_temp.csv"
    actual_train_df = read_csv(DATASET["target"][0])
    ind = sample.create_sample_index(transfer_paras["label_rate"][1], actual_train_df.values.shape[0])
    actual_train_data = actual_train_df.values[ind]
    df = DataFrame(actual_train_data)
    df.columns = list(actual_train_df.columns)
    df.to_csv(ACTUAL_TARGET_TRAIN, index=False)
    LIST_TRAIN_TEST = [
        [DATASET["source"][0], DATASET["source"][1]],
        [ACTUAL_TARGET_TRAIN, DATASET["target"][1]]
    ]
    FINISHED = False
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

    models = []
    datasets = {}
    for idx_task, task in enumerate(DATASET.items()):  ## Make training and testing dataset
        # Training set, reading based on words
        wx, y, m = make_dataset_based_on_word(LIST_TRAIN_TEST[idx_task][0], word_index, Config.MAX_SENTENCE_LEN, Config.LABEL_INDEX)
        # Testing set, reading based on words
        twx, ty, tm = make_dataset_based_on_word(LIST_TRAIN_TEST[idx_task][1], word_index, Config.MAX_SENTENCE_LEN, Config.LABEL_INDEX)

        # Training set, reading based on chars
        x, cm = make_dataset_based_on_char(LIST_TRAIN_TEST[idx_task][0], char_index, Config.MAX_WORD_LEN, Config.MAX_SENTENCE_LEN)
        # Testing set, reading based on chars
        tx, tcm = make_dataset_based_on_char(LIST_TRAIN_TEST[idx_task][1], char_index, Config.MAX_WORD_LEN, Config.MAX_SENTENCE_LEN)

        ## Create word embedding vectors
        word2vec_embedding = create_word2vec_embedding(LIST_TRAIN_TEST[idx_task], ind2word_full[idx_task])
        # word2vec_embedding = None

        model = Model(char_cnt, len(Config.LABEL_INDEX), word_cnt,
                      batch_size=transfer_paras["batch_size_train"][idx_task],
                      test_batch_size=transfer_paras["batch_size_test"][idx_task],
                      max_epoch=transfer_paras["max_epoch"][idx_task],
                      char_double_layer=False, word_double_layer=False, very_top_joint=True)
        model.build(x, y, m, wx, cm, word2vec_embedding)
        model.step_train_init()
        models.append(model)
        datasets[idx_task] = {"x": x, "y": y, "m": m, "wx": wx, "cm": cm, "tx": tx, "ty": ty, "tm": tm, "twx": twx, "tcm": tcm}

    time_train = time()
    prev_params = None
    while not FINISHED:
        for idx_md, model in enumerate(models):
            if prev_params is not None:
                lasagne.layers.set_all_param_values(model.shared_layer, prev_params)
            py, loss_epoch = model.step_train(datasets[idx_md]["tx"], datasets[idx_md]["ty"], datasets[idx_md]["tm"],
                                              datasets[idx_md]["twx"], datasets[idx_md]["tcm"])
            if py is not None:
                acc, f1, prec, recall, rand_score = model.step_evaluate(py, datasets[idx_md]["ty"], datasets[idx_md]["tm"])
                print(f"{list(DATASET.keys())[idx_md]} - Acc: {acc:.3f}, F1: {f1:.3f}, Prec: {prec:.3f}, Recall: {recall:.3f}, RandScore: {rand_score:.3f}")

            prev_params = lasagne.layers.get_all_param_values(model.shared_layer)
            if idx_md == 1 and model.epoch == model.max_epoch:
                FINISHED = True
    time_train = time() - time_train

    time_test = time()
    file_prediction = f"{transfer_paras['label_rate'][1]}-{transfer_paras['max_epoch']}-{trial}-{Config.FILE_SAVE_PREDICTION}.csv"
    models[1].predict(datasets[1]["tx"], datasets[1]["ty"], datasets[1]["tm"], datasets[1]["twx"], datasets[1]["tcm"],
                      ind2word_full, Config.LABEL_INDEX, f"{pathsave}/{file_prediction}")
    WA, TWA, rand_score, precision, recall, f_score, PA, CA, ESM, dist_mean, dist_std, TA = \
        measurement(DATASET_GROUNDTRUTH[1], f"{pathsave}/{file_prediction}")
    time_test = time() - time_test

    performance_results.append([transfer_paras["label_rate"][1], transfer_paras["max_epoch"], trial, WA, TWA, rand_score, precision,
                                recall, f_score, PA, CA, ESM, dist_mean, dist_std, TA, time_train, time_test])
    return performance_results


if __name__ == '__main__':
    pathsave = f"{Config.RESULTS_DATA}/windows/dtnn"
    Path(pathsave).mkdir(parents=True, exist_ok=True)
    with open(f'{pathsave}/{Config.FILE_METRICS_NAME}', 'a') as file:
        file.write(f"{', '.join(Config.FILE_METRICS_HEADER)}\n")
        for idx, transfer_paras in Config.TRANSFER_PARAS.items():
            for trial in range(Config.N_TRIALS):
                temp = running(transfer_paras, trial)
                file.write(f"{', '.join([str(n) for n in temp])}\n")
                file.flush()