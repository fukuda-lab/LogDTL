#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 10:47, 30/12/2020                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

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

DATASETS = {
    "linux": [Config.LINUX_TRAIN, Config.LINUX_TEST],
    # "windows": [Config.WINDOWS_TRAIN, Config.WINDOWS_TEST]
}
DATASET_GROUNDTRUTH = {
    "linux": Config.LINUX_TEST,
    # "windows": Config.WINDOWS_TEST
}


for data_name, data_list in DATASETS.items():
    pathsave = f"{Config.RESULTS_DATA}/{data_name}/dnn"
    Path(pathsave).mkdir(parents=True, exist_ok=True)

    performance_results = []
    for trial in range(Config.N_TRIALS):
        for idx_lr, labelr in enumerate(Config.LABELING_RATES):
            ## Create the actual training dataset
            ACTUAL_TRAIN = data_list[0].split(".")[0] + "_temp.csv"
            actual_train_df = read_csv(data_list[0])
            ind = sample.create_sample_index(labelr, actual_train_df.values.shape[0])
            actual_train_data = actual_train_df.values[ind]
            df = DataFrame(actual_train_data)
            df.columns = list(actual_train_df.columns)
            df.to_csv(ACTUAL_TRAIN, index=False)

            LIST_REAL_DATASET = [ACTUAL_TRAIN, data_list[1]]
            char_index, char_cnt = create_char_index(LIST_REAL_DATASET)  # char_index and char_count
            word_index, word_cnt = create_word_index(LIST_REAL_DATASET)  # word_index and word_count
            ind2word = {}
            for k, v in word_index.items():  ## All words {index: word, index: word,....}
                ind2word[v] = k

            ## Create word embedding vectors
            word2vec_embedding = create_word2vec_embedding(LIST_REAL_DATASET, ind2word)
            # word2vec_embedding = None

            ## Make training and testing MATRIX
            wx, y, m = make_dataset_based_on_word(ACTUAL_TRAIN, word_index, Config.MAX_SENTENCE_LEN, Config.LABEL_INDEX)
            # Training set, reading based on words
            twx, ty, tm = make_dataset_based_on_word(data_list[1], word_index, Config.MAX_SENTENCE_LEN, Config.LABEL_INDEX)
            # Testing set, reading based on words

            x, cm = make_dataset_based_on_char(ACTUAL_TRAIN, char_index, Config.MAX_WORD_LEN, Config.MAX_SENTENCE_LEN)
            # Training set, reading based on chars
            tx, tcm = make_dataset_based_on_char(data_list[1], char_index, Config.MAX_WORD_LEN, Config.MAX_SENTENCE_LEN)
            # Testing set, reading based on chars

            model = Model(char_cnt, len(Config.LABEL_INDEX), word_cnt, epoch=Config.EPOCHS[idx_lr],
                          batch_size=Config.BATCH_SIZE_TRAINS[idx_lr], test_batch_size=Config.BATCH_SIZE_TEST, joint=False, top_joint=False)
            model.build(x, y, m, wx, cm, word2vec_embedding)

            time_train = time()
            model.train()
            time_train = time() - time_train

            time_test = time()
            paras_name = "NoParas"
            file_prediction = f"{labelr}-{paras_name}-{trial}-{Config.FILE_SAVE_PREDICTION}.csv"

            model.predict(tx, ty, tm, twx, tcm, ind2word, Config.LABEL_INDEX, f"{pathsave}/{file_prediction}")
            time_test = time() - time_test

            WA, TWA, rand_score, precision, recall, f_score, PA, CA, ESM, dist_mean, dist_std, TA = \
                measurement(DATASET_GROUNDTRUTH[data_name], f"{pathsave}/{file_prediction}")

            performance_results.append([labelr, paras_name, trial, WA, TWA, rand_score, precision, recall, f_score, PA, CA, ESM,
                                        dist_mean, dist_std, TA, time_train, time_test])
            print(performance_results)

    df_results = DataFrame(performance_results, columns=Config.FILE_METRICS_HEADER)
    df_results.to_csv(f"{pathsave}/{Config.FILE_METRICS_NAME}", index=False)

