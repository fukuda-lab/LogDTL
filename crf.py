#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 17:30, 04/03/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from time import time
from pathlib import Path
from pandas import DataFrame
from models.transfer.crf import Model, create_features_labels
from models.utils import sample
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
ALGORITHMS = ["lbfgs"]


if __name__ == '__main__':

    for data_name, data_list in DATASETS.items():
        ## Make training and testing dataset
        X_train, y_train = create_features_labels(data_list[0])
        X_test, y_test = create_features_labels(data_list[1])
        pathsave = f"{Config.RESULTS_DATA}/{data_name}/crf"
        Path(pathsave).mkdir(parents=True, exist_ok=True)
        performance_results = []

        for trial in range(Config.N_TRIALS):
            for idx_lr, labelr in enumerate(Config.LABELING_RATES):
                for algorithm in ALGORITHMS:
                    paras_name = algorithm
                    model = Model(algorithm=algorithm)

                    ind = sample.create_sample_index(labelr, len(X_train))
                    X, y = sample.sample_arrays((X_train, y_train), ind)

                    time_train = time()
                    model.train(X, y)
                    time_train = time() - time_train

                    time_test = time()
                    file_prediction = f"{labelr}-{paras_name}-{trial}-{Config.FILE_SAVE_PREDICTION}.csv"
                    model.predict(X_test, y_test, DATASET_GROUNDTRUTH[data_name], f"{pathsave}/{file_prediction}")
                    time_test = time() - time_test

                    WA, TWA, rand_score, precision, recall, f_score, PA, CA, ESM, dist_mean, dist_std, TA = \
                        measurement(DATASET_GROUNDTRUTH[data_name], f"{pathsave}/{file_prediction}")

                    performance_results.append([labelr, paras_name, trial, WA, TWA, rand_score, precision, recall, f_score, PA, CA, ESM,
                                    dist_mean, dist_std, TA, time_train, time_test])
                    print(performance_results)

        df_results = DataFrame(performance_results, columns=Config.FILE_METRICS_HEADER)
        df_results.to_csv(f"{pathsave}/{Config.FILE_METRICS_NAME}", index=False)


