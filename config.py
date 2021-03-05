#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 17:28, 04/03/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from os.path import abspath, dirname

basedir = abspath(dirname(__file__))


class Config:
    CORE_DATA_DIR = f'{basedir}/data'
    RESULTS_DATA = f'{CORE_DATA_DIR}/results'

    DATA_OPEN = f'{CORE_DATA_DIR}/open_source'
    DATA_PROPRIETARY = f'{CORE_DATA_DIR}/proprietary'

    LINUX_RAW = f"{DATA_OPEN}/linux/2k_log_raw"
    LINUX_EVENTS = f"{DATA_OPEN}/linux/2k_log_events.csv"
    LINUX_TEMPLATES = f"{DATA_OPEN}/linux/2k_log_templates.csv"
    LINUX_TRAIN_TEST = f"{DATA_OPEN}/linux/2k_log_train_test.csv"
    LINUX_TRAIN = f"{DATA_OPEN}/linux/2k_log_train.csv"
    LINUX_TEST = f"{DATA_OPEN}/linux/2k_log_test.csv"
    LINUX_TEST_SIZE = 0.5

    WINDOWS_RAW = f"{DATA_PROPRIETARY}/windows/2k_log_raw"
    WINDOWS_EVENTS = f"{DATA_PROPRIETARY}/windows/2k_log_events.csv"
    WINDOWS_TEMPLATES = f"{DATA_PROPRIETARY}/windows/2k_log_templates.csv"
    WINDOWS_TRAIN_TEST = f"{DATA_PROPRIETARY}/windows/2k_log_train_test.csv"
    WINDOWS_TRAIN = f"{DATA_PROPRIETARY}/windows/2k_log_train.csv"
    WINDOWS_TEST = f"{DATA_PROPRIETARY}/windows/2k_log_test.csv"
    WINDOWS_TEST_SIZE = 0.5


    SPE_CHAR = "<*>"            # Special character represent variable in the dataset
    DES_CHAR = "DES"
    VAR_CHAR = "VAR"

    FILE_HEADER_CONTENT = "Content"
    FILE_HEADER_EVENT_ID = "EventId"
    FILE_HEADER_EVENT_TEMPLATE = "EventTemplate"
    FILE_HEADER_EVENT_STR = "EventStr"

    FILE_SAVE_PREDICTION = "template_prediction"
    FILE_METRICS_NAME = "performance_metrics.csv"
    FILE_METRICS_HEADER = ['labelr', 'paras', 'trial', 'WA', 'TWA', 'RandScore', 'Precision', 'Recall', 'FScore', 'PA', 'CA', 'LA',
                           'EditDistanceMean', 'EditDistanceStd', 'TemplateAccuracy', 'TimeTrain', 'TimeTest']


    N_TRIALS = 10
    LABELING_RATES = [1, 5, 10, 50, 100, 500, 1000]

    EPOCHS = [200, 200, 200, 200, 100, 50, 10]  # ==> 200 update times
    BATCH_SIZE_TRAINS = [1, 5, 10, 50, 100, 500, 1000]
    BATCH_SIZE_TEST = 200       # Test size = 1000 --> 5 batch for faster
    PREDICTED_BATCH = 100

    MAX_SENTENCE_LEN = 64  # 64
    MAX_WORD_LEN = 64  # 48
    LABEL_INDEX = ["DES", "VAR"]

    # This config for transfer models
    TRANSFER_PARAS = {
        "1": {
            # [source, target]: We use all available data in source task for training, and labelling_rate in target task
            "label_rate": [1.0, 1],
            "batch_size_train": [100, 1],
            "batch_size_test": [200, 200],
            "max_epoch": [5, 50]
        },
        "2": {
            "label_rate": [1.0, 5],
            "batch_size_train": [100, 5],
            "batch_size_test": [200, 200],
            "max_epoch": [5, 50],
        },
        "3": {
            "label_rate": [1.0, 10],
            "batch_size_train": [100, 10],
            "batch_size_test": [200, 200],
            "max_epoch": [5, 50],
        },
        "4": {
            "label_rate": [1.0, 50],
            "batch_size_train": [100, 10],
            "batch_size_test": [200, 200],
            "max_epoch": [5, 50],
        },
        "5": {
            "label_rate": [1.0, 100],
            "batch_size_train": [100, 20],
            "batch_size_test": [200, 200],
            "max_epoch": [5, 20]
        }
    }

