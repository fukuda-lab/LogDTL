#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 00:19, 07/08/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

# https://github.com/abcdw/crf-pos-tagger
# https://github.com/nikhitmago/conditional-random-field
# https://nlpforhackers.io/crf-pos-tagger/
# https://medium.com/@shabeelkandi/handling-out-of-vocabulary-words-in-natural-language-processing-based-on-context-4bbba16214d5
# https://github.com/XuezheMax/NeuroNLP2
# https://github.com/pth1993/NNVLP

from numpy import array
from pandas import read_csv, DataFrame
import sklearn_crfsuite
import hashlib
from config import Config


def simple_word2features(sentence, id_word):
    return {
        'word': sentence[id_word],
        'is_first': id_word == 0,
        'is_last': id_word == len(sentence) - 1,
        'prev_word': '' if id_word == 0 else sentence[id_word - 1],
        'next_word': '' if id_word == len(sentence) - 1 else sentence[id_word + 1],
        'has_hyphen': '-' in sentence[id_word],
        'is_numeric': sentence[id_word].isdigit(),
        'capitals_inside': sentence[id_word][1:].lower() != sentence[id_word][1:]
    }


def word2features(sentence, id_word):
    features = {
        'word': sentence[id_word],
        'is_first': id_word == 0,
        'is_last': id_word == len(sentence) - 1,
        'is_capitalized': sentence[id_word][0].upper() == sentence[id_word][0],
        'is_all_caps': sentence[id_word].upper() == sentence[id_word],
        'is_all_lower': sentence[id_word].lower() == sentence[id_word],
        'prefix-1': sentence[id_word][0],
        'prefix-2': sentence[id_word][:2],
        'prefix-3': sentence[id_word][:3],
        'suffix-1': sentence[id_word][-1],
        'suffix-2': sentence[id_word][-2:],
        'suffix-3': sentence[id_word][-3:],
        'prev_word': '' if id_word == 0 else sentence[id_word - 1],
        'next_word': '' if id_word == len(sentence) - 1 else sentence[id_word + 1],
        'has_hyphen': '-' in sentence[id_word],
        'is_numeric': sentence[id_word].isdigit(),
        'capitals_inside': sentence[id_word][1:].lower() != sentence[id_word][1:]
    }
    return features


def get_features(sent):
    return [simple_word2features(sent, i) for i in range(len(sent))]


def create_features_labels(filenames):
    if type(filenames) is str:
        filenames = [filenames]
    X_features = []
    y_tags = []
    for idx, filename in enumerate(filenames):
        df = read_csv(filename)
        df_contents = df[Config.FILE_HEADER_CONTENT].values
        df_tags = df[Config.FILE_HEADER_EVENT_STR].values
        for idx_sent, sentence in enumerate(df_contents):
            ## Features
            x_temp = get_features(sentence.strip().split())
            X_features.append(x_temp)
            ## Tagged
            y_tags.append(df_tags[idx_sent].strip().split())
    return X_features, y_tags


class Model:
    def __init__(self, algorithm="lbfgs"):
        self.algorithm = algorithm

    def train(self, X, y):
        self.model = sklearn_crfsuite.CRF(algorithm=self.algorithm)
        self.model.fit(X, y)

    def outputResult(self, pred):
        df_events = []
        templateids = []
        for pr in pred:
            template_id = hashlib.md5(pr.encode('utf-8')).hexdigest()
            templateids.append(template_id)
            df_events.append([template_id, pr])

        df_event = DataFrame(df_events, columns=[Config.FILE_HEADER_EVENT_ID, Config.FILE_HEADER_EVENT_TEMPLATE])
        return df_event

    def predict(self, X_test, y_test, real_data=None, path_file=None):
        y_pred = self.model.predict(X_test)

        ## Get original log from predicted labels
        df = read_csv(real_data)
        df_contents = df[Config.FILE_HEADER_CONTENT].values

        parsed_logs = []
        for i in range(0, len(y_test)):
            sent = []
            for j in range(0, len(y_test[i])):
                if y_pred[i][j] == Config.DES_CHAR:
                    sent.append(df_contents[i].split()[j])
                else:
                    sent.append(Config.SPE_CHAR)
            sent = " ".join(sent)
            pred_sent_labels = " ".join(y_pred[i])
            true_sent_labels = " ".join(y_test[i])
            # fout.write("{}\n{}\n{}\n".format(sent, pred_sent_labels, true_sent_labels))
            parsed_logs.append(sent)
        df_event = self.outputResult(parsed_logs)
        df_event.to_csv(path_file, index=False)

