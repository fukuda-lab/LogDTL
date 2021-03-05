#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 05:14, 14/05/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

import numpy as np
from collections import defaultdict
from scipy.special import comb
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import accuracy_score
from nltk.metrics.distance import edit_distance
from pandas import read_csv

def measurement(true_file, pred_file):
    df_groundtruth = read_csv(true_file)
    df_parsedlog = read_csv(pred_file, index_col=False)

    # Remove invalid groundtruth event Ids
    # null_logids = df_groundtruth[~df_groundtruth['EventId'].isnull()].index
    # df_groundtruth = df_groundtruth.loc[null_logids]
    # df_parsedlog = df_parsedlog.loc[null_logids]

    ## Calculate all error metrics
    labels_true = np.array(df_groundtruth.EventId.values, dtype='str')
    labels_pred = np.array(df_parsedlog.EventId.values, dtype='str')
    rand_score, precision, recall, f_score, parsing_acc, cluster_acc = get_all_errors(labels_true, labels_pred)

    ## Calculate word error
    line_true = np.array(df_groundtruth.EventTemplate.values, dtype='str')
    line_pred = np.array(df_parsedlog.EventTemplate.values, dtype='str')
    length = len(df_groundtruth.EventTemplate.values)
    words_true_matrix = []
    words_pred_matrix = []
    for idx in range(0, length):
        words_true_matrix.append(line_true[idx].split())
        words_pred_matrix.append(line_pred[idx].split())
    WA, TWA = get_words_error(words_true_matrix, words_pred_matrix)     # word-accuracy, template-word-accuracy

    ## Calculate line accuracy (exact string matching)
    ESM = accuracy_score(line_true, line_pred)      # exact-string-matching

    ## Calculate edit distance
    edit_distance_result = []
    for w_true, w_pred in zip(line_true, line_pred):
        edit_distance_result.append(edit_distance(w_true, w_pred))
    dist_mean = np.mean(edit_distance_result)
    dist_std = np.std(edit_distance_result)

    ### Calculate Template Accuracy
    TA = template_accuracy(line_true, line_pred, labels_true)

    return WA, TWA, rand_score, precision, recall, f_score, parsing_acc, cluster_acc, ESM, dist_mean, dist_std, TA


def template_accuracy(tpls_true, tpls_pred, labels_true):
    d_n_line = defaultdict(int)
    d_n_correct_line = defaultdict(int)
    for tpl_true, tpl_pred, cluster in zip(tpls_true, tpls_pred, labels_true):
        d_n_line[cluster] += 1
        if tpl_true == tpl_pred:
            d_n_correct_line[cluster] += 1

    l_line_accuracy = [1. * d_n_correct_line[label] / d_n_line[label]
                       for label in d_n_line]
    return np.average(l_line_accuracy)


def get_all_errors(labels_true, labels_pred):
    def _comb2(n):
        return comb(n, 2, exact=True)

    a_true = np.array(labels_true)
    a_pred = np.array(labels_pred)
    cm = contingency_matrix(a_true, a_pred, sparse=True)

    nz_true, nz_pred = cm.nonzero()
    nz_cnt = cm.data

    tp = sum(_comb2(n_ij) for n_ij in cm.data)

    # false negative: same cluster in labels_true, but different in labels_pred
    fn = 0
    for uniq_label, uniq_cnt in zip(*np.unique(nz_true, return_counts=True)):
        if uniq_cnt > 1:
            childs = nz_cnt[nz_true == uniq_label]
            # add combinations except true_positive part
            fn += _comb2(sum(childs)) - sum(_comb2(c) for c in childs)

    # false positive: different cluster in labels_true, but same in labels_pred
    fp = 0
    for uniq_label, uniq_cnt in zip(*np.unique(nz_pred, return_counts=True)):
        if uniq_cnt > 1:
            childs = nz_cnt[nz_pred == uniq_label]
            # add combinations except true_positive part
            fp += _comb2(sum(childs)) - sum(_comb2(c) for c in childs)

    total = _comb2(a_true.shape[0])
    tn = total - tp - fn - fp

    ## Rand score
    rand_score = (tp + tn) / (tp + fp + fn + tn)

    ## precision_recall_fscore
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = (2. * precision * recall) / (precision + recall)

    ## Parsing accuracy
    n_total = a_true.shape[0]
    n_correct1 = 0
    for uniq_label_true, uniq_cnt_true in zip(*np.unique(nz_true, return_counts=True)):
        if uniq_cnt_true == 1:
            index_uniq_true = (nz_true == uniq_label_true)
            index_uniq_pred = (nz_pred == nz_pred[index_uniq_true][0])
            if nz_true[index_uniq_pred].shape[0] == 1:
                n_correct1 += sum(nz_cnt[index_uniq_true])
    parsing_acc = 1. * n_correct1 / n_total

    ## Cluster accuracy
    n_correct2 = 0
    for uniq_label_true, uniq_cnt_true in zip(*np.unique(nz_true, return_counts=True)):
        if uniq_cnt_true == 1:
            index_uniq_true = (nz_true == uniq_label_true)
            index_uniq_pred = (nz_pred == nz_pred[index_uniq_true][0])
            if nz_true[index_uniq_pred].shape[0] == 1:
                n_correct2 += 1
    cluster_acc = 1. * n_correct2 / np.unique(a_true).shape[0]

    return rand_score, precision, recall, f_score, parsing_acc, cluster_acc


def get_words_error(answer_tpls, estimated_tpls):
    # answer_tpls = [["A", "B", "**", "D"],
    #                ["A", "B", "**", "D"],
    #                ["A", "B", "**", "D"],
    #                ["A", "B", "**", "D"],
    #                ["A", "B", "**", "E"],
    #                ["A", "B", "**", "E"],
    #                ["A", "B", "**", "E"],
    #                ["A", "B", "**", "E", "F"], ]
    #
    # estimated_tpls = [["A", "B", "**", "D"],
    #                   ["A", "B", "C", "D"],
    #                   ["A", "B", "**", "D"],
    #                   ["A", "B", "**", "D"],
    #                   ["A", "B", "**", "E"],
    #                   ["A", "B", "**", "E"],
    #                   ["A", "B", "**", "E"],
    #                   ["A", "B", "**", "E", "F"], ]
    d_answer_clusters = {}
    for atpl, etpl in zip(answer_tpls, estimated_tpls):
        tpl_key = tuple(atpl)
        if tpl_key not in d_answer_clusters:
            d_answer_clusters[tpl_key] = []
        d_answer_clusters[tpl_key].append((atpl, etpl))

    word_acc_vec = []
    n_words_total = 0
    n_words_correct = 0
    for key, cluster_members in d_answer_clusters.items():
        n_word = 0
        n_correct = 0
        for atpl, etpl in cluster_members:
            for aword, eword in zip(atpl, etpl):
                n_word += 1
                if aword == eword:
                    n_correct += 1
        n_words_total += n_word
        n_words_correct += n_correct
        word_acc_vec.append(1.0 * n_correct / n_word)

    word_acc = (1.0*n_words_correct) / n_words_total
    tpl_word_acc = np.average(word_acc_vec)

    return word_acc, tpl_word_acc

# answer_tpls = [["A", "B", "**", "D"],
#                ["A", "B", "**", "D"],
#                ["A", "B", "**", "D"],
#                ["A", "B", "**", "D"],
#                ["A", "B", "**", "E"],
#                ["A", "B", "**", "E"],
#                ["A", "B", "**", "E"],
#                ["A", "B", "**", "E", "F"], ]
#
# estimated_tpls = [["A", "B", "**", "D"],
#                   ["A", "B", "C", "D"],
#                   ["A", "B", "**", "D"],
#                   ["A", "B", "**", "D"],
#                   ["A", "B", "**", "E"],
#                   ["A", "B", "**", "E"],
#                   ["A", "B", "**", "E"],
#                   ["A", "B", "**", "E", "F"], ]
#
# print(get_words_error(answer_tpls, estimated_tpls))


def get_all_errors_without_files(labels_true, labels_pred):  ## Calculate all error metrics
    def _comb2(n):
        return comb(n, 2, exact=True)

    a_true = np.array(labels_true)
    a_pred = np.array(labels_pred)
    cm = contingency_matrix(a_true, a_pred, sparse=True)

    nz_true, nz_pred = cm.nonzero()
    nz_cnt = cm.data

    tp = sum(_comb2(n_ij) for n_ij in cm.data)

    # false negative: same cluster in labels_true, but different in labels_pred
    fn = 0
    for uniq_label, uniq_cnt in zip(*np.unique(nz_true, return_counts=True)):
        if uniq_cnt > 1:
            childs = nz_cnt[nz_true == uniq_label]
            # add combinations except true_positive part
            fn += _comb2(sum(childs)) - sum(_comb2(c) for c in childs)

    # false positive: different cluster in labels_true, but same in labels_pred
    fp = 0
    for uniq_label, uniq_cnt in zip(*np.unique(nz_pred, return_counts=True)):
        if uniq_cnt > 1:
            childs = nz_cnt[nz_pred == uniq_label]
            # add combinations except true_positive part
            fp += _comb2(sum(childs)) - sum(_comb2(c) for c in childs)

    total = _comb2(a_true.shape[0])
    tn = total - tp - fn - fp

    ## Rand score
    rand_score = (tp + tn) / (tp + fp + fn + tn)

    ## precision_recall_fscore
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = (2. * precision * recall) / (precision + recall)

    return rand_score, precision, recall, f_score
