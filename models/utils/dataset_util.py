#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 08:25, 05/08/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

import re

from pandas import read_csv
from numpy import array, zeros, int32, float32
from config import Config


WILDCARD_REGEX = re.compile(re.escape(Config.SPE_CHAR))
WHITESPACE_REGEX = re.compile(r"\s+")


# =========================== MAKE TRAIN AND TEST FILE =======================================

def template_to_regex(tpl):
    # Generate regular expression corresponding to given template
    # Not using Pattern.sub because we would like to apply re.escape for descriptive part only
    buf = [r"^"]
    tmp_tpl = tpl[:]
    while len(tmp_tpl) > 0:
        matchobj = WILDCARD_REGEX.search(tmp_tpl)
        if matchobj:
            buf.append(re.escape(tmp_tpl[:matchobj.start()]))
            buf.append(r"(.*?)")  # shortest match
            tmp_tpl = tmp_tpl[matchobj.end():]
        else:
            buf.append(re.escape(tmp_tpl))
            break
    buf.append(r"$")
    return re.compile("".join(buf))


def reformat_template(tpl, message):
    # match variable parts of the message with given template
    pattern = template_to_regex(tpl)
    matchobj = pattern.match(message)
    assert matchobj is not None

    # get boolean index of variable part in the message
    variable_index = array([False] * len(message))
    n_variables = len(matchobj.groups())
    for i in range(n_variables):
        # i+1 because matchobject group index starts from 1
        variable_index[matchobj.start(i+1):matchobj.end(i+1)] = True

    # get chr index of segmented words of the message
    segmented_word_span = []  # tuple of (start, end) corresponding to chr index of words
    start_point = 0
    while start_point < len(message):
        matchobj = WHITESPACE_REGEX.search(message[start_point:])
        if matchobj:
            segmented_word_span.append((start_point, start_point + matchobj.start()))
            start_point = start_point + matchobj.end()
        else:
            segmented_word_span.append((start_point, len(message)))
            break

    # generate new template that can be consistently segmented with whitespaces
    new_tpl = []
    for wstart, wend in segmented_word_span:
        if True in variable_index[wstart:wend]:
            # a word including variable part -> replace with one wildcard
            new_tpl.append(Config.SPE_CHAR)
        else:
            # a word without variable part -> as is
            new_tpl.append(message[wstart:wend])
    return " ".join(new_tpl)


def function_mapper(line):
    words = line.split()
    event_str = []
    for idx, word in enumerate(words):
        if word == Config.SPE_CHAR:
            event_str.append(Config.VAR_CHAR)
        else:
            event_str.append(Config.DES_CHAR)
    return " ".join(event_str)


def make_train_and_test_datafile(input_path, output_paths, test_size=0.3):
    df = read_csv(input_path)
    for row_idx, row in df.iterrows():
        tpl = row.loc[Config.FILE_HEADER_EVENT_TEMPLATE]
        line = row.loc[Config.FILE_HEADER_CONTENT]
        new_tpl = reformat_template(tpl, line)
        labels = function_mapper(new_tpl)
        assert len(line.split()) == len(new_tpl.split())
        assert len(new_tpl.split()) == len(labels.split())
        df.loc[row_idx, Config.FILE_HEADER_EVENT_TEMPLATE] = new_tpl
        df.loc[row_idx, Config.FILE_HEADER_EVENT_STR] = labels
    df = df[[Config.FILE_HEADER_CONTENT, Config.FILE_HEADER_EVENT_ID, Config.FILE_HEADER_EVENT_TEMPLATE, Config.FILE_HEADER_EVENT_STR]]
    df.to_csv(output_paths[0], index=False)  # Content, EventId, EventTemplate, EventStr

    # Shuffle data to split train and test
    df = df.sample(frac=1)
    if 0 < test_size < 1:
        num_train = df.shape[0] - int(df.shape[0] * test_size)
    else:
        num_train = df.shape[0] - int(test_size)
    df_train = df[:num_train]
    df_train.to_csv(output_paths[1], index=False)

    df_test = df[num_train:]
    df_test.to_csv(output_paths[2], index=False)
    return True


# =========================== Make Dataset based on word and character ============================


def create_char_index(filenames):
    char_index, char_cnt, rare_chars = {}, 0, []  # original code: char_cnt: 3
    ## Handle char index and char count
    for idx, filename in enumerate(filenames):
        df = read_csv(filename)
        df_values = df[Config.FILE_HEADER_CONTENT].values
        for sentence in df_values:
            words = sentence.strip().split()
            for word in words:
                for char in word:
                    if char in rare_chars:
                        char_index[char] = 3
                        continue
                    if char not in char_index:
                        char_index[char] = char_cnt
                        char_cnt += 1
    return char_index, char_cnt


def create_word_index(filenames):
    word_index, word_cnt, single_words = {}, 0, []  # original : word_cnt = 1, start from index 1 (0 meaning ignore)
    ## Handle word index and word count
    for idx, filename in enumerate(filenames):
        df = read_csv(filename)
        df_values = df[Config.FILE_HEADER_CONTENT].values
        for sentence in df_values:
            words = sentence.strip().split()
            for word in words:
                # word = process(word)
                if word in single_words:
                    word_index[word] = 1
                    continue
                if word in word_index:
                    continue
                word_index[word] = word_cnt
                word_cnt += 1
    return word_index, word_cnt


def count_number_of_lines_in_files(filename=None):  # Count number of line in the filename
    df = read_csv(filename)
    return len(df.values)


def make_dataset_based_on_word(filename, word_index=None, MAX_SENTENCE_LEN=None, LABEL_INDEX=None):
    line_cnt = count_number_of_lines_in_files(filename)
    x, y = zeros((line_cnt, MAX_SENTENCE_LEN), dtype=int32), zeros((line_cnt, MAX_SENTENCE_LEN), dtype=int32)
    mask = zeros((line_cnt, MAX_SENTENCE_LEN), dtype=float32)

    df = read_csv(filename)
    df_features = df[Config.FILE_HEADER_CONTENT].values
    df_labels = df[Config.FILE_HEADER_EVENT_STR].values
    for idx_ft, sentence in enumerate(df_features):
        words = sentence.strip().split()
        labels = df_labels[idx_ft].strip().split()
        print(len(words))
        for idx_wd, word in enumerate(words):
            # word_pro = process(word)
            word_ind, label_ind = word_index[word], LABEL_INDEX.index(labels[idx_wd])
            x[idx_ft, idx_wd] = word_ind
            y[idx_ft, idx_wd] = label_ind
            mask[idx_ft, idx_wd] = 1.0
    # y = process_labels(y, mask)
    return x, y, mask  # (n_lines, max_len_padding) ==> (index of word in vocabulary, index of label, 1.0 or 0.0)


def make_dataset_based_on_char(filename, char_index, MAX_WORD_LEN, MAX_SENTENCE_LEN):
    line_cnt = count_number_of_lines_in_files(filename)
    x = zeros((line_cnt, MAX_SENTENCE_LEN, MAX_WORD_LEN), dtype=int32)
    mask = zeros((line_cnt, MAX_SENTENCE_LEN, MAX_WORD_LEN), dtype=float32)

    df = read_csv(filename)
    df_features = df[Config.FILE_HEADER_CONTENT].values
    for idx_ft, sentence in enumerate(df_features):
        words = sentence.strip().split()
        for idx_wd, word in enumerate(words):
            for idx_ch, char in enumerate(word):
                if idx_ch + 1 >= MAX_WORD_LEN:
                    break
                x[idx_ft, idx_wd, idx_ch + 1] = char_index[char]
                mask[idx_ft, idx_wd, idx_ch + 1] = 1.0

            x[idx_ft, idx_wd, 0] = 1
            mask[idx_ft, idx_wd, 0] = 1.0
            if len(word) + 1 < MAX_WORD_LEN:
                x[idx_ft, idx_wd, len(word) + 1] = 2
                mask[idx_ft, idx_wd, len(word) + 1] = 1.0
    return x, mask


## ========================= Word 2 Vec Embedding ========================================

from gensim.models import Word2Vec


class Vectorizer:
    def __init__(self, word_list, size_n=100, min_count=0, window_n=5, iter_n=100, sg_flag=0):
        """
        :param size_n: the number of dimensions in which we wish to represent our word. This is the size of the word vector.
        :param min_count: Ignores all words with total frequency lower than this.
        :param window_n: Maximum distance between the current and predicted word within a sentence.
        :param iter_n: Number of iterations (epochs) over the corpus.
        :param sg_flag: Training algorithm: 1 for skip-gram; otherwise CBOW.
        """
        self.model = Word2Vec(word_list, vector_size=size_n, min_count=min_count, window=window_n, epochs=iter_n, sg=sg_flag)

    def vectorized(self, word):
        return self.model.__dict__['wv'][word]


def create_word2vec_embedding(filenames, ind2word):
    list_of_words = []
    for idx, filename in enumerate(filenames):
        df = read_csv(filename)
        df_values = df[Config.FILE_HEADER_CONTENT].values
        for sentence in df_values:
            words = sentence.strip().split()
            list_of_words.append(words)

    vec = Vectorizer(list_of_words, size_n=100, min_count=0, window_n=5, iter_n=100, sg_flag=0)
    # printing similarity index
    # print(vec.model.similarity('mgd', '9978'))
    # print(vec.model.similarity('mgd', '000000000000'))
    # print(vec.vectorized("mgd"))
    list_of_vectors = []
    for k, v in ind2word.items():
        list_of_vectors.append(vec.vectorized(v))
    return array(list_of_vectors)
