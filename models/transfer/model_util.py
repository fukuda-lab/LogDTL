#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 03:44, 09/08/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

import lasagne
from numpy import random as make_random
from numpy import mean, vstack, reshape, round, int32, array
import theano.tensor as T
from theano import scan
from hashlib import md5
from pandas import DataFrame
from models.utils.measurement_util import get_all_errors_without_files

CRF_INIT = True
COST = True
COST_CONST = 5.0


def theano_logsumexp(x, axis=None):
    xmax = x.max(axis=axis, keepdims=True)
    xmax_ = x.max(axis=axis)
    return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))


class CRFLayer(lasagne.layers.Layer):

    def __init__(self, incoming, num_classes, W_sim=lasagne.init.GlorotUniform(), W=lasagne.init.GlorotUniform(),
                 W_init=lasagne.init.GlorotUniform(), mask_input=None, label_input=None, **kwargs):

        super(CRFLayer, self).__init__(incoming, **kwargs)
        num_inputs = self.input_shape[-1]
        self.num_classes = num_classes
        self.W_sim = self.add_param(W_sim, (num_classes, num_classes))
        self.W = self.add_param(W, (num_inputs, num_classes))
        self.mask_input = mask_input
        self.label_input = label_input
        if CRF_INIT:
            self.W_init = self.add_param(W_init, (1, num_classes))
        else:
            self.W_init = None

    def get_output_shape_for(self, input_shape):
        return (1,)

    def get_output_for(self, input, **kwargs):
        def norm_fn(f, mask, label, previous, W_sim):
            # f: inst * class, mask: inst, previous: inst * class, W_sim: class * class
            next = previous.dimshuffle(0, 1, 'x') + f.dimshuffle(0, 'x', 1) + W_sim.dimshuffle('x', 0, 1)
            if COST:
                next = next + COST_CONST * (1.0 - T.extra_ops.to_one_hot(label, self.num_classes).dimshuffle(0, 'x', 1))
            # next: inst * prev * cur
            next = theano_logsumexp(next, axis=1)
            # next: inst * class
            mask = mask.dimshuffle(0, 'x')
            next = previous * (1.0 - mask) + next * mask
            return next

        f = T.dot(input, self.W)
        # f: inst * time * class

        initial = f[:, 0, :]
        if CRF_INIT:
            initial = initial + self.W_init[0].dimshuffle('x', 0)
        if COST:
            initial = initial + COST_CONST * (1.0 - T.extra_ops.to_one_hot(self.label_input[:, 0], self.num_classes))
        outputs, _ = scan(fn=norm_fn, sequences=[f.dimshuffle(1, 0, 2)[1:], self.mask_input.dimshuffle(1, 0)[1:],
                            self.label_input.dimshuffle(1, 0)[1:]], outputs_info=initial, non_sequences=[self.W_sim], strict=True)
        norm = T.sum(theano_logsumexp(outputs[-1], axis=1))

        f_pot = (f.reshape((-1, f.shape[-1]))[T.arange(f.shape[0] * f.shape[1]), self.label_input.flatten()] * self.mask_input.flatten()).sum()
        if CRF_INIT:
            f_pot += self.W_init[0][self.label_input[:, 0]].sum()

        labels = self.label_input
        # labels: inst * time
        shift_labels = T.roll(labels, -1, axis=1)
        mask = self.mask_input
        # mask : inst * time
        shift_mask = T.roll(mask, -1, axis=1)

        g_pot = (self.W_sim[labels.flatten(), shift_labels.flatten()] * mask.flatten() * shift_mask.flatten()).sum()

        return - (f_pot + g_pot - norm) / f.shape[0]


class CRFDecodeLayer(lasagne.layers.Layer):

    def __init__(self, incoming, num_classes, W_sim=lasagne.init.GlorotUniform(), W=lasagne.init.GlorotUniform(),
                 W_init=lasagne.init.GlorotUniform(), mask_input=None, **kwargs):

        super(CRFDecodeLayer, self).__init__(incoming, **kwargs)
        num_inputs = self.input_shape[-1]
        self.W_sim = self.add_param(W_sim, (num_classes, num_classes))
        self.W = self.add_param(W, (num_inputs, num_classes))
        self.mask_input = mask_input
        if CRF_INIT:
            self.W_init = self.add_param(W_init, (1, num_classes))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1])

    def get_output_for(self, input, **kwargs):
        def max_fn(f, mask, prev_score, prev_back, W_sim):
            next_score = prev_score.dimshuffle(0, 1, 'x') + f.dimshuffle(0, 'x', 1) + W_sim.dimshuffle('x', 0, 1)
            next_back = T.argmax(next_score, axis=1)
            next_score = T.max(next_score, axis=1)
            mask = mask.dimshuffle(0, 'x')
            next_score = next_score * mask + prev_score * (1.0 - mask)
            next_back = next_back * mask + prev_back * (1.0 - mask)
            next_back = T.cast(next_back, 'int32')
            return [next_score, next_back]

        def produce_fn(back, mask, prev_py):
            # back: inst * class, prev_py: inst, mask: inst
            next_py = back[T.arange(prev_py.shape[0]), prev_py]
            next_py = mask * next_py + (1.0 - mask) * prev_py
            next_py = T.cast(next_py, 'int32')
            return next_py

        f = T.dot(input, self.W)

        init_score, init_back = f[:, 0, :], T.zeros_like(f[:, 0, :], dtype='int32')
        if CRF_INIT:
            init_score = init_score + self.W_init[0].dimshuffle('x', 0)
        ([scores, backs], _) = scan(fn=max_fn, sequences=[f.dimshuffle(1, 0, 2)[1:], self.mask_input.dimshuffle(1, 0)[1:]],
                                           outputs_info=[init_score, init_back], non_sequences=[self.W_sim], strict=True)

        init_py = T.argmax(scores[-1], axis=1)
        init_py = T.cast(init_py, 'int32')
        # init_py: inst, backs: time * inst * class
        pys, _ = scan(fn=produce_fn, sequences=[backs, self.mask_input.dimshuffle(1, 0)[1:]], outputs_info=[init_py], go_backwards=True)
        # pys: (rev_time - 1) * inst
        pys = pys.dimshuffle(1, 0)[:, :: -1]
        # pys : inst * (time - 1)
        return T.concatenate([pys, init_py.dimshuffle(0, 'x')], axis=1)


class ElementwiseMergeLayer(lasagne.layers.MergeLayer):
    def __init__(self, incomings, merge_function, **kwargs):
        super(ElementwiseMergeLayer, self).__init__(incomings, **kwargs)
        self.merge_function = merge_function

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        output = None
        for input in inputs:
            if output is not None:
                output = self.merge_function(output, input)
            else:
                output = input
        return output


class RootModel:
    def __init__(self, char_cnt, label_cnt, word_cnt, char_emb_size=25, word_emb_size=100, epoch=10, learning_rate=1e-3,
                 batch_size=10, test_batch_size=10, char_hidden_size=80, word_hidden_size=300, max_epoch=10):
        lasagne.random.set_rng(make_random)
        make_random.seed(13)  # 13

        self.char_cnt = char_cnt
        self.label_cnt = label_cnt
        self.word_cnt = word_cnt
        self.char_embedding_size = char_emb_size
        self.word_embedding_size = word_emb_size
        self.char_hidden_size = char_hidden_size  # best of [50, 80, 100] is 80
        self.word_hidden_size = word_hidden_size  # best of [100, 200, 300, 400] is 300
        self.epoch = epoch  # best of [5, 10, 15] with reasonable time is 10
        self.learning_rate = learning_rate  # best of [1e-4, 1e-3, 1e-2, 1e-1] is 1e-3
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.max_epoch = max_epoch

        self.joint = False  # False
        self.top_joint = False  # False
        self.very_top_joint = False

        ## None Values
        self.x, self.y, self.m, self.wx, self.cm = None, None, None, None, None
        self.use_crf, self.embedding_values = None, None
        self.train_fn, self.test_fn = None, None

    def outputResult(self, pred):
        df_events = []
        templateids = []
        for pr in pred:
            template_id = md5(pr.encode('utf-8')).hexdigest()
            templateids.append(template_id)
            df_events.append([template_id, pr])

        df_event = DataFrame(df_events, columns=['EventId', 'EventTemplate'])
        return df_event

    def predict(self, tx, ty, tm, twx, tcm, ind2word=None, labels_index=None, saved_path_file=None):
        ## Start predicting
        i = 0
        pys = []
        while i < tx.shape[0]:
            j = i + self.test_batch_size
            s_x, s_m, s_wx, s_cm = tx[i: j], tm[i: j], twx[i: j], tcm[i: j]
            pys.append(self.test_fn(s_x, s_m, s_wx, s_cm))
            i = j
        py = vstack(tuple(pys))
        if not self.use_crf:
            py = py.argmax(axis=1)
            py = reshape(py, ty.shape)
        # acc, f1, prec, recall, rand_score = self.evaluate(py, ty, tm)
        # print("Accuracy: %.5f, F Score: %.5f, Precision: %.5f, Recall: %.5f, Rand Score: %.5f" % (acc, f1, prec, recall, rand_score))

        ## Saving Prediction Results
        # fout = open(path_file + 'pred_results.txt', 'w+')
        if ind2word is not None:
            parsed_logs = []
            for i in range(ty.shape[0]):
                sent = []
                pred_sent_labels = []
                true_sent_labels = []
                for j in range(ty.shape[1]):
                    if tm[i][j] == 0:
                        continue
                    if labels_index[py[i, j]] == "DES":
                        sent.append(ind2word[twx[i][j]])
                    else:
                        sent.append("**")
                    pred_sent_labels.append(labels_index[py[i, j]])
                    true_sent_labels.append(labels_index[ty[i, j]])
                sent = " ".join(sent)
                pred_sent_labels = " ".join(pred_sent_labels)
                true_sent_labels = " ".join(true_sent_labels)
                # fout.write("{}\n{}\n{}\n".format(sent, pred_sent_labels, true_sent_labels))
                parsed_logs.append(sent)
            df_event = self.outputResult(parsed_logs)
            df_event.to_csv(saved_path_file, index=False)

        # fout.write("Accuracy: %.5f, F Score: %.5f, Precision: %.5f, Recall: %.5f, Rand Score: %.5f.\n" % (acc, f1, prec, recall, rand_score))
        # fout.close()
        # return word_acc, template_word_acc, rand_score, precision, recall, f_score, parsing_acc, cluster_acc, exact_string_matching, dist_mean, dist_std

    def train(self):
        ## Start training
        for epoch in range(self.epoch):
            ind = make_random.permutation(self.x.shape[0])
            i = 0
            iter = 0
            list_loss = []
            while i < self.x.shape[0]:
                iter += 1
                j = min(self.x.shape[0], i + self.batch_size)
                s_x, s_y, s_m, s_wx = self.x[ind[i: j]], self.y[ind[i: j]], self.m[ind[i: j]], self.wx[ind[i: j]]
                s_cm = self.cm[ind[i: j]]
                loss_iter = self.train_fn(s_x, s_y, s_m, s_wx, s_cm)
                i = j
                list_loss.append(loss_iter)
            print("Epoch: {}, Loss: {}".format(epoch + 1, mean(list_loss)))

    def evaluate(self, py, y_, m_):
        y_true, mask, y_pred = y_.flatten(), m_.flatten(), py.flatten()
        acc = 1.0 * (array(y_true == y_pred, dtype=int32) * mask).sum() / mask.sum()
        labels_true = y_.flatten()[mask == 1]
        labels_pred = py.flatten()[mask == 1]
        rand_score, precision, recall, f_score = get_all_errors_without_files(labels_true, labels_pred)
        return round(acc, 5), round(f_score, 5), round(precision, 5), round(recall, 5), round(rand_score, 5)

    def step_train_init(self):
        self.epoch = 0
        self.ind = make_random.permutation(self.x.shape[0])
        self.idx_batch = 0
        self.iter_batch = 0
        self.loss_batch = []

    def step_evaluate(self, pred_y, true_y, true_mask):
        acc, f1, prec, recall, rand_score = self.evaluate(pred_y, true_y, true_mask)
        return acc, f1, prec, recall, rand_score

    def step_predict(self, tx, ty, tm, twx, tcm):  ## Start predicting after finish training 1 epoch.
        i = 0
        pys = []
        while i < tx.shape[0]:
            j = i + self.test_batch_size
            s_x, s_m, s_wx, s_cm = tx[i: j], tm[i: j], twx[i: j], tcm[i: j]
            pys.append(self.test_fn(s_x, s_m, s_wx, s_cm))
            i = j
        py = vstack(tuple(pys))
        if not self.use_crf:
            py = py.argmax(axis=1)
            py = reshape(py, ty.shape)
        return py

    def step_train(self, tx, ty, tm, twx, tcm):
        if self.epoch < self.max_epoch:
            self.iter_batch += 1
            idx_batch = self.idx_batch
            j = min(self.x.shape[0], idx_batch + self.batch_size)
            ind = self.ind
            s_x, s_y, s_m, s_wx = self.x[ind[idx_batch: j]], self.y[ind[idx_batch: j]], self.m[ind[idx_batch: j]], self.wx[ind[idx_batch: j]]
            s_cm = self.cm[ind[idx_batch: j]]
            if self.x.shape[0] > 0:
                loss_iter_batch = self.train_fn(s_x, s_y, s_m, s_wx, s_cm)
                self.loss_batch.append(loss_iter_batch)
            if j == self.x.shape[0]:
                self.idx_batch = 0
                self.epoch += 1
                self.ind = make_random.permutation(self.x.shape[0])
                loss_epoch = round(mean(self.loss_batch), 5)
                self.loss_batch = []
                return self.step_predict(tx, ty, tm, twx, tcm), loss_epoch
            else:
                self.idx_batch = j
                return None, None
        else:
            return None, None
