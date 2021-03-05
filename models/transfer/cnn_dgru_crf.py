#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 03:02, 09/08/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

import lasagne
import theano.tensor as T
from theano import function as make_function
from models.transfer.model_util import RootModel, CRFLayer, CRFDecodeLayer, ElementwiseMergeLayer


class Model(RootModel):

    def __init__(self, char_cnt, label_cnt, word_cnt, char_emb_size=25, word_emb_size=100, epoch=10, learning_rate=1e-3,
                 batch_size=10, test_batch_size=10, char_hidden_size=80, word_hidden_size=300, max_epoch=10, word_embedding_values=False,
                 char_double_layer=True, char_filter_size=(3, 4, 5), char_filter_num=20,
                 word_double_layer=True, dropout=True, dropout_rate=0.5, tanh=True, tanh_size=150,
                 joint=False, top_joint=False, very_top_joint=True):
        super().__init__(char_cnt, label_cnt, word_cnt, char_emb_size, word_emb_size, epoch, learning_rate, batch_size, test_batch_size,
                         char_hidden_size, word_hidden_size, max_epoch)
        self.use_crf = True
        self.word_embedding_values = word_embedding_values
        self.char_double_layer = char_double_layer
        self.char_filter_sizes = char_filter_size  # [3, 4, 5]
        self.char_filter_num = char_filter_num

        self.word_double_layer = word_double_layer
        self.dropout = dropout
        self.dropout_rate = dropout_rate

        self.tanh = tanh
        self.tanh_size = tanh_size

        self.joint = joint
        self.top_joint = top_joint
        self.very_top_joint = very_top_joint

    def build(self, x, y, m, wx, cm, embedding=None):
        self.x, self.y, self.m, self.wx, self.cm = x, y, m, wx, cm
        x_sym = T.itensor3('x')
        y_sym = T.imatrix('y')
        m_sym = T.matrix('mask')
        wx_sym = T.imatrix('wx')
        cm_sym = T.tensor3('cmask')

        ### There are 4 components that is matter in this kind of models
        ###     Char Embedding --> CNN
        ###     Word Embedding
        ###     Char Neural Network --> GRU
        ###     Word Neural Network --> GRU / CNN

        ###  Create Char Embedding
        model = lasagne.layers.InputLayer(shape=(None, self.x.shape[1], self.x.shape[2]), input_var=x_sym)
        model = lasagne.layers.EmbeddingLayer(model, self.char_cnt, self.char_embedding_size)
        model_char = lasagne.layers.ReshapeLayer(model, (-1, [2], [3]))

        ## Create Char Mask Embedding
        model_char_mask = lasagne.layers.InputLayer(shape=(None, self.x.shape[1], self.x.shape[2]), input_var=cm_sym)
        model_char_mask = lasagne.layers.ReshapeLayer(model_char_mask, (-1, [2]))

        ## Create Char Neural Network: GRU Network
        model_gru = lasagne.layers.GRULayer(model_char, self.char_hidden_size, mask_input=model_char_mask)
        model_gru_2 = lasagne.layers.GRULayer(model_char, self.char_hidden_size, mask_input=model_char_mask, backwards=True)

        if self.char_double_layer:
            model_char_rnn = lasagne.layers.ConcatLayer([model_gru, model_gru_2], axis=2)
            model_gru = lasagne.layers.GRULayer(model_char_rnn, self.char_hidden_size, mask_input=model_char_mask)
            model_gru_2 = lasagne.layers.GRULayer(model_char_rnn, self.char_hidden_size, mask_input=model_char_mask, backwards=True)

        model_gru = lasagne.layers.ReshapeLayer(model_gru, (-1, self.x.shape[1], [1], [2]))
        model_gru = lasagne.layers.SliceLayer(model_gru, -1, axis=2)

        model_gru_2 = lasagne.layers.ReshapeLayer(model_gru_2, (-1, self.x.shape[1], [1], [2]))
        model_gru_2 = lasagne.layers.SliceLayer(model_gru_2, 0, axis=2)

        model = lasagne.layers.ReshapeLayer(model, (-1, [3], [2]))

        ## Create CNN extracted Char Features with different filter size
        ls = []
        model_char_mask = lasagne.layers.DimshuffleLayer(model_char_mask, (0, 'x', 1))
        model_filter = ElementwiseMergeLayer([model, model_char_mask], T.mul)
        for f_size in self.char_filter_sizes:
            ls.append(lasagne.layers.Conv1DLayer(model_filter, self.char_filter_num, f_size))
        for i in range(len(ls)):  ## Max Pooling
            ls[i] = lasagne.layers.GlobalPoolLayer(ls[i], T.max)

        # List of char layers in CNN extracted Char Features.
        list_char_layers = lasagne.layers.ConcatLayer(ls)
        list_char_layers = lasagne.layers.ReshapeLayer(list_char_layers, (-1, self.x.shape[1], [1]))

        ### Create Word Embedding
        list_word_layers = lasagne.layers.InputLayer(shape=(None, self.x.shape[1]), input_var=wx_sym)
        list_word_layers = lasagne.layers.EmbeddingLayer(list_word_layers, self.word_cnt, self.word_embedding_size, W=lasagne.init.Normal(std=1e-3))
        if self.word_embedding_values:
            list_word_layers.W.set_value(embedding)

        layer_list = [list_word_layers, list_char_layers]           # Add word embedding layer    # Add char embedding
        if self.tanh:                                               # Add char network
            model_grus = lasagne.layers.ConcatLayer([model_gru, model_gru_2], axis=2)
            model_grus = lasagne.layers.ReshapeLayer(model_grus, (-1, [2]))

            model_gru = lasagne.layers.DenseLayer(model_grus, self.tanh_size, nonlinearity=lasagne.nonlinearities.tanh)
            model_gru = lasagne.layers.ReshapeLayer(model_gru, (-1, self.x.shape[1], [1]))
            self.shared_layer = model_gru
            layer_list.append(model_gru)
        elif self.joint:
            model_grus = lasagne.layers.ConcatLayer([model_gru, model_gru_2], axis=2)
            self.shared_layer = model_grus
            layer_list.append(model_grus)
            char_output = lasagne.layers.get_output(self.shared_layer)
            self.char_fn = make_function([x_sym, cm_sym], char_output, on_unused_input='ignore')
        else:
            ## Word_Embedding + Char NN (Model_GRU, Model_GRU2) ==> Word NN
            layer_list += [model_gru, model_gru_2]

        if len(layer_list) > 1:
            model = lasagne.layers.ConcatLayer(layer_list, axis=2)
        else:
            model = layer_list[0]

        if not self.joint:
            self.shared_layer = model

        ### Word NN is GRU networks
        model_mask = lasagne.layers.InputLayer(shape=(None, self.x.shape[1]), input_var=m_sym)
        model_word_1 = lasagne.layers.GRULayer(model, self.word_hidden_size, mask_input=model_mask)
        model = lasagne.layers.GRULayer(model, self.word_hidden_size, mask_input=model_mask, backwards=True)
        if self.dropout:
            model_word_1 = lasagne.layers.DropoutLayer(model_word_1, self.dropout_rate)
            model = lasagne.layers.DropoutLayer(model, self.dropout_rate)
        if self.word_double_layer:
            model = lasagne.layers.ConcatLayer([model_word_1, model], axis=2)
            model_word_1 = lasagne.layers.GRULayer(model, self.word_hidden_size, mask_input=model_mask)
            model = lasagne.layers.GRULayer(model, self.word_hidden_size, mask_input=model_mask, backwards=True)
            if self.dropout:
                model_word_1 = lasagne.layers.DropoutLayer(model_word_1, self.dropout_rate)
                model = lasagne.layers.DropoutLayer(model, self.dropout_rate)
        layer_list = [model_word_1, model]
        model = lasagne.layers.ConcatLayer(layer_list, axis=2)

        if self.top_joint:  # Char Neural Network use for CRF
            self.shared_layer = model

        ### Default CRF here
        model_train = CRFLayer(model, self.label_cnt, mask_input=m_sym, label_input=y_sym)
        model_test = CRFDecodeLayer(model, self.label_cnt, mask_input=m_sym, W=model_train.W, W_sim=model_train.W_sim, W_init=model_train.W_init)
        self.model = model_train
        if self.very_top_joint:
            self.shared_layer = self.model

        loss = lasagne.layers.get_output(model_train)
        params = lasagne.layers.get_all_params(self.model, trainable=True)
        updates = lasagne.updates.adagrad(loss, params, learning_rate=self.learning_rate)
        self.train_fn = make_function(inputs=[x_sym, y_sym, m_sym, wx_sym, cm_sym], outputs=loss, updates=updates, on_unused_input='ignore')
        py = lasagne.layers.get_output(model_test, deterministic=True)
        self.test_fn = make_function(inputs=[x_sym, m_sym, wx_sym, cm_sym], outputs=py, on_unused_input='ignore')