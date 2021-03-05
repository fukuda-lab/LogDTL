#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 09:31, 10/08/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

import lasagne
import theano.tensor as T
from theano import function as make_function
from models.transfer.model_util import RootModel


class Model(RootModel):

    def __init__(self, char_cnt, label_cnt, word_cnt, char_emb_size=25, word_emb_size=100, epoch=10, learning_rate=1e-3,
                 batch_size=10, test_batch_size=10, char_hidden_size=80, word_hidden_size=300, max_epoch=10, word_embedding_values=False,
                 word_double_layer=True, dropout=True, dropout_rate=0.5, tanh=True, tanh_size=150,
                 joint=False, top_joint=True):
        super().__init__(char_cnt, label_cnt, word_cnt, char_emb_size, word_emb_size, epoch, learning_rate, batch_size, test_batch_size,
                         char_hidden_size, word_hidden_size, max_epoch)
        self.use_crf = False
        self.word_embedding_values = word_embedding_values
        self.word_double_layer = word_double_layer
        self.dropout = dropout
        self.dropout_rate = dropout_rate

        self.tanh = tanh
        self.tanh_size = tanh_size

        self.joint = joint
        self.top_joint = top_joint

    def build(self, x, y, m, wx, cm, embedding=None):
        self.x, self.y, self.m, self.wx, self.cm = x, y, m, wx, cm
        x_sym = T.itensor3('x')
        y_sym = T.imatrix('y')
        m_sym = T.matrix('mask')
        wx_sym = T.imatrix('wx')
        cm_sym = T.tensor3('cmask')

        ### There are 2 components that is matter in this kind of models
        ###     Char Neural Network --> GRU
        ###     Word Neural Network --> GRU / CNN

        ### Create Word Embedding
        list_word_layers = lasagne.layers.InputLayer(shape=(None, self.x.shape[1]), input_var=wx_sym)
        list_word_layers = lasagne.layers.EmbeddingLayer(list_word_layers, self.word_cnt, self.word_embedding_size, W=lasagne.init.Normal(std=1e-3))
        if self.word_embedding_values:
            list_word_layers.W.set_value(embedding)

        model = list_word_layers  # Using Word Embedding and Word Neural Network only
        if not self.joint:  # Transfer Learning
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

        if self.top_joint:  # Char Neural Network use for transfer learning
            self.shared_layer = model

        ### No CRF
        model = lasagne.layers.ReshapeLayer(model, (-1, [2]))
        model = lasagne.layers.DenseLayer(model, self.label_cnt, nonlinearity=lasagne.nonlinearities.softmax)
        self.model = model
        py = lasagne.layers.get_output(model, deterministic=True)
        loss = T.dot(lasagne.objectives.categorical_crossentropy(py, y_sym.flatten()), m_sym.flatten())
        params = lasagne.layers.get_all_params(model, trainable=True)
        updates = lasagne.updates.adagrad(loss, params, learning_rate=self.learning_rate)
        self.train_fn = make_function(inputs=[x_sym, y_sym, m_sym, wx_sym, cm_sym], outputs=loss, updates=updates, on_unused_input='ignore')
        self.test_fn = make_function(inputs=[x_sym, m_sym, wx_sym, cm_sym], outputs=py, on_unused_input='ignore')
