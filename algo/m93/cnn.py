# -*- coding: utf-8 -*-
from __future__ import print_function
import tensorflow as tf
from algo.model.const import *
from algo.nn.base import BaseNNModel
from dataset.common.const import *
from algo.nn.common import dense, cnn, attention
from algo.nn.common.common import add_gaussian_noise_layer, build_dropout_keep_prob


class NNModel(BaseNNModel):
    name = 'm93_cnn'

    def build_neural_network(self, lookup_table):
        test_mode = tf.placeholder(tf.int8, None, name=TEST_MODE)
        label_gold = tf.placeholder(tf.int32, [None, ], name=LABEL_GOLD)
        sample_weights = tf.placeholder(tf.float32, [None, ], name=SAMPLE_WEIGHTS)
        lookup_table = tf.Variable(
            lookup_table, dtype=tf.float32, name=LOOKUP_TABLE,
            trainable=self.config.embedding_trainable
        )
        dropout_keep_prob = build_dropout_keep_prob(keep_prob=self.config.dropout_keep_prob, test_mode=test_mode)

        tid_ = [list() for _ in range(3)]
        seq_len_ = [list() for _ in range(3)]
        embedded_ = [list() for _ in range(3)]
        for i in range(3):
            tid_[i] = tf.placeholder(tf.int32, [self.config.batch_size, self.config.seq_len], name=TID_[i])
            seq_len_[i] = tf.placeholder(tf.int32, [None, ], name=SEQ_LEN_[i])
            embedded_[i] = tf.nn.embedding_lookup(lookup_table, tid_[i])

        if self.config.embedding_noise_type is None:
            pass
        elif self.config.embedding_noise_type == 'gaussian':
            for i in range(3):
                embedded_[i] = add_gaussian_noise_layer(
                    embedded_[i], stddev=self.config.embedding_noise_stddev, test_mode=test_mode)
        elif self.config.embedding_noise_type == 'dropout':
            emb_dropout_keep_prob = build_dropout_keep_prob(
                keep_prob=self.config.embedding_dropout_keep_prob, test_mode=test_mode)
            for i in range(3):
                embedded_[i] = tf.nn.dropout(embedded_[i], emb_dropout_keep_prob)
        else:
            raise Exception('unknown embedding noise type: {}'.format(self.config.embedding_noise_type))

        last_states = list()
        for i in range(3):
            with tf.variable_scope('turn{}'.format(i)):
                cnn_output = cnn.build2(embedded_[i], self.config.filter_num, self.config.kernel_size)
                last_state = cnn.max_pooling(cnn_output)

                if self.config.use_attention:
                    last_state, _ = attention.build(cnn_output, self.config.attention_dim)

                last_states.append(last_state)

        last_states = list(map(lambda _state: tf.nn.dropout(_state, keep_prob=dropout_keep_prob), last_states))
        dense_input = tf.concat(last_states, axis=1, name=HIDDEN_FEAT)

        l2_component = None
        for conf in self.config.dense_layers:
            dense_input, w, _ = dense.build(
                dense_input, dim_output=conf['dim'], activation=getattr(tf.nn, conf['activation']))
            if conf.get('l2', 0.) > 0:
                comp = conf['l2'] * tf.nn.l2_loss(w)
                if l2_component is None:
                    l2_component = comp
                else:
                    l2_component += comp

        l2_w_list = list()
        if self.config.max_out is None:
            y, w, b = dense.build(dense_input, dim_output=self.config.output_dim, output_name=PROB_PREDICT)
            l2_w_list.append(w)
        else:
            y_list = list()
            if len(self.config.max_out) != self.config.output_dim:
                raise ValueError('invalid max_out config')

            for dim in self.config.max_out:
                y, w, b = dense.build(dense_input, dim_output=dim)
                y = tf.expand_dims(tf.reduce_max(y, 1), axis=1)
                y_list.append(y)
                l2_w_list.append(w)

            y = tf.concat(y_list, axis=1, name=PROB_PREDICT)

        # 计算loss
        _loss_1 = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
            logits=y, labels=label_gold, weights=sample_weights))

        _loss_2 = tf.constant(0., dtype=tf.float32)
        if self.config.l2_reg_lambda is not None and self.config.l2_reg_lambda > 0:
            for w in l2_w_list:
                _loss_2 += self.config.l2_reg_lambda * tf.nn.l2_loss(w)
        if l2_component is not None:
            _loss_2 += l2_component
        loss = tf.add(_loss_1, _loss_2, name=LOSS)

        # 预测标签
        tf.cast(tf.argmax(y, 1), tf.int32, name=LABEL_PREDICT)

        # 统一的后处理
        self.build_optimizer(loss=loss)
        self.set_graph(graph=tf.get_default_graph())