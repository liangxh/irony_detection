# -*- coding: utf-8 -*-
import tensorflow as tf

from algo.model.const import *
from algo.model.nn_config import BaseNNConfig
from algo.nn.base import BaseNNModel
from algo.nn.common import dense, rnn_cell


class NNConfig(BaseNNConfig):
    @property
    def rnn_dim(self):
        return self.data['rnn']['dim']


class NNModel(BaseNNModel):
    name = 'gru'

    def build_neural_network(self, lookup_table):
        label_gold = tf.placeholder(tf.int32, [None, ], name=LABEL_GOLD)
        token_id_seq = tf.placeholder(tf.int32, [None, self.config.seq_len], name=TOKEN_ID_SEQ)
        seq_len = tf.placeholder(tf.int32, [None, ], name=SEQ_LEN)
        sample_weights = tf.placeholder(tf.float32, [None, ], name=SAMPLE_WEIGHTS)
        dropout_keep_prob = tf.placeholder(tf.float32, name=DROPOUT_KEEP_PROB)
        lookup_table = tf.Variable(
            lookup_table, dtype=tf.float32, name=LOOKUP_TABLE,
            trainable=self.config.embedding_trainable
        )
        embedded = tf.nn.embedding_lookup(lookup_table, token_id_seq)

        rnn_outputs, rnn_last_states = tf.nn.dynamic_rnn(
            rnn_cell.build_gru(self.config.rnn_dim, dropout_keep_prob=dropout_keep_prob),
            inputs=embedded,
            sequence_length=seq_len,
            dtype=tf.float32
        )
        dense_input = tf.concat([rnn_last_states, ], axis=1, name=HIDDEN_FEAT)

        if not self.config.binary_classification:
            y, w, b = dense.build(dense_input, dim_output=self.config.output_dim, output_name=PROB_PREDICT)
            # 计算loss
            _loss_1 = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
                logits=y, labels=label_gold, weights=sample_weights))
        else:
            y, w, b = dense.build(dense_input, dim_output=1, activation=tf.nn.sigmoid)
            _loss_1 = -tf.reduce_mean(
                y * tf.log(tf.clip_by_value(label_gold, 1e-10, 1.0))
                + (1 - y) * tf.log(tf.clip_by_value(1 - label_gold, 1e-10, 1.0))
            )

        _loss_2 = tf.constant(0., dtype=tf.float32)
        if self.config.l2_reg_lambda is not None and self.config.l2_reg_lambda > 0:
            _loss_2 += self.config.l2_reg_lambda * tf.nn.l2_loss(w)
        loss = tf.add(_loss_1, _loss_2, name=LOSS)

        # 预测标签
        tf.cast(tf.argmax(y, 1), tf.int32, name=LABEL_PREDICT)

        # 统一的后处理
        self.build_optimizer(loss=loss)
        self.set_graph(graph=tf.get_default_graph())
