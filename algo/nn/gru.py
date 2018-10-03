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
    def __init__(self, config):
        super(NNModel, self).__init__(config=config)
        self.variable_keys += [PROB_PREDICT, ]

    def build_neural_network(self, lookup_table):
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
        dense_input = tf.concat([rnn_last_states, ], axis=1)
        y, w, b = dense.build(dense_input, self.config.output_dim, output_name=PROB_PREDICT)

        # 计算loss
        label_gold = tf.placeholder(tf.int32, [None, ], name=LABEL_GOLD)
        prob_gold = tf.one_hot(label_gold, self.config.output_dim)
        #_loss_1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=prob_gold))
        _loss_1 = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
            logits=y, labels=label_gold, weights=sample_weights))
        _loss_2 = tf.constant(0., dtype=tf.float32)
        if self.config.l2_reg_lambda is not None and self.config.l2_reg_lambda > 0:
            _loss_2 += self.config.l2_reg_lambda * tf.nn.l2_loss(w)
        loss = tf.add(_loss_1, _loss_2, name=LOSS)

        global_step = tf.Variable(0, trainable=False, name=GLOBAL_STEP)
        learning_rate = tf.train.exponential_decay(
            self.config.learning_rate_init,
            global_step=global_step,
            decay_steps=self.config.learning_rate_decay_steps,
            decay_rate=self.config.learning_rate_decay_rate
        )
        # 预测标签
        tf.cast(tf.argmax(y, 1), tf.int32, name=LABEL_PREDICT)
        # Optimizer
        tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step, name=OPTIMIZER)

        self.preserve_variables(tf.get_default_graph())
