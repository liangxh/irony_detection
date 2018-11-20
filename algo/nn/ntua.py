# -*- coding: utf-8 -*-
import tensorflow as tf
from algo.model.const import *
from algo.model.nn_config import BaseNNConfig
from algo.nn.base import BaseNNModel
from algo.nn.common import dense, rnn_cell, attention
from algo.nn.common.common import add_gaussian_noise_layer


class NNConfig(BaseNNConfig):
    @property
    def rnn_dim(self):
        return self.data['rnn']['dim']

    @property
    def attention_dim(self):
        return self.data['attention']['dim']

    @property
    def embedding_noise_stddev(self):
        return self.data['embedding']['noise_stddev']


class NNModel(BaseNNModel):
    name = 'ntua'

    def __init__(self, *args, **kwargs):
        super(NNModel, self).__init__(*args, **kwargs)
        self.optimizer = None

    def build_neural_network(self, lookup_table):
        test_mode = tf.placeholder(tf.int8, None, name=TEST_MODE)

        label_gold = tf.placeholder(tf.int32, [None, ], name=LABEL_GOLD)
        token_id_seq = tf.placeholder(tf.int32, [self.config.batch_size, self.config.seq_len], name=TOKEN_ID_SEQ)
        seq_len = tf.placeholder(tf.int32, [None, ], name=SEQ_LEN)
        sample_weights = tf.placeholder(tf.float32, [None, ], name=SAMPLE_WEIGHTS)
        dropout_keep_prob = tf.placeholder(tf.float32, name=DROPOUT_KEEP_PROB)
        lookup_table = tf.Variable(
            lookup_table, dtype=tf.float32, name=LOOKUP_TABLE,
            trainable=self.config.embedding_trainable
        )
        embedded = tf.nn.embedding_lookup(lookup_table, token_id_seq)

        if self.config.embedding_noise_stddev != 0.:
            embedded = add_gaussian_noise_layer(embedded, stddev=self.config.embedding_noise_stddev, test_mode=test_mode)

        with tf.variable_scope("blstm_layer_1") as scope:
            cell_fw = rnn_cell.build_lstm(self.config.rnn_dim, dropout_keep_prob=dropout_keep_prob)
            cell_bw = rnn_cell.build_lstm(self.config.rnn_dim, dropout_keep_prob=dropout_keep_prob)

            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, embedded, seq_len,
                cell_fw.zero_state(self.config.batch_size, tf.float32),
                cell_bw.zero_state(self.config.batch_size, tf.float32)
            )
            outputs = tf.concat(outputs, axis=-1)

        with tf.variable_scope("blstm_layer_2") as scope:
            cell_fw = rnn_cell.build_lstm(self.config.rnn_dim, dropout_keep_prob=dropout_keep_prob)
            cell_bw = rnn_cell.build_lstm(self.config.rnn_dim, dropout_keep_prob=dropout_keep_prob)

            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, outputs, seq_len,
                cell_fw.zero_state(self.config.batch_size, tf.float32),
                cell_bw.zero_state(self.config.batch_size, tf.float32)
            )
            outputs = tf.concat(outputs, axis=-1)
        attention_output, _ = attention.build(outputs, self.config.attention_dim)

        dense_input = tf.concat([attention_output, ], axis=1, name=HIDDEN_FEAT)

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

    def build_optimizer(self, loss):
        # 计算loss
        global_step = tf.Variable(0, trainable=False, name=GLOBAL_STEP)
        learning_rate = tf.train.exponential_decay(
            self.config.learning_rate_init,
            global_step=global_step,
            decay_steps=self.config.learning_rate_decay_steps,
            decay_rate=self.config.learning_rate_decay_rate
        )
        # Optimizer

        optimizer = tf.train.AdamOptimizer(learning_rate)
        gvs = optimizer.compute_gradients(loss)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        self.optimizer = optimizer.apply_gradients(capped_gvs, name=OPTIMIZER)

    def var(self, key):
        if key == OPTIMIZER:
            return self.optimizer
        return self.graph.get_operation_by_name(key).outputs[0]
