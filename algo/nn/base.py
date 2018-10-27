# -*- coding: utf-8 -*-
import tensorflow as tf
from algo.model.const import *


class BaseNNModel(object):
    def __init__(self, config):
        self.config = config
        self.graph = None

    def var(self, key):
        return self.graph.get_operation_by_name(key).outputs[0]

    def set_graph(self, graph):
        self.graph = graph

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
        tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step, name=OPTIMIZER)
