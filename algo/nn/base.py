# -*- coding: utf-8 -*-
import tensorflow as tf
from algo.model.const import *


class BaseNNModel(object):
    def __init__(self, config):
        self.config = config
        self.graph = None
        self.optimizer = None

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

        optimizer = tf.train.AdamOptimizer(learning_rate)
        gvs = optimizer.compute_gradients(loss)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        self.optimizer = optimizer.apply_gradients(capped_gvs, name=OPTIMIZER)

    def var(self, key):
        if key == OPTIMIZER:
            return self.optimizer
        return self.graph.get_operation_by_name(key).outputs[0]
