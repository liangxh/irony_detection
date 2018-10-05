# -*- coding: utf-8 -*-
from algo.model.const import *


class BaseNNModel(object):
    def __init__(self, config):
        self.config = config
        self._variables = None
        self.variable_keys = [
            TOKEN_ID_SEQ, SEQ_LEN,
            LABEL_PREDICT, LABEL_GOLD, SAMPLE_WEIGHTS, HIDDEN_FEAT,
            DROPOUT_KEEP_PROB, LOSS, GLOBAL_STEP, OPTIMIZER,
        ]

    def var(self, key):
        return self._variables[key]

    def preserve_variables(self, tf_graph):
        self._variables = {key: tf_graph.get_operation_by_name(key).outputs[0] for key in self.variable_keys}
