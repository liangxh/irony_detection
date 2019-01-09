# -*- coding: utf-8 -*-
from algo.model.const import *


class TrainConfig(object):
    def __init__(self, data):
        self.data = data

    @property
    def text_type(self):
        return self.data.get('text_type', 'txt')

    @property
    def epoch(self):
        return self.data['epoch']

    @property
    def valid_rate(self):
        return self.data['valid_rate']

    @property
    def dropout_keep_prob(self):
        return self.data['dropout_keep_prob']

    @property
    def batch_size(self):
        return self.data['batch_size']

    @property
    def use_class_weights(self):
        return self.data['use_class_weights']

    @property
    def early_stop_metric(self):
        return self.data['early_stop_metric'] if 'early_stop_metric' in self.data else F1_SCORE
