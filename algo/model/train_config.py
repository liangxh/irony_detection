# -*- coding: utf-8 -*-


class TrainConfig(object):
    def __init__(self, data):
        self.data = data

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
