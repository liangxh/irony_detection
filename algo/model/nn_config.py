# -*- coding: utf-8 -*-


class BaseNNConfig(object):
    def __init__(self, data):
        self.data = data

    def set_embedding_dim(self, dim):
        self.data['embedding']['dim'] = dim

    def set_output_dim(self, dim):
        self.data['output']['dim'] = dim

    def set_seq_len(self, seq_len):
        self.data['seq_len'] = seq_len

    @property
    def seq_len(self):
        return self.data['seq_len']

    @property
    def learning_rate_init(self):
        return self.data['learning_rate']['init']

    @property
    def learning_rate_decay_steps(self):
        return self.data['learning_rate']['decay_steps']

    @property
    def learning_rate_decay_rate(self):
        return self.data['learning_rate']['decay_rate']

    @property
    def l2_reg_lambda(self):
        return self.data['l2_reg_lambda']

    @property
    def embedding_trainable(self):
        return self.data['embedding']['trainable']

    @property
    def embedding_dim(self):
        return self.data['embedding']['dim']

    @property
    def rnn_dim(self):
        return self.data['rnn']['dim']

    @property
    def output_dim(self):
        return self.data['output']['dim']
