# -*- coding: utf-8 -*-


class BaseNNConfig(object):
    def __init__(self, data):
        self.full_data = data
        self.data = data['nn']

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
    def output_dim(self):
        return self.data['output']['dim']

    @property
    def binary_classification(self):
        return self.output_dim == 2 and self.data['binary_classification']

    @property
    def batch_size(self):
        return self.full_data['train']['batch_size']

    @property
    def dropout_keep_prob(self):
        return self.data['dropout_keep_prob']

    @property
    def embedding_noise_type(self):
        return self.data['embedding'].get('noise_type', 'gaussian')

    @property
    def embedding_noise_stddev(self):
        return self.data['embedding']['noise_stddev']

    @property
    def embedding_dropout_keep_prob(self):
        return self.data['embedding']['dropout_keep_prob']

    @property
    def dense_layers(self):
        return self.data['dense']

    @property
    def max_out(self):
        return self.data['max_out']

    @property
    def use_attention(self):
        return self.data['use_attention']
