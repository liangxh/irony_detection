# -*- coding: utf-8 -*-
import os


class BaseConfig(object):
    dataset_key = None
    path_data = os.path.join(os.environ['HOME'], 'lab', 'irony_detection', 'data')

    word2vec_google = os.path.join(os.environ['HOME'], 'Downloads', 'GoogleNews-vectors-negative300.bin')

    def path(self, mode, dtype, version=None):
        suffix = '{}.{}'.format(mode, dtype)
        if version is not None:
            suffix += '.{}'.format(version)
        return os.path.join(self.path_data, self.dataset_key, suffix)
