# -*- coding: utf-8 -*-
import os


class BaseConfig(object):
    dataset_key = None
    path_data = os.path.join(os.environ['HOME'], 'lab', 'irony_detection_data')
    path_output = os.path.join(os.environ['HOME'], 'lab', 'irony_detection_output')

    word2vec_google = os.path.join(os.environ['HOME'], 'Downloads', 'GoogleNews-vectors-negative300.bin')

    def path(self, mode, dtype, version=None):
        suffix = '{}.{}'.format(mode, dtype)
        if version is not None:
            suffix += '.{}'.format(version)
        return os.path.join(self.path_data, self.dataset_key, suffix)

    def output_path(self, key, mode, dtype):
        return os.path.join(self.path_output, self.dataset_key, '{}.{}.{}'.format(key, mode, dtype))

    def prepare_data_folder(self):
        path = os.path.join(self.path_data, self.dataset_key)
        if not os.path.exists(path):
            os.mkdir(path)

    def prepare_output_folder(self):
        path = os.path.join(self.path_output, self.dataset_key)
        if not os.path.exists(path):
            os.mkdir(path)
