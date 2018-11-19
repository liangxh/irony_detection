# -*- coding: utf-8 -*-
import os


class BaseConfig(object):
    dataset_key = None
    path_data = os.path.join(os.environ['HOME'], 'lab', 'irony_detection_data')
    path_output = os.path.join(os.environ['HOME'], 'lab', 'irony_detection_output')
    path_model = os.path.join(os.environ['HOME'], 'lab', 'irony_detection_model')

    word2vec_google = os.path.join(os.environ['HOME'], 'Downloads', 'GoogleNews-vectors-negative300.bin')

    def path(self, mode, dtype, version=None):
        suffix = '{}.{}'.format(mode, dtype)
        if version is not None:
            suffix += '.{}'.format(version)
        return os.path.join(self.path_data, self.dataset_key, suffix)

    def output_path(self, key, mode, dtype):
        return os.path.join(self.path_output, self.dataset_key, key, '{}.{}'.format(mode, dtype))

    def model_path(self, key):
        return os.path.join(self.path_model, self.dataset_key, key)

    @classmethod
    def prepare_folder(cls, paths):
        for path in paths:
            if not os.path.exists(path):
                os.mkdir(path)

    def prepare_data_folder(self):
        path = os.path.join(self.path_data, self.dataset_key)
        self.prepare_folder([path, ])

    def prepare_output_folder(self, output_key=None):
        paths = [self.path_output, os.path.join(self.path_output, self.dataset_key), ]
        if output_key is not None:
            path = os.path.join(paths[-1], output_key)
            paths.append(path)
        self.prepare_folder(paths)

    def prepare_model_folder(self, output_key=None):
        paths = [self.path_model,  os.path.join(self.path_model, self.dataset_key), ]
        if output_key is not None:
            path = os.path.join(paths[-1], output_key)
            paths.append(path)
        self.prepare_folder(paths)

    def output_folder(self, output_key):
        return os.path.join(self.path_output, self.dataset_key, output_key)

    def model_folder(self, output_key):
        return os.path.join(self.path_model, self.dataset_key, output_key)
