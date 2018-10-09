# -*- coding: utf-8 -*-
import os
from dataset.common.config import BaseConfig


class Config(BaseConfig):
    dataset_key = 'semeval2018_task1'
    path_E_c_train = os.path.join(BaseConfig.path_data, dataset_key, 'E-c-En-train.txt')
    path_E_c_dev = os.path.join(BaseConfig.path_data, dataset_key, 'E-c-En-dev.txt')


config = Config()
