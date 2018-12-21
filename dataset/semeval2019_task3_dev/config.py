# -*- coding: utf-8 -*-
import os
from dataset.common.config import BaseConfig


class Config(BaseConfig):
    path_root = os.path.join(os.environ['HOME'], 'lab', 'semeval2019_data', 'task3')

    path_train = os.path.join(path_root, 'train.txt')
    path_dev = os.path.join(path_root, 'dev.txt')
    path_dev_no_labels = os.path.join(path_root, 'devwithoutlabels.txt')

    dataset_key = 'semeval2019_task3_dev'


config = Config()
