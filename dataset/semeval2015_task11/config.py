# -*- coding: utf-8 -*-
import os
from dataset.common.config import BaseConfig


class Config(BaseConfig):
    dataset_key = 'semeval2015_task11'
    path_train_raw = os.path.join(BaseConfig.path_data, dataset_key, 'task-11-train.json')
    path_trial_raw = os.path.join(BaseConfig.path_data, dataset_key, 'task-11-trial.json')
    path_new_raw = os.path.join(BaseConfig.path_data, dataset_key, 'task-11-new.json')

    path_train_id_score = os.path.join(BaseConfig.path_data, dataset_key, 'task-11-train-data.csv')
    path_trial_id_score = os.path.join(BaseConfig.path_data, dataset_key, 'task-11-trial-data.csv')
    path_new_id_score = os.path.join(BaseConfig.path_data, dataset_key, 'newid_weightedtweetdata.csv')


config = Config()
