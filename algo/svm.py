# -*- coding: utf-8 -*-
from __future__ import print_function
import commandr
import importlib
import yaml
import numpy as np
from sklearn import svm
from dataset.common.const import *
from dataset.common.load import *
from algo.model.const import *
from algo.lib.evaluate import basic_evaluate
from algo.lib.common import print_evaluation

FEATS = 'feats'


class Config(object):
    def __init__(self, data):
        self.data = data

    @property
    def feat_keys(self):
        return self.data['feat_keys']

    @property
    def use_class_weights(self):
        return self.data['train']['use_class_weights']


def load_dataset(data_config, train_config, label_version=None):
    datasets = dict()
    for mode in [TRAIN, TEST]:
        feats = list()
        for key in train_config.feat_keys:
            path = data_config.output_path(key, mode, HIDDEN_FEAT)
            feat_list, _ = load_feat(path)
            feats.append(feat_list)
        feats = np.concatenate(feats, axis=1) if len(feats) > 1 else np.asarray(feats[0])

        label_path = data_config.path(mode, LABEL, label_version)
        labels_gold = load_label_list(label_path)

        datasets[mode] = {
            LABEL_GOLD: np.asarray(labels_gold),
            FEATS: feats
        }
    return datasets


@commandr.command
def main(dataset_key, label_version=None, config_path='config_svm.yaml'):
    """
    python algo/svm.py main semeval2018_task3 A

    :param dataset_key:
    :param label_version:
    :param config_path:
    :return:
    """
    pos_label = None
    if dataset_key == 'semeval2018_task3' and label_version == 'A':
        pos_label = 1

    config_data = yaml.load(open(config_path))
    train_config = Config(data=config_data)

    data_config = getattr(importlib.import_module('dataset.{}.config'.format(dataset_key)), 'config')
    datasets = load_dataset(data_config, train_config, label_version)

    if train_config.use_class_weights:
        class_weight = 'balanced'
    else:
        class_weight = None

    clf = svm.SVC(class_weight=class_weight)
    clf.fit(X=datasets[TRAIN][FEATS], y=datasets[TRAIN][LABEL_GOLD])

    for mode in [TRAIN, TEST]:
        labels_predict = clf.predict(X=datasets[mode][FEATS])
        labels_gold = datasets[mode][LABEL_GOLD]
        res = basic_evaluate(gold=labels_gold, pred=labels_predict, pos_label=pos_label)

        print(mode)
        print_evaluation(res)
        print()


if __name__ == '__main__':
    commandr.Run()
