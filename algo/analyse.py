# -*- coding: utf-8 -*-
from __future__ import print_function
import commandr
import importlib
import numpy as np
import yaml
from algo.model.const import *
from dataset.common.const import *
from dataset.common.load import *

ANALYSIS = 'analysis'
WRONG_PREDICT = 'wrong_predict'


@commandr.command
def main(dataset_key, label_version, output_key):
    """
    [Usage]
    python algo/ensemble.py main -d semeval2018_task3 -l A -e sv

    :param dataset_key:
    :param label_version:
    :param ensemble_mode:
    :param config_path:
    :return:
    """
    data_config = getattr(importlib.import_module('dataset.{}.config'.format(dataset_key)), 'config')

    for mode in [TRAIN, TEST]:
        path = data_config.path(mode, TEXT)
        texts = load_text_list(path)

        path = data_config.path(mode, LABEL, label_version)
        labels_gold = load_label_list(path)

        path = data_config.output_path(output_key, mode, LABEL_PREDICT)
        labels_predict = load_label_list(path)

        res = list()
        for l_gold, l_predict, t in zip(labels_gold, labels_predict, texts):
            if l_gold != l_predict:



    for mode in [TRAIN, TEST]:
        res = basic_evaluate(gold=labels_gold[mode], pred=labels_predict[mode], pos_label=pos_label)
        print(mode)
        print_evaluation(res)
        print()


if __name__ == '__main__':
    commandr.Run()
