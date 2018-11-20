# -*- coding: utf-8 -*-
from __future__ import print_function
import commandr
import importlib
import numpy as np
import yaml
from algo.model.const import *
from dataset.common.const import *
from dataset.common.load import *


MAJORITY_VOTING = 'mv'
WEIGHTED_MAJORITY_VOTE = 'wmv'
SOFT_VOTING = 'sv'


class Config(object):
    def __init__(self, data):
        self.data = data

    @property
    def components(self):
        return self.data['components']


def argmax(value_list):
    idx = 0
    max_value = value_list[idx]
    for i, value in enumerate(value_list[1:]):
        if value > max_value:
            max_value = value
            idx = i + 1
    return idx, max_value


@commandr.command
def main(dataset_key, label_version, ensemble_mode, config_path='config_ensemble.yaml')
    """
    [Usage]
    python algo/ensemble.py -d semeval2018_task3 -l A -m sv
    
    :param dataset_key: 
    :param label_version: 
    :param ensemble_mode: 
    :param config_path: 
    :return: 
    """

    config_data = yaml.load(open(config_path))
    config = Config(data=config_data)
    data_config = getattr(importlib.import_module('dataset.{}.config'.format(dataset_key)), 'config')

    labels = dict()
    n_sample = dict()
    for mode in [TRAIN, TEST]:
        label_path = data_config.path(mode, LABEL, label_version)
        labels[mode] = load_label_list(label_path)
        n_sample[mode] = len(labels[mode])

    if ensemble_mode == SOFT_VOTING:
        components = dict()

        for mode in [TRAIN, TEST]:
            component = list()

            for output_key in config.components:
                path = data_config.output_path(output_key, mode, PROB_PREDICT)
                prob_list = list()
                with open(path) as file_obj:
                    for line in file_obj:
                        line = line.strip()
                        if line == '':
                            continue
                        prob = map(float, line.split('\t'))
                        prob_list.append(prob)
                components[output_key] = prob_list



    elif ensemble_mode == MAJORITY_VOTING:
        components = dict()

        for output_key in config.components:
            path = data_config.output_path(output_key, mode, LABEL_PREDICT)
            label_list = list()
            with open(path) as file_obj:
                for line in file_obj:
                    line = line.strip()
                    if line == '':
                        continue
                    label = int(line)
                    label_list.append(label)
            components[output_key] = label_list

    elif ensemble_mode == WEIGHTED_MAJORITY_VOTE:
        pass
    else:
        raise ValueError('unknown mode: {}'.format(ensemble_mode))


if __name__ == '__main__':
    commandr.command()
