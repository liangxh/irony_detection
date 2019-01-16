# -*- coding: utf-8 -*-
from __future__ import print_function
import commandr
import numpy as np
import yaml
from algo.model.const import *
from dataset.semeval2019_task3_dev.config import config as data_config
from dataset.common.const import *
from dataset.common.load import *
from algo.lib.evaluate93 import basic_evaluate
from algo.lib.common import print_evaluation
from algo.lib.common import generate_wrong_prediction_report

MAJORITY_VOTING = 'mv'
WEIGHTED_MAJORITY_VOTE = 'wmv'
SOFT_VOTING = 'sv'


ANALYSIS = 'analysis'
WRONG_PREDICT = 'wrong_predict'


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
def main(ensemble_mode, config_path='config93_ensemble.yaml', build_analysis=False):
    """
    [Usage]
    python3 -m algo.ensemble93 main -e mv --build-analysis

    :param ensemble_mode: 
    :param config_path:
    :param build_analysis: bool
    :return: 
    """
    dataset_key = 'semeval2019_task3_dev'

    config_data = yaml.load(open(config_path))
    config = Config(data=config_data)

    labels_predict = dict()
    labels_gold = dict()
    n_sample = dict()
    for mode in [TRAIN, TEST]:
        label_path = data_config.path(mode, LABEL, None)
        labels_gold[mode] = load_label_list(label_path)
        n_sample[mode] = len(labels_gold[mode])
    output_dim = max(labels_gold[TEST]) + 1

    if ensemble_mode == SOFT_VOTING:
        for mode in [TRAIN, TEST]:
            components = dict()
            for output_key in config.components:
                path = data_config.output_path(output_key, mode, PROB_PREDICT)

                prob_list = list()
                with open(path) as file_obj:
                    for line in file_obj:
                        line = line.strip()
                        if line == '':
                            continue
                        prob = list(map(float, line.split('\t')))
                        prob_list.append(prob)
                components[output_key] = prob_list

            labels = list()
            for i in range(n_sample[mode]):
                prob = np.zeros((output_dim, ))
                for output_key, prob_list in components.items():
                    prob += np.asarray(prob_list[i])
                labels.append(np.argmax(prob))
            labels_predict[mode] = labels

    elif ensemble_mode == MAJORITY_VOTING:
        components = dict()

        for mode in [TRAIN, TEST]:
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

            labels = list()
            for i in range(n_sample[mode]):
                prob = np.zeros((output_dim, ))
                for output_key, label_list in components.items():
                    label = label_list[i]
                    prob[label] += 1
                labels.append(np.argmax(prob))
            labels_predict[mode] = labels

    elif ensemble_mode == WEIGHTED_MAJORITY_VOTE:
        raise NotImplementedError
    else:
        raise ValueError('unknown mode: {}'.format(ensemble_mode))

    for mode in [TRAIN, TEST]:
        res = basic_evaluate(gold=labels_gold[mode], pred=labels_predict[mode])
        print(mode)
        print_evaluation(res)
        print()

        if build_analysis:
            output_path = data_config.path(mode, ANALYSIS, WRONG_PREDICT)
            text_list = load_text_list(data_config.path(mode, TEXT, EK))
            res = generate_wrong_prediction_report(
                labels_gold=labels_gold[mode], labels_predict=labels_predict[mode], text_list=text_list)
            with open(output_path, 'w') as file_obj:
                file_obj.write('gold\tpredict\ttext')
                for l_gold, l_predict, t in res:
                    file_obj.write('{} {} {}\n'.format(l_gold, l_predict, t))


if __name__ == '__main__':
    commandr.Run()
