# -*- coding: utf-8 -*-
from __future__ import print_function
import commandr
import numpy as np
import yaml
from algo.model.const import *
from dataset.semeval2019_task3_dev.config import config as data_config
from dataset.semeval2019_task3_dev.process import label_str
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

    @property
    def others(self):
        return self.data['others']['components']

    @property
    def others_enabled(self):
        return self.data['others']['enabled']

    @property
    def others_min_vote(self):
        return self.data['others']['min_vote']

    @property
    def tri(self):
        return self.data['tri']['components']

    @property
    def tri_enabled(self):
        return self.data['tri']['enabled']

    @property
    def tri_min_vote(self):
        return self.data['tri']['min_vote']

    @property
    def oh(self):
        return self.data['oh']['components']

    @property
    def oh_enabled(self):
        return self.data['oh']['enabled']

    @property
    def oh_min_vote(self):
        return self.data['oh']['min_vote']


def argmax(value_list):
    idx = 0
    max_value = value_list[idx]
    for i, value in enumerate(value_list[1:]):
        if value > max_value:
            max_value = value
            idx = i + 1
    return idx, max_value


@commandr.command
def improve_others(main_output_key, sub_output_key):
    labels_predict = dict()
    labels_predict_after = dict()

    labels_gold = dict()
    for mode in [TEST]:
        label_path = data_config.path(mode, LABEL, None)
        labels_gold[mode] = load_label_list(label_path)

        path = data_config.output_path(main_output_key, mode, LABEL_PREDICT)
        labels_predict_base = load_label_list(path)
        labels_predict[mode] = list() + labels_predict_base

        path = data_config.output_path(sub_output_key, mode, LABEL_PREDICT)
        labels = load_label_list(path)

        for i, (p_base, p_sub) in enumerate(zip(labels_predict_base, labels)):
            if p_sub == 0:
                labels_predict_base[i] = 0
        labels_predict_after[mode] = labels_predict_base

    for mode in [TEST]:
        res = basic_evaluate(gold=labels_gold[mode], pred=labels_predict[mode])
        print(mode)
        print_evaluation(res)
        for col in res[CONFUSION_MATRIX]:
            print(','.join(map(str, col)))
        print()

        res = basic_evaluate(gold=labels_gold[mode], pred=labels_predict_after[mode])
        print(mode, '(AFTER)')
        print_evaluation(res)
        for col in res[CONFUSION_MATRIX]:
            print(','.join(map(str, col)))
        print()


@commandr.command
def main(ensemble_mode, config_path='config93_ensemble.yaml', final_output=None):
    """
    [Usage]
    python3 -m algo.ensemble93 main -e mv --build-analysis

    :param ensemble_mode:
    :param config_path:
    :param build_analysis: bool
    :return:
    """
    config_data = yaml.load(open(config_path))
    config = Config(data=config_data)

    labels_predict = dict()
    labels_predict_last = dict()
    labels_gold = dict()

    n_sample = dict()
    for mode in [TRAIN, TEST, FINAL]:
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

        for mode in [TRAIN, TEST, FINAL]:
            components = list()

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
                components.append(label_list)

            labels = list()
            for i in range(n_sample[mode]):
                prob = np.zeros((output_dim, ))
                for label_list in components:
                    label = label_list[i]
                    prob[label] += 1
                labels.append(np.argmax(prob))
            labels_predict[mode] = labels

    elif ensemble_mode == WEIGHTED_MAJORITY_VOTE:
        raise NotImplementedError
    else:
        raise ValueError('unknown mode: {}'.format(ensemble_mode))

    for mode in [TRAIN, TEST, FINAL]:
        if not mode == FINAL:
            res = basic_evaluate(gold=labels_gold[mode], pred=labels_predict[mode])
            print(mode)
            print_evaluation(res)
            for col in res[CONFUSION_MATRIX]:
                print(','.join(map(str, col)))
            print()

        n_sample = len(labels_predict[mode])
        labels_predict_last[mode] = labels_predict[mode]

        # 修正HAS
        if config.tri_enabled:
            votes = [[0 for _ in range(4)] for _ in range(n_sample)]
            for output_key in config.tri:
                path = data_config.output_path(output_key, mode, LABEL_PREDICT)
                labels = load_label_list(path)
                if len(labels) != n_sample:
                    raise Exception('mismatch {}({}) != {}'.format(output_key, len(labels), n_sample))

                for i, label in enumerate(labels):
                    votes[i][label] += 1

            base = list() + labels_predict_last[mode]
            for i, vote in enumerate(votes):
                if base[i] != 0:
                    arg_max = int(np.argmax(vote))
                    if arg_max != 0 and vote[arg_max] >= config.tri_min_vote:
                        base[i] = arg_max

            labels_predict_last[mode] = base
            if not mode == FINAL:
                res = basic_evaluate(gold=labels_gold[mode], pred=base)
                print(mode, '(after TRI)')
                print_evaluation(res)
                for col in res[CONFUSION_MATRIX]:
                    print(','.join(map(str, col)))
                print()

        # 将判成HAS的样本修正为Others
        if config.others_enabled:
            votes = [0 for i in range(n_sample)]

            for output_key in config.others:
                path = data_config.output_path(output_key, mode, LABEL_PREDICT)
                labels = load_label_list(path)
                if len(labels) != n_sample:
                    raise Exception('mismatch {}({}) != {}'.format(output_key, len(labels), n_sample))

                for i, label in enumerate(labels):
                    if label == 0:
                        votes[i] += 1
            if config.others_min_vote == 'all':
                min_vote = len(config.others)
            else:
                min_vote = int(config.others_min_vote)
            base = list() + labels_predict_last[mode]
            for i, vote in enumerate(votes):
                if vote >= min_vote:
                    base[i] = 0

            labels_predict_last[mode] = base
            if not mode == FINAL:
                res = basic_evaluate(gold=labels_gold[mode], pred=base)
                print(mode, '(after OTHERS)')
                print_evaluation(res)
                for col in res[CONFUSION_MATRIX]:
                    print(','.join(map(str, col)))
                print()

        if config.oh_enabled:
            votes = [[0 for _ in range(4)] for _ in range(n_sample)]
            for output_key in config.oh:
                path = data_config.output_path(output_key, mode, LABEL_PREDICT)
                labels = load_label_list(path)
                if len(labels) != n_sample:
                    raise Exception('mismatch {}({}) != {}'.format(output_key, len(labels), n_sample))

                for i, label in enumerate(labels):
                    votes[i][label] += 1

            base = list() + labels_predict_last[mode]
            for i, vote in enumerate(votes):
                if base[i] != 0:
                    arg_max = int(np.argmax(vote))
                    if base[i] == 1 and arg_max == 0 and vote[arg_max] >= config.oh_min_vote:
                        base[i] = arg_max

            labels_predict_last[mode] = base
            if not mode == FINAL:
                res = basic_evaluate(gold=labels_gold[mode], pred=base)
                print(mode, '(after OH)')
                print_evaluation(res)
                for col in res[CONFUSION_MATRIX]:
                    print(','.join(map(str, col)))
                print()

        if mode == FINAL and final_output is not None:
            first_line = open(data_config.path_train, 'r').readline()
            with open(final_output, 'w') as o_obj:
                o_obj.write(first_line)

                lines = open(data_config.path_test_no_labels).read().strip().split('\n')
                lines = lines[1:]
                lines = list(map(lambda l: l.strip(), lines))

                labels = labels_predict_last[FINAL]
                labels = list(map(lambda l: label_str[l], labels))
                assert len(labels) == len(lines)

                for line, label in zip(lines, labels):
                    o_obj.write('{}\t{}\n'.format(line, label))


if __name__ == '__main__':
    commandr.Run()
