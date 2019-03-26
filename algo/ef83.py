# -*- coding: utf-8 -*-
from __future__ import print_function
import commandr
import numpy as np
import yaml
from algo.model.const import *
from dataset.semeval2018_task3.config import config as data_config
from dataset.common.const import *
from dataset.common.load import *
from algo.lib.evaluate93 import basic_evaluate
from algo.lib.common import print_evaluation

MAJORITY_VOTING = 'mv'
WEIGHTED_MAJORITY_VOTE = 'wmv'
SOFT_VOTING = 'sv'


ANALYSIS = 'analysis'
WRONG_PREDICT = 'wrong_predict'


class Config(object):
    def __init__(self, data):
        self.data = data

    def components(self, key=None):
        if key is None:
            return self.data['components']
        else:
            return self.data['components_{}'.format(key)]

    def thr(self, key=None):
        if key is None:
            return self.data['thr']
        else:
            return self.data['thr_{}'.format(key)]

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


def argmax(value_list):
    idx = 0
    max_value = value_list[idx]
    for i, value in enumerate(value_list[1:]):
        if value > max_value:
            max_value = value
            idx = i + 1
    return idx, max_value


def combine(output_keys, mode, full_output=False):
    dim = 0
    label_lists = list()
    for output_key in output_keys:
        path = data_config.output_path(output_key, mode, LABEL_PREDICT)
        label_list = load_label_list(path)
        label_lists.append(label_list)
        dim = max(dim, max(label_list) + 1)

    res = list()
    counts = list()
    for votes in zip(*label_lists):
        count = [0] * dim
        for v in votes:
            count[v] += 1
        vote, vote_count = argmax(count)
        res.append((vote, vote_count))
        counts.append(count)

    if not full_output:
        return res
    else:
        return res, counts


@commandr.command
def main(config_path='e83.yaml'):
    """
    [Usage]
    python3 -m algo.ensemble93 main -e mv --build-analysis

    :param config_path:
    :return:
    """
    config_data = yaml.load(open(config_path))
    config = Config(data=config_data)

    for mode in [TRAIN, TEST]:
        b_result = combine(output_keys=config.components('b'), mode=mode)
        b_vote = list(map(lambda _item: _item[0], b_result))

        b2_result = combine(output_keys=config.components('b2'), mode=mode)
        b2_vote = list(map(lambda _item: _item[0], b2_result))

        last_vote = list()
        for b_v, b2_v in zip(b_vote, b2_vote):
            if b_v == 0:
                label = 0
            elif b2_v == 0:
                label = 1
            else:
                label = 2
            last_vote.append(label)

        b3_result = combine(output_keys=config.components('b3'), mode=mode)
        b3_vote = list(map(lambda _item: _item[0], b3_result))

        labels_predict = list()
        for last_v, b3_v in zip(last_vote, b3_vote):
            if last_v != 2:
                label = last_v
            elif b3_v == 0:
                label = 2
            else:
                label = 3
            labels_predict.append(label)

        labels_gold = load_label_list(data_config.path(mode, LABEL, 'B'))

        res = basic_evaluate(gold=labels_gold, pred=labels_predict)

        print(mode)
        print_evaluation(res)
        for col in res[CONFUSION_MATRIX]:
            print(','.join(map(str, col)))


@commandr.command
def m2(config_path='e83.yaml'):
    """
    [Usage]
    python3 -m algo.ensemble93 main -e mv --build-analysis

    :param config_path:
    :return:
    """
    config_data = yaml.load(open(config_path))
    config = Config(data=config_data)

    for mode in [TEST, ]:
        labels_gold = load_label_list(data_config.path(mode, LABEL, 'B'))

        b_result = combine(output_keys=config.components(), mode=mode)
        b_vote = list(map(lambda _item: _item[0], b_result))

        b0_result = dict()
        b0_vote = dict()

        last_vote = b_vote

        res = basic_evaluate(gold=labels_gold, pred=last_vote)

        print('{}'.format(mode))
        print_evaluation(res)
        for col in res[CONFUSION_MATRIX]:
            print(','.join(map(str, col)))

        for i in [1, 2, 3]:
            b0_result[i] = combine(output_keys=config.components('b0{}'.format(i)), mode=mode)
            b0_vote[i] = list(map(lambda _item: _item[0], b0_result[i]))

            new_vote = list()
            for l_v, b0_v in zip(last_vote, b0_vote[i]):
                if l_v in {0, i}:
                    new_vote.append(b0_v)
                else:
                    new_vote.append(l_v)
            last_vote = new_vote

            res = basic_evaluate(gold=labels_gold, pred=new_vote)

            print('{} - {}'.format(mode, i))
            print_evaluation(res)
            for col in res[CONFUSION_MATRIX]:
                print(','.join(map(str, col)))


@commandr.command
def m3(config_path='e83.yaml'):
    """
    [Usage]
    python3 -m algo.ensemble93 main -e mv --build-analysis

    :param config_path:
    :return:
    """
    config_data = yaml.load(open(config_path))
    config = Config(data=config_data)

    for mode in [TEST, ]:
        labels_gold = load_label_list(data_config.path(mode, LABEL, 'B'))

        b_result = combine(output_keys=config.components(), mode=mode)
        b_vote = list(map(lambda _item: _item[0], b_result))

        b0_result = dict()
        b0_vote = dict()

        last_vote = b_vote

        res = basic_evaluate(gold=labels_gold, pred=last_vote)

        print('{}'.format(mode))
        print_evaluation(res)
        for col in res[CONFUSION_MATRIX]:
            print(','.join(map(str, col)))

        for i in [1, 2, 3]:
            key = 'b0{}'.format(i)
            thr = config.thr(key)
            b0_result[i] = combine(output_keys=config.components(key), mode=mode)

            new_vote = list()
            for l_v, b0_res in zip(last_vote, b0_result[i]):
                this_vote = 0 if b0_res[0] == 0 else i
                if l_v in {0, i} and b0_res[1] >= thr:
                    new_vote.append(this_vote)
                else:
                    new_vote.append(l_v)
            last_vote = new_vote

            res = basic_evaluate(gold=labels_gold, pred=new_vote)

            print('{} - {}'.format(mode, i))
            print_evaluation(res)
            for col in res[CONFUSION_MATRIX]:
                print(','.join(map(str, col)))


@commandr.command
def m3a(target=0, thr=1, config_path='e83a.yaml'):
    target = int(target)
    thr = int(thr)
    config_data = yaml.load(open(config_path))
    config = Config(data=config_data)

    for mode in [TEST, ]:
        labels_gold = load_label_list(data_config.path(mode, LABEL, 'A'))

        b_result = combine(output_keys=config.components(), mode=mode)
        new_vote = list()
        for r in b_result:
            if r[0] == target and r[1] >= thr:
                new_vote.append(target)
            else:
                new_vote.append(1 - target)
        res = basic_evaluate(gold=labels_gold, pred=new_vote)

        print('{}'.format(mode))
        print_evaluation(res)
        for col in res[CONFUSION_MATRIX]:
            print(','.join(map(str, col)))

        last_vote = new_vote
        output_keys = config.components('b')
        b_result, counts = combine(output_keys=output_keys, mode=mode, full_output=True)
        new_vote = list()
        for count, l_v in zip(counts, last_vote):
            if count[0] <= 1:
                new_vote.append(0)
            else:
                new_vote.append(l_v)
        res = basic_evaluate(gold=labels_gold, pred=new_vote)
        print('{}'.format(mode))
        print_evaluation(res)
        for col in res[CONFUSION_MATRIX]:
            print(','.join(map(str, col)))


if __name__ == '__main__':
    commandr.Run()
