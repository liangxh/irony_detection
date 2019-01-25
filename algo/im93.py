# -*- coding: utf-8 -*-
from __future__ import print_function
import commandr
import numpy as np
import yaml
from algo.model.const import *
from dataset.semeval2019_task3_dev.config import config as data_config
from dataset.semeval2019_task3_dev.process import label_str, Processor
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
    def tri_out_vote(self):
        return self.data['tri']['out_vote']


def argmax(value_list):
    idx = 0
    max_value = value_list[idx]
    for i, value in enumerate(value_list[1:]):
        if value > max_value:
            max_value = value
            idx = i + 1
    return idx, max_value


modes = {
    TRAIN: [TRAIN, TEST],
    FINAL: [FINAL, ]
}


def export_final(output_filename, labels):
    first_line = open(data_config.path_train, 'r').readline()
    with open(output_filename, 'w') as o_obj:
        o_obj.write(first_line)

        lines = open(data_config.path_test_no_labels).read().strip().split('\n')
        lines = lines[1:]
        lines = list(map(lambda l: l.strip(), lines))

        labels = list(map(lambda l: label_str[l], labels))
        assert len(labels) == len(lines)

        for line, label in zip(lines, labels):
            o_obj.write('{}\t{}\n'.format(line, label))



def load_tri_votes(config, modes):
    n_sample = 5509
    votes = [[0 for _ in range(4)] for _ in range(n_sample)]
    for output_key in config.tri:
        labels = list()
        for _mode in modes:
            path = data_config.output_path(output_key, _mode, LABEL_PREDICT)
            labels += load_label_list(path)

        for i, label in enumerate(labels):
            votes[i][label] += 1
    return votes


def load_others_votes(config, modes):
    n_sample = 5509
    votes = [0 for _ in range(n_sample)]
    for output_key in config.others:
        labels = list()
        for _mode in modes:
            path = data_config.output_path(output_key, _mode, LABEL_PREDICT)
            labels += load_label_list(path)
        if len(labels) != n_sample:
            raise Exception('mismatch {}({}) != {}'.format(output_key, len(labels), n_sample))

        for i, label in enumerate(labels):
            if label == 0:
                votes[i] += 1
    return votes


@commandr.command
def main(input_filename, config_path='e93.yaml', final_output=None):
    """
    [Usage]
    python3 -m algo.ensemble93 main -e mv --build-analysis
    """
    config_data = yaml.load(open(config_path))
    config = Config(data=config_data)

    labels_gold = dict()
    labels_predict = dict()
    labels_predict_last = dict()

    dataset = Processor.load_origin(input_filename)
    labels_predict[FINAL] = list(map(lambda _item: _item[-1], dataset))

    for mode in [FINAL, ]:
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
            n_changed = 0

            votes = [[0 for _ in range(4)] for _ in range(n_sample)]
            for output_key in config.tri:
                labels = list()
                for _mode in modes[mode]:
                    path = data_config.output_path(output_key, _mode, LABEL_PREDICT)
                    labels += load_label_list(path)
                if len(labels) != n_sample:
                    raise Exception('mismatch {}({}) != {}'.format(output_key, len(labels), n_sample))

                for i, label in enumerate(labels):
                    votes[i][label] += 1

            base = list() + labels_predict_last[mode]
            for i, vote in enumerate(votes):
                arg_max = int(np.argmax(vote))
                if arg_max == 0:
                    continue
                if base[i] != 0:
                    if vote[arg_max] >= config.tri_min_vote:
                        if base[i] != arg_max:
                            n_changed += 1
                        base[i] = arg_max
                elif vote[arg_max] >= config.tri_out_vote:
                    base[i] = arg_max
                    n_changed += 1

            print('n_exchanged within "HAS": {}'.format(n_changed))

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
            votes = [0 for _ in range(n_sample)]
            n_changed = 0

            for output_key in config.others:
                labels = list()
                for _mode in modes[mode]:
                    path = data_config.output_path(output_key, _mode, LABEL_PREDICT)
                    labels += load_label_list(path)
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
                    if base[i] != 0:
                        n_changed += 1
                    base[i] = 0
            print('n_changed to "OTHERS": {}'.format(n_changed))

            labels_predict_last[mode] = base
            if not mode == FINAL:
                res = basic_evaluate(gold=labels_gold[mode], pred=base)
                print(mode, '(after OTHERS)')
                print_evaluation(res)
                for col in res[CONFUSION_MATRIX]:
                    print(','.join(map(str, col)))
                print()

        if mode == FINAL and final_output is not None:
            labels = labels_predict_last[FINAL]
            export_final(final_output, labels)


@commandr.command('diff')
def diff(a_filename, b_filename, output_filename, config_path='e93.yaml'):
    config_data = yaml.load(open(config_path))
    config = Config(data=config_data)

    votes = None

    for output_key in config.others:
        labels = list()
        for _mode in modes[FINAL]:
            path = data_config.output_path(output_key, _mode, LABEL_PREDICT)
            labels += load_label_list(path)

        if votes is None:
            n_sample = len(labels)
            votes = [0 for _ in range(n_sample)]

        for i, label in enumerate(labels):
            if label == 0:
                votes[i] += 1

    dataset = Processor.load_origin(a_filename)
    labels_a = list(map(lambda _item: _item[-1], dataset))

    dataset = Processor.load_origin(b_filename)
    labels_b = list(map(lambda _item: _item[-1], dataset))

    assert len(votes) == len(labels_a) == len(labels_b)

    n_match = 0
    with open(output_filename, 'w') as file_obj:
        for i, (a, b, d) in enumerate(zip(labels_a, labels_b, dataset)):
            if a == 3:
                if b == 0:
                    file_obj.write('{}\t{}\t{}\t{}\t{}->{} ({})\n'.format(
                        i, d[0], d[1], d[2], label_str[a], label_str[b], votes[i]
                    ))
                else:
                    n_match += 1
    print(n_match)


@commandr.command('others')
def filter_by_others(input_filename, output_filename, thr, config_path='e93.yaml'):
    thr = int(thr)
    config_data = yaml.load(open(config_path))
    config = Config(data=config_data)

    votes = None

    for output_key in config.others:
        labels = list()
        for _mode in modes[FINAL]:
            path = data_config.output_path(output_key, _mode, LABEL_PREDICT)
            labels += load_label_list(path)

        if votes is None:
            n_sample = len(labels)
            votes = [0 for _ in range(n_sample)]

        for i, label in enumerate(labels):
            if label == 0:
                votes[i] += 1

    dataset = Processor.load_origin(input_filename)
    labels = list(map(lambda _item: _item[-1], dataset))

    assert len(votes) == len(labels)

    with open(output_filename, 'w') as file_obj:
        for i, (p, d) in enumerate(zip(labels, dataset)):
            if p != 0 and votes[i] >= thr:
                file_obj.write('{}\t{}\t{}\t{}\t{} ({})\n'.format(
                    i, d[0], d[1], d[2], p, votes[i]
                ))
                labels[i] = 0
    export_final('test.txt', labels)


@commandr.command
def export(config_path='e93.yaml'):
    """
    [Usage]
    python3 -m algo.ensemble93 main -e mv --build-analysis
    """
    config_data = yaml.load(open(config_path))
    config = Config(data=config_data)

    votes = load_tri_votes(config, [FINAL, ])
    with open('out/vote_tri.txt', 'w') as file_obj:
        for i, vote in enumerate(votes):
            file_obj.write('{} {}\n'.format(i, vote))

    votes = load_others_votes(config, [FINAL, ])
    with open('out/vote_bin.txt', 'w') as file_obj:
        for i, vote in enumerate(votes):
            file_obj.write('{} {}\n'.format(i, vote))


@commandr.command('oout')
def others_out(filename, thr, output_file, config_path='e93.yaml'):
    """
    [Usage]
    python3 -m algo.ensemble93 main -e mv --build-analysis
    """
    thr = int(thr)
    config_data = yaml.load(open(config_path))
    config = Config(data=config_data)

    votes = load_tri_votes(config, [FINAL, ])
    with open('out/vote_tri.txt', 'w') as file_obj:
        for i, vote in enumerate(votes):
            file_obj.write('{} {}\n'.format(i, vote))

    votes_others = load_others_votes(config, [FINAL, ])
    votes_tri = load_tri_votes(config, [FINAL, ])

    dataset = Processor.load_origin(filename)
    with open(output_file, 'w') as file_obj:
        for i, (d, v_others, v_tri) in enumerate(zip(dataset, votes_others, votes_tri)):
            if p == 0 and v_others <= thr:
                file_obj.write('{}\t{}\t{}\t{}\t{} ({}->{})\n'.format(
                    i, d[0], d[1], d[2], d[-1], v_others, v_tri
                ))


if __name__ == '__main__':
    commandr.Run()
