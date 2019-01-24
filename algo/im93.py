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


@commandr.command
def main(filename, config_path='e93.yaml', final_output=None):
    """
    [Usage]
    python3 -m algo.ensemble93 main -e mv --build-analysis
    """
    config_data = yaml.load(open(config_path))
    config = Config(data=config_data)

    labels_gold = dict()
    labels_predict = dict()
    labels_predict_last = dict()

    dataset = Processor.load_origin(filename)
    labels_predict[FINAL] = list(map(lambda _item: _item[-1], dataset))

    modes = {
        TRAIN: [TRAIN, TEST],
        FINAL: [FINAL, ]
    }

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
