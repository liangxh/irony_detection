# -*- coding: utf-8 -*-
import re
import commandr
from dataset.common.const import *
from dataset.semeval2019_task3_dev.process import Processor, label_str
from dataset.semeval2019_task3_dev.config import config


@commandr.command
def build_basic():
    config.prepare_data_folder()

    labels = list()
    text_turns = [[] for _ in range(3)]
    for turn_1, turn_2, turn_3, label_idx in Processor.load_origin_train():
        turns = [turn_1, turn_2, turn_3]
        for i, r in enumerate(turns):
            text_turns[i].append(r)
        labels.append(label_idx)

    for i, texts in enumerate(text_turns):
        path = config.path(TRAIN, 'turn', str(i))
        open(path, 'w').write('\n'.join(texts) + '\n')

    path = config.path(TRAIN, LABEL)
    open(path, 'w').write('\n'.join(map(str, labels)) + '\n')

    labels = list()
    text_turns = [[] for _ in range(3)]
    for turn_1, turn_2, turn_3 in Processor.load_origin_dev_no_labels():
        turns = [turn_1, turn_2, turn_3]
        for i, r in enumerate(turns):
            text_turns[i].append(r)
        labels.append(0)

    for i, texts in enumerate(text_turns):
        path = config.path(TEST, 'turn', str(i))
        open(path, 'w').write('\n'.join(texts) + '\n')

    path = config.path(TEST, LABEL)
    open(path, 'w').write('\n'.join(map(str, labels)) + '\n')


@commandr.command
def build_ek():
    config.prepare_data_folder()

    for mode in [TRAIN, TEST]:
        text_ek_turns = []
        for i in range(3):
            path = config.path(mode, 'turn', '{}.ek'.format(i))
            texts = open(path).read().strip().split('\n')
            text_ek_turns.append(texts)

        n_sample = len(text_ek_turns[0])
        for texts in text_ek_turns:
            assert len(texts) == n_sample

        out_path = config.path(mode, TEXT, EK)
        with open(out_path, 'w') as file_obj:
            for turn_1, turn_2, turn_3 in zip(*text_ek_turns):
                text = '{} <turn> {} <turn> {}'.format(turn_1, turn_2, turn_3)
                file_obj.write(text + '\n')


@commandr.command
def build_dev_submit(output_path, path_labels=None):
    first_line = open(config.path_train, 'r').readline()
    with open(output_path, 'w') as o_obj:
        o_obj.write(first_line)

        lines = open(config.path_dev_no_labels).read().strip().split('\n')
        lines = lines[1:]
        lines = map(lambda l: l.strip(), lines)

        if path_labels is not None:
            labels = open(path_labels, 'r').read().strip().split('\n')
            labels = map(int, labels)
            labels = map(lambda l: label_str[l], labels)
            assert len(labels) == len(lines)
        else:
            labels = ['others'] * len(lines)

        for line, label in zip(lines, labels):
            o_obj.write('{}\t{}\n'.format(line, label))


if __name__ == '__main__':
    commandr.Run()
