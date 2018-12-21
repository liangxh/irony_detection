# -*- coding: utf-8 -*-
import re
import emoji
import json
import commandr
from string import punctuation
from collections import defaultdict
from dataset.common.const import *
from dataset.common.load import load_label_list, load_tokenized_list, load_vocab_list
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
    for turn_1, turn_2, turn_3, label_idx in Processor.load_origin_dev():
        turns = [turn_1, turn_2, turn_3]
        for i, r in enumerate(turns):
            text_turns[i].append(r)
        labels.append(label_idx)

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
def build_data_split():
    filenames = ['{}.csv'.format(label) for label in label_str]
    file_objs = map(lambda _f: open(_f, 'w'), filenames)

    mode = TRAIN
    text_ek_turns = []
    labels = load_label_list(config.path(TRAIN, LABEL))
    for i in range(3):
        path = config.path(mode, 'turn', '{}.ek'.format(i))
        texts = open(path).read().strip().split('\n')
        text_ek_turns.append(texts)

    n_sample = len(text_ek_turns[0])
    for texts in text_ek_turns:
        assert len(texts) == n_sample

    for label, turn_1, turn_2, turn_3 in zip(labels, *text_ek_turns):
        text = '{} || {} || {}'.format(turn_1, turn_2, turn_3)
        file_objs[label].write(text + '\n')

    for obj in file_objs:
        obj.close()


@commandr.command
def build_dev_submit(output_key=None):
    """
    python -m scripts.semeval2019_task3_dev build_dev_submit -o gru_ek_1543492018
    python3 -m scripts.semeval2019_task3_dev build_dev_submit -o

    :param output_key: string
    :return:
    """
    output_path = 'test.txt'

    first_line = open(config.path_train, 'r').readline()
    with open(output_path, 'w') as o_obj:
        o_obj.write(first_line)

        lines = open(config.path_dev_no_labels).read().strip().split('\n')
        lines = lines[1:]
        lines = list(map(lambda l: l.strip(), lines))

        if output_key is not None:
            path_labels = config.output_path(output_key, TEST, LABEL_PREDICT)

            labels = open(path_labels, 'r').read().strip().split('\n')
            labels = list(map(int, labels))
            labels = list(map(lambda l: label_str[l], labels))
            assert len(labels) == len(lines)
        else:
            labels = ['others'] * len(lines)

        for line, label in zip(lines, labels):
            o_obj.write('{}\t{}\n'.format(line, label))


@commandr.command('analyse_slang')
def analyse_slang(filename_vocab, key, mode=TRAIN):
    """
    python scripts/semeval2019_task3_dev.py analyse_slang out/google_vocabs.txt --key google
    python scripts/semeval2019_task3_dev.py analyse_slang out/google_vocabs.txt --key google --mode test

    python scripts/semeval2019_task3_dev.py analyse_slang out/ntua_vocabs.txt --key ntua
    python scripts/semeval2019_task3_dev.py analyse_slang out/ntua_vocabs.txt --key ntua --mode test

    :param filename_vocab:
    :return:
    """
    slang_count = defaultdict(lambda: 0)
    label_count = defaultdict(lambda: 0)

    vocabs = open(filename_vocab, 'r').read().strip().split('\n')
    vocabs = set(vocabs)
    labels = load_label_list(config.path(mode, LABEL))
    tokenized_list = load_tokenized_list(config.path(mode, TEXT, EK))
    for tokens, label in zip(tokenized_list, labels):
        contain_slang = False
        for token in tokens:
            if token.startswith('<') and token.endswith('>'):
                continue
            if token in emoji.UNICODE_EMOJI or token in punctuation:
                continue
            if not re.match('^[a-zA-Z]+$', token):
                continue

            if token in {'to', 'a', 'and', 'of'}:
                continue

            if token not in vocabs and token.replace('our', 'or') not in vocabs and token.replace('ise', 'ize') not in vocabs:
                slang_count[token] += 1
                contain_slang = True
        if contain_slang:
            label_count[label] += 1

    slang_count = sorted(slang_count.items(), key=lambda _item: _item[1], reverse=True)
    with open('out/{}_{}_slang_tf.txt'.format(key, mode), 'w') as file_obj:
        for slang, count in slang_count:
            file_obj.write(u'{}\t{}\n'.format(count, slang).encode('utf8'))
    json.dump(label_count, open('out/{}_{}_slang_label.json'.format(key, mode), 'w'))


if __name__ == '__main__':
    commandr.Run()
