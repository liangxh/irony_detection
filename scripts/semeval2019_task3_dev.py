# -*- coding: utf-8 -*-
import re
import emoji
import json
import commandr
import numpy as np
from string import punctuation
from collections import defaultdict
from algo.model.const import *
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

    binary_labels = [0 if label == 0 else 1 for label in labels]
    path = config.path(TRAIN, LABEL, 'binary')
    open(path, 'w').write('\n'.join(map(str, binary_labels)) + '\n')

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

    binary_labels = [0 if label == 0 else 1 for label in labels]
    path = config.path(TEST, LABEL, 'binary')
    open(path, 'w').write('\n'.join(map(str, binary_labels)) + '\n')


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


@commandr.command
def analyse_cover():
    train_tokenized_list = load_tokenized_list(config.path(TRAIN, TEXT, EK))
    test_tokenized_list = load_tokenized_list(config.path(TEST, TEXT, EK))
    test_labels = load_label_list(config.path(TEST, LABEL))

    train_vocab = set()
    for tokens in train_tokenized_list:
        train_vocab |= set(tokens)

    unknown_vocab_dist = defaultdict(lambda: defaultdict(lambda: list()))
    unknown_sample = defaultdict(lambda: 0)

    test_vocab = set()
    for tokens, label in zip(test_tokenized_list, test_labels):
        line = ' '.join(tokens)
        tokens = set(tokens)
        test_vocab |= tokens
        unknown = tokens - train_vocab
        if len(unknown) > 0:
            unknown_sample[label] += 1
            for token in unknown:
                unknown_vocab_dist[label][token] += [line, ]

    print len(train_vocab - test_vocab)
    print len(test_vocab - train_vocab)
    new_vocab = test_vocab - train_vocab
    with open('out/new_vocab_in_test.txt', 'w') as file_obj:
        file_obj.write(u'\n'.join(new_vocab).encode('utf8'))

    with open('out/new_vocab_dist.txt', 'w') as file_obj:
        for i in range(4):
            label = label_str[i]
            tf = unknown_vocab_dist[i]
            tf = sorted(tf.items(), key=lambda _item: -len(_item[1]))
            file_obj.write('{}: n_sample: {}; n_vocab: {}\n'.format(label, unknown_sample[i], len(tf)))
            for t, f in tf:
                file_obj.write(u'\t{}\t{}\n'.format(len(f), t).encode('utf8'))
                for fi in f:
                    file_obj.write(u'\t\t{}\n'.format(fi).encode('utf8'))


def confusion_matrix_to_score(mat):
    dim = mat.shape[0]
    n_right = (mat[1:, 1:] * np.eye(dim - 1)).sum()
    n_pred = np.sum(mat[:, 1:])
    n_gold = np.sum(mat[1:, :])
    precision = float(n_right) / n_pred if n_pred > 0 else 0.
    recall = float(n_right) / n_gold if n_gold > 0 else 0.
    score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.
    return score


def estimate_new_score(confusion_matrix, real_distribution=None):
    if real_distribution is None:
        real_distribution = [0.88, 0.04, 0.04, 0.04]

    mat = np.asarray(confusion_matrix)
    known_dist = mat.sum(axis=1).astype(float) / mat.sum()
    real_dist = np.asarray(real_distribution)

    multi = real_dist / known_dist
    new_mat = mat * multi.reshape([1, -1]).T
    score = confusion_matrix_to_score(new_mat)
    return score


@commandr.command('eval')
def build_eval_report(filename='out/eval.json', output_filename='out/out_eval.csv'):
    data = json.load(open(filename))
    content = ''
    keys = [ACCURACY, PRECISION, RECALL, F1_SCORE, 'test_score']

    line = ''
    for i in range(3):
        for key in keys:
            line += '{},'.format(key)
        line += ','
    content += line + '\n'

    for reses in zip(data[TRAIN], data[DEV], data[TEST]):
        line = ''
        for res in reses:
            for key in keys:
                line += '{},'.format(res[key])
            line += ','
        content += line + '\n'

    open(output_filename, 'w').write(content)


if __name__ == '__main__':
    commandr.Run()
