# -*- coding: utf-8 -*-
import re
import commandr
import json
from collections import defaultdict
from nlp.process import naive_tokenize
from dataset.common.const import *
from dataset.semeval2018.task3.process import Processor
from dataset.semeval2018.task3.config import config


@commandr.command
def check_data():
    for sA, sB in zip(Processor.load_origin_train('A'), Processor.load_origin_train('B')):
        if sA[1] != sB[1]:
            print sA[1]
            print sB[1]
        if sA[0] != (0 if sB[0] == 0 else 1):
            print sA[0], sB[0]

    for sA, sB in zip(Processor.load_origin_test('A'), Processor.load_origin_test('B')):
        if sA[1] != sB[1]:
            print sA[1]
            print sB[1]
        if sA[0] != (0 if sB[0] == 0 else 1):
            print sA[0], sB[0]


@commandr.command
def build_text_label():
    for key, func in {TRAIN: Processor.load_origin_train, TEST: Processor.load_origin_test}.items():
        text_path = config.path(key, TEXT)
        label_path = config.path(key, LABEL)
        with open(text_path, 'w') as text_obj, open(label_path, 'w') as label_obj:
            for label, text in func('B'):
                text = re.sub('\s+', ' ', text)
                text_obj.write(text + '\n')
                label_obj.write(str(label) + '\n')


@commandr.command
def build_vocab(version='v0'):
    for key in [TRAIN, TEST]:
        text_path = config.path(key, TEXT)
        output_path = config.path(key, VOCAB, version)

        tf = defaultdict(lambda: 0)
        df = defaultdict(lambda: 0)

        with open(text_path, 'r') as file_obj:
            for line in file_obj:
                line = line.strip()
                if line == '':
                    continue
                tokens = naive_tokenize(line)
                for token in tokens:
                    tf[token] += 1
                for token in set(tokens):
                    df[token] += 1

        token_list = tf.keys()
        token_list = sorted(token_list, key=lambda _t: (-df[_t], -tf[_t]))
        with open(output_path, 'w') as file_obj:
            for token in token_list:
                data = {'t': token, 'tf': tf[token], 'df': df[token]}
                file_obj.write('{}\n'.format(json.dumps(data)))


@commandr.command
def full_vocab(version='v0'):
    output_path = config.path(ALL, VOCAB, version)

    tf = defaultdict(lambda: 0)
    df = defaultdict(lambda: 0)

    for key in [TRAIN, TEST]:
        path = config.path(key, VOCAB, version)
        with open(path, 'r') as file_obj:
            for line in file_obj:
                line = line.strip()
                if line == '':
                    continue
                data = json.loads(line)
                _t = data['t']
                _tf = data['tf']
                _df = data['df']
                tf[_t] += _tf
                df[_t] += _df

    token_list = tf.keys()
    token_list = sorted(token_list, key=lambda _t: (-df[_t], -tf[_t]))
    with open(output_path, 'w') as file_obj:
        for token in token_list:
            # data = {'t': token, 'tf': tf[token], 'df': df[token]}
            # file_obj.write('{}\n'.format(json.dumps(data)))
            file_obj.write('{}\n'.format(token.encode('utf8')))


if __name__ == '__main__':
    commandr.Run()
