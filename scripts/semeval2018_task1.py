# -*- coding: utf-8 -*-
from __future__ import print_function
import commandr
import json
from collections import defaultdict
from dataset.semeval2018_task1.config import config
from dataset.common.const import *
from dataset.semeval2018_task1.process import Processor
from nlp.process import naive_tokenize


@commandr.command
def build_text_label():
    path_list = {
        TRAIN: config.path_E_c_train,
        TEST: config.path_E_c_dev
    }
    for key, path in path_list.items():
        text_path = config.path(key, TEXT)
        label_path = config.path(key, LABEL)
        with open(text_path, 'w') as text_obj, open(label_path, 'w') as label_obj:
            for label, text in Processor.load_origin(path):
                text_obj.write(text + '\n')
                label_obj.write(str(label) + '\n')


@commandr.command
def build_vocab(version='v0'):
    all_token = set()

    for key in [TRAIN, TEST]:
        text_path = config.path(key, TEXT)
        output_path = config.path(key, VOCAB, version)
        print(output_path)

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

        all_token |= set(token_list)

    output_path = config.path(ALL, VOCAB, version)
    with open(output_path, 'w') as file_obj:
        for token in all_token:
            file_obj.write('{}\n'.format(token.encode('utf8')))
    print(output_path)


if __name__ == '__main__':
    commandr.Run()
